"""Iteration 13: relation-faithful generation (see docs/ITERATION_13_PROTOCOL.md).

  python scripts/iteration13.py prepare             # freeze CBC splits (reference / development pool / confirmation pool)
  python scripts/iteration13.py develop --workers 5 # development samples only; writes configs/iteration13_choice.json
  python scripts/iteration13.py confirm --workers 5 # confirmation samples evaluated on the reference half
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import pandas as pd

import iteration12 as it12

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("MHD_OUT", ROOT / "runs" / "iteration13"))
SPLITS = ROOT / "configs" / "iteration13_splits.json"
CHOICE = ROOT / "configs" / "iteration13_choice.json"
DATASET = "nhanes_cbc"
DEV_SIZES, DEV_SEEDS = (250, 500), (0, 1, 2)
SIZES, REPLICATES = (250, 1000), (0, 1, 2, 3, 4)
GUIDANCE_GRID = (0., 1., 2., 5., 10.)
MULTIPLIER, EVALUATOR_SEED = 5, 2026
MHD = {**it12.MHD}
LITERATURE = ("tabdiff", "tabddpm_upstream", "ctgan", "tvae", "copula")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False, default=float) + "\n")


def load(dataset=DATASET):
    from medhyperdiff.data import SCHEMAS, normalize
    schema = SCHEMAS[dataset]
    frame = normalize(pd.read_csv(ROOT / "data" / "processed" / f"{dataset}.csv",
                                  dtype={c: str for c in (*schema.categorical, schema.target)}), schema)
    return frame, schema


def prepare():
    from sklearn.model_selection import StratifiedGroupKFold
    assert not SPLITS.exists(), "Splits are frozen"
    frame, schema = load()
    y = (frame[schema.target] == schema.positive).astype(int).to_numpy()
    groups = pd.util.hash_pandas_object(frame[schema.features], index=False).to_numpy()
    pool, reference = next(StratifiedGroupKFold(2, shuffle=True, random_state=2027).split(frame, y, groups))
    rng = np.random.default_rng(13001)
    development = np.sort(rng.choice(pool, int(round(.3 * len(pool))), replace=False))
    confirmation = np.setdiff1d(pool, development)
    def draw(source, size, seed):
        rng = np.random.default_rng(seed)
        chosen = np.sort(np.concatenate([rng.choice(source[y[source] == label], int(round(size * (y[source] == label).mean())), replace=False)
                                         for label in (0, 1)]))
        fit_local, val_local = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed % 1000).split(chosen, y[chosen], groups[chosen]))
        return {"fit": chosen[fit_local].tolist(), "validation": chosen[val_local].tolist()}
    samples = {f"dev_n{n}_s{s}": draw(development, n, 13100 + 10 * n + s) for n in DEV_SIZES for s in DEV_SEEDS}
    samples.update({f"n{n}_r{r}": draw(confirmation, n, 13500 + 10 * n + r) for n in SIZES for r in REPLICATES})
    for key, sample in samples.items():
        rows = set(sample["fit"] + sample["validation"])
        holdout = set(development) - rows if key.startswith("dev") else set(reference)
        sample["evaluation"] = sorted(holdout)
    assert not set(groups[pool]) & set(groups[reference])
    dump(SPLITS, {"created_at": datetime.now(timezone.utc).isoformat(), "data_sha256": hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest(),
                  "reference": reference.tolist(), "development_pool": development.tolist(), "confirmation_pool": confirmation.tolist(),
                  "samples": samples})
    print(json.dumps({"rows": len(frame), "reference": len(reference), "development": len(development), "confirmation": len(confirmation)}))


def rounder_for(fit, schema):
    from medhyperdiff.preprocessing import precision_bands, round_banded
    bands = {c: precision_bands(fit[c]) for c in schema.numeric}
    return lambda name, values: round_banded(values, bands[name]) if name in bands else values


def relation_variants(raw, fit, schema, coder, seed):
    """Post-hoc knowledge-equivalent processing of any raw output: hard derivation and derivation + resampled residuals."""
    from medhyperdiff.relations import residual_name
    rounder = rounder_for(fit, schema)
    out = {}
    fit_encoded = coder.encode(fit)
    rng = np.random.default_rng(seed)
    for mode in ("derived", "resampled"):
        encoded = coder.encode(raw)
        for relation in coder.relations:
            column = residual_name(relation)
            if mode == "derived":
                encoded[column] = np.where(encoded[column].notna() | raw[relation.derived].notna(), 0., np.nan)
            else:
                pool = fit_encoded[column].dropna().to_numpy()
                encoded[column] = np.where(encoded[column].notna() | raw[relation.derived].notna(), rng.choice(pool, len(encoded)), np.nan)
        out[mode] = coder.decode(encoded, rounder)
    return out


def build_mhd(params, schema, fit, coder=None, structure=True):
    from medhyperdiff.constraints import ConstraintSet, Rule, build_constraints
    from medhyperdiff.diffusion import DiffusionConfig, MixedDiffusion
    from medhyperdiff.preprocessing import Preprocessor
    from medhyperdiff.relations import residual_name
    from medhyperdiff.structure import dependency_graph
    p = dict(params)
    p.pop("guidance", None)
    transform = p.pop("numeric_transform")
    if coder is None:
        frame, model_schema, constraints = fit, schema, build_constraints(schema, fit)
    else:
        frame, model_schema = coder.encode(fit), coder.encoded_schema
        rules = [r for r in build_constraints(schema, fit).rules
                 if r.kind in ("lower", "upper") and all(c in model_schema.numeric for c in r.columns)]
        if structure:
            derived_by = {rel.derived: rel for rel in coder.relations}
            def expand(column):
                # A reconstructed operand is represented by its residual and, recursively, its own operands.
                if column not in derived_by:
                    return [column]
                rel = derived_by[column]
                return [residual_name(rel)] + [c for op in rel.operands for c in expand(op)]
            for rel in coder.relations:
                members = list(dict.fromkeys([residual_name(rel)] + [c for op in rel.operands for c in expand(op)]))
                rules.append(Rule(f"{rel.name}_structure", "structure", tuple(members), status="structure"))
        constraints = ConstraintSet(rules)
    prep = Preprocessor(model_schema, transform, False).fit(frame)
    graph, _ = dependency_graph(frame, prep, seed=2026, bootstraps=10)
    model = MixedDiffusion(prep, graph, constraints, DiffusionConfig(**p, guidance=0., schedule="none"))
    return model, frame


def relation_w1(table, reference, fit, schema, constraints):
    from medhyperdiff.evaluation import rule_realism
    realism = rule_realism(fit, reference, table, constraints)
    relational = {k: v for k, v in realism["rules"].items() if v.get("residual_w1_fit_sd") is not None}
    return float(np.mean([v["residual_w1_fit_sd"] for v in relational.values()])), realism


def run_job(job):
    import torch
    torch.set_num_threads(2)
    from medhyperdiff.cli import source_hash
    from medhyperdiff.constraints import build_constraints
    from medhyperdiff.diffusion import guidance_schedule
    from medhyperdiff.evaluation import c2st, density_ratio_attack, evaluate
    from medhyperdiff.relations import RELATIONS, RelationCoder
    out = OUT / job["id"]
    if (out / "manifest.json").exists() and json.loads((out / "manifest.json").read_text()).get("status") == "complete":
        return job["id"], "skipped"
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    try:
        frame, schema = load()
        plan = json.loads(SPLITS.read_text())
        sample = plan["samples"][job["sample"]]
        fit, validation = frame.iloc[sample["fit"]], frame.iloc[sample["validation"]]
        evaluation = frame.iloc[sample["evaluation"]]
        constraints = build_constraints(schema, fit)
        coder = RelationCoder(schema, RELATIONS[DATASET]).fit(fit)  # conditioning-based targets, fit rows only
        labels = np.tile((fit[schema.target] == schema.positive).astype(int).to_numpy(), MULTIPLIER)
        negative = next(v for v in fit[schema.target].unique() if v != schema.positive)
        variant, seed, extra = job["variant"], job["seed"], {}
        rounder = rounder_for(fit, schema)
        t0 = time.perf_counter()
        def finish(table):
            table[schema.target] = np.where(labels[:len(table)] == 1, schema.positive, negative)
            return table
        if variant in ("mhd_guided", "mhd_unguided"):
            model, _ = build_mhd({**MHD, "seed": seed}, schema, fit)
            model.fit(fit, validation)
            fit_seconds = time.perf_counter() - t0
            grid = GUIDANCE_GRID if variant == "mhd_guided" else (0.,)
            calibration = []
            val_labels = (fit[schema.target] == schema.positive).astype(int).to_numpy()
            for g in grid:
                model.config.guidance, model.config.schedule = g, ("adaptive" if g else "none")
                model.schedule = guidance_schedule(model.config.steps, model.config.schedule)
                trial = it12.decode(model.sample(val_labels, seed=90001 + seed), fit, schema, "banded")
                trial[schema.target] = np.where(val_labels == 1, schema.positive, negative)
                score, _ = relation_w1(trial, validation, fit, schema, constraints)
                calibration.append({"guidance": g, "validation_relation_w1": score})
            chosen = min(calibration, key=lambda r: (r["validation_relation_w1"], r["guidance"]))["guidance"]
            model.config.guidance, model.config.schedule = chosen, ("adaptive" if chosen else "none")
            model.schedule = guidance_schedule(model.config.steps, model.config.schedule)
            raw = model.sample(labels, seed=70001 + seed)
            primary = finish(it12.decode(raw, fit, schema, "banded"))
            extra.update(guidance=chosen, calibration=calibration)
        elif variant in ("mhd_reparam", "mhd_reparam_backbone"):
            structure = variant == "mhd_reparam"
            params = {**MHD, "seed": seed}
            if not structure:
                params.update(graph_enabled=False, hypergraph_enabled=False)
            model, _ = build_mhd(params, schema, fit, coder, structure=structure)
            model.fit(coder.encode(fit), coder.encode(validation))
            fit_seconds = time.perf_counter() - t0
            encoded = model.sample(labels, seed=70001 + seed)
            encoded[schema.target] = np.where(labels == 1, schema.positive, negative)
            raw = coder.decode(encoded, lambda n, v: v)
            primary = finish(coder.decode(coder.encode(raw), rounder))
        elif variant == "tabdiff_reparam":
            from medhyperdiff.diffusion import DiffusionConfig
            from medhyperdiff.external_baselines import TabDiffAdapter
            from medhyperdiff.preprocessing import Preprocessor
            encoded_fit = coder.encode(fit)
            model = TabDiffAdapter(Preprocessor(coder.encoded_schema, "affine", False).fit(encoded_fit), DiffusionConfig(seed=seed)).fit(encoded_fit)
            fit_seconds = time.perf_counter() - t0
            encoded = model.sample(labels, seed=70001 + seed)
            encoded[schema.target] = np.where(labels == 1, schema.positive, negative)
            raw = coder.decode(encoded, lambda n, v: v)
            primary = finish(coder.decode(coder.encode(raw), rounder))
        else:
            method, params = it12.VARIANTS[variant]
            generator = it12.build(method, params, schema, fit, seed)
            generator.fit(fit, validation) if method in ("mhd", "tabddpm") else generator.fit(fit)
            fit_seconds = time.perf_counter() - t0
            raw = generator.sample(labels, seed=70001 + seed)
            primary = finish(it12.decode(raw, fit, schema, "banded"))
        raw[schema.target] = np.where(labels == 1, schema.positive, negative)
        tables = {"primary": primary}
        if not variant.startswith(("mhd", "tabdiff_reparam")) or variant == "mhd_unguided":
            for mode, table in relation_variants(raw, fit, schema, coder, seed).items():
                tables[mode] = finish(table)
        t0 = time.perf_counter()
        metrics = {}
        for name, table in tables.items():
            table.to_csv(out / f"synthetic_{name}.csv", index=False)
            table = pd.read_csv(out / f"synthetic_{name}.csv", dtype={c: str for c in (*schema.categorical, schema.target)}, float_precision="round_trip")
            score, realism = relation_w1(table, evaluation, fit, schema, constraints)
            entry = {"relation_w1": score, "realism": realism, "c2st": c2st(evaluation, table, schema, EVALUATOR_SEED)["auc"]}
            if name == "primary" or job["stage"] == "confirm":
                full = evaluate(fit, evaluation, table, schema, constraints, EVALUATOR_SEED, {})
                rng = np.random.default_rng(EVALUATOR_SEED + seed)
                index = rng.choice(len(evaluation), min(len(fit), len(evaluation) // 2), replace=False)
                entry.update(fidelity=full["fidelity"], utility=full["utility"], privacy=full["privacy"],
                             density_attack=density_ratio_attack(fit, evaluation.iloc[index], table,
                                                                 evaluation.drop(evaluation.index[index]), schema, seed=EVALUATOR_SEED))
            metrics[name] = entry
        dump(out / "metrics.json", metrics)
        dump(out / "manifest.json", {**job, "status": "complete", "started_at": started, "completed_at": datetime.now(timezone.utc).isoformat(),
             "fit_rows": len(fit), "evaluation_rows": len(evaluation), "fit_seconds": fit_seconds, "evaluation_seconds": time.perf_counter() - t0,
             "source_sha256": source_hash(), "runner_sha256": sha(__file__), "splits_sha256": sha(SPLITS),
             "python": platform.python_version(), "torch": torch.__version__, "derivations": coder.describe(), **extra})
        return job["id"], "complete"
    except Exception as exc:
        dump(out / "manifest.json", {**job, "status": "failed", "started_at": started, "error": f"{type(exc).__name__}: {exc}",
                                     "traceback": traceback.format_exc()})
        return job["id"], f"failed: {type(exc).__name__}: {exc}"


DEV_VARIANTS = ("mhd_unguided", "mhd_guided", "mhd_reparam", "mhd_reparam_backbone")


def development_jobs():
    return [dict(id=f"develop/{v}/n{n}_s{s}", stage="develop", variant=v, sample=f"dev_n{n}_s{s}", size=n, seed=s)
            for v in DEV_VARIANTS for n in DEV_SIZES for s in DEV_SEEDS]


def confirmation_jobs():
    choice = json.loads(CHOICE.read_text())["primary"]
    variants = sorted({choice, "mhd_unguided", "tabdiff_reparam", *LITERATURE, "bootstrap"})
    return [dict(id=f"confirm/{v}/n{n}_r{r}", stage="confirm", variant=v, sample=f"n{n}_r{r}", size=n, seed=r,
                 choice_sha256=sha(CHOICE)) for v in variants for n in SIZES for r in REPLICATES]


def choose():
    """Predeclared rule: lowest mean development relation W1 among variants whose mean development C2ST
    is within 0.02 of the lowest; ties go to the simpler variant (order in DEV_VARIANTS)."""
    assert not CHOICE.exists()
    rows = []
    for job in development_jobs():
        m = json.loads((OUT / job["id"] / "metrics.json").read_text())["primary"]
        rows.append({"variant": job["variant"], "size": job["size"], "seed": job["seed"], "relation_w1": m["relation_w1"], "c2st": m["c2st"],
                     "membership": max(m["privacy"]["nearest_synthetic_membership_auc"], m["density_attack"]["auc"])})
    table = pd.DataFrame(rows)
    means = table.groupby("variant")[["relation_w1", "c2st", "membership"]].mean()
    eligible = means[means.c2st <= means.c2st.min() + .02]
    order = {v: i for i, v in enumerate(DEV_VARIANTS)}
    primary = min(eligible.index, key=lambda v: (round(eligible.at[v, "relation_w1"], 6), order[v]))
    dump(CHOICE, {"created_at": datetime.now(timezone.utc).isoformat(), "primary": primary,
                  "rule": choose.__doc__.strip(), "means": means.reset_index().to_dict("records"), "development": rows})
    print(means.round(4).to_string(), "\nprimary:", primary)


def execute(jobs, workers):
    pending = [j for j in jobs if not ((OUT / j["id"] / "manifest.json").exists() and
               json.loads((OUT / j["id"] / "manifest.json").read_text()).get("status") == "complete")]
    cost = {"tabdiff": 40, "tabdiff_reparam": 40, "tabddpm_upstream": 30, "ctgan": 8, "tvae": 4}
    pending.sort(key=lambda j: cost.get(j["variant"], 5) * j["size"], reverse=True)
    print(json.dumps(dict(total=len(jobs), pending=len(pending), workers=workers, out=str(OUT))), flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_job, j): j["id"] for j in pending}
        for done, future in enumerate(as_completed(futures), 1):
            identifier, status = future.result()
            print(json.dumps(dict(done=done, of=len(pending), id=identifier, status=status, at=datetime.now(timezone.utc).isoformat())), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "develop", "choose", "confirm"))
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "develop":
        execute(development_jobs(), args.workers)
    elif args.stage == "choose":
        choose()
    else:
        assert CHOICE.exists(), "Run develop and choose first"
        execute(confirmation_jobs(), args.workers)


if __name__ == "__main__":
    main()
