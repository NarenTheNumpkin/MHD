"""Iteration 14: robustness of relation-faithful generation (see docs/ITERATION_14_PROTOCOL.md).

  python scripts/iteration14.py prepare               # build populations, freeze all splits
  python scripts/iteration14.py develop --workers 8   # MHD size-scaling candidates on the 2015-2016 development populations
  python scripts/iteration14.py choose                # predeclared rule; writes configs/iteration14_choice.json
  python scripts/iteration14.py confirm --workers 8 --arms comparators   # published generators, plug-ins, controls
  python scripts/iteration14.py queue --workers 8     # development first, then comparator arms, in one pool
  python scripts/iteration14.py confirm --workers 8 --arms mhd           # requires the choice
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

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import iteration12 as it12
import iteration13 as it13

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("MHD_OUT", ROOT / "runs" / "iteration14"))
SPLITS = ROOT / "configs" / "iteration14_splits.json"
CHOICE = ROOT / "configs" / "iteration14_choice.json"
DEVELOPMENT = {"nhanes_cbc_i": "cbc", "nhanes_chem_i": "chem"}
CONFIRMATION = {"nhanes_cbc_h": "cbc", "nhanes_cbc_l": "cbc", "nhanes_chem_g": "chem", "nhanes_chem_l": "chem"}
PANELS = {**DEVELOPMENT, **CONFIRMATION, "nhanes_cbc": "cbc"}
DEV_SIZES, DEV_SEEDS = (250, 1000, 2000), (0, 1, 2)
# Amendment 1 (10 October 2026): n = 100 removed from confirmation to shorten the run; see the protocol.
SIZES, REPLICATES = (250, 1000), (0, 1, 2, 3, 4)
LARGE_SIZE, LARGE_REPLICATES, LARGE_POPULATIONS = 2000, (0, 1, 2), ("nhanes_cbc_l", "nhanes_chem_l")
SECONDARY_SIZES = (250, 1000)  # iteration 13 population (nhanes_cbc), reference half already examined
CANDIDATES = ("c0", "c1", "c2", "c3")
PUBLISHED = ("tabdiff", "tabddpm_upstream", "ctgan", "tvae", "copula")
MULTIPLIER, EVALUATOR_SEED = 5, 2026
COST = {"tabdiff": 40, "tabddpm_upstream": 12, "ctgan": 3, "tvae": 2, "mhd": 6}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False, default=float) + "\n")


def load(dataset):
    return it13.load(dataset)


def mhd_params(candidate, size):
    """Size-scaling candidates for the reparameterized MedHyperDiff (fixed before development).

    c0: iteration 13 configuration (5,000 steps, width 32) at every size.
    c1: optimizer steps proportional to sample size, i.e. a constant number of passes over the fit rows.
    c2: c1 with width 64 at every size.
    c3: c1 with width growing with the square root of sample size (32 at n <= 250, 64 at 1,000, 88 at 2,000).
    """
    params = dict(it12.MHD)
    scale = max(1., size / 250)
    if candidate in ("c1", "c2", "c3"):
        params["train_steps"] = int(round(5000 * scale))
    if candidate == "c2":
        params["width"] = 64
    if candidate == "c3":
        params["width"] = int(8 * round(32 * np.sqrt(scale) / 8))
    return params


def config_key(candidate, size):
    p = mhd_params(candidate, size)
    return f"s{p['train_steps']}_w{p['width']}"


def prepare():
    from sklearn.model_selection import StratifiedGroupKFold
    from medhyperdiff.data import build_nhanes_cycle
    assert not SPLITS.exists(), "Splits are frozen"
    plan = {"created_at": datetime.now(timezone.utc).isoformat(), "datasets": {}}
    for index, (dataset, panel) in enumerate({**DEVELOPMENT, **CONFIRMATION}.items()):
        build_nhanes_cycle(ROOT / "data", panel, dataset.rsplit("_", 1)[1])
        frame, schema = load(dataset)
        y = (frame[schema.target] == schema.positive).astype(int).to_numpy()
        groups = pd.util.hash_pandas_object(frame[schema.features], index=False).to_numpy()
        pool, reference = next(StratifiedGroupKFold(2, shuffle=True, random_state=14000 + index).split(frame, y, groups))
        assert not set(groups[pool]) & set(groups[reference])

        def draw(size, seed):
            rng = np.random.default_rng(seed)
            chosen = np.sort(np.concatenate([rng.choice(pool[y[pool] == label], int(round(size * (y[pool] == label).mean())), replace=False)
                                             for label in (0, 1)]))
            fit_local, val_local = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed % 1000).split(chosen, y[chosen], groups[chosen]))
            return {"fit": chosen[fit_local].tolist(), "validation": chosen[val_local].tolist()}
        base = 140000 + 1000 * index
        if dataset in DEVELOPMENT:
            samples = {f"dev_n{n}_s{s}": draw(n, base + 10 * (n // 50) + s) for n in DEV_SIZES for s in DEV_SEEDS}
        else:
            samples = {f"n{n}_r{r}": draw(n, base + 10 * (n // 50) + r) for n in SIZES for r in REPLICATES}
            if dataset in LARGE_POPULATIONS:
                samples.update({f"n{LARGE_SIZE}_r{r}": draw(LARGE_SIZE, base + 10 * (LARGE_SIZE // 50) + r) for r in LARGE_REPLICATES})
        plan["datasets"][dataset] = {"panel": panel, "role": "development" if dataset in DEVELOPMENT else "confirmation",
                                     "rows": len(frame), "data_sha256": hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest(),
                                     "reference": reference.tolist(), "pool": pool.tolist(), "samples": samples}
        print(json.dumps({"dataset": dataset, "rows": len(frame), "reference": len(reference), "pool": len(pool),
                          "prevalence": round(float(y.mean()), 4)}), flush=True)
    dump(SPLITS, plan)


def relation_metrics(table, reference, fit, schema, constraints, coder):
    """Iteration 13 relation distance plus the iteration 14 version whose per-relation scale is floored at the SD of
    the absolute rounding error of the derived value (needed when a relation is exact, as NHANES globulin is)."""
    from scipy.stats import wasserstein_distance
    from medhyperdiff.evaluation import rule_realism, rule_residuals
    from medhyperdiff.preprocessing import precision_bands
    realism = rule_realism(fit, reference, table, constraints)
    relational = [r for r in constraints.rules if len(r.columns) >= 2]
    per = {}
    for rule in relational:
        fit_res, real_res, syn_res = (rule_residuals(f, rule, constraints) for f in (fit, reference, table))
        decimals = precision_bands(pd.to_numeric(fit[rule.columns[0]], errors="coerce").to_numpy(float))["global"]
        floor = 10.0 ** -decimals / np.sqrt(48) if decimals is not None else 1e-8
        scale = max(float(np.std(fit_res)) if len(fit_res) > 1 else 0., floor)
        per[rule.name] = float(wasserstein_distance(real_res, syn_res) / scale) if len(real_res) and len(syn_res) else None
    values = [v for v in per.values() if v is not None]
    legacy = [v["residual_w1_fit_sd"] for v in realism["rules"].values() if v.get("residual_w1_fit_sd") is not None]
    return {"relation_w1": float(np.mean(values)), "relation_w1_by_rule": per,
            "relation_w1_iteration13": float(np.mean(legacy)), "realism": realism}


def fit_generator(variant, schema, fit, validation, coder, size, seed, labels, negative):
    """Returns (raw table in the original schema, primary decoded table or None, fit seconds, extra manifest fields)."""
    t0 = time.perf_counter()
    rounder = it13.rounder_for(fit, schema)
    if variant.startswith("mhd:"):
        candidate = variant.split(":", 1)[1]
        params = {**mhd_params(candidate, size), "seed": seed}
        model, _ = it13.build_mhd(params, schema, fit, coder, structure=True)
        model.fit(coder.encode(fit), coder.encode(validation))
        seconds = time.perf_counter() - t0
        encoded = model.sample(labels, seed=70001 + seed)
        encoded[schema.target] = np.where(labels == 1, schema.positive, negative)
        # Decode once with recorded-resolution rounding (operands first, then reconstruction and re-derivation).
        return coder.decode(encoded, lambda n, v: v), coder.decode(encoded, rounder), seconds, {"params": params}
    if variant.endswith("_rp"):
        method, params = it12.VARIANTS[variant[:-3]]
        encoded_fit = coder.encode(fit)
        generator = it12.build(method, params, coder.encoded_schema, encoded_fit, seed)
        generator.fit(encoded_fit, coder.encode(validation)) if method in ("mhd", "tabddpm") else generator.fit(encoded_fit)
        seconds = time.perf_counter() - t0
        encoded = generator.sample(labels, seed=70001 + seed)
        encoded[schema.target] = np.where(labels == 1, schema.positive, negative)
        return coder.decode(encoded, lambda n, v: v), coder.decode(encoded, rounder), seconds, {}
    method, params = it12.VARIANTS[variant]
    generator = it12.build(method, params, schema, fit, seed)
    generator.fit(fit, validation) if method in ("mhd", "tabddpm") else generator.fit(fit)
    seconds = time.perf_counter() - t0
    raw = generator.sample(labels, seed=70001 + seed)
    return raw, it12.decode(raw, fit, schema, "banded"), seconds, {}


def run_job(job):
    import torch
    torch.set_num_threads(1)
    from medhyperdiff.cli import source_hash
    from medhyperdiff.constraints import build_constraints
    from medhyperdiff.evaluation import c2st, clinical_category_tv, density_ratio_attack, evaluate
    from medhyperdiff.relations import RelationCoder, relations_for
    out = OUT / job["id"]
    if (out / "manifest.json").exists() and json.loads((out / "manifest.json").read_text()).get("status") == "complete":
        return job["id"], "skipped"
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    try:
        dataset = job["dataset"]
        frame, schema = load(dataset)
        if dataset == "nhanes_cbc":
            plan = json.loads(it13.SPLITS.read_text())
            sample, reference_index = plan["samples"][job["sample"]], plan["reference"]
            splits_file = it13.SPLITS
        else:
            plan = json.loads(SPLITS.read_text())["datasets"][dataset]
            assert plan["data_sha256"] == hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
            sample, reference_index, splits_file = plan["samples"][job["sample"]], plan["reference"], SPLITS
        fit, validation = frame.iloc[sample["fit"]], frame.iloc[sample["validation"]]
        evaluation = frame.iloc[reference_index]
        constraints = build_constraints(schema, fit)
        coder = RelationCoder(schema, relations_for(dataset)).fit(fit)
        labels = np.tile((fit[schema.target] == schema.positive).astype(int).to_numpy(), MULTIPLIER)
        negative = next(v for v in fit[schema.target].unique() if v != schema.positive)
        variant, seed = job["variant"], job["seed"]

        def finish(table):
            table = table.copy()
            table[schema.target] = np.where(labels[:len(table)] == 1, schema.positive, negative)
            return table
        raw, primary, fit_seconds, extra = fit_generator(variant, schema, fit, validation, coder, job["size"], seed, labels, negative)
        raw[schema.target] = np.where(labels == 1, schema.positive, negative)
        tables = {"primary": finish(primary)}
        if variant in PUBLISHED or variant == "bootstrap":
            for mode, table in it13.relation_variants(raw, fit, schema, coder, seed).items():
                tables[mode] = finish(table)
        t0 = time.perf_counter()
        metrics = {}
        for name, table in tables.items():
            table.to_csv(out / f"synthetic_{name}.csv", index=False)
            table = pd.read_csv(out / f"synthetic_{name}.csv", dtype={c: str for c in (*schema.categorical, schema.target)}, float_precision="round_trip")
            entry = relation_metrics(table, evaluation, fit, schema, constraints, coder)
            entry["c2st"] = c2st(evaluation, table, schema, EVALUATOR_SEED)["auc"]
            entry["clinical_category_tv"] = clinical_category_tv(evaluation, table, PANELS[dataset])
            if name == "primary" or job["stage"] != "develop":
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
             "source_sha256": source_hash(), "runner_sha256": sha(__file__), "splits_sha256": sha(splits_file),
             "python": platform.python_version(), "torch": torch.__version__, "derivations": coder.describe(), **extra})
        return job["id"], "complete"
    except Exception as exc:
        dump(out / "manifest.json", {**job, "status": "failed", "started_at": started, "error": f"{type(exc).__name__}: {exc}",
                                     "traceback": traceback.format_exc()})
        return job["id"], f"failed: {type(exc).__name__}: {exc}"


def development_jobs():
    jobs = []
    for dataset in DEVELOPMENT:
        for size in DEV_SIZES:
            for seed in DEV_SEEDS:
                keys = {}
                for candidate in CANDIDATES:
                    keys.setdefault(config_key(candidate, size), candidate)
                for key, candidate in keys.items():
                    jobs.append(dict(id=f"develop/{dataset}/mhd_{key}/n{size}_s{seed}", stage="develop", dataset=dataset,
                                     variant=f"mhd:{candidate}", config=key, sample=f"dev_n{size}_s{seed}", size=size, seed=seed))
                if size == 1000:
                    jobs.append(dict(id=f"develop/{dataset}/tabdiff/n{size}_s{seed}", stage="develop", dataset=dataset,
                                     variant="tabdiff", sample=f"dev_n{size}_s{seed}", size=size, seed=seed))
    return jobs


def confirmation_cells():
    for dataset in CONFIRMATION:
        for size in SIZES:
            for replicate in REPLICATES:
                yield dataset, size, replicate
        if dataset in LARGE_POPULATIONS:
            for replicate in LARGE_REPLICATES:
                yield dataset, LARGE_SIZE, replicate


def confirmation_jobs(arms):
    variants = []
    if arms in ("comparators", "all"):
        variants += [*PUBLISHED, *(f"{v}_rp" for v in PUBLISHED), "bootstrap", "mhd:c0"]
    jobs = []
    for dataset, size, replicate in confirmation_cells():
        names = list(variants)
        if arms in ("mhd", "all"):
            primary = json.loads(CHOICE.read_text())["primary"]
            names += [f"mhd:{primary}"] if primary != "c0" else []
        for variant in dict.fromkeys(names):
            jobs.append(dict(id=f"confirm/{dataset}/{variant.replace(':', '_')}/n{size}_r{replicate}", stage="confirm", dataset=dataset,
                             variant=variant, sample=f"n{size}_r{replicate}", size=size, seed=replicate))
    if arms in ("mhd", "all"):
        primary = json.loads(CHOICE.read_text())["primary"]
        for size in SECONDARY_SIZES:
            for replicate in it13.REPLICATES:
                jobs.append(dict(id=f"secondary/nhanes_cbc/mhd_{primary}/n{size}_r{replicate}", stage="secondary", dataset="nhanes_cbc",
                                 variant=f"mhd:{primary}", sample=f"n{size}_r{replicate}", size=size, seed=replicate))
    return jobs


def choose():
    """Predeclared rule: the candidate with the lowest mean development C2ST over both development populations and the
    three development sizes, among candidates whose mean development membership AUROC is at most 0.60 in every
    population x size cell; candidates within 0.002 of the lowest are tied and the earliest in CANDIDATES wins."""
    assert not CHOICE.exists()
    rows = []
    for job in development_jobs():
        if job["variant"] == "tabdiff":
            continue
        m = json.loads((OUT / job["id"] / "metrics.json").read_text())["primary"]
        record = {"dataset": job["dataset"], "size": job["size"], "seed": job["seed"], "config": job["config"], "relation_w1": m["relation_w1"],
                  "c2st": m["c2st"], "membership": max(m["privacy"]["nearest_synthetic_membership_auc"], m["density_attack"]["auc"]),
                  "forest_tstr": m["utility"]["forest"]["TSTR"]["auroc"]}
        for candidate in CANDIDATES:
            if config_key(candidate, job["size"]) == job["config"]:
                rows.append({**record, "candidate": candidate})
    table = pd.DataFrame(rows)
    cells = table.groupby(["candidate", "dataset", "size"])[["c2st", "membership", "relation_w1", "forest_tstr"]].mean()
    means = cells.groupby("candidate").mean()
    eligible = [c for c in CANDIDATES if (cells.loc[c, "membership"] <= .60).all()]
    best = min(means.loc[eligible, "c2st"])
    primary = next(c for c in CANDIDATES if c in eligible and means.at[c, "c2st"] <= best + .002)
    dump(CHOICE, {"created_at": datetime.now(timezone.utc).isoformat(), "primary": primary, "rule": choose.__doc__.strip(),
                  "means": means.reset_index().to_dict("records"), "cells": cells.reset_index().to_dict("records"), "development": rows})
    print(cells.round(4).to_string(), "\n", means.round(4).to_string(), "\nprimary:", primary)


def execute(jobs, workers, keep_order=False):
    pending = [j for j in jobs if not ((OUT / j["id"] / "manifest.json").exists() and
               json.loads((OUT / j["id"] / "manifest.json").read_text()).get("status") == "complete")]
    def cost(job):
        variant = job["variant"].split(":")[0].replace("_rp", "")
        weight = COST.get(variant, 1)
        if variant == "mhd":
            weight *= mhd_params(job["variant"].split(":")[1], job["size"])["train_steps"] / 5000
        return weight * job["size"]
    if keep_order:  # stable: blocks keep their order, each block sorted by cost
        rank = {id(j): i for i, j in enumerate(pending)}
        pending.sort(key=lambda j: (j["stage"] != "develop", -cost(j), rank[id(j)]))
    else:
        pending.sort(key=cost, reverse=True)
    print(json.dumps(dict(total=len(jobs), pending=len(pending), workers=workers, out=str(OUT))), flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_job, j): j["id"] for j in pending}
        for done, future in enumerate(as_completed(futures), 1):
            identifier, status = future.result()
            print(json.dumps(dict(done=done, of=len(pending), id=identifier, status=status, at=datetime.now(timezone.utc).isoformat())), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "develop", "choose", "confirm", "queue"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--arms", choices=("comparators", "mhd", "all"), default="all")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "develop":
        execute(development_jobs(), args.workers)
    elif args.stage == "choose":
        choose()
    elif args.stage == "queue":
        execute(development_jobs() + confirmation_jobs("comparators"), args.workers, keep_order=True)
    else:
        if args.arms != "comparators":
            assert CHOICE.exists(), "Run develop and choose first"
        execute(confirmation_jobs(args.arms), args.workers)


if __name__ == "__main__":
    main()
