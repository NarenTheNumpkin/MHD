"""Iteration 12: controlled study of training and evaluation pitfalls (see docs/ITERATION_12_PROTOCOL.md).

  python scripts/iteration12.py prepare            # freeze population splits and training samples
  python scripts/iteration12.py run --workers 5    # fit, sample and evaluate every job (resumable)

Outputs go to runs/iteration12 (override with MHD_OUT). Every manifest records source, runner, data and
split hashes. Nothing here touches the iteration 2-11 reserved test folds.
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

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("MHD_OUT", ROOT / "runs" / "iteration12"))
SPLITS = ROOT / "configs" / "iteration12_splits.json"
DATASETS = ("nhanes", "thyroid")
SIZES = (250, 1000)
REPLICATES = (0, 1, 2, 3, 4)
MULTIPLIER = 5
EVALUATOR_SEED = 2026
CURVE_SIZES = (50, 100, 200, 500, 1000)
CURVE_REPEATS = 10

MHD = dict(training="steps", train_steps=5000, numeric_transform="quantile", steps=100, width=32, layers=2, heads=4,
           batch_size=64, learning_rate=1e-3, numeric_prediction="v", clinical_weight=.01, guidance=0.,
           paired_initialization=True, observation_prior=True)
TABDDPM = dict(learning_rate=0.00016780238571462852, mlp_layers=(512, 1024), steps=100)
VARIANTS = {
    "mhd_corrected": ("mhd", dict(MHD)),
    "mhd_earlystop": ("mhd", {**MHD, "training": "epochs", "epochs": 100, "patience": 20}),
    "mhd_affine": ("mhd", {**MHD, "numeric_transform": "affine"}),
    "tabddpm_upstream": ("tabddpm", {**TABDDPM, "training": "steps", "train_steps": 30000, "batch_size": 4096, "weight_decay": 0.}),
    "tabddpm_adapter": ("tabddpm", {**TABDDPM, "training": "epochs", "epochs": 300, "patience": 20, "batch_size": 64,
                                    "weight_decay": 1e-4, "width": 128}),
    "tabdiff": ("tabdiff", {}),
    "ctgan": ("ctgan", dict(epochs=300, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                            batch_size=500, pac=10, generator_lr=2e-4, discriminator_lr=2e-4)),
    "tvae": ("tvae", dict(epochs=300, embedding_dim=128, compress_dims=(128, 128), decompress_dims=(128, 128),
                          batch_size=500, l2scale=1e-5, loss_factor=2.)),
    "copula": ("copula", {}), "copula_repair": ("copula_repair", {}),
    "independent": ("independent", {}), "bootstrap": ("bootstrap", {}),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False, default=float) + "\n")


def load(dataset):
    from medhyperdiff.data import SCHEMAS, normalize
    schema = SCHEMAS[dataset]
    frame = normalize(pd.read_csv(ROOT / "data" / "processed" / f"{dataset}.csv", dtype={c: str for c in (*schema.categorical, schema.target)}), schema)
    return frame, schema


def prepare():
    from sklearn.model_selection import StratifiedGroupKFold
    assert not SPLITS.exists(), "Splits are frozen"
    plan = {"created_at": datetime.now(timezone.utc).isoformat(), "datasets": {}}
    for dataset in DATASETS:
        frame, schema = load(dataset)
        y = (frame[schema.target] == schema.positive).astype(int).to_numpy()
        groups = pd.util.hash_pandas_object(frame[schema.features], index=False).to_numpy()
        pool, reference = next(StratifiedGroupKFold(2, shuffle=True, random_state=2026).split(frame, y, groups))
        samples = {}
        for size in SIZES:
            for replicate in REPLICATES:
                rng = np.random.default_rng(12000 + 100 * size + replicate)
                chosen = []
                for label in (0, 1):
                    candidates = pool[y[pool] == label]
                    take = int(round(size * (y[pool] == label).mean()))
                    chosen.append(rng.choice(candidates, take, replace=False))
                chosen = np.sort(np.concatenate(chosen))
                fit_local, val_local = next(StratifiedGroupKFold(5, shuffle=True, random_state=size + replicate).split(
                    chosen, y[chosen], groups[chosen]))
                samples[f"n{size}_r{replicate}"] = {"fit": chosen[fit_local].tolist(), "validation": chosen[val_local].tolist()}
        assert not set(groups[pool]) & set(groups[reference])
        plan["datasets"][dataset] = {"data_sha256": hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest(),
                                     "rows": len(frame), "pool": pool.tolist(), "reference": reference.tolist(), "samples": samples}
    dump(SPLITS, plan)
    print(json.dumps({d: {"pool": len(v["pool"]), "reference": len(v["reference"])} for d, v in plan["datasets"].items()}))


def decode(frame, fit, schema, mode):
    """Post-hoc recorded-precision decoding with resolution detected on fit rows only."""
    from medhyperdiff.preprocessing import Preprocessor, round_banded
    if mode == "raw":
        return frame
    prep = Preprocessor(schema, "affine", mode).fit(fit)
    out = frame.copy()
    for i, col in enumerate(schema.numeric):
        values = out[col].to_numpy(float)
        if mode == "global" and prep.decimals[i] is not None:
            values = np.where(np.isnan(values), values, np.round(values, prep.decimals[i]))
        elif mode == "banded":
            values = round_banded(values, prep.bands[i])
        out[col] = values
    return out


def build(method, params, schema, fit, seed):
    from medhyperdiff.constraints import build_constraints
    from medhyperdiff.diffusion import DiffusionConfig, MixedDiffusion, guidance_schedule
    from medhyperdiff.preprocessing import Preprocessor
    p = dict(params)
    if method == "mhd":
        prep = Preprocessor(schema, p.pop("numeric_transform"), False).fit(fit)
        from medhyperdiff.structure import dependency_graph
        graph, _ = dependency_graph(fit, prep, seed=2026, bootstraps=10)
        guidance = p.pop("guidance")
        model = MixedDiffusion(prep, graph, build_constraints(schema, fit), DiffusionConfig(**p, seed=seed, guidance=0., schedule="none"))
        model.target_guidance = guidance
        return model
    if method == "tabddpm":
        from medhyperdiff.external_baselines import TabDDPMAdapter
        prep = Preprocessor(schema, "affine", False).fit(fit)
        return TabDDPMAdapter(prep, DiffusionConfig(**p, seed=seed))
    if method == "tabdiff":
        from medhyperdiff.external_baselines import TabDiffAdapter
        return TabDiffAdapter(Preprocessor(schema, "affine", False).fit(fit), DiffusionConfig(seed=seed), p)
    if method in ("ctgan", "tvae"):
        from medhyperdiff.baselines import CTGANAdapter
        epochs = p.pop("epochs")
        return CTGANAdapter(schema, method, epochs, seed, p, decode_precision=False)
    from medhyperdiff.baselines import ReferenceGenerator
    return ReferenceGenerator(schema, "copula" if method == "copula_repair" else method, seed, decode_precision=False)


def light_metrics(reference, synthetic, fit, schema):
    from scipy.stats import wasserstein_distance
    from medhyperdiff.evaluation import c2st
    w = {}
    for col in schema.numeric:
        r, s = reference[col].dropna(), synthetic[col].dropna()
        spread = float(fit[col].std())
        w[col] = float(wasserstein_distance(r, s) / (spread if np.isfinite(spread) and spread > 1e-6 else 1)) if len(r) and len(s) else None
    return {"c2st": c2st(reference, synthetic, schema, EVALUATOR_SEED)["auc"], "wasserstein_fit_sd": w}


def run_job(job):
    import torch
    torch.set_num_threads(2)
    from medhyperdiff.cli import source_hash
    from medhyperdiff.constraints import build_constraints, repair_simple
    from medhyperdiff.diffusion import guidance_schedule
    from medhyperdiff.evaluation import evaluate, c2st_curve, density_ratio_attack
    out = OUT / job["id"]
    if (out / "manifest.json").exists() and json.loads((out / "manifest.json").read_text()).get("status") == "complete":
        return job["id"], "skipped"
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    try:
        frame, schema = load(job["dataset"])
        plan = json.loads(SPLITS.read_text())["datasets"][job["dataset"]]
        assert plan["data_sha256"] == hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
        sample = plan["samples"][job["sample"]]
        fit, validation = frame.iloc[sample["fit"]], frame.iloc[sample["validation"]]
        reference = frame.iloc[plan["reference"]]
        method, params, seed = job["method"], job["params"], job["seed"]
        constraints = build_constraints(schema, fit)
        labels = np.tile((fit[schema.target] == schema.positive).astype(int).to_numpy(), MULTIPLIER)
        extra = {}
        t0 = time.perf_counter()
        generator = build(method, params, schema, fit, seed)
        if method in ("mhd", "tabddpm"):
            generator.fit(fit, validation)
            extra.update(selected_epoch=generator.best_epoch, executed=len(generator.history),
                         parameters=sum(p.numel() for p in generator.model.parameters()))
            if method == "mhd" and generator.target_guidance:
                generator.config.guidance, generator.config.schedule = generator.target_guidance, "adaptive"
                generator.schedule = guidance_schedule(generator.config.steps, "adaptive")
        else:
            generator.fit(fit)
        fit_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        raw = generator.sample(labels, seed=70001 + seed)
        negative = next(v for v in fit[schema.target].unique() if v != schema.positive)
        raw[schema.target] = np.where(labels == 1, schema.positive, negative)
        if method == "copula_repair":
            raw, changed = repair_simple(raw, constraints)
            extra["repair_changed_row_fraction"] = changed
        sample_seconds = time.perf_counter() - t0
        tables = {}
        for mode in ("raw", "global", "banded"):
            table = decode(raw, fit, schema, mode)
            table.to_csv(out / f"synthetic_{mode}.csv", index=False)
            tables[mode] = pd.read_csv(out / f"synthetic_{mode}.csv", dtype={c: str for c in (*schema.categorical, schema.target)},
                                       float_precision="round_trip")
        t0 = time.perf_counter()
        predictions = {}
        primary = tables["banded"]
        metrics = evaluate(fit, reference, primary, schema, constraints, EVALUATOR_SEED, predictions)
        pd.DataFrame(predictions).to_csv(out / "utility_predictions.csv", index=False)
        metrics["c2st_curve"] = c2st_curve(reference, primary, schema, CURVE_SIZES, CURVE_REPEATS, seed=EVALUATOR_SEED)
        rng = np.random.default_rng(EVALUATOR_SEED + seed)
        nonmember_index = rng.choice(len(reference), len(fit), replace=False)
        nonmembers = reference.iloc[nonmember_index]
        background = reference.drop(reference.index[nonmember_index])
        metrics["density_attack"] = density_ratio_attack(fit, nonmembers, primary, background, schema, seed=EVALUATOR_SEED)
        metrics["decodings"] = {mode: light_metrics(reference, tables[mode], fit, schema) for mode in ("raw", "global", "banded")}
        dump(out / "metrics.json", metrics)
        if hasattr(generator, "history"):
            dump(out / "history.json", generator.history)
        manifest = {**job, "status": "complete", "started_at": started, "completed_at": datetime.now(timezone.utc).isoformat(),
                    "fit_rows": len(fit), "validation_rows": len(validation), "reference_rows": len(reference),
                    "synthetic_rows": len(raw), "fit_seconds": fit_seconds, "sample_seconds": sample_seconds,
                    "evaluation_seconds": time.perf_counter() - t0, "source_sha256": source_hash(), "runner_sha256": sha(__file__),
                    "splits_sha256": sha(SPLITS), "metrics_sha256": sha(out / "metrics.json"),
                    "python": platform.python_version(), "torch": torch.__version__, **extra}
        dump(out / "manifest.json", manifest)
        return job["id"], "complete"
    except Exception as exc:
        dump(out / "manifest.json", {**job, "status": "failed", "started_at": started,
                                     "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        return job["id"], f"failed: {type(exc).__name__}: {exc}"


def jobs():
    result = []
    for dataset in DATASETS:
        for size in SIZES:
            for replicate in REPLICATES:
                for variant, (method, params) in VARIANTS.items():
                    result.append(dict(id=f"{dataset}/n{size}/r{replicate}/{variant}", dataset=dataset, size=size,
                                       replicate=replicate, sample=f"n{size}_r{replicate}", variant=variant, method=method,
                                       params={k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()},
                                       seed=replicate))
    return result


COST = {"tabdiff": 40, "tabddpm_upstream": 30, "ctgan": 8, "tvae": 4, "mhd_corrected": 5, "mhd_affine": 5,
        "tabddpm_adapter": 4, "mhd_earlystop": 1}


def execute(selected, workers):
    pending = [j for j in selected if not ((OUT / j["id"] / "manifest.json").exists() and
               json.loads((OUT / j["id"] / "manifest.json").read_text()).get("status") == "complete")]
    pending.sort(key=lambda j: COST.get(j["variant"], 0) * j["size"], reverse=True)
    print(json.dumps(dict(total=len(selected), pending=len(pending), workers=workers, out=str(OUT))), flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_job, j): j["id"] for j in pending}
        for done, future in enumerate(as_completed(futures), 1):
            identifier, status = future.result()
            print(json.dumps(dict(done=done, of=len(pending), id=identifier, status=status,
                                  at=datetime.now(timezone.utc).isoformat())), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "smoke"))
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "smoke":
        # One fast job per method family on the smallest sample, written under OUT/smoke.
        global OUT
        OUT = OUT / "smoke"
        picked = [j for j in jobs() if j["sample"] == "n250_r0" and j["dataset"] == "thyroid"]
        tiny = {"mhd": {"train_steps": 30, "epochs": 2}, "tabddpm": {"train_steps": 30, "epochs": 2},
                "tabdiff": {"train_steps": 6, "steps": 4, "dim_t": 32, "factor": 4}, "ctgan": {"epochs": 2}, "tvae": {"epochs": 2}}
        for j in picked:
            j["params"] = {**j["params"], **{k: v for k, v in tiny.get(j["method"], {}).items()
                                             if k in j["params"] or j["method"] == "tabdiff"}}
            print(run_job(j), flush=True)
    else:
        execute(jobs(), args.workers)


if __name__ == "__main__":
    main()
