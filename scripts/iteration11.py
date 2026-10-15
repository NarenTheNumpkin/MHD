"""Iteration 11: corrected training/representation, equal-budget tuning and confirmation.

Stages (each resumable; completed job directories are never overwritten):
  ablation  ILPD development partitions; one correction at a time on the full_prior recipe.
  search    equal-budget random search per neural method and dataset (inner validation only).
  select    freeze per-dataset configurations from search results with the predeclared rule.
  guidance  sampling-only guidance calibration for the selected MedHyperDiff fits.
  confirm   frozen single pass on the reserved outer test fold 0 (opens the test set).

Outputs default to runs/iteration11 under the project; set MHD_OUT to redirect them.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
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
OUT = Path(os.environ.get("MHD_OUT", ROOT / "runs" / "iteration11"))
DATASETS = ("ilpd", "heart_failure", "ckd", "diabetes")
INNER_SEEDS = (2027, 3027, 4027)
SAMPLING_OFFSET = 70001
EVALUATOR_SEED = 2026
MULTIPLIER = 5
SEARCH_SIZE = 8
CONFIRM_SEEDS = (0, 1, 2, 3, 4)
GUIDANCE_GRID = (0., 2., 5., 10., 20.)

LEGACY = dict(training="epochs", numeric_transform="affine", decode_precision=False, steps=30, width=32, layers=2,
              heads=4, batch_size=64, learning_rate=1e-3, epochs=100, patience=20, numeric_prediction="v",
              clinical_weight=.01, guidance=10., paired_initialization=True, observation_prior=True)
CORRECTED_BASE = dict(training="steps", train_steps=5000, numeric_transform="quantile", decode_precision=True,
                      steps=100, width=32, layers=2, heads=4, batch_size=64, learning_rate=1e-3,
                      numeric_prediction="v", clinical_weight=.01, guidance=10., paired_initialization=True,
                      observation_prior=True)
# One correction at a time, cumulatively, then leave-one-out and structure ablations.
ABLATION = {
    "a0_legacy_full_prior": dict(LEGACY),
    "a1_fixed_steps_ema": {**LEGACY, "training": "steps", "train_steps": 5000},
    "a2_plus_quantile": {**LEGACY, "training": "steps", "train_steps": 5000, "numeric_transform": "quantile"},
    "a3_plus_T100": {**LEGACY, "training": "steps", "train_steps": 5000, "numeric_transform": "quantile", "steps": 100},
    "a4_plus_precision": dict(CORRECTED_BASE),
    "a4_but_epoch_selection": {**CORRECTED_BASE, "training": "epochs", "epochs": 100, "patience": 20},
    "a4_no_guidance": {**CORRECTED_BASE, "guidance": 0.},
    "a4_backbone": {**CORRECTED_BASE, "mode": "base", "guidance": 0., "clinical_weight": 0.},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False, default=float) + "\n")


def search_space(method, rng):
    """Predeclared spaces. TabDDPM follows upstream scripts/tune_ddpm.py."""
    if method == "mhd":
        return dict(train_steps=int(rng.choice([2000, 5000, 10000])),
                    learning_rate=float(np.exp(rng.uniform(np.log(3e-4), np.log(3e-3)))),
                    width=int(rng.choice([32, 64])), layers=int(rng.choice([2, 3])),
                    steps=int(rng.choice([50, 100, 200])), batch_size=int(rng.choice([64, 128])),
                    numeric_prediction=str(rng.choice(["x0", "v"])),
                    clinical_weight=float(rng.choice([.01, .1])))
    if method == "tabddpm":
        n_layers = 2 * int(rng.integers(1, 5))
        dim = lambda: int(2 ** rng.integers(7, 11))
        first, middle, last = dim(), dim(), dim()
        layers = [first] + [middle] * (n_layers - 2) + [last]
        return dict(learning_rate=float(np.exp(rng.uniform(np.log(1e-5), np.log(3e-3)))),
                    mlp_layers=layers, batch_size=int(rng.choice([256, 4096])),
                    train_steps=int(rng.choice([5000, 20000, 30000])), steps=int(rng.choice([100, 1000])))
    if method == "tabdiff":
        return dict(learning_rate=float(np.exp(rng.uniform(np.log(3e-4), np.log(3e-3)))),
                    train_steps=int(rng.choice([6000, 8000, 10000])), num_layers=int(rng.choice([2, 3])),
                    d_token=int(rng.choice([4, 8])), factor=int(rng.choice([16, 32])),
                    dim_t=int(rng.choice([512, 1024])), steps=int(rng.choice([50, 100])))
    if method == "ctgan":
        dims = [(64, 64), (128, 128), (256, 256)][int(rng.integers(0, 3))]
        return dict(epochs=int(rng.choice([100, 300, 1000])), embedding_dim=int(rng.choice([32, 128])),
                    generator_dim=list(dims), discriminator_dim=list(dims), batch_size=int(rng.choice([100, 500])),
                    pac=10, generator_lr=float(rng.choice([2e-4, 1e-3])), discriminator_lr=float(rng.choice([2e-4, 1e-3])))
    if method == "tvae":
        dims = [(64, 64), (128, 128), (256, 256)][int(rng.integers(0, 3))]
        return dict(epochs=int(rng.choice([100, 300, 1000])), embedding_dim=int(rng.choice([32, 128])),
                    compress_dims=list(dims), decompress_dims=list(dims), batch_size=int(rng.choice([64, 500])),
                    l2scale=float(rng.choice([1e-5, 1e-4])), loss_factor=float(rng.choice([1., 2.])))
    raise ValueError(method)


# Configuration 0 of every method is its recommended default; the other seven are random.
DEFAULTS = {
    "mhd": {k: CORRECTED_BASE[k] for k in ("train_steps", "learning_rate", "width", "layers", "steps",
                                           "batch_size", "numeric_prediction", "clinical_weight")},
    # Upstream exp/diabetes/ddpm_cb_best/config.toml (Pima diabetes, 768 rows) at the pinned commit.
    "tabddpm": dict(learning_rate=1.1510940031144828e-05, mlp_layers=[128, 512], batch_size=4096,
                    train_steps=30000, steps=1000),
    # Upstream tabdiff/configs/tabdiff_configs.toml at the pinned commit.
    "tabdiff": dict(learning_rate=1e-3, train_steps=8000, num_layers=2, d_token=4, factor=32, dim_t=1024, steps=50),
    "ctgan": dict(epochs=300, embedding_dim=128, generator_dim=[256, 256], discriminator_dim=[256, 256],
                  batch_size=500, pac=10, generator_lr=2e-4, discriminator_lr=2e-4),
    "tvae": dict(epochs=300, embedding_dim=128, compress_dims=[128, 128], decompress_dims=[128, 128],
                 batch_size=500, l2scale=1e-5, loss_factor=2.),
}


def search_configs(method):
    rng = np.random.default_rng(int(hashlib.sha256(f"iteration11-{method}".encode()).hexdigest()[:8], 16))
    return [dict(DEFAULTS[method])] + [search_space(method, rng) for _ in range(SEARCH_SIZE - 1)]


def load(dataset, inner):
    from medhyperdiff.data import SCHEMAS, nested_splits, normalize
    schema = SCHEMAS[dataset]
    frame = normalize(pd.read_csv(ROOT / "data" / "processed" / f"{dataset}.csv"), schema)
    split = next(nested_splits(frame, schema, seed=2026, inner_seed=inner))
    reserved = next(nested_splits(frame, schema, seed=2026))
    assert split["test"] == reserved["test"]
    assert not set(split["test"]) & set(split["train"] + split["validation"])
    return frame, split, schema


def build_mhd(params, schema, train):
    from medhyperdiff.constraints import build_constraints
    from medhyperdiff.diffusion import DiffusionConfig, MixedDiffusion, guidance_schedule
    from medhyperdiff.preprocessing import Preprocessor
    from medhyperdiff.structure import dependency_graph
    p = dict(params)
    mode = p.pop("mode", "full")
    prep = Preprocessor(schema, p.pop("numeric_transform"), p.pop("decode_precision")).fit(train)
    constraints = build_constraints(schema, train)
    graph, _ = dependency_graph(train, prep, seed=2026, bootstraps=10)
    full = mode == "full"
    guidance = p.pop("guidance")
    cfg = DiffusionConfig(**p, graph_enabled=full, hypergraph_enabled=full, schedule="none", guidance=0.)
    if not full:
        cfg.clinical_weight = 0.
    model = MixedDiffusion(prep, graph, constraints, cfg)
    return model, guidance, guidance_schedule


def set_guidance(model, guidance, guidance_schedule):
    model.config.guidance = float(guidance)
    model.config.schedule = "adaptive" if guidance else "none"
    model.schedule = guidance_schedule(model.config.steps, model.config.schedule)


def run_job(job):
    """Fit, sample and evaluate one generator; write a complete, hash-stamped record."""
    import torch
    torch.set_num_threads(2)
    from medhyperdiff.cli import source_hash
    from medhyperdiff.constraints import build_constraints, repair_simple
    from medhyperdiff.evaluation import evaluate
    out = OUT / job["id"]
    if (out / "manifest.json").exists() and json.loads((out / "manifest.json").read_text()).get("status") == "complete":
        return job["id"], "skipped"
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    try:
        frame, split, schema = load(job["dataset"], job["inner_seed"])
        outer = job["partition"] == "outer_test"
        fit_rows = split["train"] + split["validation"] if outer else split["train"]
        train = frame.iloc[fit_rows]
        holdout = frame.iloc[split["test"] if outer else split["validation"]]
        method, params, seed = job["method"], dict(job["params"]), job["training_seed"]
        labels = np.tile((train[schema.target].to_numpy() == schema.positive).astype(int), MULTIPLIER)
        constraints = build_constraints(schema, train)
        t0 = time.perf_counter()
        extra = {}
        if method in ("mhd", "mhd_legacy"):
            model, guidance, schedule_fn = build_mhd({**params, "seed": seed}, schema, train)
            if model.config.training == "epochs":
                # The original recipe fits the inner fit rows and selects its checkpoint on the
                # inner validation rows of the job's inner seed, in both stages.
                fit_frame = frame.iloc[split["train"]]
                model.fit(fit_frame, frame.iloc[split["validation"]])
                extra["epoch_selection_fit_rows"] = len(fit_frame)
            else:
                model.fit(train, None if outer else holdout)
            set_guidance(model, guidance, schedule_fn)
            generator = model
            extra.update(parameters=sum(p.numel() for p in model.model.parameters()), selected_epoch=model.best_epoch)
        elif method == "tabddpm":
            from medhyperdiff.diffusion import DiffusionConfig
            from medhyperdiff.external_baselines import TabDDPMAdapter
            from medhyperdiff.preprocessing import Preprocessor
            prep = Preprocessor(schema, "affine", True).fit(train)
            cfg = DiffusionConfig(training="steps", weight_decay=0., seed=seed, width=128, layers=2,
                                  **{k: (tuple(v) if k == "mlp_layers" else v) for k, v in params.items()})
            # Upstream validation-loss monitoring needs records; the outer stage reuses fit rows.
            generator = TabDDPMAdapter(prep, cfg).fit(train, holdout if not outer else train.iloc[:64])
            extra.update(parameters=sum(p.numel() for p in generator.model.parameters()), provenance=generator.provenance)
        elif method == "tabdiff":
            from medhyperdiff.diffusion import DiffusionConfig
            from medhyperdiff.external_baselines import TabDiffAdapter
            from medhyperdiff.preprocessing import Preprocessor
            prep = Preprocessor(schema, "affine", True).fit(train)
            generator = TabDiffAdapter(prep, DiffusionConfig(seed=seed), params).fit(train)
            extra.update(parameters=sum(p.numel() for p in generator.diffusion.parameters()),
                         provenance=generator.provenance, selected_epoch=generator.best_epoch)
        elif method in ("ctgan", "tvae"):
            from medhyperdiff.baselines import CTGANAdapter
            native = {k: (tuple(v) if isinstance(v, list) else v) for k, v in params.items() if k != "epochs"}
            generator = CTGANAdapter(schema, method, params["epochs"], seed, native, decode_precision=True).fit(train)
        elif method in ("copula", "independent", "bootstrap", "copula_repair"):
            from medhyperdiff.baselines import ReferenceGenerator
            kind = "copula" if method == "copula_repair" else method
            generator = ReferenceGenerator(schema, kind, seed, decode_precision=True).fit(train)
        else:
            raise ValueError(method)
        fit_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        synthetic = generator.sample(labels, seed=SAMPLING_OFFSET + seed)
        negative = next(v for v in train[schema.target].unique() if v != schema.positive)
        synthetic[schema.target] = np.where(labels == 1, schema.positive, negative)
        if method == "copula_repair":
            synthetic, changed = repair_simple(synthetic, constraints)
            extra["repair_changed_row_fraction"] = changed
        sample_seconds = time.perf_counter() - t0
        synthetic.to_csv(out / "synthetic.csv", index=False)
        synthetic = pd.read_csv(out / "synthetic.csv", dtype={c: str for c in (*schema.categorical, schema.target)},
                                float_precision="round_trip")
        t0 = time.perf_counter()
        predictions = {}
        metrics = evaluate(train, holdout, synthetic, schema, constraints, EVALUATOR_SEED, predictions)
        for kind in ("forest", "logistic"):
            metrics["utility"][kind][f"AUGMENT_{MULTIPLIER}x"] = metrics["utility"][kind].pop("AUGMENT_1x")
            predictions[f"{kind}_AUGMENT_{MULTIPLIER}x"] = predictions.pop(f"{kind}_AUGMENT_1x")
        pd.DataFrame(predictions).to_csv(out / "utility_predictions.csv", index=False)
        dump(out / "metrics.json", metrics)
        if hasattr(generator, "history"):
            dump(out / "history.json", generator.history)
        if job.get("save_checkpoint") and hasattr(generator, "save"):
            generator.save(out / "checkpoint.pt")
        manifest = {**job, "status": "complete", "started_at": started, "completed_at": datetime.now(timezone.utc).isoformat(),
                    "outer_test_opened": outer, "fit_rows": len(train), "holdout_rows": len(holdout),
                    "synthetic_rows": len(synthetic), "fit_seconds": fit_seconds, "sample_seconds": sample_seconds,
                    "evaluation_seconds": time.perf_counter() - t0, "source_sha256": source_hash(),
                    "runner_sha256": sha(__file__), "data_sha256": hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest(),
                    "metrics_sha256": sha(out / "metrics.json"), "synthetic_sha256": sha(out / "synthetic.csv"),
                    "python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
                    "threads": torch.get_num_threads(), **extra}
        dump(out / "manifest.json", manifest)
        return job["id"], "complete"
    except Exception as exc:
        dump(out / "manifest.json", {**job, "status": "failed", "started_at": started,
                                     "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        return job["id"], f"failed: {type(exc).__name__}: {exc}"


def ablation_jobs():
    jobs = []
    for name, params in ABLATION.items():
        for inner in INNER_SEEDS:
            for seed in (0, 1, 2):
                method = "mhd_legacy" if name == "a0_legacy_full_prior" else "mhd"
                jobs.append(dict(id=f"ablation/ilpd/{name}/i{inner}_t{seed}", stage="ablation", dataset="ilpd",
                                 partition="inner_validation", inner_seed=inner, training_seed=seed,
                                 method=method, variant=name, params=params))
    return jobs


def search_jobs(methods=("mhd", "tabddpm", "ctgan", "tvae")):
    jobs = []
    for dataset in DATASETS:
        for method in methods:
            for k, cfg in enumerate(search_configs(method)):
                params = {**CORRECTED_BASE, **cfg} if method == "mhd" else cfg
                for inner in INNER_SEEDS:
                    jobs.append(dict(id=f"search/{dataset}/{method}/c{k}/i{inner}_t0", stage="search", dataset=dataset,
                                     partition="inner_validation", inner_seed=inner, training_seed=0,
                                     method=method, config_index=k, params=params))
    return jobs


def execute(jobs, workers):
    pending = [j for j in jobs if not ((OUT / j["id"] / "manifest.json").exists() and
               json.loads((OUT / j["id"] / "manifest.json").read_text()).get("status") == "complete")]
    print(json.dumps(dict(total=len(jobs), pending=len(pending), workers=workers, out=str(OUT))), flush=True)
    # Longest expected jobs first keeps the pool busy.
    cost = lambda j: (j["method"] == "tabddpm") * j["params"].get("train_steps", 0) * sum(j["params"].get("mlp_layers", [0])) / 1e3 \
        + j["params"].get("train_steps", 0) * j["params"].get("width", 0) / 1e3 + j["params"].get("epochs", 0) \
        + (j["method"] == "tabdiff") * j["params"].get("train_steps", 0)
    pending.sort(key=cost, reverse=True)
    failures = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_job, j): j["id"] for j in pending}
        for done, future in enumerate(as_completed(futures), 1):
            identifier, status = future.result()
            failures += status.startswith("failed")
            print(json.dumps(dict(done=done, of=len(pending), id=identifier, status=status,
                                  at=datetime.now(timezone.utc).isoformat())), flush=True)
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("ablation", "search", "search_tabdiff", "select", "guidance", "confirm", "list"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = parser.parse_args()
    if args.stage == "ablation":
        raise SystemExit(execute(ablation_jobs(), args.workers))
    if args.stage == "search":
        raise SystemExit(execute([j for j in search_jobs() if j["dataset"] in args.datasets], args.workers))
    if args.stage == "search_tabdiff":
        raise SystemExit(execute([j for j in search_jobs(("tabdiff",)) if j["dataset"] in args.datasets], args.workers))
    if args.stage == "list":
        for method in ("mhd", "tabddpm", "tabdiff", "ctgan", "tvae"):
            for k, cfg in enumerate(search_configs(method)):
                print(method, k, json.dumps(cfg))
        return
    raise SystemExit(f"stage {args.stage} is implemented in a later step")


if __name__ == "__main__":
    main()
