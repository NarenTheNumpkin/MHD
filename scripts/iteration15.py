"""Iteration 15: decomposing the training-recipe correction (see docs/ITERATION_15_PROTOCOL.md).

  python scripts/iteration15.py run --workers 8
  python scripts/iteration15.py report
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import iteration12 as it12

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("MHD_OUT", ROOT / "runs" / "iteration15"))
REPORT = ROOT / "reports" / "iteration15"
DATASETS, SIZES, REPLICATES = ("nhanes", "thyroid"), (250, 1000), (0, 1, 2, 3, 4)
EVALUATOR_SEED, MULTIPLIER = it12.EVALUATOR_SEED, it12.MULTIPLIER
LABELS = {"V0": "Legacy recipe", "V1": "Corrected recipe", "V2": "V1 + validation-selected checkpoint",
          "V3": "V2 + patience 20", "V4": "V2 + 16 corruption draws", "V5": "V1 with legacy update budget",
          "V6": "V1 with constant rate, raw weights", "V7": "Legacy rebuilt in fixed-budget loop"}


def params_for(variant, fit_rows):
    base = dict(it12.MHD)
    per_pass = math.ceil(fit_rows / base["batch_size"])
    legacy_updates = 100 * per_pass
    if variant == "V0":
        return {**base, "training": "epochs", "epochs": 100, "patience": 20}
    if variant == "V1":
        return base
    if variant == "V2":
        return {**base, "checkpoint": "validation", "check_every": per_pass}
    if variant == "V3":
        return {**base, "checkpoint": "validation", "check_every": per_pass, "selection_patience": 20}
    if variant == "V4":
        return {**base, "checkpoint": "validation", "check_every": per_pass, "validation_draws": 16}
    if variant == "V5":
        return {**base, "train_steps": legacy_updates, "ema_decay": 1 - 5 / legacy_updates}
    if variant == "V6":
        return {**base, "lr_schedule": "constant", "use_ema": False}
    if variant == "V7":
        return {**base, "train_steps": legacy_updates, "lr_schedule": "constant", "use_ema": False,
                "checkpoint": "validation", "check_every": per_pass, "selection_patience": 20}
    raise ValueError(variant)


def jobs():
    return [dict(id=f"{d}/n{n}/r{r}/{v}", dataset=d, size=n, replicate=r, sample=f"n{n}_r{r}", variant=v, seed=r)
            for d in DATASETS for n in SIZES for r in REPLICATES for v in LABELS]


def run_job(job):
    import torch
    torch.set_num_threads(1)
    from medhyperdiff.cli import source_hash
    from medhyperdiff.constraints import build_constraints
    from medhyperdiff.evaluation import density_ratio_attack, evaluate
    out = OUT / job["id"]
    if (out / "manifest.json").exists() and json.loads((out / "manifest.json").read_text()).get("status") == "complete":
        return job["id"], "skipped"
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    try:
        frame, schema = it12.load(job["dataset"])
        plan = json.loads(it12.SPLITS.read_text())["datasets"][job["dataset"]]
        sample = plan["samples"][job["sample"]]
        fit, validation, reference = frame.iloc[sample["fit"]], frame.iloc[sample["validation"]], frame.iloc[plan["reference"]]
        params = params_for(job["variant"], len(fit))
        t0 = time.perf_counter()
        model = it12.build("mhd", params, schema, fit, job["seed"])
        model.fit(fit, validation)
        fit_seconds = time.perf_counter() - t0
        labels = np.tile((fit[schema.target] == schema.positive).astype(int).to_numpy(), MULTIPLIER)
        raw = model.sample(labels, seed=70001 + job["seed"])
        negative = next(v for v in fit[schema.target].unique() if v != schema.positive)
        raw[schema.target] = np.where(labels == 1, schema.positive, negative)
        table = it12.decode(raw, fit, schema, "banded")
        table.to_csv(out / "synthetic_banded.csv", index=False)
        table = pd.read_csv(out / "synthetic_banded.csv", dtype={c: str for c in (*schema.categorical, schema.target)}, float_precision="round_trip")
        constraints = build_constraints(schema, fit)
        metrics = evaluate(fit, reference, table, schema, constraints, EVALUATOR_SEED, {})
        rng = np.random.default_rng(EVALUATOR_SEED + job["seed"])
        index = rng.choice(len(reference), len(fit), replace=False)
        metrics["density_attack"] = density_ratio_attack(fit, reference.iloc[index], table, reference.drop(reference.index[index]),
                                                         schema, seed=EVALUATOR_SEED)
        it12.dump(out / "metrics.json", metrics)
        it12.dump(out / "history.json", model.history)
        if params.get("training") == "epochs":
            selected, executed = model.best_epoch * math.ceil(len(fit) / params["batch_size"]), len(model.history) * math.ceil(len(fit) / params["batch_size"])
        else:
            selected, executed = getattr(model, "selected_step", None), getattr(model, "executed_steps", None)
        it12.dump(out / "manifest.json", {**job, "status": "complete", "started_at": started, "completed_at": datetime.now(timezone.utc).isoformat(),
                  "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()}, "fit_rows": len(fit),
                  "selected_update": selected, "executed_updates": executed, "fit_seconds": fit_seconds,
                  "source_sha256": source_hash(), "runner_sha256": it12.sha(__file__), "splits_sha256": it12.sha(it12.SPLITS),
                  "python": platform.python_version(), "torch": torch.__version__})
        return job["id"], "complete"
    except Exception as exc:
        it12.dump(out / "manifest.json", {**job, "status": "failed", "started_at": started, "error": f"{type(exc).__name__}: {exc}",
                                         "traceback": traceback.format_exc()})
        return job["id"], f"failed: {type(exc).__name__}: {exc}"


def execute(workers):
    selected = jobs()
    pending = [j for j in selected if not ((OUT / j["id"] / "manifest.json").exists() and
               json.loads((OUT / j["id"] / "manifest.json").read_text()).get("status") == "complete")]
    cost = {"V0": 1, "V5": 1, "V7": 1, "V3": 3}
    pending.sort(key=lambda j: cost.get(j["variant"], 5) * j["size"], reverse=True)
    print(json.dumps(dict(total=len(selected), pending=len(pending), workers=workers, out=str(OUT))), flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_job, j): j["id"] for j in pending}
        for done, future in enumerate(as_completed(futures), 1):
            identifier, status = future.result()
            print(json.dumps(dict(done=done, of=len(pending), id=identifier, status=status, at=datetime.now(timezone.utc).isoformat())), flush=True)


def report():
    rows = []
    for manifest in sorted(OUT.rglob("manifest.json")):
        m = json.loads(manifest.read_text())
        if m.get("status") != "complete":
            continue
        x = json.loads((manifest.parent / "metrics.json").read_text())
        rows.append(dict(dataset=m["dataset"], size=m["size"], replicate=m["replicate"], variant=m["variant"],
                         c2st=x["c2st"]["auc"], association=x["fidelity"]["association_rmse"],
                         numeric_w=x["fidelity"]["mean_numeric_wasserstein_train_sd"], rf=x["utility"]["forest"]["TSTR"]["auroc"],
                         membership=max(x["privacy"]["nearest_synthetic_membership_auc"], x["density_attack"]["auc"]),
                         selected=m["selected_update"], executed=m["executed_updates"]))
    d = pd.DataFrame(rows)
    REPORT.mkdir(parents=True, exist_ok=True)
    d.to_csv(REPORT / "all.csv", index=False)
    means = d.groupby(["dataset", "size", "variant"])[["c2st", "association", "numeric_w", "rf", "membership", "selected", "executed"]].mean()
    means.to_csv(REPORT / "means.csv")
    base = d[d.variant == "V1"].set_index(["dataset", "size", "replicate"])
    effects = []
    for variant, reference in (("V2", "V1"), ("V3", "V1"), ("V4", "V2"), ("V5", "V1"), ("V6", "V1"), ("V0", "V1"), ("V7", "V1"), ("V7", "V0")):
        a = d[d.variant == variant].set_index(["dataset", "size", "replicate"])
        b = d[d.variant == reference].set_index(["dataset", "size", "replicate"])
        keys = a.index.intersection(b.index)
        for metric in ("c2st", "association", "rf", "membership"):
            diff = a.loc[keys, metric] - b.loc[keys, metric]
            effects.append(dict(contrast=f"{variant} - {reference}", metric=metric, n=len(diff), mean=float(diff.mean()),
                                low=float(diff.min()), high=float(diff.max()), positive=int((diff > 0).sum()), negative=int((diff < 0).sum())))
    effects = pd.DataFrame(effects)
    effects.to_csv(REPORT / "effects.csv", index=False)
    pd.set_option("display.width", 200)
    print(means.round(3).to_string())
    print(effects.round(4).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "report"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    execute(args.workers) if args.stage == "run" else report()


if __name__ == "__main__":
    main()
