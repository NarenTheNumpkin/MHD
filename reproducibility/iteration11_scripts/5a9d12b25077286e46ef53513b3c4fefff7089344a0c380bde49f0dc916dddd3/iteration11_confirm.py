"""Iteration 11 follow-on stages; the frozen runner `iteration11.py` is imported unchanged.

  select    apply the predeclared selection rule to completed search records
  guidance  sampling-only guidance calibration of the selected MedHyperDiff configuration
  confirm   single frozen pass on the reserved outer test fold 0 of every dataset

Selection and guidance files are written once and refuse to be overwritten. The confirm
stage records their hashes in every manifest and is the only code path that opens the
reserved outer test partition.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import iteration11 as it

SELECTION = it.ROOT / "configs" / "iteration11_selection.json"
GUIDANCE = it.ROOT / "configs" / "iteration11_guidance.json"
METHODS = ("mhd", "tabddpm", "tabdiff", "ctgan", "tvae")
PRIVACY_THRESHOLD = .60
STATISTICAL = ("copula", "copula_repair", "independent", "bootstrap")


def read_record(job_id):
    path = it.OUT / job_id
    manifest = json.loads((path / "manifest.json").read_text()) if (path / "manifest.json").exists() else {}
    if manifest.get("status") != "complete":
        return None
    return json.loads((path / "metrics.json").read_text())


def endpoints(metrics):
    realism = metrics["constraints"].get("realism", {})
    return dict(c2st=metrics["c2st"]["auc"], numeric_w=metrics["fidelity"]["mean_numeric_wasserstein_train_sd"],
                rf=metrics["utility"]["forest"]["TSTR"]["auroc"], logistic=metrics["utility"]["logistic"]["TSTR"]["auroc"],
                association=metrics["fidelity"]["association_rmse"], violation_gap=realism.get("mean_violation_gap"),
                membership=metrics["privacy"]["nearest_synthetic_membership_auc"])


def select():
    assert not SELECTION.exists(), "Selection is frozen; delete nothing, create a new iteration instead."
    jobs = it.search_jobs(METHODS)
    table, chosen = [], {}
    for dataset in it.DATASETS:
        chosen[dataset] = {}
        for method in METHODS:
            rows = []
            for k, params in enumerate(it.search_configs(method)):
                records = [read_record(f"search/{dataset}/{method}/c{k}/i{inner}_t0") for inner in it.INNER_SEEDS]
                complete = all(r is not None for r in records) and all(r["c2st"]["auc"] is not None for r in records)
                row = dict(dataset=dataset, method=method, config_index=k, complete=complete)
                if complete:
                    values = [endpoints(r) for r in records]
                    row.update({key: float(np.mean([v[key] for v in values])) for key in ("c2st", "numeric_w", "rf", "logistic",
                                "association", "membership")})
                rows.append(row)
            valid = [r for r in rows if r["complete"]]
            if not valid:
                raise RuntimeError(f"No complete configuration for {dataset}/{method}")
            def choose(candidates, key):
                ranks = np.mean([rankdata([r["c2st"] for r in candidates]), rankdata([r["numeric_w"] for r in candidates]),
                                 rankdata([-r["rf"] for r in candidates])], axis=0)
                for r, score in zip(candidates, ranks):
                    r[key] = float(score)
                return min(candidates, key=lambda r: (r[key], r["c2st"], r["config_index"]))
            best = choose(valid, "selection_score")
            best["selected"] = True
            params = it.search_configs(method)[best["config_index"]]
            entry = dict(config_index=best["config_index"], membership=best["membership"],
                         params={**it.CORRECTED_BASE, **params} if method == "mhd" else params)
            # Amendment 2: secondary privacy-constrained choice (membership AUROC <= 0.60).
            eligible = [r for r in valid if r["membership"] <= PRIVACY_THRESHOLD]
            if eligible:
                private = choose(eligible, "private_score")
                private["selected_private"] = True
                pparams = it.search_configs(method)[private["config_index"]]
                entry["private"] = dict(config_index=private["config_index"], membership=private["membership"],
                    params={**it.CORRECTED_BASE, **pparams} if method == "mhd" else pparams)
            else:
                entry["private"] = None
            chosen[dataset][method] = entry
            table.extend(rows)
    expected = len(it.DATASETS) * len(METHODS) * it.SEARCH_SIZE
    assert len(table) == expected and len(jobs) == expected * len(it.INNER_SEEDS)
    it.dump(SELECTION, dict(created_at=datetime.now(timezone.utc).isoformat(), rule=json.loads(
        (it.ROOT / "configs" / "iteration11.json").read_text())["selection_rule"], selected=chosen, table=table))
    pd.DataFrame(table).to_csv(it.OUT / "search_selection_table.csv", index=False)
    print(json.dumps({d: {m: v["config_index"] for m, v in c.items()} for d, c in chosen.items()}))


def guidance_job(dataset, inner, params):
    """Fit once (training seed 0) and sample every guidance value from the same weights."""
    import torch
    torch.set_num_threads(2)
    from medhyperdiff.constraints import build_constraints
    from medhyperdiff.evaluation import evaluate
    frame, split, schema = it.load(dataset, inner)
    train, holdout = frame.iloc[split["train"]], frame.iloc[split["validation"]]
    model, _, schedule_fn = it.build_mhd({**params, "seed": 0}, schema, train)
    model.fit(train, holdout)
    labels = np.tile((train[schema.target].to_numpy() == schema.positive).astype(int), it.MULTIPLIER)
    negative = next(v for v in train[schema.target].unique() if v != schema.positive)
    constraints = build_constraints(schema, train)
    results = {}
    for guidance in it.GUIDANCE_GRID:
        out = it.OUT / "guidance" / dataset / f"g{guidance:g}" / f"i{inner}_t0"
        it.set_guidance(model, guidance, schedule_fn)
        synthetic = model.sample(labels, seed=it.SAMPLING_OFFSET)
        synthetic[schema.target] = np.where(labels == 1, schema.positive, negative)
        out.mkdir(parents=True, exist_ok=True)
        synthetic.to_csv(out / "synthetic.csv", index=False)
        synthetic = pd.read_csv(out / "synthetic.csv", dtype={c: str for c in (*schema.categorical, schema.target)},
                                float_precision="round_trip")
        metrics = evaluate(train, holdout, synthetic, schema, constraints, it.EVALUATOR_SEED, {})
        it.dump(out / "metrics.json", metrics)
        results[guidance] = endpoints(metrics)
    return dataset, inner, results


def calibrate(workers):
    assert SELECTION.exists() and not GUIDANCE.exists()
    selected = json.loads(SELECTION.read_text())["selected"]
    collected = {d: {} for d in it.DATASETS}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(guidance_job, d, inner, selected[d]["mhd"]["params"]) for d in it.DATASETS for inner in it.INNER_SEEDS]
        for future in as_completed(futures):
            dataset, inner, results = future.result()
            collected[dataset][inner] = results
            print(json.dumps(dict(dataset=dataset, inner=inner, gaps={g: r["violation_gap"] for g, r in results.items()})), flush=True)
    chosen, table = {}, []
    for dataset in it.DATASETS:
        means = {g: {k: float(np.mean([collected[dataset][i][g][k] for i in it.INNER_SEEDS]))
                     for k in ("violation_gap", "c2st", "numeric_w", "rf", "membership")} for g in it.GUIDANCE_GRID}
        best_gap = min(m["violation_gap"] for m in means.values())
        chosen[dataset] = min(g for g, m in means.items() if m["violation_gap"] <= best_gap + 1e-9)
        table += [dict(dataset=dataset, guidance=g, **m, selected=g == chosen[dataset]) for g, m in means.items()]
    it.dump(GUIDANCE, dict(created_at=datetime.now(timezone.utc).isoformat(),
        rule="minimize partition-mean violation gap versus real validation; ties within 1e-9 choose smaller guidance",
        selection_sha256=it.sha(SELECTION), guidance=chosen, table=table))
    print(json.dumps(chosen))


def confirm_jobs():
    selected = json.loads(SELECTION.read_text())["selected"]
    guidance = json.loads(GUIDANCE.read_text())["guidance"]
    frozen = dict(selection_sha256=it.sha(SELECTION), guidance_sha256=it.sha(GUIDANCE), confirm_script_sha256=it.sha(__file__))
    jobs = []
    for dataset in it.DATASETS:
        mhd = {**selected[dataset]["mhd"]["params"], "guidance": float(guidance[dataset])}
        arms = {"mhd": ("mhd", mhd),
                "mhd_backbone": ("mhd", {**mhd, "mode": "base", "guidance": 0., "clinical_weight": 0.}),
                "mhd_legacy": ("mhd_legacy", dict(it.LEGACY)),
                **{m: (m, selected[dataset][m]["params"]) for m in ("tabddpm", "tabdiff", "ctgan", "tvae")},
                **{m: (m, {}) for m in STATISTICAL}}
        # Amendment 2: privacy-constrained alternatives where they differ from the primary choice.
        for m in METHODS:
            private = selected[dataset][m].get("private")
            if private and private["config_index"] != selected[dataset][m]["config_index"]:
                params = private["params"]
                if m == "mhd":
                    params = {**params, "guidance": float(guidance[dataset])}
                arms[f"{m}_private"] = (m, params)
        for arm, (method, params) in arms.items():
            for seed in it.CONFIRM_SEEDS:
                jobs.append(dict(id=f"confirm/{dataset}/{arm}/t{seed}", stage="confirm", dataset=dataset, arm=arm,
                                 partition="outer_test", inner_seed=it.INNER_SEEDS[0], training_seed=seed,
                                 method=method, params=params, save_checkpoint=method != "tabddpm", **frozen))
    return jobs


MECHANISM = {
    "legacy_full": ("mhd_legacy", dict(it.LEGACY)),
    "legacy_dense": ("mhd_legacy", {**it.LEGACY, "dense_structure": True}),
    "corrected_full": ("mhd", dict(it.CORRECTED_BASE)),
    "corrected_dense": ("mhd", {**it.CORRECTED_BASE, "dense_structure": True}),
    "corrected_full_no_guidance": ("mhd", {**it.CORRECTED_BASE, "guidance": 0.}),
    "corrected_backbone": ("mhd", {**it.CORRECTED_BASE, "mode": "base", "guidance": 0., "clinical_weight": 0.}),
}


def mechanism_jobs():
    """Controlled simulator (supplied XOR parity and protein identity); development partitions only."""
    from medhyperdiff.data import toy_data
    path = it.ROOT / "data" / "processed" / "toy.csv"
    if not path.exists():
        toy_data(512, seed=7301).to_csv(path, index=False)  # the simulator draw used since iteration 2
    return [dict(id=f"mechanism/toy/{arm}/i{inner}_t{seed}", stage="mechanism", dataset="toy", arm=arm,
                 partition="inner_validation", inner_seed=inner, training_seed=seed, method=method, params=params)
            for arm, (method, params) in MECHANISM.items() for inner in it.INNER_SEEDS for seed in (0, 1, 2)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "guidance", "confirm", "mechanism"))
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if args.stage == "select":
        select()
    elif args.stage == "guidance":
        calibrate(args.workers)
    elif args.stage == "mechanism":
        raise SystemExit(it.execute(mechanism_jobs(), args.workers))
    else:
        assert SELECTION.exists() and GUIDANCE.exists(), "Freeze selection and guidance before opening the test set"
        raise SystemExit(it.execute(confirm_jobs(), args.workers))


if __name__ == "__main__":
    main()
