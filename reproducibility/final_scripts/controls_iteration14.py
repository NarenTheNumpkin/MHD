"""Post hoc knowledge-equivalent controls for the relation studies (second publication review, 13 October 2026).

For every published generator's decoded output in the iteration 13 and 14 confirmations, two further post-processings
are evaluated with the full metric set of the confirmation runs (relation distance, C2ST, utility, membership,
clinical categories, relation diagnostics). No generator is refitted.

  textbook   every derived variable is recomputed from the generated operands in its textbook direction and rounded to
             its recorded resolution (deterministic rule-based reconstruction).
  joint      row-wise joint residual resampling: each synthetic row receives the complete residual vector of one
             randomly drawn fit row (fit rows with every residual observed), in the reconstruction direction of the
             conditioning rule; a residual stays missing where the synthetic row lacks the participants. The table is
             then decoded exactly like MedHyperDiff output, including re-derivation of reported-exact relations.

  python scripts/controls_iteration14.py run --workers 8
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import iteration13 as it13
import iteration14 as it

OUT = Path(os.environ.get("MHD_CONTROLS", it.ROOT / "runs" / "iteration14_controls"))


def context(study, dataset, sample):
    frame, schema = it.load(dataset)
    if study == "iteration13":
        plan = json.loads(it13.SPLITS.read_text())
        return frame, schema, frame.iloc[plan["samples"][sample]["fit"]], frame.iloc[plan["reference"]]
    plan = json.loads(it.SPLITS.read_text())["datasets"][dataset]
    return frame, schema, frame.iloc[plan["samples"][sample]["fit"]], frame.iloc[plan["reference"]]


def textbook(table, fit, schema, dataset):
    from medhyperdiff.relations import relations_for
    out, rounder = table.copy(), it13.rounder_for(fit, schema)
    for relation in relations_for(dataset):
        values = {c: pd.to_numeric(out[c], errors="coerce").to_numpy(float) for c in relation.operands}
        out[relation.derived] = rounder(relation.derived, relation.solve(relation.derived, values))
    return out


def joint(table, fit, schema, dataset, size, seed):
    from medhyperdiff.relations import RelationCoder, relations_for, residual_name
    coder = RelationCoder(schema, relations_for(dataset)).fit(fit)
    columns = [residual_name(r) for r in coder.relations]
    pool = coder.encode(fit)[columns].dropna().to_numpy()
    encoded = coder.encode(table)
    rng = np.random.default_rng([2026, size, seed])
    draw = pool[rng.integers(0, len(pool), len(encoded))]
    for j, column in enumerate(columns):
        encoded[column] = np.where(encoded[column].notna(), draw[:, j], np.nan)
    encoded[schema.target] = table[schema.target].to_numpy()
    return coder.decode(encoded, it13.rounder_for(fit, schema))


def evaluate_table(table, fit, reference, schema, dataset, seed):
    from medhyperdiff.constraints import build_constraints
    from medhyperdiff.evaluation import c2st, clinical_category_tv, density_ratio_attack, evaluate
    import reanalysis_iteration14 as ra
    constraints = build_constraints(schema, fit)
    metrics = it.relation_metrics(table, reference, fit, schema, constraints, None)
    full = evaluate(fit, reference, table, schema, constraints, it.EVALUATOR_SEED, {})
    rng = np.random.default_rng(it.EVALUATOR_SEED + seed)
    index = rng.choice(len(reference), min(len(fit), len(reference) // 2), replace=False)
    density = density_ratio_attack(fit, reference.iloc[index], table, reference.drop(reference.index[index]), schema, seed=it.EVALUATOR_SEED)
    row = dict(relation_w1=metrics["relation_w1"], c2st=c2st(reference, table, schema, it.EVALUATOR_SEED)["auc"],
               clinical_tv=clinical_category_tv(reference, table, it.PANELS[dataset]),
               membership=max(full["privacy"]["nearest_synthetic_membership_auc"], density["auc"]),
               membership_nn=full["privacy"]["nearest_synthetic_membership_auc"], membership_density=density["auc"],
               rf=full["utility"]["forest"]["TSTR"]["auroc"], rf_real=full["utility"]["forest"]["TRTR"]["auroc"],
               numeric_w=full["fidelity"]["mean_numeric_wasserstein_train_sd"], association=full["fidelity"]["association_rmse"])
    row.update({f"w1_{k}": v for k, v in metrics["relation_w1_by_rule"].items()})
    row.update(ra.relation_diagnostics(table, reference, fit, schema, dataset))
    return row


def job(task):
    study, manifest = task
    m = json.loads(Path(manifest).read_text())
    dataset = m.get("dataset", "nhanes_cbc")
    frame, schema, fit, reference = context(study, dataset, m["sample"])
    source = pd.read_csv(Path(manifest).parent / "synthetic_primary.csv", dtype={c: str for c in (*schema.categorical, schema.target)},
                         float_precision="round_trip")
    rows = []
    for name, table in (("textbook", textbook(source, fit, schema, dataset)), ("joint", joint(source, fit, schema, dataset, m["size"], m["seed"]))):
        target = OUT / study / dataset / m["variant"] / m["sample"]
        target.mkdir(parents=True, exist_ok=True)
        table.to_csv(target / f"synthetic_{name}.csv.gz", index=False)
        table = pd.read_csv(target / f"synthetic_{name}.csv.gz", dtype={c: str for c in (*schema.categorical, schema.target)}, float_precision="round_trip")
        rows.append(dict(study=study, dataset=dataset, size=m["size"], seed=m["seed"], variant=m["variant"], processing=name,
                         source_id=m["id"], **evaluate_table(table, fit, reference, schema, dataset, m["seed"])))
    return rows


def tasks():
    out = []
    for manifest in sorted((it.OUT / "confirm").rglob("manifest.json")):
        m = json.loads(manifest.read_text())
        if m.get("status") == "complete" and m["variant"] in it.PUBLISHED and m["size"] in (250, 1000, 2000):
            out.append(("iteration14", str(manifest)))
    for manifest in sorted((it13.OUT / "confirm").rglob("manifest.json")):
        m = json.loads(manifest.read_text())
        if m.get("status") == "complete" and m["variant"] in it.PUBLISHED:
            out.append(("iteration13", str(manifest)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run",))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    selected = tasks()
    print(json.dumps({"tasks": len(selected)}), flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(job, t) for t in selected]
        for done, future in enumerate(as_completed(futures), 1):
            rows.extend(future.result())
            if done % 20 == 0 or done == len(futures):
                print(json.dumps({"done": done, "of": len(futures)}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "controls.csv", index=False)
    print(json.dumps({"rows": len(rows), "file": str(OUT / "controls.csv")}), flush=True)


if __name__ == "__main__":
    main()
