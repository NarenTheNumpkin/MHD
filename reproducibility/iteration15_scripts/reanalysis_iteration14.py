"""Post hoc reanalysis of the iteration 14 confirmation runs (publication review of 12 October 2026).

Nothing here refits a generator. Every quantity is computed from saved synthetic tables, saved metrics and the frozen
splits; all analyses are exploratory and labelled as such in the report.

  python scripts/reanalysis_iteration14.py relations --workers 3   # relation diagnostics for every saved table
  python scripts/reanalysis_iteration14.py canonical --workers 3   # canonical-direction re-derivation control
  python scripts/reanalysis_iteration14.py report                  # tables for the manuscript
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wasserstein_distance

import iteration13 as it13
import iteration14 as it

REPORT = it.ROOT / "reports" / "iteration14" / "reanalysis"
CYCLE = {"nhanes_cbc_h": "2013-14", "nhanes_cbc_l": "2021-23", "nhanes_chem_g": "2011-12", "nhanes_chem_l": "2021-23"}
SIZES = (250, 1000, 2000)


def context(dataset, sample):
    frame, schema = it.load(dataset)
    plan = json.loads(it.SPLITS.read_text())["datasets"][dataset]
    s = plan["samples"][sample]
    return frame, schema, frame.iloc[s["fit"]], frame.iloc[plan["reference"]]


def read(path, schema):
    return pd.read_csv(path, dtype={c: str for c in (*schema.categorical, schema.target)}, float_precision="round_trip")


def signed(frame, relation):
    values = {c: pd.to_numeric(frame[c], errors="coerce").to_numpy(float) for c in relation.participants}
    return values[relation.derived] - relation.solve(relation.derived, values)


def relation_diagnostics(table, reference, fit, schema, dataset):
    """Signed distance, conditional and cross-relation dependence, tails, coverage and invalid values."""
    from medhyperdiff.preprocessing import precision_bands
    from medhyperdiff.relations import relations_for
    relations = relations_for(dataset)
    label = lambda f: (f[schema.target].astype(str) == schema.positive).astype(float).to_numpy()
    out, residuals = {}, {}
    signed_d, conditional, tails, coverage = [], [], [], []
    for relation in relations:
        r_fit, r_ref, r_syn = signed(fit, relation), signed(reference, relation), signed(table, relation)
        residuals[relation.name] = (r_ref, r_syn)
        decimals = precision_bands(pd.to_numeric(fit[relation.derived], errors="coerce").to_numpy(float))["global"]
        floor = 10.0 ** -decimals / np.sqrt(12) if decimals is not None else 1e-8
        finite_fit = r_fit[np.isfinite(r_fit)]
        scale = max(float(np.std(finite_fit)) if len(finite_fit) > 1 else 0., floor)
        a, b = r_ref[np.isfinite(r_ref)], r_syn[np.isfinite(r_syn)]
        signed_d.append(wasserstein_distance(a, b) / scale if len(a) and len(b) else np.nan)
        q = np.quantile(np.abs(a), .99) if len(a) else np.nan
        tails.append(float(np.mean(np.abs(b) > q + 1e-12)) if len(b) else np.nan)
        coverage.append(abs(np.isfinite(r_syn).mean() - np.isfinite(r_ref).mean()))
        gaps = []
        for column in (*relation.participants, "age", "__label__"):
            def rho(frame, res):
                x = label(frame) if column == "__label__" else pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
                ok = np.isfinite(res) & np.isfinite(x)
                if ok.sum() < 20 or np.std(res[ok]) == 0 or np.std(x[ok]) == 0:
                    return 0.
                return float(spearmanr(res[ok], x[ok]).statistic)
            gaps.append(abs(rho(table, r_syn) - rho(reference, r_ref)))
        conditional.append(float(np.mean(gaps)))
    cross = []
    names = list(residuals)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            def rho(k):
                x, y = residuals[first][k], residuals[second][k]
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < 20 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
                    return 0.
                return float(spearmanr(x[ok], y[ok]).statistic)
            cross.append(abs(rho(1) - rho(0)))
    participants = sorted({c for r in relations for c in r.participants})
    values = table[participants].apply(pd.to_numeric, errors="coerce")
    invalid = float(((values < 0) | np.isinf(values)).any(axis=1).mean())
    out.update(signed_distance=float(np.nanmean(signed_d)), conditional_gap=float(np.mean(conditional)),
               cross_relation_gap=float(np.mean(cross)) if cross else np.nan, tail_exceedance=float(np.nanmean(tails)),
               coverage_gap=float(np.mean(coverage)), invalid_rate=invalid)
    return out


def relations_job(manifest):
    m = json.loads(Path(manifest).read_text())
    if m.get("status") != "complete" or m["size"] not in SIZES:
        return []
    frame, schema, fit, reference = context(m["dataset"], m["sample"])
    rows = []
    for path in sorted(Path(manifest).parent.glob("synthetic_*.csv")):
        table = read(path, schema)
        rows.append(dict(id=m["id"], dataset=m["dataset"], size=m["size"], seed=m["seed"], variant=m["variant"],
                         processing=path.stem.split("_", 1)[1], **relation_diagnostics(table, reference, fit, schema, m["dataset"])))
    return rows


def canonical_job(manifest):
    """Re-derive every supplied relation in its textbook direction from a published generator's decoded output
    (deterministic reconstruction of the derived variable, as in rule-based frameworks), then score it."""
    from medhyperdiff.constraints import build_constraints
    from medhyperdiff.evaluation import c2st, clinical_category_tv
    from medhyperdiff.relations import relations_for
    m = json.loads(Path(manifest).read_text())
    if m.get("status") != "complete" or m["size"] not in SIZES:
        return []
    frame, schema, fit, reference = context(m["dataset"], m["sample"])
    table = read(Path(manifest).parent / "synthetic_primary.csv", schema)
    rounder = it13.rounder_for(fit, schema)
    for relation in relations_for(m["dataset"]):
        values = {c: pd.to_numeric(table[c], errors="coerce").to_numpy(float) for c in relation.operands}
        table[relation.derived] = rounder(relation.derived, relation.solve(relation.derived, values))
    constraints = build_constraints(schema, fit)
    metrics = it.relation_metrics(table, reference, fit, schema, constraints, None)
    return [dict(id=m["id"], dataset=m["dataset"], size=m["size"], seed=m["seed"], variant=m["variant"], processing="canonical",
                 relation_w1=metrics["relation_w1"], c2st=c2st(reference, table, schema, it.EVALUATOR_SEED)["auc"],
                 clinical_tv=clinical_category_tv(reference, table, it.PANELS[m["dataset"]]),
                 **relation_diagnostics(table, reference, fit, schema, m["dataset"]))]


def run(kind, workers):
    manifests = sorted(str(p) for p in (it.OUT / "confirm").rglob("manifest.json"))
    if kind == "canonical":
        manifests = [p for p in manifests if json.loads(Path(p).read_text())["variant"] in it.PUBLISHED]
    job = relations_job if kind == "relations" else canonical_job
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [r for chunk in pool.map(job, manifests, chunksize=4) for r in chunk]
    REPORT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORT / f"{kind}.csv", index=False)
    print(kind, len(rows))


def eligible_arms(cell):
    means = cell.groupby(["variant", "processing"])[["c2st", "rf", "membership", "clinical_tv", "relation_w1"]].mean()
    published = means[~means.index.get_level_values(0).str.startswith("mhd") & (means.index.get_level_values(0) != "bootstrap")]
    return published[published.membership <= .60]


def report():
    d = pd.read_csv(it.ROOT / "reports" / "iteration14" / "confirm_all.csv")
    d = d[d.status == "complete"]
    primary = json.loads(it.CHOICE.read_text())["primary"]
    ours = f"mhd:{primary}"
    main = d[d["size"].isin(SIZES)]
    rows, units = [], []
    for (dataset, size), cell in main.groupby(["dataset", "size"]):
        arms = eligible_arms(cell)
        realism = arms.c2st.idxmin()
        utility = arms.rf.idxmax()
        category = arms.clinical_tv.idxmin()
        mine = cell[(cell.variant == ours) & (cell.processing == "primary")].set_index("seed")
        def paired(arm, metric):
            other = cell[(cell.variant == arm[0]) & (cell.processing == arm[1])].set_index("seed")
            keys = mine.index.intersection(other.index)
            diff = mine.loc[keys, metric] - other.loc[keys, metric]
            return diff
        for label, arm, metric in (("realism-selected", realism, "c2st"), ("realism-selected", realism, "rf"),
                                   ("utility-selected", utility, "rf"), ("utility-selected", utility, "c2st"),
                                   ("TabDiff + reparameterization", ("tabdiff_rp", "primary"), "c2st"),
                                   ("TabDiff + reparameterization", ("tabdiff_rp", "primary"), "rf"),
                                   ("category-selected", category, "clinical_tv")):
            diff = paired(arm, metric)
            rows.append(dict(dataset=dataset, cycle=CYCLE[dataset], size=size, comparator=label, arm="/".join(arm), metric=metric,
                             ours=float(mine[metric].mean()), other=float(mine[metric].mean() - diff.mean()), difference=float(diff.mean()),
                             min=float(diff.min()), max=float(diff.max()), better=int((diff < 0).sum() if metric in ("c2st", "clinical_tv") else (diff > 0).sum()),
                             n=len(diff)))
            for s, value in diff.items():
                units.append(dict(dataset=dataset, cycle=CYCLE[dataset], size=size, comparator=label, metric=metric, seed=s, difference=value))
    effects = pd.DataFrame(rows)
    REPORT.mkdir(parents=True, exist_ok=True)
    effects.to_csv(REPORT / "per_dataset_effects.csv", index=False)
    u = pd.DataFrame(units)
    loco = []
    for (size, comparator, metric), g in u.groupby(["size", "comparator", "metric"]):
        loco.append(dict(size=size, comparator=comparator, metric=metric, left_out="none", mean=float(g.difference.mean()), n=len(g)))
        for cycle in sorted(g.cycle.unique()):
            h = g[g.cycle != cycle]
            if len(h):
                loco.append(dict(size=size, comparator=comparator, metric=metric, left_out=cycle, mean=float(h.difference.mean()), n=len(h)))
    loco = pd.DataFrame(loco)
    loco.to_csv(REPORT / "leave_one_cycle_out.csv", index=False)
    small = d[(d["size"] == 100) & (d.variant == ours)].groupby("dataset")[["membership", "c2st", "relation_w1", "rf"]].agg(["mean", "min", "max"])
    small.to_csv(REPORT / "n100_ours_only.csv")
    pd.set_option("display.width", 220)
    print(effects.round(3).to_string(index=False))
    print(loco[loco["size"].isin([250, 1000])].round(4).to_string(index=False))
    print(small.round(3).to_string())
    for kind in ("relations", "canonical"):
        path = REPORT / f"{kind}.csv"
        if path.exists():
            x = pd.read_csv(path)
            keep = x[x.variant.isin([ours, "tabdiff", "tabdiff_rp", "copula", "copula_rp"])] if kind == "relations" else x
            summary = keep.groupby(["dataset", "size", "variant", "processing"]).mean(numeric_only=True).drop(columns=["seed"])
            summary.to_csv(REPORT / f"{kind}_summary.csv")
            print(summary.round(3).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("relations", "canonical", "report"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    report() if args.stage == "report" else run(args.stage, args.workers)


if __name__ == "__main__":
    main()
