"""Final comparison for the manuscript: one comparator registry, symmetric membership eligibility, LaTeX tables.

Combines the confirmation arms of iterations 13 and 14 with the post hoc textbook and joint-resampling controls
(`runs/iteration14_controls/controls.csv`). Every arm, including MedHyperDiff, is eligible for privacy-constrained
comparisons only when its mean membership AUROC over the samples of a setting is at most 0.60. The bootstrap copy
control and MedHyperDiff variants are never comparators.

  python scripts/report_final.py <latex table folder>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import iteration14 as it

ROOT = it.ROOT
REPORT = ROOT / "reports" / "final"
TABLES = Path(sys.argv[1]) if len(sys.argv) > 1 else REPORT / "tables"
THRESHOLD = .60
NAMES = {"tabdiff": "TabDiff", "tabddpm_upstream": "TabDDPM", "ctgan": "CTGAN", "tvae": "TVAE", "copula": "Copula"}
PROCESSING = {"primary": "as generated", "derived": "rule-direction re-derivation", "resampled": "independent residuals",
              "textbook": "textbook re-derivation", "joint": "joint residuals"}
DATASETS = {"nhanes_cbc_h": "Blood count 2013-14", "nhanes_cbc_l": "Blood count 2021-23", "nhanes_chem_g": "Chemistry 2011-12",
            "nhanes_chem_l": "Chemistry 2021-23"}
CYCLE = {"nhanes_cbc_h": "2013-14", "nhanes_cbc_l": "2021-23", "nhanes_chem_g": "2011-12", "nhanes_chem_l": "2021-23"}
OURS14, OURS13 = "mhd:c1", "mhd_reparam"


def label(variant, processing):
    if variant in (OURS14, OURS13):
        return "MedHyperDiff"
    if variant == "mhd:c0":
        return "MedHyperDiff, fixed budget"
    if variant == "mhd_unguided":
        return "MedHyperDiff without relation coding"
    if variant.endswith("_rp") or variant == "tabdiff_reparam":
        return f"{NAMES[variant.replace('_rp', '').replace('_reparam', '')]} + reparameterization"
    if variant == "bootstrap":
        return f"Bootstrap copy, {PROCESSING[processing]}"
    return f"{NAMES[variant]}, {PROCESSING[processing]}"


def registry14():
    base = pd.read_csv(ROOT / "reports" / "iteration14" / "confirm_all.csv")
    base = base[(base.status == "complete") & base["size"].isin([250, 1000, 2000])]
    diag = pd.read_csv(ROOT / "reports" / "iteration14" / "reanalysis" / "relations.csv")
    keys = ["id", "processing"]
    diag_cols = ["signed_distance", "conditional_gap", "cross_relation_gap", "tail_exceedance", "coverage_gap", "invalid_rate"]
    base = base.merge(diag[keys + diag_cols], on=keys, how="left")
    controls = pd.read_csv(ROOT / "runs" / "iteration14_controls" / "controls.csv")
    controls = controls[controls.study == "iteration14"].rename(columns={"source_id": "id"})
    d = pd.concat([base, controls], ignore_index=True)
    d["arm"] = [label(v, p) for v, p in zip(d.variant, d.processing)]
    d["comparator"] = ~d.variant.str.startswith("mhd") & (d.variant != "bootstrap")
    return d


def registry13():
    base = pd.read_csv(ROOT / "reports" / "iteration13" / "confirm_all.csv")
    base = base[base.status == "complete"]
    controls = pd.read_csv(ROOT / "runs" / "iteration14_controls" / "controls.csv")
    controls = controls[controls.study == "iteration13"].rename(columns={"source_id": "id"})
    d = pd.concat([base, controls], ignore_index=True)
    d["arm"] = [label(v, p) for v, p in zip(d.variant, d.processing)]
    d["comparator"] = ~d.variant.str.startswith("mhd") & (d.variant != "bootstrap")
    return d


def cell_summary(cell, ours):
    means = cell.groupby(["variant", "processing"]).mean(numeric_only=True)
    comp = means[[not v.startswith("mhd") and v != "bootstrap" for v in means.index.get_level_values(0)]]
    eligible = comp[comp.membership <= THRESHOLD]
    mine = cell[(cell.variant == ours) & (cell.processing == "primary")].set_index("seed")
    def paired(arm, metric, lower=True):
        other = cell[(cell.variant == arm[0]) & (cell.processing == arm[1])].set_index("seed")
        keys = mine.index.intersection(other.index)
        diff = mine.loc[keys, metric] - other.loc[keys, metric]
        wins = int((diff < 0).sum() if lower else (diff > 0).sum())
        return diff, wins, len(keys)
    out = {"ours": mine.mean(numeric_only=True), "ours_eligible": bool(mine.membership.mean() <= THRESHOLD)}
    for key, metric, pool, lower in (("relation", "relation_w1", comp, True), ("realism", "c2st", eligible, True),
                                     ("realism_any", "c2st", comp, True), ("utility", "rf", eligible, False),
                                     ("category", "clinical_tv", eligible, True)):
        if metric not in pool or pool[metric].isna().all():
            continue
        arm = pool[metric].idxmin() if lower else pool[metric].idxmax()
        diff, wins, n = paired(arm, metric, lower)
        out[key] = dict(arm=arm, label=label(*arm), value=float(pool.loc[arm, metric]), diff=diff, wins=wins, n=n,
                        membership=float(pool.loc[arm, "membership"]))
    return out


def fmt(x, digits=3):
    return f"{x:.{digits}f}"


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    d = registry14()
    d.to_csv(REPORT / "registry_iteration14.csv", index=False)
    rows, units, facts = [], [], {}
    for dataset in it.CONFIRMATION:
        for size in sorted(d[d.dataset == dataset]["size"].unique()):
            cell = d[(d.dataset == dataset) & (d["size"] == size)]
            s = cell_summary(cell, OURS14)
            o = s["ours"]
            ours_best_eligible = s["ours_eligible"] and o.c2st < s["realism"]["value"]
            rows.append(dict(dataset=dataset, size=size, ours_relation=o.relation_w1, control_relation=s["relation"]["value"],
                             control_arm=s["relation"]["label"], relation_wins=f"{s['relation']['wins']}/{s['relation']['n']}",
                             ours_c2st=o.c2st, ours_membership=o.membership, ours_eligible=s["ours_eligible"],
                             comparator_c2st=s["realism"]["value"], comparator_arm=s["realism"]["label"],
                             comparator_membership=s["realism"]["membership"], c2st_wins=f"{s['realism']['wins']}/{s['realism']['n']}",
                             any_c2st=s["realism_any"]["value"], any_arm=s["realism_any"]["label"], any_membership=s["realism_any"]["membership"],
                             ours_best_eligible=ours_best_eligible,
                             ours_rf=o.rf, comparator_rf=s["utility"]["value"], utility_arm=s["utility"]["label"],
                             rf_wins=f"{s['utility']['wins']}/{s['utility']['n']}",
                             ours_category=o.clinical_tv, comparator_category=s["category"]["value"], category_arm=s["category"]["label"],
                             category_wins=f"{s['category']['wins']}/{s['category']['n']}"))
            for key, metric in (("realism", "c2st"), ("utility", "rf"), ("category", "clinical_tv")):
                for seed, value in s[key]["diff"].items():
                    units.append(dict(dataset=dataset, cycle=CYCLE[dataset], size=size, comparison=key, metric=metric, seed=seed, difference=value))
    table = pd.DataFrame(rows)
    table.to_csv(REPORT / "per_dataset.csv", index=False)
    u = pd.DataFrame(units)
    pooled = []
    for (size, comparison), g in u.groupby(["size", "comparison"]):
        entry = dict(size=size, comparison=comparison, mean=float(g.difference.mean()), n=len(g))
        loco = [float(g[g.cycle != c].difference.mean()) for c in sorted(g.cycle.unique()) if len(g[g.cycle != c])]
        entry.update(loco_min=min(loco) if loco else np.nan, loco_max=max(loco) if loco else np.nan)
        pooled.append(entry)
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(REPORT / "pooled.csv", index=False)

    # Table: per-dataset confirmation.
    lines = []
    for dataset in it.CONFIRMATION:
        sub = table[table.dataset == dataset]
        for i, r in enumerate(sub.itertuples()):
            first = DATASETS[dataset] if i == 0 else ""
            dagger = r"$^{\dagger}$" if not r.ours_eligible else ""
            def pair(ours, other, lower, eligible_rule):
                o, c = fmt(ours) + (dagger if eligible_rule else ""), fmt(other)
                ours_better = (ours < other - 5e-4) if lower else (ours > other + 5e-4)
                other_better = (other < ours - 5e-4) if lower else (other > ours + 5e-4)
                if ours_better and (r.ours_eligible or not eligible_rule):
                    return rf"\textbf{{{o}}} & {c}"
                if other_better or (eligible_rule and not r.ours_eligible):
                    return rf"{o} & \textbf{{{c}}}"
                return rf"{o} & {c}"
            rel = pair(r.ours_relation, r.control_relation, True, False)
            c2 = pair(r.ours_c2st, r.comparator_c2st, True, True)
            ut = pair(r.ours_rf, r.comparator_rf, False, True)
            lines.append(rf"{first} & {r.size:,} & {rel} & {r.relation_wins} & {c2} & {r.c2st_wins} & {ut} & {r.rf_wins} \\")
    (TABLES / "table_robust_rows.tex").write_text("\n".join(lines) + "\n")

    # Table: relation diagnostics by panel (n = 250 and 1,000), including the new controls.
    arms = [((OURS14, "primary"), "MedHyperDiff"), (("tabdiff_rp", "primary"), "TabDiff + reparameterization"),
            (("copula", "joint"), "Copula, joint residuals"), (("tabdiff", "joint"), "TabDiff, joint residuals"),
            (("copula", "resampled"), "Copula, indep. residuals"), (("tabdiff", "textbook"), "TabDiff, re-der. (textbook)"),
            (("tabdiff", "derived"), "TabDiff, re-der. (rule)"), (("tabdiff", "primary"), "TabDiff, as generated")]
    sub = d[d["size"].isin([250, 1000])].copy()
    sub["panel"] = np.where(sub.dataset.str.contains("cbc"), "blood", "chem")
    agg = sub.groupby(["variant", "processing", "panel"])[["relation_w1", "signed_distance", "cross_relation_gap", "conditional_gap"]].mean()
    diag_rows, best = [], {}
    metrics = [("relation_w1", "blood"), ("relation_w1", "chem"), ("cross_relation_gap", "blood"), ("cross_relation_gap", "chem"),
               ("conditional_gap", "blood"), ("conditional_gap", "chem")]
    for metric, panel in metrics:
        best[(metric, panel)] = min(agg.loc[(v, p, panel), metric] for (v, p), _ in arms)
    for (v, p), name in arms:
        cells = []
        for metric, panel in metrics:
            value = agg.loc[(v, p, panel), metric]
            text = f"{value:.3f}" if value < 10 else f"{value:.0f}" if value >= 100 else f"{value:.2f}"
            cells.append(rf"\textbf{{{text}}}" if abs(value - best[(metric, panel)]) < 5e-4 else text)
        diag_rows.append(name + " & " + " & ".join(cells) + r" \\")
    (TABLES / "table_diagnostics_rows.tex").write_text("\n".join(diag_rows) + "\n")
    agg.reset_index().to_csv(REPORT / "diagnostics_by_panel.csv", index=False)

    # First test (2017-18 blood counts): ours against the best comparator per measure.
    d13 = registry13()
    d13.to_csv(REPORT / "registry_iteration13.csv", index=False)
    first = []
    for size in (250, 1000):
        cell = d13[d13["size"] == size]
        s = cell_summary(cell, OURS13)
        o = s["ours"]
        first.append(dict(size=size, ours_relation=o.relation_w1, relation=s["relation"]["value"], relation_arm=s["relation"]["label"],
                          relation_wins=s["relation"]["wins"], ours_c2st=o.c2st, c2st=s["realism"]["value"], c2st_arm=s["realism"]["label"],
                          c2st_wins=s["realism"]["wins"], c2st_membership=s["realism"]["membership"], ours_rf=o.rf, rf=s["utility"]["value"],
                          rf_arm=s["utility"]["label"], rf_wins=s["utility"]["wins"], ours_membership=o.membership,
                          tabdiff_raw=float(cell[(cell.variant == "tabdiff") & (cell.processing == "primary")].relation_w1.mean()),
                          tabdiff_derived=float(cell[(cell.variant == "tabdiff") & (cell.processing == "derived")].relation_w1.mean())))
    first = pd.DataFrame(first)
    first.to_csv(REPORT / "first_test.csv", index=False)

    # Figure data and facts for the text.
    facts.update(pooled=pooled.round(4).to_dict("records"), per_dataset=table.drop(columns=[]).round(4).to_dict("records"),
                 first_test=first.round(4).to_dict("records"))
    (REPORT / "facts.json").write_text(json.dumps(facts, indent=1, default=str))
    pd.set_option("display.width", 250)
    print(table[["dataset", "size", "ours_relation", "control_relation", "control_arm", "relation_wins", "ours_c2st", "ours_eligible",
                 "comparator_c2st", "comparator_arm", "comparator_membership", "c2st_wins", "any_c2st", "any_arm", "ours_best_eligible"]].round(3).to_string(index=False))
    print(table[["dataset", "size", "ours_rf", "comparator_rf", "utility_arm", "rf_wins", "ours_category", "comparator_category",
                 "category_arm", "category_wins"]].round(3).to_string(index=False))
    print(pooled.round(4).to_string(index=False))
    print(first.round(3).to_string(index=False))
    print(agg.round(3).to_string())


if __name__ == "__main__":
    main()
