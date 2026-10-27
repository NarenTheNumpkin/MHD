"""Iteration 14 report: pre-registered robustness hypotheses (docs/ITERATION_14_PROTOCOL.md)."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import iteration13 as it13
import iteration14 as it

REPORT = Path(os.environ.get("MHD_REPORT", it.ROOT / "reports" / "iteration14"))
NAMES = {"tabdiff": "TabDiff", "tabddpm_upstream": "TabDDPM", "ctgan": "CTGAN", "tvae": "TVAE", "copula": "Gaussian copula",
         "bootstrap": "Bootstrap copy control"}
PROCESSING = {"primary": "as generated", "derived": "hard derivation", "resampled": "derivation + resampled residuals"}
POPULATION = {"nhanes_cbc_h": "CBC 2013-14", "nhanes_cbc_l": "CBC 2021-23", "nhanes_chem_g": "Chemistry 2011-12",
              "nhanes_chem_l": "Chemistry 2021-23", "nhanes_cbc_i": "CBC 2015-16 (dev)", "nhanes_chem_i": "Chemistry 2015-16 (dev)",
              "nhanes_cbc": "CBC 2017-18 (iteration 13)"}


def arm_name(variant, processing):
    if variant.startswith("mhd:"):
        return f"MedHyperDiff {variant.split(':')[1]}"
    if variant.endswith("_rp"):
        return f"{NAMES[variant[:-3]]} + reparameterization"
    return f"{NAMES[variant]}, {PROCESSING[processing]}"


def rows(root, stage):
    out = []
    for manifest in sorted((root / stage).rglob("manifest.json")):
        m = json.loads(manifest.read_text())
        if m.get("status") != "complete":
            out.append(dict(id=m.get("id"), status=m.get("status"), error=m.get("error")))
            continue
        metrics = json.loads((manifest.parent / "metrics.json").read_text())
        for table, x in metrics.items():
            row = dict(id=m["id"], status="complete", stage=m["stage"], dataset=m["dataset"], variant=m["variant"], processing=table,
                       arm=arm_name(m["variant"], table), size=m["size"], seed=m["seed"], config=m.get("config"),
                       relation_w1=x["relation_w1"], relation_w1_it13=x.get("relation_w1_iteration13"), c2st=x["c2st"],
                       clinical_tv=x.get("clinical_category_tv"), fit_seconds=m["fit_seconds"])
            for name, value in x.get("relation_w1_by_rule", {}).items():
                row[f"w1_{name}"] = value
            if "privacy" in x:
                row.update(membership=max(x["privacy"]["nearest_synthetic_membership_auc"], x["density_attack"]["auc"]),
                           rf=x["utility"]["forest"]["TSTR"]["auroc"], rf_real=x["utility"]["forest"]["TRTR"]["auroc"],
                           numeric_w=x["fidelity"]["mean_numeric_wasserstein_train_sd"], association=x["fidelity"]["association_rmse"])
            out.append(row)
    return pd.DataFrame(out)


def wins(a, b, metric="relation_w1"):
    """Paired replicates in which arm a is lower than arm b."""
    seeds = a.index.intersection(b.index)
    return int((a.loc[seeds, metric] < b.loc[seeds, metric]).sum()), len(seeds)


def cell_arms(cell):
    return {name: g.set_index("seed") for name, g in cell.groupby(["variant", "processing"])}


def hypotheses(d, primary):
    """H1, H1x, H2, H3, H4 exactly as declared in the protocol; one row per tested unit."""
    out = []
    for (dataset, size), cell in d.groupby(["dataset", "size"]):
        panel = it.PANELS[dataset]
        arms = cell_arms(cell)
        need = 3 if size == it.LARGE_SIZE else 4
        mhd = arms[(f"mhd:{primary}", "primary")]
        means = cell.groupby(["variant", "processing"])[["c2st", "membership", "relation_w1", "rf"]].mean()
        competitors = means[~means.index.get_level_values(0).str.startswith("mhd") & (means.index.get_level_values(0) != "bootstrap")]
        non_memorizing = competitors[competitors.membership <= .60]
        best = non_memorizing.c2st.idxmin()
        base = dict(dataset=dataset, population=POPULATION[dataset], panel=panel, size=size)
        # H1 / H1x
        relation = {}
        for method in it.PUBLISHED:
            relation[f"{method}/primary"] = wins(mhd, arms[(method, "primary")])
            if panel == "cbc":
                relation[f"{method}/derived"] = wins(mhd, arms[(method, "derived")])
        best_derived = min(means.loc[(m, "derived"), "relation_w1"] for m in it.PUBLISHED)
        ok = all(w >= need for w, _ in relation.values())
        if panel == "chem":
            ok = ok and mhd.relation_w1.mean() <= best_derived + .10
        out.append(dict(**base, hypothesis="H1" if panel == "cbc" else "H1x", unit="MedHyperDiff", holds=bool(ok),
                        min_wins=min(w for w, _ in relation.values()), of=need + (1 if need == 4 else 0),
                        value=float(mhd.relation_w1.mean()), reference=float(best_derived), detail=json.dumps(relation)))
        # H2
        for method in it.PUBLISHED:
            plug, raw, derived = arms[(f"{method}_rp", "primary")], arms[(method, "primary")], arms[(method, "derived")]
            a = wins(plug, raw)[0] >= need
            if panel == "cbc":
                b = wins(plug, derived)[0] >= need
            else:
                b = plug.relation_w1.mean() <= derived.relation_w1.mean() + .10
            reference = min(raw.c2st.mean(), derived.c2st.mean())
            c = plug.c2st.mean() <= reference + .02
            out.append(dict(**base, hypothesis="H2", unit=NAMES[method], holds=bool(a and b and c), a=bool(a), b=bool(b), c=bool(c),
                            value=float(plug.c2st.mean()), reference=float(reference),
                            relation_plugin=float(plug.relation_w1.mean()), relation_raw=float(raw.relation_w1.mean()),
                            relation_derived=float(derived.relation_w1.mean()), membership_plugin=float(plug.membership.mean())))
        # H3 / H4
        if size <= 250:
            holds = bool(mhd.c2st.mean() < non_memorizing.c2st.min() and mhd.membership.mean() <= .60)
            hypothesis = "H3"
        else:
            holds = bool(mhd.c2st.mean() <= non_memorizing.c2st.min() + .02)
            hypothesis = "H4"
        out.append(dict(**base, hypothesis=hypothesis, unit="MedHyperDiff", holds=holds, value=float(mhd.c2st.mean()),
                        reference=float(non_memorizing.c2st.min()), best_arm=arm_name(*best),
                        membership=float(mhd.membership.mean()), rf=float(mhd.rf.mean()), best_rf=float(non_memorizing.rf.max())))
    return pd.DataFrame(out)


def verdicts(h):
    v = {}
    for name, rule in (("H1", lambda g: g.holds.all()), ("H1x", lambda g: g.holds.all()),
                       ("H2", lambda g: (g.a & g.b).all() and g.c.sum() >= .9 * len(g)), ("H3", lambda g: g.holds.sum() >= 3),
                       ("H4", lambda g: g.holds.sum() >= 4)):
        g = h[h.hypothesis == name]
        v[name] = dict(units=len(g), holding=int(g.holds.sum()), supported=bool(rule(g)) if len(g) else None)
        if name == "H2":
            v[name].update(a=int(g.a.sum()), b=int(g.b.sum()), c=int(g.c.sum()))
    return v


def pooled(d, primary, reps=2000, seed=14):
    """Per size: paired MedHyperDiff-primary minus comparator differences, bootstrap over population x replicate."""
    rng = np.random.default_rng(seed)
    out = []
    for size, g in d.groupby("size"):
        units = []
        for dataset, cell in g.groupby("dataset"):
            means = cell.groupby(["variant", "processing"])[["c2st", "membership"]].mean()
            comp = means[~means.index.get_level_values(0).str.startswith("mhd") & (means.index.get_level_values(0) != "bootstrap")
                         & ~means.index.get_level_values(0).str.endswith("_rp")]
            best = comp[comp.membership <= .60].c2st.idxmin()
            arms = cell_arms(cell)
            mhd = arms[(f"mhd:{primary}", "primary")]
            for label, other in (("best published", arms[best]), ("TabDiff + reparameterization", arms[("tabdiff_rp", "primary")])):
                for s in mhd.index.intersection(other.index):
                    units.append(dict(comparison=label, dataset=dataset, seed=s, c2st=mhd.at[s, "c2st"] - other.at[s, "c2st"],
                                      rf=mhd.at[s, "rf"] - other.at[s, "rf"], relation=mhd.at[s, "relation_w1"] - other.at[s, "relation_w1"]))
        u = pd.DataFrame(units)
        for label, x in u.groupby("comparison"):
            for metric in ("c2st", "rf"):
                values = x[metric].to_numpy()
                boot = [rng.choice(values, len(values)).mean() for _ in range(reps)]
                out.append(dict(size=size, comparison=label, metric=metric, n=len(values), mean=float(values.mean()),
                                low=float(np.quantile(boot, .025)), high=float(np.quantile(boot, .975))))
    return pd.DataFrame(out)


def style(ax, ink2="#52514e", grid="#e4e3df"):
    ax.grid(axis="y", color=grid, linewidth=.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=ink2, labelsize=7, length=0)


def figures(d, primary, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2 = "#0b0b0b", "#52514e"
    colors = {"mhd": "#2a78d6", "c0": "#9ab8e0", "tabdiff": "#c2410c", "tabdiff_rp": "#f59e0b", "best": "#52514e", "copula": "#8a8984"}
    datasets = list(it.CONFIRMATION)
    lines = [((f"mhd:{primary}", "primary"), f"MedHyperDiff ({primary})", colors["mhd"], "-"),
             (("mhd:c0", "primary"), "MedHyperDiff (iteration 13 configuration)", colors["c0"], "--"),
             (("tabdiff", "primary"), "TabDiff, as generated", colors["tabdiff"], ":"),
             (("tabdiff", "derived"), "TabDiff, hard derivation", colors["tabdiff"], "-"),
             (("tabdiff_rp", "primary"), "TabDiff + reparameterization", colors["tabdiff_rp"], "-"),
             (("copula", "resampled"), "Gaussian copula, derivation + resampled residuals", colors["copula"], "-")]
    for metric, label, log, name in (("relation_w1", "Relation residual distance (log scale, lower is better)", True, "relation_by_size"),
                                     ("c2st", "C2ST AUROC (lower is better)", False, "c2st_by_size"),
                                     ("rf", "Forest TSTR AUROC (higher is better)", False, "utility_by_size")):
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=metric != "relation_w1" and False)
        for ax, dataset in zip(axes, datasets):
            cell = d[d.dataset == dataset]
            for key, text, color, dash in lines:
                g = cell[(cell.variant == key[0]) & (cell.processing == key[1])].groupby("size")[metric]
                if not len(g.mean()):
                    continue
                m, s = g.mean(), g.std().fillna(0)
                ax.plot(m.index, m.values, dash, color=color, lw=1.6, marker="o", ms=3.5, label=text)
                ax.fill_between(m.index, (m - s).values, (m + s).values, color=color, alpha=.12, lw=0)
            ax.set_xscale("log")
            ax.set_xticks(sorted(cell["size"].unique()), [f"{n:,}" for n in sorted(cell["size"].unique())])
            ax.minorticks_off()
            if log:
                ax.set_yscale("log")
            ax.set_title(POPULATION[dataset], fontsize=9, color=ink, loc="left")
            ax.set_xlabel("Training records (n)", fontsize=8, color=ink2)
            style(ax)
        axes[0].set_ylabel(label, fontsize=8, color=ink2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(.5, -.02))
        fig.tight_layout(rect=(0, .1, 1, 1))
        fig.savefig(folder / f"{name}.png", dpi=200, facecolor="white")
        plt.close(fig)
    # Plug-in effect: relation distance and C2ST of each published generator before and after reparameterization.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    order = list(it.PUBLISHED)
    for ax, metric, label in ((axes[0], "relation_w1", "Relation residual distance (log)"), (axes[1], "c2st", "C2ST AUROC")):
        for i, method in enumerate(order):
            for k, (processing, variant, marker, color) in enumerate((("primary", method, "o", "#8a8984"), ("derived", method, "s", "#52514e"),
                                                                      ("primary", f"{method}_rp", "D", "#2a78d6"))):
                g = d[(d.variant == variant) & (d.processing == processing)].groupby(["dataset", "size"])[metric].mean()
                x = i + (k - 1) * .22 + np.linspace(-.06, .06, len(g))
                ax.scatter(x, g.values, s=10, marker=marker, color=color, alpha=.8,
                           label=["as generated", "hard derivation", "+ reparameterization"][k] if i == 0 else None)
        ax.set_xticks(range(len(order)), [NAMES[m] for m in order], fontsize=7.5)
        if metric == "relation_w1":
            ax.set_yscale("log")
        ax.set_ylabel(label, fontsize=8, color=ink2)
        style(ax)
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.suptitle("Each point is one population x size cell (mean of replicates)", fontsize=8.5, color=ink2, x=.01, ha="left")
    fig.tight_layout()
    fig.savefig(folder / "plugin_effect.png", dpi=200, facecolor="white")
    plt.close(fig)


def secondary(primary):
    """Iteration 13 population: the chosen configuration against the stored iteration 13 comparators."""
    root = it.OUT
    mine = rows(root, "secondary")
    if not len(mine):
        return None
    import report_iteration13 as r13
    old = r13.rows("confirm")
    old = old[old.status == "complete"]
    old = old.assign(arm=[f"iteration 13: {v}, {PROCESSING.get(p, p)}" for v, p in zip(old.variant, old.processing)])
    mine = mine.assign(arm=f"iteration 14: MedHyperDiff {primary}")
    keep = ["size", "arm", "relation_w1", "c2st", "membership", "rf"]
    table = pd.concat([mine[keep], old[keep]]).groupby(["size", "arm"]).mean().reset_index()
    return table


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "figures").mkdir(exist_ok=True)
    development = rows(it.OUT, "develop")
    development.to_csv(REPORT / "development_all.csv", index=False)
    if not it.CHOICE.exists():
        print(development.groupby(["dataset", "size", "config"])[["relation_w1", "c2st", "membership", "rf"]].mean().round(4))
        return
    primary = json.loads(it.CHOICE.read_text())["primary"]
    confirm = rows(it.OUT, "confirm")
    confirm.to_csv(REPORT / "confirm_all.csv", index=False)
    failed = confirm[confirm.status != "complete"]
    if len(failed):
        print("Incomplete or failed:", failed[["id", "status"]].to_string(index=False))
    d = confirm[confirm.status == "complete"]
    # Amendment 1: n = 100 was dropped; MedHyperDiff n = 100 runs finished before the amendment are kept on disk only.
    d = d[d["size"].isin([*it.SIZES, it.LARGE_SIZE])]
    metrics = ["relation_w1", "c2st", "membership", "rf", "clinical_tv", "numeric_w", "association"] + sorted(c for c in d if c.startswith("w1_"))
    summary = d.groupby(["dataset", "size", "variant", "processing", "arm"])[metrics].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" if b == "std" else a for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(REPORT / "confirm_summary.csv", index=False)
    h = hypotheses(d, primary)
    h.to_csv(REPORT / "hypotheses.csv", index=False)
    v = verdicts(h)
    (REPORT / "verdicts.json").write_text(json.dumps(v, indent=2) + "\n")
    p = pooled(d, primary)
    p.to_csv(REPORT / "pooled.csv", index=False)
    figures(d, primary, REPORT / "figures")
    s = secondary(primary)
    if s is not None:
        s.to_csv(REPORT / "secondary_iteration13_population.csv", index=False)
    print(json.dumps(v, indent=2))
    print(h.drop(columns=[c for c in ("detail",) if c in h]).round(3).to_string(index=False))
    print(p.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
