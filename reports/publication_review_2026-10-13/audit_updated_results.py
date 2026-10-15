"""Read-only audit of the revised paper; numpy/pandas only, no generator fits.

Checks iteration 15 reporting and the newly added canonical copula control.
The independent density attack implementation is validated against all five
saved primary-control attack scores before applying it to canonical outputs.
Run with a Python environment containing numpy and pandas.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def selected_definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = []
    for node in tree.body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
        if name in names:
            selected.append(node)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


def auc(y, score):
    positive, negative = score[y == 1], score[y == 0]
    delta = positive[:, None] - negative[None, :]
    return float(np.mean((delta > 0) + .5 * (delta == 0)))


def attack_features(background, frames, schema):
    normal = NormalDist()
    known = {c: np.sort(background[c].dropna().to_numpy(float)) for c in schema.numeric}
    levels = {c: sorted(background[c].fillna("__MISSING__").astype(str).unique()) for c in schema.categorical}
    result = []
    for frame in frames:
        parts = []
        for c in schema.numeric:
            values = frame[c].to_numpy(float)
            ranks = (np.searchsorted(known[c], np.nan_to_num(values), side="left") +
                     np.searchsorted(known[c], np.nan_to_num(values), side="right")) / 2
            p = np.clip((ranks + .5) / (len(known[c]) + 1), 1e-4, 1 - 1e-4)
            scores = np.array([normal.inv_cdf(float(v)) for v in p]) if len(known[c]) else np.zeros(len(values))
            parts += [np.where(np.isnan(values), 0., scores), np.isnan(values).astype(float)]
        for c in schema.categorical:
            values = frame[c].fillna("__MISSING__").astype(str).to_numpy()
            parts += [(values == level).astype(float) for level in levels[c]]
        result.append(np.column_stack(parts))
    return result


def density_attack(members, nonmembers, synthetic, background, schema):
    ref, syn, mem, non = attack_features(background, [background, synthetic, members, nonmembers], schema)
    keep = ref.std(0) > 1e-9
    center = ref[:, keep].mean(0)
    centered = ref[:, keep] - center
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
    basis = eigenvectors[:, np.argsort(eigenvalues)[::-1][:min(8, int(keep.sum()))]]
    projected = [(x[:, keep] - center) @ basis for x in (ref, syn, mem, non)]
    scale = projected[0].std(0)
    scale[scale < 1e-9] = 1
    ref_z, syn_z, mem_z, non_z = [x / scale for x in projected]

    def log_density(training, query):
        dimension = training.shape[1]
        bandwidth = len(training) ** (-1 / (dimension + 4))
        constant = np.log(len(training)) + dimension * np.log(bandwidth) + .5 * dimension * np.log(2 * np.pi)
        output = []
        for start in range(0, len(query), 100):
            chunk = query[start:start + 100]
            distances = (chunk ** 2).sum(1)[:, None] + (training ** 2).sum(1)[None, :] - 2 * chunk @ training.T
            exponent = -np.maximum(distances, 0) / (2 * bandwidth ** 2)
            largest = exponent.max(1)
            output.append(largest + np.log(np.exp(exponent - largest[:, None]).sum(1)) - constant)
        return np.concatenate(output)

    queries = np.concatenate([mem_z, non_z])
    scores = log_density(syn_z, queries) - log_density(ref_z, queries)
    return auc(np.r_[np.ones(len(members)), np.zeros(len(nonmembers))], scores)


def main():
    ns = {"np": np, "pd": pd, "dataclass": dataclass, "roc_auc_score": auc}
    selected_definitions(ROOT / "src/medhyperdiff/data.py", {"Schema", "SCHEMAS", "canonical_category", "normalize"}, ns)
    selected_definitions(ROOT / "src/medhyperdiff/preprocessing.py", {"_value_decimals", "precision_bands", "round_banded"}, ns)
    selected_definitions(ROOT / "src/medhyperdiff/relations.py", {"Relation", "RELATIONS"}, ns)
    selected_definitions(ROOT / "src/medhyperdiff/evaluation.py", {"gower_distances", "nearest", "privacy"}, ns)
    dataset = "nhanes_chem_l"
    schema = ns["SCHEMAS"][dataset]
    frame = ns["normalize"](pd.read_csv(ROOT / f"data/processed/{dataset}.csv", dtype={"sex": str, "diabetes": str}), schema)
    plan = json.loads((ROOT / "configs/iteration14_splits.json").read_text())["datasets"][dataset]
    reference = frame.iloc[plan["reference"]]
    canonical_scores = pd.read_csv(ROOT / "reports/iteration14/reanalysis/canonical.csv")
    reported = pd.read_csv(ROOT / "reports/iteration14/confirm_all.csv")
    rows = []
    for seed in range(5):
        directory = ROOT / f"runs/iteration14/confirm/{dataset}/copula/n250_r{seed}"
        sample = plan["samples"][f"n250_r{seed}"]
        fit = frame.iloc[sample["fit"]]
        table = pd.read_csv(directory / "synthetic_primary.csv", dtype={"sex": str, "diabetes": str}, float_precision="round_trip")
        saved = json.loads((directory / "metrics.json").read_text())["primary"]
        indices = np.random.default_rng(2026 + seed).choice(len(reference), min(len(fit), len(reference) // 2), replace=False)
        nonmembers, background = reference.iloc[indices], reference.drop(reference.index[indices])
        nearest_primary = ns["privacy"](fit, reference, table, schema, 2026)["nearest_synthetic_membership_auc"]
        density_primary = density_attack(fit, nonmembers, table, background, schema)
        assert abs(nearest_primary - saved["privacy"]["nearest_synthetic_membership_auc"]) < 1e-10
        assert abs(density_primary - saved["density_attack"]["auc"]) < 1e-10
        bands = {c: ns["precision_bands"](fit[c]) for c in schema.numeric}
        for relation in ns["RELATIONS"]["nhanes_chem"]:
            values = {c: pd.to_numeric(table[c], errors="coerce").to_numpy(float) for c in relation.operands}
            table[relation.derived] = ns["round_banded"](relation.solve(relation.derived, values), bands[relation.derived])
        nearest_canonical = ns["privacy"](fit, reference, table, schema, 2026)["nearest_synthetic_membership_auc"]
        density_canonical = density_attack(fit, nonmembers, table, background, schema)
        cell = canonical_scores[(canonical_scores.dataset == dataset) & (canonical_scores["size"] == 250) &
                                (canonical_scores.variant == "copula") & (canonical_scores.seed == seed)].iloc[0]
        ours = reported[(reported.dataset == dataset) & (reported["size"] == 250) &
                        (reported.variant == "mhd:c1") & (reported.processing == "primary") & (reported.seed == seed)].iloc[0]
        rows.append(dict(seed=seed, nearest_primary=nearest_primary, density_primary=density_primary,
                         nearest_canonical=nearest_canonical, density_canonical=density_canonical,
                         membership_canonical=max(nearest_canonical, density_canonical),
                         c2st_canonical=float(cell.c2st), c2st_ours=float(ours.c2st)))
        print(rows[-1], flush=True)
    validation = pd.read_csv(ROOT / "reports/iteration15/all.csv")
    mismatches = []
    manifests = list((ROOT / "runs/iteration15").rglob("manifest.json"))
    for path in manifests:
        m = json.loads(path.read_text())
        x = json.loads((path.parent / "metrics.json").read_text())
        r = validation[(validation.dataset == m["dataset"]) & (validation["size"] == m["size"]) &
                       (validation.replicate == m["replicate"]) & (validation.variant == m["variant"])].iloc[0]
        for key, value in (("c2st", x["c2st"]["auc"]),
                           ("membership", max(x["privacy"]["nearest_synthetic_membership_auc"], x["density_attack"]["auc"])),
                           ("rf", x["utility"]["forest"]["TSTR"]["auroc"])):
            if abs(r[key] - value) > 1e-12:
                mismatches.append([m["id"], key])
    result = dict(pdf_sha256=hashlib.sha256((ROOT / "paper/ieee/main.pdf").read_bytes()).hexdigest(),
                  iteration15_manifests=len(manifests), iteration15_report_mismatches=mismatches,
                  canonical_copula_rows=rows, canonical_copula_means=pd.DataFrame(rows).mean().to_dict(),
                  canonical_copula_c2st_wins=int(sum(r["c2st_canonical"] < r["c2st_ours"] for r in rows)),
                  audit_scope="Saved-output audit; C2ST values from canonical.csv, attacks recomputed without fitting a generator.")
    (OUT / "audit_evidence.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
