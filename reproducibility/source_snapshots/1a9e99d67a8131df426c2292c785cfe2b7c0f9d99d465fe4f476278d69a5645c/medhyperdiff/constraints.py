"""Auditable constraints with physical-unit residuals and observed-value coverage.

Confidence here is a fixed sensitivity weight, not calibrated uncertainty.
Real-data algebraic rules are soft until assay/rounding semantics are reviewed.
"""
from dataclasses import dataclass, asdict, replace

import numpy as np
import pandas as pd
import torch


PROTEIN_SOURCE = "https://medlineplus.gov/lab-tests/total-protein-and-albumin-globulin-a-g-ratio/"
BILIRUBIN_SOURCE = "https://medlineplus.gov/ency/article/003479.htm"


@dataclass(frozen=True)
class Rule:
    name: str
    kind: str
    columns: tuple[str, ...]
    confidence: float = 1.0
    tolerance: float = 0.0
    scale: float = 1.0
    bound: float = 0.0
    status: str = "schema"
    source: str = ""
    units: str = "source measurement units; assay compatibility requires review"
    review_status: str = "not clinically reviewed"


class ConstraintSet:
    def __init__(self, rules):
        self.rules = list(rules)
        for rule in self.rules:
            if rule.scale <= 0 or rule.tolerance < 0 or not 0 <= rule.confidence <= 1:
                raise ValueError("Constraint scale, tolerance, or confidence is invalid")

    def to_dict(self):
        return [asdict(r) for r in self.rules]

    def hyperedges(self, names):
        return [[names.index(c) for c in r.columns] for r in self.rules if len(r.columns) >= 2]

    @staticmethod
    def residual(rule, values):
        v = [values[c] for c in rule.columns]
        if rule.kind == "lower":
            return rule.bound - v[0]
        if rule.kind == "upper":
            return v[0] - rule.bound
        if rule.kind == "le":
            return v[0] - v[1]
        if rule.kind == "protein":
            # Algebraically equivalent to r = a/(t-a) for t>a, without singularity.
            total, albumin, ratio = v
            return abs(ratio * (total - albumin) - albumin)
        if rule.kind == "parity":
            # Expected violation under factorized categorical probabilities.
            a, b, c = v
            return (1 - (1 - 2*a) * (1 - 2*b) * (1 - 2*c)) / 2
        raise ValueError(f"Unknown rule kind {rule.kind}")

    def evaluate(self, frame):
        results = {}
        any_eligible = np.zeros(len(frame), dtype=bool)
        any_violation = np.zeros(len(frame), dtype=bool)
        for rule in self.rules:
            valid = frame[list(rule.columns)].notna().all(axis=1).to_numpy()
            vals = {c: pd.to_numeric(frame[c], errors="coerce").to_numpy() for c in rule.columns}
            raw = np.asarray(self.residual(rule, vals), dtype=float)
            valid &= np.isfinite(raw)
            severity = np.maximum(raw - rule.tolerance, 0) / rule.scale
            n = int(valid.sum())
            violated = valid & (severity > 1e-6)
            any_eligible |= valid
            any_violation |= violated
            results[rule.name] = {"eligible": n, "coverage": n / max(len(frame), 1),
                "violation_rate": float(violated.sum()/n) if n else None,
                "mean_severity": float(severity[valid].mean()) if n else None,
                "status": rule.status, "tolerance": rule.tolerance, "scale": rule.scale,
                "confidence": rule.confidence}
        return {"rules": results, "any_rule_violation_rate": float(any_violation.sum()/any_eligible.sum()) if any_eligible.any() else None,
                "any_rule_coverage": float(any_eligible.mean()) if len(frame) else 0}

    def energy(self, numeric, logits, preprocessor, observed=None, detach_masks=True):
        """Per-row energy. Do not backpropagate into eligibility/missingness masks.

        At training use real observation masks. At sampling use detached predicted
        observation probabilities, avoiding a direct incentive to hide violations.
        Final observed coverage still needs auditing for indirect evasion.
        """
        values = {}
        eligibility = {}
        for i, col in enumerate(preprocessor.schema.numeric):
            values[col] = numeric[:, i] * float(preprocessor.scales[i]) + float(preprocessor.means[i])
            mask_index = len(preprocessor.cat_names) + i
            p_observed = logits[mask_index].softmax(-1)[:, 1]
            eligibility[col] = observed[:, i] if observed is not None else p_observed.detach() if detach_masks else p_observed
        for i, col in enumerate(preprocessor.cat_names):
            levels = preprocessor.levels[i]
            probs = logits[i].softmax(-1)
            if "1" in levels:
                values[col] = probs[:, levels.index("1")]
            eligibility[col] = 1 - probs[:, levels.index(preprocessor.missing)].detach() - probs[:, levels.index(preprocessor.unknown)].detach()
        energy = numeric.new_zeros(numeric.shape[0])
        weights = 0.0
        for rule in self.rules:
            residual = self.residual(rule, values)
            severity = torch.relu(residual - rule.tolerance) / rule.scale
            mask = torch.ones_like(energy)
            for col in rule.columns:
                mask = mask * eligibility[col]
            # SmoothL1-like growth prevents one corrupt lab from dominating gradients.
            penalty = torch.where(severity < 1, .5 * severity.square(), severity - .5)
            energy = energy + rule.confidence * mask * penalty
            weights += rule.confidence
        return energy / max(weights, 1.0)


def build_constraints(schema, train):
    source = f"https://archive.ics.uci.edu/dataset/{schema.uci_id}" if schema.uci_id else "controlled toy generator"
    rules = []
    for col in schema.numeric:
        scale = float(train[col].std())
        scale = scale if np.isfinite(scale) and scale > 1e-6 else 1
        rules.append(Rule(f"{col}_nonnegative", "lower", (col,), scale=scale, source=source))
    if schema.name == "heart_failure":
        rules.append(Rule("ejection_fraction_percentage", "upper", ("ejection_fraction",), bound=100, scale=100, source=source))
    if schema.name in ("ilpd", "toy"):
        rules.append(Rule("albumin_le_total", "le", ("albumin", "total_protein"),
                          status="soft_biochemical" if schema.name == "ilpd" else "controlled", source=PROTEIN_SOURCE))
        # Tolerance estimated solely from fit data is a descriptive residual envelope,
        # not a clinical acceptance threshold or calibrated confidence probability.
        raw = train.ag_ratio * (train.total_protein - train.albumin) - train.albumin
        raw = raw.abs().dropna()
        tolerance = float(raw.quantile(.95)) if len(raw) and schema.name == "ilpd" else .02
        rules.append(Rule("protein_identity", "protein", ("total_protein", "albumin", "ag_ratio"),
            confidence=.5 if schema.name == "ilpd" else 1, tolerance=tolerance,
            status="exploratory_train_calibrated" if schema.name == "ilpd" else "controlled", source=PROTEIN_SOURCE))
    if schema.name == "ilpd":
        rules.append(Rule("direct_le_total_bilirubin", "le", ("direct_bilirubin", "total_bilirubin"),
            confidence=.75, tolerance=.1, scale=1, status="soft_biochemical", source=BILIRUBIN_SOURCE))
    if schema.name == "toy":
        rules.append(Rule("parity", "parity", ("symptom_a", "symptom_b", "symptom_c"), status="controlled", source="toy XOR mechanism"))
    return ConstraintSet(rules)


def with_unit_confidence(constraints):
    return ConstraintSet([replace(r, confidence=1.) for r in constraints.rules])


def repair_simple(frame, constraints):
    """Transparent postprocessing comparator; return corrected copy and change rate.

    Only supports the simple constraints used here; never used silently by diffusion.
    Missing values are preserved. This is not an implementation of HFGF/C-DGM.
    """
    fixed = frame.copy()
    for rule in constraints.rules:
        cols = rule.columns
        if rule.kind == "lower":
            fixed[cols[0]] = fixed[cols[0]].clip(lower=rule.bound)
        elif rule.kind == "upper":
            fixed[cols[0]] = fixed[cols[0]].clip(upper=rule.bound)
        elif rule.kind == "le":
            eligible = fixed[list(cols)].notna().all(axis=1)
            fixed.loc[eligible, cols[0]] = np.minimum(fixed.loc[eligible, cols[0]], fixed.loc[eligible, cols[1]])
        elif rule.kind == "protein":
            total, albumin, ratio = cols
            eligible = fixed[list(cols)].notna().all(axis=1)
            globulin = (fixed[total] - fixed[albumin]).clip(lower=.1)
            fixed.loc[eligible, total] = fixed.loc[eligible, albumin] + globulin[eligible]
            fixed.loc[eligible, ratio] = fixed.loc[eligible, albumin] / globulin[eligible]
        elif rule.kind == "parity":
            a, b, c = cols
            eligible = fixed[list(cols)].notna().all(axis=1)
            fixed.loc[eligible, c] = (fixed.loc[eligible, a].astype(int) ^ fixed.loc[eligible, b].astype(int)).astype(str)
    changed = ~(fixed.eq(frame) | (fixed.isna() & frame.isna())).all(axis=1)
    return fixed, float(changed.mean())
