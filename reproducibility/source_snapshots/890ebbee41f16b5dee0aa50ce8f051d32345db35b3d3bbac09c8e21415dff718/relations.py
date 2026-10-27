"""Relation reparameterization: generate a well-conditioned basis plus learned residuals, then reconstruct.

A supplied relation is an equation among participants (for example haematocrit = RBC x MCV / 10). For each
relation one participant is reconstructed and the generator models a residual column
`res__<target> = target - solve(target | other participants)` instead of the target. After sampling, the
generated columns are decoded to recorded resolution first, then each target is reconstructed in an acyclic
order and decoded. Exact and approximate relations are handled alike: the residual distribution is learned,
so rounding noise and rare discrepancies are reproduced rather than forced to zero or ignored.

Which participant to reconstruct matters. Errors in generated operands propagate into the reconstructed
variable; reconstructing a narrow variable (a small percentage, a tightly regulated index) from wide ones is
ill-conditioned. `RelationCoder.fit` therefore chooses, from fit rows only, one target per relation so that an
acyclic reconstruction order exists and the summed spread of the targets is largest, measuring spread on the
scale on which the relation is additive (log scale for products and ratios, raw scale for sums and differences,
coefficient-weighted contributions for linear formulas).

Some relations are reported exactly: the laboratory computes the derived value from the reported (rounded)
operands and rounds it, so real residuals are only the derived value's own rounding. `fit` detects these from fit
rows (at least 99% of complete rows within half a recording step of the formula), and `decode` then re-derives the
reported value from the decoded operands, as the laboratory does, after the chosen participant has been
reconstructed. Without this step the rounding of a reconstructed operand (for example urine albumin recorded to
0.1 mg/L) would add relation error that real records do not have.
"""
from dataclasses import dataclass, replace
import itertools

import numpy as np
import pandas as pd

MULTIPLICATIVE = ("scaled_product", "scaled_ratio", "bmi")


@dataclass(frozen=True)
class Relation:
    """Canonical form `derived = f(operands)`; any participant can be solved for."""
    name: str
    derived: str
    operands: tuple
    kind: str          # "scaled_product" (k*a*b), "scaled_ratio" (k*a/b), "difference" (a-b), "complement" (k - sum), "bmi",
                       # "linear" (k + sum(w*x))
    constant: float = 1.
    weights: tuple = ()

    @property
    def participants(self):
        return (self.derived, *self.operands)

    def solve(self, target, values):
        """Value of `target` implied by the other participants (NaN where undefined)."""
        safe = lambda x: np.where(np.abs(x) < 1e-9, np.nan, x)
        d, k = self.derived, self.constant
        if target == d:
            ops = [values[c] for c in self.operands]
            if self.kind == "scaled_product":
                return k * ops[0] * ops[1]
            if self.kind == "scaled_ratio":
                return k * ops[0] / safe(ops[1])
            if self.kind == "difference":
                return ops[0] - ops[1]
            if self.kind == "complement":
                return k - np.sum(ops, axis=0)
            if self.kind == "bmi":
                return ops[0] / (np.where(ops[1] < 30, np.nan, ops[1]) / 100) ** 2
            if self.kind == "linear":
                return k + np.sum([w * x for w, x in zip(self.weights, ops)], axis=0)
            raise ValueError(self.kind)
        if self.kind == "linear":
            others = [w * values[c] for w, c in zip(self.weights, self.operands) if c != target]
            return (values[d] - k - np.sum(others, axis=0)) / self.weight(target)
        if self.kind == "complement":
            others = [values[c] for c in self.operands if c != target]
            return k - values[d] - np.sum(others, axis=0)
        a, b = self.operands
        if self.kind == "scaled_product":
            return values[d] / safe(k * values[b if target == a else a])
        if self.kind == "scaled_ratio":
            return values[d] * values[b] / k if target == a else k * values[a] / safe(values[d])
        if self.kind == "difference":
            return values[d] + values[b] if target == a else values[a] - values[d]
        if self.kind == "bmi":
            return values[d] * (values[b] / 100) ** 2 if target == a else 100 * np.sqrt(np.clip(values[a] / safe(values[d]), 0, None))
        raise ValueError(self.kind)

    def weight(self, column):
        """Coefficient of an operand on the relation's additive scale (1 for non-linear kinds)."""
        return self.weights[self.operands.index(column)] if self.kind == "linear" and column in self.operands else 1.

    def solved_for(self, target):
        return Derivation(self, target)


@dataclass(frozen=True)
class Derivation:
    """`target = g(other participants)`; exposes `name`, `derived`, `operands` and `formula`."""
    relation: Relation
    target: str

    @property
    def name(self):
        return self.relation.name

    @property
    def derived(self):
        return self.target

    @property
    def operands(self):
        return tuple(c for c in self.relation.participants if c != self.target)

    def formula(self, values):
        return self.relation.solve(self.target, values)


def residual_name(derivation):
    return f"res__{derivation.derived}"


def spread(relation, column, frame):
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    values = values[np.isfinite(values)]
    if relation.kind in MULTIPLICATIVE:
        values = np.log(values[values > 0])
    return abs(relation.weight(column)) * float(np.std(values)) if len(values) > 1 else 0.


def choose_derivations(relations, frame):
    """One target per relation, distinct, acyclic, maximizing summed relation-scale spread (fit rows only)."""
    best, best_score = None, -np.inf
    for targets in itertools.product(*[r.participants for r in relations]):
        if len(set(targets)) < len(targets):
            continue
        pending = [r.solved_for(t) for r, t in zip(relations, targets)]
        derived, available, order = set(targets), set(), []
        while pending:
            ready = [d for d in pending if all(op not in derived or op in available for op in d.operands)]
            if not ready:
                break
            order.append(ready[0])
            available.add(ready[0].derived)
            pending.remove(ready[0])
        if pending:
            continue
        score = sum(spread(r, t, frame) for r, t in zip(relations, targets))
        if score > best_score + 1e-12:
            best, best_score = order, score
    if best is None:
        raise ValueError("No acyclic reconstruction order exists for these relations")
    return best


class RelationCoder:
    """Encode a frame to generated columns + residuals and decode back."""

    def __init__(self, schema, relations, derivations=None, exact=None):
        self.schema, self.supplied = schema, list(relations)
        self.relations = list(derivations) if derivations is not None else [r.solved_for(r.derived) for r in self.supplied]
        self.exact = dict(exact or {})
        self._update_schema()

    def fit(self, frame, detect_exact=True):
        self.relations = choose_derivations(self.supplied, frame)
        self.exact = {r.name: reported_exact(r, frame) for r in self.supplied} if detect_exact else {}
        self._update_schema()
        return self

    def _update_schema(self):
        derived = [r.derived for r in self.relations]
        if len(set(derived)) != len(derived):
            raise ValueError("Each derived column may be reparameterized once")
        numeric = [c for c in self.schema.numeric if c not in derived] + [residual_name(r) for r in self.relations]
        self.encoded_schema = replace(self.schema, name=f"{self.schema.name}__reparam", numeric=tuple(numeric))

    def describe(self):
        return [{"relation": r.name, "target": r.derived, "from": list(r.operands), "reported_exact": self.exact.get(r.name, False)}
                for r in self.relations]

    def encode(self, frame):
        out = frame.copy()
        for relation in self.relations:
            values = {c: pd.to_numeric(out[c], errors="coerce").to_numpy(float) for c in (relation.derived, *relation.operands)}
            out[residual_name(relation)] = values[relation.derived] - relation.formula(values)
        return out[self.encoded_schema.columns]

    def decode(self, frame, round_fn=None):
        """Reconstruct targets in order. `round_fn(column_name, values)` decodes recorded resolution."""
        out = frame.copy()
        rounder = round_fn or (lambda name, values: values)
        derived = {r.derived for r in self.relations}
        for column in self.schema.numeric:
            if column not in derived:
                out[column] = rounder(column, pd.to_numeric(out[column], errors="coerce").to_numpy(float))
        used = set()
        for relation in self.relations:
            values = {c: pd.to_numeric(out[c], errors="coerce").to_numpy(float) for c in relation.operands}
            target = relation.formula(values) + pd.to_numeric(out[residual_name(relation)], errors="coerce").to_numpy(float)
            out[relation.derived] = rounder(relation.derived, target)
            canonical = relation.relation
            # Reported-exact relation: derive the reported value from the decoded operands, unless an earlier
            # reconstruction already consumed it.
            if self.exact.get(canonical.name) and canonical.derived not in used:
                values = {c: pd.to_numeric(out[c], errors="coerce").to_numpy(float) for c in canonical.operands}
                out[canonical.derived] = rounder(canonical.derived, canonical.solve(canonical.derived, values))
            used.update(relation.operands)
        return out[self.schema.columns]


def reported_exact(relation, frame, share=.99):
    """True when the derived value equals the formula of the reported operands up to its own recording step."""
    from .preprocessing import precision_bands
    values = {c: pd.to_numeric(frame[c], errors="coerce").to_numpy(float) for c in relation.participants}
    residual = values[relation.derived] - relation.solve(relation.derived, values)
    residual = residual[np.isfinite(residual)]
    decimals = precision_bands(values[relation.derived])["global"]
    if not len(residual) or decimals is None:
        return False
    step = 10.0 ** -decimals
    return bool(np.mean(np.abs(residual) <= step / 2 * (1 + 1e-6) + 1e-9) >= share)


RELATIONS = {
    # NHANES complete blood count (CBC_J): the analyser measures RBC, haemoglobin and MCV and computes
    # haematocrit, MCH and MCHC; the five differential percentages are reported to 0.1 and sum to ~100.
    "nhanes_cbc": [Relation("hct", "hct", ("rbc", "mcv"), "scaled_product", .1),
                   Relation("mch", "mch", ("hgb", "rbc"), "scaled_ratio", 10.),
                   Relation("mchc", "mchc", ("hgb", "hct"), "scaled_ratio", 100.),
                   Relation("differential", "baso_pct", ("lymph_pct", "mono_pct", "neut_pct", "eos_pct"), "complement", 100.)],
    "thyroid": [Relation("fti", "fti", ("tt4", "t4u"), "scaled_ratio", 1.)],
    "nhanes": [Relation("globulin", "globulin", ("total_protein", "albumin"), "difference"),
               Relation("bmi", "bmi", ("weight", "height"), "bmi")],
    # NHANES chemistry, urine and body-measure relations as reported (cycles G, I, L): globulin = total protein -
    # albumin, BMI = weight/height^2, ACR = 100 x urine albumin / urine creatinine, and the calculated serum
    # osmolality 1.86 x Na + glucose/18 + BUN/2.8 + 9.
    "nhanes_chem": [Relation("globulin", "globulin", ("total_protein", "albumin"), "difference"),
                    Relation("bmi", "bmi", ("weight", "height"), "bmi"),
                    Relation("acr", "acr", ("urine_albumin", "urine_creatinine"), "scaled_ratio", 100.),
                    Relation("osmolality", "osmolality", ("sodium", "glucose", "bun"), "linear", 9., (1.86, 1 / 18, 1 / 2.8))],
}


def relations_for(dataset):
    """Supplied relations of a dataset; further NHANES cycles share their panel's relations."""
    for panel in ("nhanes_cbc", "nhanes_chem"):
        if dataset.startswith(panel):
            return RELATIONS[panel]
    return RELATIONS[dataset]
