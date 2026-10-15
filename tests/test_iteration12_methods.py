"""Iteration-12 additions: definitional rule kinds, banded recording precision, C2ST curve, density attack."""
import numpy as np
import pandas as pd
import pytest
import torch

from medhyperdiff.constraints import ConstraintSet, Rule, repair_simple
from medhyperdiff.data import SCHEMAS
from medhyperdiff.evaluation import c2st_curve, density_ratio_attack
from medhyperdiff.preprocessing import Preprocessor, precision_bands, round_banded


def test_definitional_residuals_numpy_torch_and_repair_agree():
    rules = ConstraintSet([Rule("g", "difference", ("c", "a", "b"), tolerance=.05),
                           Rule("r", "ratio", ("q", "a", "b"), tolerance=.5),
                           Rule("m", "bmi", ("bmi", "w", "h"), tolerance=.05)])
    frame = pd.DataFrame({"a": [7.1, 6.0], "b": [4.3, 2.0], "c": [2.8, 3.1], "q": [1.65, 3.0],
                          "w": [80., 50.], "h": [180., 160.], "bmi": [24.7, 30.]})
    values = {k: frame[k].to_numpy() for k in frame}
    tensors = {k: torch.as_tensor(v) for k, v in values.items()}
    for rule in rules.rules:
        np.testing.assert_allclose(rules.residual(rule, values), rules.residual(rule, tensors).numpy())
    result = rules.evaluate(frame)["rules"]
    assert result["g"]["violation_rate"] == .5 and result["m"]["violation_rate"] == .5
    fixed, changed = repair_simple(frame, rules)
    assert changed == 1. and all(v["violation_rate"] == 0 for v in rules.evaluate(fixed)["rules"].values())


def test_precision_bands_follow_significant_figures_and_ignore_float_artifacts():
    values = np.r_[np.round(np.linspace(1.1, 9.9, 80), 1), [9.599999, 9.799999],
                   np.round(np.linspace(.011, .099, 40), 3), np.arange(12., 90.)]
    spec = precision_bands(values)
    assert spec["bands"][0] == 1 and spec["bands"][-2] == 3 and spec["bands"][1] == 0
    out = round_banded(np.array([5.123, .04567, 42.6, 0., np.nan, 1234.5]), spec)
    np.testing.assert_allclose(out[:4], [5.1, .046, 43., 0.])
    assert np.isnan(out[4]) and out[5] == 1234. or out[5] == 1235.


def test_banded_preprocessor_roundtrip_serializes():
    schema = SCHEMAS["toy"]
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"total_protein": np.round(rng.uniform(5, 9, 60), 1), "albumin": np.round(rng.uniform(2, 5, 60), 1),
                          "ag_ratio": np.round(rng.uniform(.5, 2, 60), 2), "age": rng.integers(20, 80, 60).astype(float),
                          "symptom_a": rng.integers(0, 2, 60).astype(str), "symptom_b": rng.integers(0, 2, 60).astype(str),
                          "symptom_c": rng.integers(0, 2, 60).astype(str), "disease": np.r_[np.ones(30, int), np.zeros(30, int)].astype(str)})
    prep = Preprocessor(schema, "quantile", "banded").fit(frame)
    numeric = rng.normal(size=(40, 4)).astype(np.float32)
    _, categorical = prep.transform(frame.iloc[:40])
    decoded = prep.inverse(numeric, categorical)
    assert np.allclose(decoded["albumin"] * 10, np.round(decoded["albumin"] * 10))
    pd.testing.assert_frame_equal(Preprocessor.from_dict(schema, prep.to_dict()).inverse(numeric, categorical), decoded)


def _toy_tables(seed=0, n=300):
    rng = np.random.default_rng(seed)
    make = lambda m: pd.DataFrame({"total_protein": rng.normal(7, .5, m), "albumin": rng.normal(4, .4, m),
        "ag_ratio": rng.normal(1.2, .2, m), "age": rng.normal(50, 10, m),
        "symptom_a": rng.integers(0, 2, m).astype(str), "symptom_b": rng.integers(0, 2, m).astype(str),
        "symptom_c": rng.integers(0, 2, m).astype(str), "disease": rng.integers(0, 2, m).astype(str)})
    return make(n), make(n), make(3 * n)


def test_density_attack_flags_copies_but_not_independent_draws():
    schema = SCHEMAS["toy"]
    members, nonmembers, background = _toy_tables()
    copies = pd.concat([members] * 3, ignore_index=True)
    fresh = _toy_tables(seed=9)[2]
    assert density_ratio_attack(members, nonmembers, copies, background, schema)["auc"] > .8
    assert abs(density_ratio_attack(members, nonmembers, fresh, background, schema)["auc"] - .5) < .1


def test_c2st_curve_sizes_and_identity():
    schema = SCHEMAS["toy"]
    real, _, _ = _toy_tables(n=240)
    rows = c2st_curve(real, real.sample(frac=1, random_state=1).reset_index(drop=True), schema, sizes=(50, 100, 500), repeats=2)
    assert sorted({r["size"] for r in rows}) == [50, 100] and len(rows) == 4


def test_relation_coder_roundtrip_and_rounding_order():
    from medhyperdiff.data import Schema
    from medhyperdiff.relations import Relation, RelationCoder
    schema = Schema("cbc_toy", ("rbc", "mcv", "hct", "a", "b", "c"), ("sex",), "y", "1")
    rng = np.random.default_rng(1)
    rbc, mcv = np.round(rng.uniform(3.5, 6, 50), 2), np.round(rng.uniform(80, 100, 50), 1)
    frame = pd.DataFrame({"rbc": rbc, "mcv": mcv, "hct": np.round(rbc * mcv / 10, 1), "a": np.round(rng.uniform(10, 60, 50), 1),
                          "b": np.round(rng.uniform(10, 30, 50), 1), "sex": "f", "y": "1"})
    frame["c"] = np.round(100 - frame.a - frame.b, 1)
    frame.loc[3, "mcv"] = np.nan
    relations = [Relation("hct", "hct", ("rbc", "mcv"), "scaled_product", .1), Relation("sum", "c", ("a", "b"), "complement", 100.)]
    coder = RelationCoder(schema, relations)
    encoded = coder.encode(frame)
    assert "hct" not in encoded and "res__hct" in encoded and np.isnan(encoded.loc[3, "res__hct"])
    assert np.abs(encoded["res__c"]).max() < 1e-9 and np.abs(encoded["res__hct"].dropna()).max() <= .05 + 1e-9
    decoded = coder.decode(encoded)
    pd.testing.assert_frame_equal(decoded[["rbc", "hct", "c"]].iloc[[0, 1, 2]], frame[["rbc", "hct", "c"]].iloc[[0, 1, 2]], atol=1e-9)
    noisy = encoded.copy()
    noisy["rbc"] += .0042
    rounded = coder.decode(noisy, lambda name, v: np.round(v, 2 if name == "rbc" else 1))
    assert np.allclose(rounded.rbc.dropna() * 100, np.round(rounded.rbc.dropna() * 100))
    assert np.allclose(rounded.c, np.round(100 - rounded.a - rounded.b, 1))


def test_structure_rules_have_zero_energy_but_create_hyperedges():
    from medhyperdiff.constraints import ConstraintSet, Rule
    rules = ConstraintSet([Rule("s", "structure", ("x", "y", "z"))])
    assert rules.hyperedges(["x", "y", "z", "w"]) == [[0, 1, 2]]
    values = {k: np.array([1., 2.]) for k in "xyz"}
    assert np.all(rules.residual(rules.rules[0], values) == 0)


def test_solve_for_every_participant_and_conditioning_choice():
    from medhyperdiff.relations import Relation, RELATIONS, choose_derivations
    rng = np.random.default_rng(3)
    rbc, mcv, mchc = rng.normal(4.7, .5, 400), rng.normal(90, 6, 400), rng.normal(33.5, .9, 400)
    hct = rbc * mcv / 10
    hgb = mchc * hct / 100
    frame = pd.DataFrame({"rbc": rbc, "mcv": mcv, "mchc": mchc, "hct": hct, "hgb": hgb, "mch": 10 * hgb / rbc})
    values = {c: frame[c].to_numpy() for c in frame}
    for relation in RELATIONS["nhanes_cbc"][:3]:
        for target in relation.participants:
            np.testing.assert_allclose(relation.solve(target, values), values[target], rtol=1e-10)
    order = choose_derivations(RELATIONS["nhanes_cbc"][:3], frame)
    targets = [d.derived for d in order]
    assert "mchc" not in targets and len(set(targets)) == 3
    available = set(frame) - set(targets)
    for derivation in order:  # acyclic: every operand is generated or reconstructed earlier
        assert set(derivation.operands) <= available
        available.add(derivation.derived)
    bmi = Relation("bmi", "bmi", ("weight", "height"), "bmi")
    v = {"weight": np.array([80.]), "height": np.array([180.]), "bmi": np.array([80 / 1.8 ** 2])}
    np.testing.assert_allclose(bmi.solve("height", v), [180.])
