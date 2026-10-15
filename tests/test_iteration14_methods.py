"""Iteration 14: linear relations, reported-exact re-derivation, clinical categories and size-scaling candidates."""
import numpy as np
import pandas as pd
import torch

from medhyperdiff.constraints import ConstraintSet, Rule
from medhyperdiff.data import SCHEMAS, Schema
from medhyperdiff.evaluation import clinical_categories, clinical_category_tv
from medhyperdiff.relations import RELATIONS, Relation, RelationCoder, choose_derivations, relations_for, reported_exact


def chemistry_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    sodium, glucose, bun = np.round(rng.normal(140, 2.5, n)), np.round(rng.lognormal(4.6, .3, n)), np.round(rng.lognormal(2.6, .35, n))
    albumin, total_protein = np.round(rng.normal(4.2, .3, n), 1), np.round(rng.normal(7.1, .45, n), 1)
    weight, height = np.round(rng.lognormal(4.4, .22, n), 1), np.round(rng.normal(168, 10, n), 1)
    urine_albumin, urine_creatinine = np.round(rng.lognormal(2.2, 1.2, n), 1), np.round(rng.lognormal(4.6, .6, n))
    return pd.DataFrame({"age": rng.integers(20, 80, n).astype(float), "height": height, "weight": weight,
                         "bmi": np.round(weight / (height / 100) ** 2, 1), "albumin": albumin, "total_protein": total_protein,
                         "globulin": np.round(total_protein - albumin, 1), "sodium": sodium, "potassium": np.round(rng.normal(4, .3, n), 1),
                         "chloride": np.round(rng.normal(103, 2.5, n)), "bicarbonate": np.round(rng.normal(25, 2, n)), "glucose": glucose,
                         "bun": bun, "creatinine": np.round(rng.lognormal(-.1, .25, n), 2),
                         "osmolality": np.round(1.86 * sodium + glucose / 18 + bun / 2.8 + 9), "calcium": np.round(rng.normal(9.4, .35, n), 1),
                         "uric_acid": np.round(rng.normal(5.4, 1.3, n), 1), "urine_albumin": urine_albumin, "urine_creatinine": urine_creatinine,
                         "acr": np.round(100 * urine_albumin / urine_creatinine, 2), "sex": np.where(rng.random(n) < .5, "male", "female"),
                         "diabetes": np.where(rng.random(n) < .15, "1", "0")})


def test_linear_relation_solves_every_participant_and_weights_spread():
    osmolality = RELATIONS["nhanes_chem"][3]
    rng = np.random.default_rng(2)
    values = {"sodium": rng.normal(140, 3, 50), "glucose": rng.normal(100, 20, 50), "bun": rng.normal(14, 4, 50)}
    values["osmolality"] = osmolality.solve("osmolality", values)
    np.testing.assert_allclose(values["osmolality"], 1.86 * values["sodium"] + values["glucose"] / 18 + values["bun"] / 2.8 + 9)
    for target in osmolality.participants:
        np.testing.assert_allclose(osmolality.solve(target, values), values[target], rtol=1e-10)
    from medhyperdiff.relations import spread
    frame = pd.DataFrame(values)
    np.testing.assert_allclose(spread(osmolality, "sodium", frame), 1.86 * frame.sodium.std(ddof=0))
    np.testing.assert_allclose(spread(osmolality, "glucose", frame), frame.glucose.std(ddof=0) / 18)


def test_linear_rule_residual_numpy_and_torch_agree():
    rule = Rule("osm", "linear", ("osmolality", "sodium", "glucose", "bun"), bound=9., weights=(1.86, 1 / 18, 1 / 2.8))
    values = {"osmolality": np.array([290., 280.]), "sodium": np.array([140., 138.]), "glucose": np.array([90., 108.]), "bun": np.array([14., 7.])}
    expected = np.abs(values["osmolality"] - (1.86 * values["sodium"] + values["glucose"] / 18 + values["bun"] / 2.8 + 9))
    np.testing.assert_allclose(ConstraintSet.residual(rule, values), expected)
    torch_values = {k: torch.tensor(v) for k, v in values.items()}
    np.testing.assert_allclose(ConstraintSet.residual(rule, torch_values).numpy(), expected)


def test_reported_exact_detection_and_rederivation():
    frame = chemistry_frame()
    schema = SCHEMAS["nhanes_chem_g"]
    coder = RelationCoder(schema, relations_for("nhanes_chem_g")).fit(frame)
    assert all(coder.exact.values()) and len(coder.exact) == 4
    assert {d.name: d.derived for d in coder.relations}["osmolality"] == "osmolality"
    # Reconstruct urine albumin, as the conditioning rule does on NHANES fit rows (urine albumin is the widest participant there).
    coder.relations = [d.relation.solved_for("urine_albumin") if d.name == "acr" else d for d in coder.relations]
    coder._update_schema()
    # An approximate relation (computed from unrounded values) is not reported-exact.
    rng = np.random.default_rng(5)
    rbc, mcv = rng.uniform(3.5, 6, 300), rng.uniform(80, 100, 300)
    cbc = pd.DataFrame({"rbc": np.round(rbc, 2), "mcv": np.round(mcv, 1), "hct": np.round(rbc * mcv / 10, 1)})
    assert not reported_exact(RELATIONS["nhanes_cbc"][0], cbc)
    # Perturbed generated values: decoding re-derives ACR from the rounded urine values, so the relation holds as reported.
    encoded = coder.encode(frame)
    encoded["acr"] *= np.exp(np.random.default_rng(1).normal(0, .1, len(encoded)))
    encoded["res__urine_albumin"] = np.random.default_rng(2).normal(0, .03, len(encoded))
    decimals = {c: (2 if c in ("acr", "creatinine") else 0 if c in ("sodium", "glucose", "bun", "osmolality", "chloride", "bicarbonate",
                                                                   "urine_creatinine", "age") else 1) for c in schema.numeric}
    decoded = coder.decode(encoded, lambda name, v: np.round(v, decimals[name]))
    residual = decoded.acr - 100 * decoded.urine_albumin / decoded.urine_creatinine
    assert np.abs(residual).max() <= .005 + 1e-9
    assert np.allclose(decoded.urine_albumin * 10, np.round(decoded.urine_albumin * 10))
    plain = RelationCoder(schema, relations_for("nhanes_chem_g"), derivations=coder.relations)  # no exact flags
    undone = plain.decode(encoded, lambda name, v: np.round(v, decimals[name]))
    assert np.abs(undone.acr - 100 * undone.urine_albumin / undone.urine_creatinine).max() > .01


def test_chemistry_constraints_and_choice_are_acyclic():
    from medhyperdiff.constraints import build_constraints
    frame = chemistry_frame()
    rules = {r.name: r for r in build_constraints(SCHEMAS["nhanes_chem_l"], frame).rules if len(r.columns) >= 2}
    assert set(rules) == {"globulin_identity", "bmi_identity", "acr_index", "osmolality_formula"}
    for rule in rules.values():
        raw = ConstraintSet.residual(rule, {c: frame[c].to_numpy(float) for c in rule.columns})
        assert np.nanmax(raw) <= rule.tolerance + 1e-9 or np.mean(raw <= rule.tolerance + 1e-9) >= .95
    order = choose_derivations(relations_for("nhanes_chem_l"), frame)
    assert len({d.derived for d in order}) == 4


def test_clinical_categories_and_total_variation():
    cbc = pd.DataFrame({"hgb": [12.5, 12.5, 15.], "mcv": [75., 75., 105.], "mch": [24., 24., 34.], "mchc": [31., 31., 33.],
                        "sex": ["male", "female", "male"]})
    labels = clinical_categories(cbc, "cbc")
    assert list(labels) == ["1|0|1|1", "0|0|1|1", "0|2|0|0"]
    assert clinical_category_tv(cbc, cbc, "cbc") == 0
    assert clinical_category_tv(cbc.iloc[[0]], cbc.iloc[[2]], "cbc") == 1
    chem = chemistry_frame(50)
    chem.loc[0, "globulin"] = 0.
    assert pd.isna(clinical_categories(chem, "chem").iloc[0]) and clinical_categories(chem, "chem").iloc[1:].notna().all()


def test_size_scaling_candidates_and_development_dedup():
    import iteration14 as it
    assert it.mhd_params("c0", 2000)["train_steps"] == 5000
    assert it.mhd_params("c1", 1000)["train_steps"] == 20000 and it.mhd_params("c1", 100)["train_steps"] == 5000
    assert [it.mhd_params("c3", n)["width"] for n in (100, 250, 1000, 2000)] == [32, 32, 64, 88]
    assert it.mhd_params("c2", 250)["width"] == 64
    jobs = it.development_jobs()
    assert len({j["id"] for j in jobs}) == len(jobs)
    per_cell = pd.Series([(j["dataset"], j["size"], j["seed"]) for j in jobs if j["variant"].startswith("mhd")]).value_counts()
    assert {k[1]: v for k, v in per_cell.items()} == {250: 2, 1000: 3, 2000: 4}


def test_training_decomposition_options():
    """Iteration 15 options: validation-selected checkpoints share the final-checkpoint trajectory."""
    import iteration12 as it12
    frame, schema = it12.load("thyroid")
    plan = __import__("json").loads(it12.SPLITS.read_text())["datasets"]["thyroid"]["samples"]["n250_r0"]
    fit, validation = frame.iloc[plan["fit"]], frame.iloc[plan["validation"]]
    base = {**it12.MHD, "train_steps": 40}
    final = it12.build("mhd", {**base, "check_every": 4}, schema, fit, 0).fit(fit, validation)
    chosen = it12.build("mhd", {**base, "checkpoint": "validation", "check_every": 4, "validation_draws": 2}, schema, fit, 0).fit(fit, validation)
    losses = [h["ema_validation_loss"] for h in chosen.history]
    assert chosen.executed_steps == final.executed_steps == 40
    assert chosen.selected_step == 4 * (int(np.argmin(losses)) + 1)
    # Validation uses its own random generator, so the training trajectory is identical to the final-checkpoint run.
    assert [h["train_loss"] for h in chosen.history] == [h["train_loss"] for h in final.history]
    stopped = it12.build("mhd", {**base, "checkpoint": "validation", "check_every": 1, "selection_patience": 2}, schema, fit, 0).fit(fit, validation)
    assert stopped.executed_steps <= 40
    plain = it12.build("mhd", {**base, "lr_schedule": "constant", "use_ema": False}, schema, fit, 0).fit(fit, validation)
    assert all(h["learning_rate"] == base["learning_rate"] for h in plain.history)


def test_bmi_log_scale_coefficient_and_unchanged_choice():
    bmi = RELATIONS["nhanes_chem"][1]
    assert bmi.weight("height") == 2. and bmi.weight("weight") == 1. and bmi.weight("bmi") == 1.
    frame = chemistry_frame()
    from medhyperdiff.relations import spread
    assert spread(bmi, "height", frame) == 2 * np.std(np.log(frame.height))
    chosen = {d.name: d.derived for d in choose_derivations(relations_for("nhanes_chem_l"), frame)}["bmi"]
    assert chosen == max(bmi.participants, key=lambda c: spread(bmi, c, frame))
