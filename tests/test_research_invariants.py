import numpy as np
import pandas as pd
import pytest
import torch

from medhyperdiff.data import SCHEMAS, normalize, nested_splits, toy_data
from medhyperdiff.preprocessing import Preprocessor
from medhyperdiff.constraints import build_constraints, Rule, ConstraintSet
from medhyperdiff.diffusion import (DiffusionConfig, MixedDiffusion, categorical_reverse_probs,
    guidance_schedule, cosine_schedule, clean_categorical_logits)
from medhyperdiff.model import HypergraphBlock
from medhyperdiff.evaluation import fidelity, privacy


@pytest.fixture
def prepared():
    torch.set_num_threads(1)
    schema = SCHEMAS["toy"]
    frame = toy_data(96, 17)
    train, validation = frame.iloc[:72].copy(), frame.iloc[72:].copy()
    prep = Preprocessor(schema).fit(train)
    constraints = build_constraints(schema, train)
    return schema, train, validation, prep, constraints


def test_preprocessing_train_only_unknown_and_missing(prepared):
    schema, train, val, prep, _ = prepared
    means = prep.means.copy()
    val.loc[val.index[0], "age"] = 1e9
    val.loc[val.index[1], "symptom_a"] = "unseen"
    val.loc[val.index[2], "albumin"] = np.nan
    numeric, categorical = prep.transform(val)
    assert np.array_equal(prep.means, means)
    assert categorical[1, 0] == prep.levels[0].index(prep.unknown)
    decoded = prep.inverse(numeric, categorical)
    assert pd.isna(decoded.loc[2, "albumin"])
    assert not prep.allowed[0][prep.levels[0].index(prep.unknown)]
    raw_num, raw_cat = prep.transform(train)
    restored = prep.inverse(raw_num, raw_cat)
    np.testing.assert_allclose(restored[list(schema.numeric)], train[list(schema.numeric)], rtol=1e-6)


def test_duplicate_predictors_do_not_cross_splits():
    schema = SCHEMAS["toy"]
    frame = toy_data(160, 2)
    duplicate = frame.iloc[:20].copy()
    duplicate[schema.target] = np.where(duplicate[schema.target] == "1", "0", "1")
    frame = pd.concat([frame, duplicate], ignore_index=True)
    hashes = pd.util.hash_pandas_object(frame[schema.features], index=False)
    for split in nested_splits(frame, schema):
        groups = [set(hashes.iloc[split[key]]) for key in ("train", "validation", "test")]
        assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


def test_categorical_posterior_and_terminal_reveal():
    logits = torch.zeros(3, 2)
    state = torch.tensor([2, 0, 1])
    posterior = categorical_reverse_probs(logits, state, 2, 4)
    torch.testing.assert_close(posterior, torch.tensor([[.125, .125, .75], [1., 0., 0.], [0., 1., 0.]]))
    final = categorical_reverse_probs(logits, state, 2, 1)
    assert torch.all(final[:, -1] == 0)
    torch.testing.assert_close(final.sum(-1), torch.ones(3))


def test_guidance_uses_known_unmasked_categories():
    logits = [torch.zeros(3, 2)]
    result = clean_categorical_logits(logits, torch.tensor([[0], [1], [2]]), [2])
    torch.testing.assert_close(result[0].softmax(-1), torch.tensor([[1., 0.], [0., 1.], [.5, .5]]))


def test_diffusion_endpoints_and_equal_guidance_budget():
    _, _, abar = cosine_schedule(50)
    assert abar[0] == 1 and abar[-1] < 1e-4
    assert torch.all(abar[:-1] > abar[1:])
    schedules = [guidance_schedule(50, m) for m in ("constant", "adaptive", "late")]
    assert all(abs(s.sum() - 50) < 1e-10 for s in schedules)
    assert schedules[1][1] > schedules[1][-1]


def test_constraint_gradients_decrease_energy(prepared):
    _, train, _, prep, rules = prepared
    model = MixedDiffusion(prep, np.zeros((len(prep.token_names),)*2), rules,
                           DiffusionConfig(epochs=1, width=16, layers=1, steps=4))
    numeric, cat, labels = model.tensors(train.iloc[:8])
    numeric[:, 2] += 4
    logits = [torch.nn.functional.one_hot(cat[:, j], k).float() * 20 for j, k in enumerate(prep.cardinalities)]
    before = rules.energy(numeric, logits, prep).mean()
    guided, guided_logits = model.guide(numeric, logits, .1)
    after = rules.energy(guided, guided_logits, prep).mean()
    assert after < before
    # Per-row guidance cannot vary just because the same batch is repeated.
    repeated, _ = model.guide(numeric.repeat(2, 1), [v.repeat(2, 1) for v in logits], .1)
    torch.testing.assert_close(repeated[:8], guided)


def test_missing_values_are_not_counted_valid(prepared):
    _, train, _, _, constraints = prepared
    frame = train.copy()
    frame["ag_ratio"] = np.nan
    metric = constraints.evaluate(frame)["rules"]["protein_identity"]
    assert metric["eligible"] == 0
    assert metric["violation_rate"] is None


def test_incidence_block_nonlinear_and_isolated():
    block = HypergraphBlock(8, 4, [[0, 1, 2]])
    tokens = torch.randn(2, 4, 8, requires_grad=True)
    output = block(tokens)
    torch.testing.assert_close(output[:, 3], tokens[:, 3])
    gradient = torch.autograd.grad(output[:, 0].square().sum(), tokens)[0]
    assert gradient[:, 2].abs().sum() > 0


def test_tiny_training_sampling_and_reload(prepared, tmp_path):
    _, train, val, prep, rules = prepared
    model = MixedDiffusion(prep, np.zeros((len(prep.token_names),)*2), rules,
        DiffusionConfig(epochs=2, width=16, layers=1, steps=4, batch_size=32, guidance=.1))
    model.fit(train, val)
    sample = model.sample([0, 1]*4, seed=3)
    assert sample.shape == (8, 7)
    assert np.isfinite(sample[list(prep.schema.numeric)]).all().all()
    assert set(sample.symptom_c) <= {"0", "1"}
    pd.testing.assert_frame_equal(sample, model.sample([0, 1]*4, seed=3))
    path = tmp_path / "checkpoint.pt"
    model.save(path)
    checkpoint = torch.load(path, weights_only=True)
    restored = MixedDiffusion.load(path)
    pd.testing.assert_frame_equal(sample, restored.sample([0, 1]*4, seed=3))


def test_fidelity_identity_and_memorization_control(prepared):
    schema, train, val, _, _ = prepared
    metrics = fidelity(train, train, train.copy(), schema)
    assert metrics["association_rmse"] == 0
    assert metrics["mean_numeric_wasserstein_train_sd"] == 0
    audit = privacy(train, val, train.copy(), schema)
    assert audit["exact_predictor_match_train"] == 1
    assert audit["nearest_synthetic_membership_auc"] > .95


def test_reversed_numeric_dependency_is_detected(prepared):
    schema, train, _, _, _ = prepared
    synthetic = train.copy()
    synthetic["albumin"] = -synthetic["albumin"]
    assert fidelity(train, train, synthetic, schema)["signed_numeric_spearman_rmse"] > 0


def test_heart_failure_followup_is_excluded():
    assert "time" not in SCHEMAS["heart_failure"].features
    assert "time" in SCHEMAS["heart_failure"].exclude


def test_all_missing_numeric_column_roundtrip(prepared):
    schema, train, _, _, _ = prepared
    train["age"] = np.nan
    prep = Preprocessor(schema).fit(train)
    num, cat = prep.transform(train)
    assert np.isfinite(num).all()
    assert prep.inverse(num, cat).age.isna().all()


def test_copula_with_constant_and_imbalanced_columns(prepared):
    from medhyperdiff.baselines import ReferenceGenerator
    schema, train, _, _, _ = prepared
    train["symptom_a"] = "1"
    train["albumin"] = 3.
    generator = ReferenceGenerator(schema).fit(train)
    sample = generator.sample([0, 1] * 30, seed=4)
    assert set(sample.symptom_a) == {"1"}
    np.testing.assert_allclose(sample.albumin, 3.)
    assert np.isfinite(sample[list(schema.numeric)]).all().all()


def test_velocity_reconstructs_clean_at_all_noise_levels():
    from medhyperdiff.diffusion import numeric_target, clean_from_prediction
    clean, noise = torch.randn(51, 4), torch.randn(51, 4)
    abar = cosine_schedule(50)[2][:, None]
    noisy = abar.sqrt() * clean + (1 - abar).sqrt() * noise
    velocity = numeric_target(clean, noise, abar, "v")
    torch.testing.assert_close(clean_from_prediction(velocity, noisy, abar, "v"), clean)


def test_inner_seed_keeps_reserved_test_unchanged():
    schema = SCHEMAS["toy"]
    frame = toy_data(160, 2)
    original = next(nested_splits(frame, schema))
    changed = next(nested_splits(frame, schema, inner_seed=101))
    assert original["test"] == changed["test"]
    assert original["validation"] != changed["validation"]
    assert set(original["train"] + original["validation"]) == set(changed["train"] + changed["validation"])


def test_gated_model_starts_at_common_backbone_and_dense_has_equal_capacity(prepared):
    from dataclasses import replace
    _, train, _, prep, rules = prepared
    graph = np.ones((len(prep.token_names),) * 2)
    cfg = DiffusionConfig(epochs=1, width=16, layers=1, steps=4,
        gated_structure=True, paired_initialization=True)
    base = MixedDiffusion(prep, graph, rules, replace(cfg, graph_enabled=False, hypergraph_enabled=False))
    full = MixedDiffusion(prep, graph, rules, cfg)
    dense = MixedDiffusion(prep, graph, rules, replace(cfg, dense_structure=True))
    assert sum(p.numel() for p in full.model.parameters()) == sum(p.numel() for p in dense.model.parameters())
    num, cat, labels = full.tensors(train.iloc[:8])
    inputs = (num, cat, torch.full((8,), .5), labels)
    a, b = base.model(*inputs), full.model(*inputs)
    torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
    for x, y in zip(a[1], b[1]):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    b[0].square().sum().backward()
    assert full.model.hyper_blocks[0].residual_gate.grad.abs() > 0


def test_velocity_gated_checkpoint_roundtrip(prepared, tmp_path):
    _, train, val, prep, rules = prepared
    model = MixedDiffusion(prep, np.zeros((len(prep.token_names),) * 2), rules,
        DiffusionConfig(epochs=2, width=16, layers=1, steps=4, numeric_prediction="v",
            gated_structure=True, paired_initialization=True))
    model.fit(train, val)
    path = tmp_path / "velocity.pt"
    model.save(path)
    restored = MixedDiffusion.load(path)
    pd.testing.assert_frame_equal(model.sample([0, 1]*4, 4), restored.sample([0, 1]*4, 4))


def test_official_tabddpm_train_and_checkpoint_roundtrip(prepared, tmp_path):
    from pathlib import Path
    if not Path("external/tab-ddpm/tab_ddpm").exists():
        pytest.skip("Optional pinned official TabDDPM source is not installed")
    from medhyperdiff.external_baselines import TabDDPMAdapter
    _, train, val, prep, _ = prepared
    cfg = DiffusionConfig(epochs=2, width=16, layers=1, steps=4, batch_size=32)
    model = TabDDPMAdapter(prep, cfg).fit(train, val)
    sample = model.sample([0, 1]*4, seed=4)
    assert np.isfinite(sample[list(prep.schema.numeric)]).all().all()
    assert set(sample.symptom_a) <= {"0", "1"}
    path = tmp_path / "tabddpm.pt"
    model.save(path)
    pd.testing.assert_frame_equal(sample, TabDDPMAdapter.load(path).sample([0, 1]*4, seed=4))


def test_duplicate_grouped_discriminator_on_identical_tables(prepared):
    from medhyperdiff.evaluation import c2st
    schema, train, _, _, _ = prepared
    result = c2st(train, train.copy(), schema, seed=4)
    assert result["duplicate_grouped"]
    assert result["auc"] == .5
