"""Iteration-11 corrections: quantile coding, recorded precision, fixed-step EMA training,
official-recipe TabDDPM and rule-realism metrics. Legacy defaults must stay unchanged."""
import numpy as np
import pandas as pd
import pytest
import torch

from medhyperdiff.constraints import build_constraints
from medhyperdiff.data import SCHEMAS, toy_data
from medhyperdiff.diffusion import DiffusionConfig, MixedDiffusion
from medhyperdiff.evaluation import evaluate, rule_realism
from medhyperdiff.preprocessing import Preprocessor, recorded_decimals
from medhyperdiff.structure import dependency_graph


@pytest.fixture
def toy():
    frame = toy_data(160, seed=3)
    frame.loc[[2, 9, 17], "ag_ratio"] = np.nan
    return frame, SCHEMAS["toy"]


def test_recorded_decimals_detects_measurement_resolution():
    assert recorded_decimals([1.0, 2.0, 30.0]) == 0
    assert recorded_decimals([0.1, 0.7, 12.3, np.nan]) == 1
    assert recorded_decimals([0.35, 1.0]) == 2
    assert recorded_decimals([60.667, 61.0]) == 3
    assert recorded_decimals([np.pi]) is None
    assert recorded_decimals([np.nan]) is None


def test_quantile_coding_roundtrips_fit_values_and_keeps_missingness(toy):
    frame, schema = toy
    prep = Preprocessor(schema, "quantile").fit(frame)
    numeric, categorical = prep.transform(frame)
    assert np.isfinite(numeric).all() and abs(float(numeric[:, 0].mean())) < .1
    decoded = prep.inverse(numeric, categorical)
    for col in schema.numeric:
        observed = frame[col].notna()
        np.testing.assert_allclose(decoded.loc[observed, col], frame.loc[observed, col], rtol=0, atol=2e-5)
        assert decoded[col].isna().equals(frame[col].isna())


def test_quantile_inverse_is_bounded_and_matches_differentiable_decode(toy):
    frame, schema = toy
    prep = Preprocessor(schema, "quantile").fit(frame)
    z = torch.linspace(-9, 9, 101)[:, None].repeat(1, len(schema.numeric)).requires_grad_(True)
    columns = prep.physical_torch(z)
    numpy_decode = prep.physical_numeric(z.detach().numpy())
    for i, col in enumerate(schema.numeric):
        np.testing.assert_allclose(columns[i].detach().numpy(), numpy_decode[:, i], rtol=1e-5, atol=1e-5)
        assert numpy_decode[:, i].min() >= frame[col].min() - 1e-9
        assert numpy_decode[:, i].max() <= frame[col].max() + 1e-9
    torch.stack(columns, 1).sum().backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0


def test_affine_differentiable_decode_is_the_original_expression(toy):
    frame, schema = toy
    prep = Preprocessor(schema).fit(frame)
    z = torch.randn(17, len(schema.numeric))
    for i, column in enumerate(prep.physical_torch(z)):
        assert torch.equal(column, z[:, i] * float(prep.scales[i]) + float(prep.means[i]))
    assert "numeric_transform" not in prep.to_dict()  # archived checkpoint format unchanged


def test_precision_decoding_rounds_to_fit_grid_and_serializes(toy):
    frame, schema = toy
    frame = frame.copy()
    frame["total_protein"] = frame["total_protein"].round(1)
    prep = Preprocessor(schema, "quantile", decode_precision=True).fit(frame)
    assert prep.decimals[schema.numeric.index("total_protein")] == 1
    numeric = np.random.default_rng(0).normal(size=(50, len(schema.numeric))).astype(np.float32)
    _, categorical = prep.transform(frame.iloc[:50])
    decoded = prep.inverse(numeric, categorical)
    values = decoded["total_protein"].dropna().to_numpy()
    assert np.allclose(values * 10, np.round(values * 10))
    clone = Preprocessor.from_dict(schema, prep.to_dict())
    pd.testing.assert_frame_equal(clone.inverse(numeric, categorical), decoded)


def _fitted(frame, schema, validation=None, **overrides):
    prep = Preprocessor(schema, "quantile", decode_precision=True).fit(frame)
    constraints = build_constraints(schema, frame)
    graph, _ = dependency_graph(frame, prep, seed=0, bootstraps=3)
    cfg = DiffusionConfig(steps=8, width=16, layers=1, heads=2, batch_size=32, training="steps",
                          train_steps=30, numeric_prediction="v", clinical_weight=.01,
                          observation_prior=True, paired_initialization=True, guidance=1., **overrides)
    return MixedDiffusion(prep, graph, constraints, cfg).fit(frame, validation)


def test_fixed_step_training_ignores_validation_for_weights(toy):
    frame, schema = toy
    with_validation = _fitted(frame, schema, validation=frame.iloc[:40])
    without_validation = _fitted(frame, schema, validation=None)
    for name, tensor in with_validation.model.state_dict().items():
        assert torch.equal(tensor, without_validation.model.state_dict()[name]), name
    assert with_validation.best_epoch is None and "ema_validation_loss" in with_validation.history[-1]
    assert with_validation.history[-1]["learning_rate"] < with_validation.config.learning_rate


def test_fixed_step_checkpoint_roundtrip_samples_identically(toy, tmp_path):
    frame, schema = toy
    model = _fitted(frame, schema)
    labels = np.r_[np.ones(20, int), np.zeros(20, int)]
    first = model.sample(labels, seed=11)
    model.save(tmp_path / "m.pt")
    again = MixedDiffusion.load(tmp_path / "m.pt").sample(labels, seed=11)
    pd.testing.assert_frame_equal(first, again)
    for col in schema.numeric:
        assert first[col].dropna().between(frame[col].min(), frame[col].max()).all()


def test_rule_realism_scores_real_rates_not_zero(toy):
    frame, schema = toy
    constraints = build_constraints(schema, frame)
    identical = rule_realism(frame, frame, frame, constraints)
    assert identical["mean_violation_gap"] == 0 and identical["mean_relational_residual_w1_fit_sd"] == 0
    noisy = frame.copy()
    noisy["ag_ratio"] = noisy["ag_ratio"] * 1.3
    worse = rule_realism(frame, frame, noisy, constraints)
    assert worse["rules"]["protein_identity"]["residual_w1"] > 0
    assert worse["rules"]["protein_identity"]["violation_gap"] > 0
    metrics = evaluate(frame, frame, frame, schema, constraints, seed=0)
    assert "realism" in metrics["constraints"]


def test_official_recipe_tabddpm_trains_with_fixed_steps(toy, tmp_path):
    pytest.importorskip("ctgan")
    from pathlib import Path
    if not (Path("external/tab-ddpm") / "tab_ddpm").exists():
        pytest.skip("pinned TabDDPM source not available")
    from medhyperdiff.external_baselines import TabDDPMAdapter
    frame, schema = toy
    prep = Preprocessor(schema, decode_precision=True).fit(frame)
    cfg = DiffusionConfig(steps=10, width=16, layers=2, batch_size=4096, training="steps", train_steps=12,
                          learning_rate=1e-3, weight_decay=0., mlp_layers=(32, 64))
    model = TabDDPMAdapter(prep, cfg).fit(frame, frame.iloc[:30])
    assert model.best_epoch is None and model.history[-1]["step"] == 12
    labels = np.r_[np.ones(10, int), np.zeros(10, int)]
    first = model.sample(labels, seed=4)
    model.save(tmp_path / "t.pt")
    pd.testing.assert_frame_equal(first, TabDDPMAdapter.load(tmp_path / "t.pt").sample(labels, seed=4))


def test_tabdiff_adapter_trains_and_returns_exact_class_counts(toy):
    from pathlib import Path
    if not (Path("external/TabDiff") / "tabdiff").exists():
        pytest.skip("pinned TabDiff source not available")
    from medhyperdiff.external_baselines import TabDiffAdapter
    frame, schema = toy
    prep = Preprocessor(schema, decode_precision=True).fit(frame)
    model = TabDiffAdapter(prep, DiffusionConfig(seed=0), dict(train_steps=6, steps=4, dim_t=32, factor=4)).fit(frame)
    assert model.best_epoch == 6  # short runs fall back to the final EMA state
    labels = np.r_[np.ones(7, int), np.zeros(5, int)]
    synthetic = model.sample(labels, seed=3)
    assert len(synthetic) == 12 and list(synthetic.columns) == schema.features
    for col in schema.numeric:
        # float32 affine codes precede the quantile transform: allow round-off at the bounds
        slack = 1e-6 * max(1., float(frame[col].abs().max()))
        assert synthetic[col].dropna().between(frame[col].min() - slack, frame[col].max() + slack).all()
