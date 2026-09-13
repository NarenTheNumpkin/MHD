from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from iteration6_sampler import VarianceSampler, reverse_variance, distribution_shape
from medhyperdiff.constraints import build_constraints
from medhyperdiff.data import SCHEMAS, toy_data
from medhyperdiff.diffusion import MixedDiffusion, DiffusionConfig
from medhyperdiff.preprocessing import Preprocessor


def sampler():
    torch.set_num_threads(1)
    frame = toy_data(80, 7)
    prep = Preprocessor(SCHEMAS["toy"]).fit(frame)
    config = DiffusionConfig(width=16, layers=1, steps=5, epochs=2, batch_size=32,
                             paired_initialization=True, guidance=.2)
    model = VarianceSampler(prep, np.zeros((len(prep.token_names),)*2), build_constraints(SCHEMAS["toy"], frame), config)
    model.fit(frame.iloc[:60], frame.iloc[60:])
    return frame, model


def test_variance_endpoints_and_log_midpoint():
    a, b = torch.tensor(.1), torch.tensor(.025)
    assert torch.equal(reverse_variance(a, b, 0, 2), b)
    assert torch.equal(reverse_variance(a, b, 1, 2), a)
    torch.testing.assert_close(reverse_variance(a, b, .5, 2), (a*b).sqrt())
    for m in (0, .5, 1):
        assert reverse_variance(a, b, m, 1) == 0
    for m in (-.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            reverse_variance(a, b, m, 2)


def test_zero_variance_mixture_matches_original_complete_sampling_loop(tmp_path):
    _, new = sampler()
    checkpoint = tmp_path/"model.pt"
    new.save(checkpoint)
    base = MixedDiffusion.load(checkpoint)
    labels = np.array([0, 1]*11)
    # Partial final batch and nonzero guidance exercise ordering and all transition code.
    pd.testing.assert_frame_equal(new.sample(labels, 71, 7), base.sample(labels, 71, 7), check_exact=True)
    states = {k: v.clone() for k, v in new.model.state_dict().items()}
    new.variance_mix = 1
    a = new.sample(labels, 71, 7)
    pd.testing.assert_frame_equal(a, new.sample(labels, 71, 7), check_exact=True)
    assert all(torch.equal(states[k], v) for k, v in new.model.state_dict().items())
    assert not a.equals(base.sample(labels, 71, 7))


def test_no_last_step_noise_even_with_larger_variance():
    _, model = sampler()
    model.config.guidance = 0
    forward = model.model.forward
    def zero_clean(numeric, categorical, time, labels):
        prediction, logits = forward(numeric, categorical, time, labels)
        return torch.zeros_like(prediction), logits
    model.model.forward = zero_clean
    for mix in (.5, 1):
        model.variance_mix = mix
        generated = model.sample([0, 1]*8, 92)
        expected = np.tile(model.preprocessor.means, (16, 1))
        np.testing.assert_allclose(generated[list(model.preprocessor.schema.numeric)], expected, atol=1e-6, rtol=0)


def test_shape_diagnostics_use_fit_thresholds_and_detect_tail_growth():
    schema = SCHEMAS["toy"]
    fit = toy_data(100, 9)
    exact = distribution_shape(fit, fit, fit, schema)
    assert all(v["q05_error"] == v["q95_error"] == 0 and v["iqr_ratio"] == 1 for v in exact.values())
    extreme = fit.copy()
    extreme.loc[:19, "age"] = fit.age.max()+100
    measured = distribution_shape(fit, fit, extreme, schema)["age"]
    assert measured["synthetic_outside_fit_range"] == .2
    assert measured["q95_error"] > 0
    assert measured["fit_quantiles"] == exact["age"]["fit_quantiles"]
