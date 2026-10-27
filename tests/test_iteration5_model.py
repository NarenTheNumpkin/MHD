"""Verify isolated training interventions without using clinical validation outcomes."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from iteration5_model import ObservationInitializedDiffusion
from medhyperdiff.constraints import build_constraints
from medhyperdiff.data import SCHEMAS, toy_data
from medhyperdiff.diffusion import DiffusionConfig, MixedDiffusion
from medhyperdiff.preprocessing import Preprocessor


def models(weight=1):
    torch.set_num_threads(1)
    schema = SCHEMAS["toy"]
    fit = toy_data(80, 23)
    fit.loc[:3, "ag_ratio"] = np.nan
    prep = Preprocessor(schema).fit(fit)
    config = DiffusionConfig(width=16, layers=1, steps=5, epochs=2, batch_size=32,
                             numeric_prediction="v", paired_initialization=True)
    graph = np.zeros((len(prep.token_names), len(prep.token_names)))
    args = prep, graph, build_constraints(schema, fit), config
    return fit, MixedDiffusion(*args), ObservationInitializedDiffusion(*args, observation_weight=weight)


def test_weight_one_loss_and_gradients_equal_original():
    fit, base, new = models()
    tensors = base.tensors(fit)
    a, ad = base.loss(*tensors, generator=torch.Generator().manual_seed(111))
    b, bd = new.loss(*tensors, generator=torch.Generator().manual_seed(111))
    assert torch.equal(a, b) and ad == bd
    a.backward(); b.backward()
    for x, y in zip(base.model.parameters(), new.model.parameters()):
        if x.grad is not None:
            assert torch.equal(x.grad, y.grad)


def test_prior_initialization_changes_only_variable_observation_head():
    fit, base, new = models()
    new.initialize_observation_heads(fit)
    index = len(new.preprocessor.cat_names)+list(new.preprocessor.schema.numeric).index("ag_ratio")
    for name, value in base.model.state_dict().items():
        if name not in (f"category_heads.{index}.weight", f"category_heads.{index}.bias"):
            assert torch.equal(value, new.model.state_dict()[name]), name
    head = new.model.category_heads[index]
    assert (head.weight == 0).all()
    torch.testing.assert_close(head.bias.softmax(0), torch.tensor([.05, .95]))
    assert head.weight.requires_grad and head.bias.requires_grad
    assert new.observation_initialization["ag_ratio"]["counts"] == [4, 76]


def test_observation_weight_only_changes_mask_ce():
    fit, base, new = models(4)
    tensors = base.tensors(fit)
    capture = []
    handles = []
    for head in new.model.category_heads:
        handles.append(head.register_forward_hook(lambda module, inputs, output: capture.append(output)))
    # With the same explicit generator, reconstruct the exact mask CE independently.
    a, ad = base.loss(*tensors, generator=torch.Generator().manual_seed(92))
    b, bd = new.loss(*tensors, generator=torch.Generator().manual_seed(92))
    g = torch.Generator().manual_seed(92)
    t = torch.randint(1, 6, (len(fit),), generator=g)
    torch.randn(tensors[0].shape, generator=g)
    mask = torch.rand(tensors[1].shape, generator=g) < t[:, None]/5
    terms = []
    for j in range(len(new.preprocessor.cat_names), len(capture)):
        logits = capture[j].masked_fill(~getattr(new.model, f"allowed_{j}"), -1e9)
        terms.append(torch.nn.functional.cross_entropy(logits, tensors[1][:, j], reduction="none")*mask[:, j]*5/t)
    expected_extra = 3*torch.stack(terms, 1).sum()/tensors[1].numel()
    torch.testing.assert_close(b-a, expected_extra, atol=1e-6, rtol=1e-5)
    assert ad["numeric"] == bd["numeric"] and ad["clinical"] == bd["clinical"]
    for h in handles:
        h.remove()


def test_new_checkpoint_loads_in_unmodified_sampling_cli(tmp_path):
    fit, _, model = models(4)
    val = toy_data(24, 12)
    model.fit(fit, val)
    path = tmp_path/"new.pt"
    model.save(path)
    experimental = ObservationInitializedDiffusion.load(path)
    generic = MixedDiffusion.load(path)
    assert experimental.observation_weight == 4
    assert experimental.observation_initialization == model.observation_initialization
    labels = np.array([0, 1]*5)
    pd.testing.assert_frame_equal(model.sample(labels, 73), experimental.sample(labels, 73), check_exact=True)
    pd.testing.assert_frame_equal(model.sample(labels, 73), generic.sample(labels, 73), check_exact=True)
