"""Reproduce the analytic-sampler and learned-tail diagnostic, without fitting."""
import contextlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from medhyperdiff.data import normalize, SCHEMAS
from medhyperdiff.external_baselines import TabDDPMAdapter


torch.set_num_threads(2)
model = TabDDPMAdapter.load("runs/iteration2/external_diagnostic/ilpd_tabddpm_long_s0/checkpoint.pt")
# Loading the adapter verifies the pinned, unmodified source and imports its modules.
from tab_ddpm import GaussianMultinomialDiffusion


class Oracle(torch.nn.Module):
    def forward(self, x, t, y=None):
        # Analytic E[epsilon | x_t] when clean numerical data are N(0, I).
        numerical = (1 - diffusion.alphas_cumprod[t, None]).sqrt() * x[:, :4]
        return torch.cat((numerical, torch.zeros(len(x), 2)), dim=1)


diffusion = GaussianMultinomialDiffusion(np.array([2]), 4, Oracle(), num_timesteps=1000)
torch.manual_seed(71)
with contextlib.redirect_stdout(io.StringIO()):
    x, _ = diffusion.sample(4000, torch.tensor([.5, .5]))
result = {"gaussian_oracle_mean": x[:, :4].mean(0).tolist(),
          "gaussian_oracle_std": x[:, :4].std(0).tolist()}
frame = normalize(pd.read_csv("data/processed/ilpd.csv"), SCHEMAS["ilpd"])
split = json.loads(Path("runs/iteration2/external_diagnostic/ilpd_tabddpm_long_s0/splits.json").read_text())["folds"][0]
num, _ = model.preprocessor.transform(frame.iloc[split["train"]])
back = model.quantiles.inverse_transform(model.quantiles.transform(num))
result["quantile_roundtrip_max_absolute_error"] = float(np.abs(num - back).max())
torch.manual_seed(50000)
with contextlib.redirect_stdout(io.StringIO()):
    sampled, _ = model.diffusion.sample(1000, torch.tensor([.3, .7]))
result.update(learned_transformed_sample_std=sampled[:, :9].std(0).tolist(),
    learned_transformed_sample_mean=sampled[:, :9].mean(0).tolist(), selected_epoch=model.best_epoch,
    best_validation_loss=min(h["validation_loss"] for h in model.history),
    last_train_loss=model.history[-1]["train_loss"])
Path("reports/iteration2/TABDDPM_DIAGNOSTIC.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
