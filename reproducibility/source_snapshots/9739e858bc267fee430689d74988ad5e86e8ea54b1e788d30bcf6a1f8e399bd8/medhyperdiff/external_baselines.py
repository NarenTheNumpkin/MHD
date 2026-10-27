"""Adapter around pinned, unmodified official TabDDPM diffusion and MLP code.

This is a small-budget adaptation, not a reproduction of published benchmark scores.
The fit-only quantile transform and compact categorical/missingness encoding live here.
"""
from contextlib import redirect_stdout
import copy
from dataclasses import asdict
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from sklearn.preprocessing import QuantileTransformer
import torch

from .diffusion import seed_all


TABDDPM_COMMIT = "b476257dd460b778ba09eb97f7a51d6490fa17f8"


class TabDDPMAdapter:
    def __init__(self, preprocessor, config):
        self.preprocessor, self.config = preprocessor, config
        root = Path(os.environ.get("TABDDPM_SOURCE", "external/tab-ddpm")).resolve()
        if not (root / "tab_ddpm" / "gaussian_multinomial_diffsuion.py").exists():
            raise FileNotFoundError("Clone the pinned official TabDDPM repository; see docs/EXTERNAL_BASELINES.md")
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--", "tab_ddpm"], text=True).strip()
        if commit != TABDDPM_COMMIT or dirty:
            raise ValueError("TabDDPM source must match the unmodified pinned commit")
        sys.path.insert(0, str(root))
        from tab_ddpm import GaussianMultinomialDiffusion, MLPDiffusion
        self.diffusion_class, self.denoiser_class = GaussianMultinomialDiffusion, MLPDiffusion
        self.provenance = {"external_method": "official-source TabDDPM, small-budget adapter",
            "upstream_repository": "https://github.com/yandex-research/tab-ddpm",
            "upstream_commit": commit,
            "upstream_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((root / "tab_ddpm").glob("*.py"))},
            "numeric_preprocessing": "fit-only empirical normal quantile transform after median imputation",
            "categorical_preprocessing": "compact fit-observed states; explicit numeric observation flags; unseen validation states mapped to fit mode",
            "optimizer": "AdamW, constant learning rate, weight decay 1e-4; no EMA; fixed-noise validation checkpoint",
            "gaussian_parameterization": "epsilon (official default)", "published_benchmark_reproduction": False}

    def encode(self, frame):
        num, cat = self.preprocessor.transform(frame)
        num = self.quantiles.transform(num).astype(np.float32)
        compact = np.column_stack([np.array([mapping.get(int(v), self.fallback[j]) for v in cat[:, j]])
            for j, mapping in enumerate(self.maps)])
        data = torch.as_tensor(np.column_stack([num, compact]), dtype=torch.float32, device=self.config.device)
        y = (frame[self.preprocessor.schema.target].to_numpy() == self.preprocessor.schema.positive).astype(int)
        return data, torch.as_tensor(y, device=self.config.device)

    def fit(self, train, validation):
        cfg = self.config
        seed_all(cfg.seed)
        num, cat = self.preprocessor.transform(train)
        self.quantiles = QuantileTransformer(n_quantiles=min(len(train), 1000),
            output_distribution="normal", subsample=None, random_state=cfg.seed).fit(num)
        self.states = [np.unique(cat[:, j]).astype(int) for j in range(cat.shape[1])]
        self.maps = [{int(v): i for i, v in enumerate(states)} for states in self.states]
        self.fallback = [int(np.argmax(np.bincount([self.maps[j][int(v)] for v in cat[:, j]])))
            for j in range(cat.shape[1])]
        _, val_cat = self.preprocessor.transform(validation)
        self.provenance["unseen_validation_categorical_cells"] = int(sum(
            (~np.isin(val_cat[:, j], states)).sum() for j, states in enumerate(self.states)))
        cardinalities = np.array([len(s) for s in self.states])
        d_in = num.shape[1] + int(cardinalities.sum())
        self.model = self.denoiser_class(d_in, 2, True,
            dict(d_layers=self.mlp_layers(cfg), dropout=0.), dim_t=self.time_dim(cfg)).to(cfg.device)
        self.diffusion = self.diffusion_class(cardinalities, num.shape[1], self.model,
            num_timesteps=cfg.steps, device=torch.device(cfg.device)).to(cfg.device)
        x, y = self.encode(train)
        vx, vy = self.encode(validation)
        if cfg.training == "steps":
            return self.fit_official(x, y, vx, vy)
        seed_all(cfg.seed + 200003)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
        self.history, best, state, stale = [], float("inf"), None, 0
        for epoch in range(cfg.epochs):
            self.diffusion.train()
            total = 0.
            for idx in torch.randperm(len(x), device=cfg.device).split(cfg.batch_size):
                optimizer.zero_grad(set_to_none=True)
                multi, gauss = self.diffusion.mixed_loss(x[idx], {"y": y[idx]})
                loss = multi + gauss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite official TabDDPM loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                optimizer.step()
                total += float(loss.detach()) * len(idx)
            self.diffusion.eval()
            devices = [torch.device(cfg.device).index or 0] if cfg.device.startswith("cuda") else []
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                torch.manual_seed(cfg.seed + 100003)
                multi, gauss = self.diffusion.mixed_loss(vx, {"y": vy})
            value = float(multi + gauss)
            self.history.append({"epoch": epoch + 1, "train_loss": total / len(x), "validation_loss": value})
            if value < best:
                best, state, stale = value, copy.deepcopy(self.diffusion.state_dict()), 0
                self.best_epoch = epoch + 1
            else:
                stale += 1
            if cfg.patience and stale >= cfg.patience:
                break
        if state is None:
            raise FloatingPointError("No finite official TabDDPM checkpoint")
        self.diffusion.load_state_dict(state)
        return self

    @staticmethod
    def time_dim(cfg):
        # Upstream MLPDiffusion uses its default time embedding (128) in the official recipe.
        return 128 if cfg.training == "steps" else cfg.width

    @staticmethod
    def mlp_layers(cfg):
        return [int(v) for v in cfg.mlp_layers] if cfg.mlp_layers else [cfg.width] * cfg.layers

    def fit_official(self, x, y, vx, vy):
        """Upstream `scripts/train.py` recipe at the pinned commit.

        Fixed step budget, shuffled minibatches cycling over the fit rows (the last
        partial batch is kept), AdamW, learning rate annealed linearly after each step,
        no gradient clipping and no validation checkpoint selection. Upstream
        `pipeline.py` samples from the final non-EMA weights (`model.pt`), so we do too.
        The fixed-noise validation loss is only logged.
        """
        cfg = self.config
        seed_all(cfg.seed + 200003)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        batch = min(cfg.batch_size, len(x))
        self.history, order, position, running, count = [], torch.randperm(len(x), device=cfg.device), 0, 0., 0
        log_every = max(1, cfg.train_steps // 50)
        for step in range(cfg.train_steps):
            if position >= len(x):
                order, position = torch.randperm(len(x), device=cfg.device), 0
            idx = order[position:position + batch]
            position += batch
            self.diffusion.train()
            optimizer.zero_grad(set_to_none=True)
            multi, gauss = self.diffusion.mixed_loss(x[idx], {"y": y[idx]})
            loss = multi + gauss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite official TabDDPM loss")
            loss.backward()
            optimizer.step()
            for group in optimizer.param_groups:
                group["lr"] = cfg.learning_rate * (1 - step / cfg.train_steps)
            running += float(loss.detach()) * len(idx)
            count += len(idx)
            if (step + 1) % log_every == 0 or step + 1 == cfg.train_steps:
                self.diffusion.eval()
                devices = [torch.device(cfg.device).index or 0] if cfg.device.startswith("cuda") else []
                with torch.random.fork_rng(devices=devices), torch.no_grad():
                    torch.manual_seed(cfg.seed + 100003)
                    vm, vg = self.diffusion.mixed_loss(vx, {"y": vy})
                self.history.append({"step": step + 1, "train_loss": running / count,
                                     "validation_loss": float(vm + vg)})
                running, count = 0., 0
        self.best_epoch = None
        self.provenance.update(optimizer="upstream recipe: AdamW, linear LR annealing, no clipping, "
            "fixed steps, final non-EMA weights, no validation checkpoint selection",
            train_steps=cfg.train_steps, mlp_layers=self.mlp_layers(cfg))
        return self

    def sample(self, labels, seed=0):
        seed_all(seed)
        labels = np.asarray(labels, dtype=int)
        if not len(labels) or not np.isin(labels, [0, 1]).all():
            raise ValueError("Provide nonempty binary labels")
        self.diffusion.eval()
        numeric_count = len(self.preprocessor.schema.numeric)
        numeric = np.empty((len(labels), numeric_count), dtype=np.float32)
        categorical = np.empty((len(labels), len(self.states)), dtype=int)
        for label in np.unique(labels):
            indices = np.flatnonzero(labels == label)
            distribution = torch.tensor([1. - label, float(label)], device=self.config.device)
            # A degenerate class distribution uses the official sampler with exact requested counts.
            with redirect_stdout(io.StringIO()):
                sampled, output = self.diffusion.sample(len(indices), distribution)
            assert (output["y"].cpu().numpy() == label).all()
            sampled = sampled.numpy()
            numeric[indices] = self.quantiles.inverse_transform(sampled[:, :numeric_count])
            for j, states in enumerate(self.states):
                categorical[indices, j] = states[sampled[:, numeric_count + j].astype(int)]
        return self.preprocessor.inverse(numeric, categorical)

    def save(self, path):
        torch.save({"model_type": "official_tabddpm_adapter", "state_dict": self.diffusion.state_dict(),
            "config": asdict(self.config), "schema": asdict(self.preprocessor.schema),
            "preprocessing": self.preprocessor.to_dict(), "states": [v.tolist() for v in self.states],
            "fallback": self.fallback, "quantiles": self.quantiles.quantiles_.tolist(),
            "references": self.quantiles.references_.tolist(), "history": self.history,
            "best_epoch": self.best_epoch, "provenance": self.provenance}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        from .data import Schema
        from .preprocessing import Preprocessor
        from .diffusion import DiffusionConfig
        state = torch.load(path, map_location="cpu", weights_only=True)
        schema_data = state["schema"]
        for name in ("numeric", "categorical", "exclude"):
            schema_data[name] = tuple(schema_data[name])
        prep = Preprocessor.from_dict(Schema(**schema_data), state["preprocessing"])
        cfg = DiffusionConfig(**state["config"])
        cfg.device = device
        result = cls(prep, cfg)
        result.states = [np.asarray(v, dtype=int) for v in state["states"]]
        result.maps = [{int(v): i for i, v in enumerate(s)} for s in result.states]
        result.fallback = state["fallback"]
        result.quantiles = QuantileTransformer(output_distribution="normal")
        result.quantiles.quantiles_ = np.asarray(state["quantiles"])
        result.quantiles.references_ = np.asarray(state["references"])
        result.quantiles.n_quantiles_ = len(result.quantiles.references_)
        result.quantiles.n_features_in_ = len(prep.schema.numeric)
        cards = np.array([len(s) for s in result.states])
        result.model = result.denoiser_class(len(prep.schema.numeric) + int(cards.sum()), 2, True,
            dict(d_layers=cls.mlp_layers(cfg), dropout=0.), dim_t=cls.time_dim(cfg)).to(device)
        result.diffusion = result.diffusion_class(cards, len(prep.schema.numeric), result.model,
            num_timesteps=cfg.steps, device=torch.device(device)).to(device)
        result.diffusion.load_state_dict(state["state_dict"])
        result.history, result.best_epoch = state["history"], state["best_epoch"]
        return result
