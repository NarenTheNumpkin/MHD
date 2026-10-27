"""Gaussian DDPM numerics + absorbing-mask categorical diffusion.

This is an original compact research backbone, NOT a TabDDPM/TabDiff reproduction.
Guidance is approximate descent on clean predictions; it is not an exact sampler
from a constrained distribution and does not guarantee feasibility.
"""
from dataclasses import dataclass, asdict
import copy
import random

import numpy as np
import torch
from torch.nn import functional as F

from .model import Denoiser


@dataclass
class DiffusionConfig:
    steps: int = 50
    epochs: int = 100
    batch_size: int = 64
    width: int = 48
    layers: int = 2
    heads: int = 4
    learning_rate: float = .001
    clinical_weight: float = .1
    guidance: float = .2
    schedule: str = "adaptive"
    patience: int = 20
    device: str = "cpu"
    seed: int = 0
    graph_enabled: bool = True
    hypergraph_enabled: bool = True
    unit_confidence: bool = False


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_schedule(steps, device="cpu"):
    if steps < 2:
        raise ValueError("At least two diffusion steps are required")
    t = torch.linspace(0, steps, steps + 1, device=device)
    curve = torch.cos(((t / steps + .008) / 1.008) * torch.pi / 2).square()
    curve = curve / curve[0]
    beta = (1 - curve[1:] / curve[:-1]).clamp(.0001, .999)
    beta = torch.cat([torch.zeros(1, device=device), beta])
    alpha = 1 - beta
    abar = torch.cumprod(alpha, dim=0)
    return beta, alpha, abar


def guidance_schedule(steps, mode):
    """Index 1 is the last reverse step. Schedules have equal integrated weight."""
    t = np.arange(1, steps + 1)
    if mode == "none":
        values = np.zeros(steps)
    elif mode == "constant":
        values = np.ones(steps)
    elif mode == "adaptive":
        values = (1 - (t - 1) / steps) ** 2
    elif mode == "late":
        values = (t <= max(1, steps // 4)).astype(float)
    else:
        raise ValueError(f"Unknown guidance schedule {mode}")
    if values.sum() > 0:
        values *= steps / values.sum()
    return np.r_[0., values]


def categorical_reverse_probs(logits, state, mask_id, step):
    """Exact absorbing posterior under the learned x0 parameterization.

    For a(t)=1-t/T, a masked value reveals with probability 1/t;
    already observed values copy through. No gradient is taken through integer IDs.
    """
    probs = logits.softmax(-1)
    out = torch.zeros(len(state), mask_id + 1, device=state.device)
    masked = state == mask_id
    out[masked, :mask_id] = probs[masked] / step
    out[masked, mask_id] = 1 - 1 / step
    out[~masked, state[~masked]] = 1
    return out


def clean_categorical_logits(logits, categorical, cardinalities):
    """The clean state of an already revealed category is known exactly."""
    result = []
    for j, (values, k) in enumerate(zip(logits, cardinalities)):
        unmasked = categorical[:, j] < k
        observed = F.one_hot(categorical[:, j].clamp(max=k-1), k).bool()
        fixed = torch.where(observed, 0., -1e9).to(values)
        result.append(torch.where(unmasked[:, None], fixed, values))
    return result


class MixedDiffusion:
    def __init__(self, preprocessor, graph, constraints, config=None):
        self.preprocessor = preprocessor
        self.constraints = constraints
        self.config = config or DiffusionConfig()
        cfg = self.config
        if cfg.epochs < 1 or cfg.batch_size < 1 or cfg.guidance < 0 or cfg.clinical_weight < 0:
            raise ValueError("Invalid training or guidance configuration")
        seed_all(cfg.seed)
        self.device = torch.device(cfg.device)
        graph = np.asarray(graph) if cfg.graph_enabled else np.zeros_like(graph)
        edges = constraints.hyperedges(preprocessor.token_names) if cfg.hypergraph_enabled else []
        self.model = Denoiser(len(preprocessor.schema.numeric), preprocessor.cardinalities,
            graph, edges, preprocessor.allowed, cfg.width, cfg.layers, cfg.heads).to(self.device)
        self.beta, self.alpha, self.abar = cosine_schedule(cfg.steps, self.device)
        self.schedule = guidance_schedule(cfg.steps, cfg.schedule)

    def tensors(self, frame):
        num, cat = self.preprocessor.transform(frame)
        labels = (frame[self.preprocessor.schema.target].to_numpy() == self.preprocessor.schema.positive).astype(np.int64)
        return (torch.as_tensor(num, device=self.device), torch.as_tensor(cat, device=self.device),
                torch.as_tensor(labels, device=self.device))

    def loss(self, numeric, categorical, labels, generator=None):
        cfg = self.config
        n = len(numeric)
        t = torch.randint(1, cfg.steps + 1, (n,), device=self.device, generator=generator)
        noise = torch.randn(numeric.shape, device=self.device, generator=generator)
        abar = self.abar[t, None]
        noisy_num = abar.sqrt() * numeric + (1 - abar).sqrt() * noise
        mask = torch.rand(categorical.shape, device=self.device, generator=generator) < (t[:, None] / cfg.steps)
        noisy_cat = categorical.clone()
        for j, k in enumerate(self.preprocessor.cardinalities):
            noisy_cat[:, j] = torch.where(mask[:, j], k, categorical[:, j])
        clean, logits = self.model(noisy_num, noisy_cat, t / cfg.steps, labels)
        observed = categorical[:, len(self.preprocessor.cat_names):].float()
        numeric_loss = ((clean - numeric).square() * observed).sum() / observed.sum().clamp(min=1)
        # Uniformly sample t; T/t times masked CE is the absorbing variational term.
        terms = [F.cross_entropy(logit, categorical[:, j], reduction="none") * mask[:, j] * (cfg.steps / t)
                 for j, logit in enumerate(logits)]
        categorical_loss = torch.stack(terms, 1).mean() if terms else numeric_loss * 0
        energy_logits = clean_categorical_logits(logits, noisy_cat, self.preprocessor.cardinalities)
        clinical = self.constraints.energy(clean, energy_logits, self.preprocessor, observed)
        # Late weighting protects noisy clean estimates during training.
        clinical_loss = (clinical * (1 - (t - 1) / cfg.steps).square()).mean()
        total = numeric_loss + categorical_loss + cfg.clinical_weight * clinical_loss
        return total, {"numeric": float(numeric_loss.detach()), "categorical": float(categorical_loss.detach()),
                       "clinical": float(clinical_loss.detach())}

    def fit(self, train, validation):
        numeric, categorical, labels = self.tensors(train)
        val_tensors = self.tensors(validation)
        cfg = self.config
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
        history, best, best_state, stale = [], float("inf"), None, 0
        for epoch in range(cfg.epochs):
            self.model.train()
            order = torch.randperm(len(train), device=self.device)
            running, count = 0., 0
            for idx in order.split(cfg.batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss, _ = self.loss(numeric[idx], categorical[idx], labels[idx])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite diffusion training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                optimizer.step()
                running += float(loss.detach()) * len(idx)
                count += len(idx)
            self.model.eval()
            # Fixed validation corruption prevents checkpoint selection by fresh noise luck.
            generator = torch.Generator(device=self.device).manual_seed(cfg.seed + 100003)
            with torch.no_grad():
                val_loss, details = self.loss(*val_tensors, generator=generator)
            value = float(val_loss)
            history.append({"epoch": epoch + 1, "train_loss": running / count, "validation_loss": value, **details})
            if value < best:
                best, best_state, stale = value, copy.deepcopy(self.model.state_dict()), 0
                self.best_epoch = epoch + 1
            else:
                stale += 1
            if cfg.patience and stale >= cfg.patience:
                break
        if best_state is None:
            raise FloatingPointError("No finite validation checkpoint")
        self.model.load_state_dict(best_state)
        self.history = history
        return self

    def guide(self, clean, logits, strength):
        if not strength or not self.constraints.rules:
            return clean, logits
        with torch.enable_grad():
            clean = clean.detach().requires_grad_(True)
            logits = [v.detach().requires_grad_(True) for v in logits]
            energy = self.constraints.energy(clean, logits, self.preprocessor).sum()
            gradients = torch.autograd.grad(energy, [clean, *logits], allow_unused=True)
            variables = [clean, *logits]
            grads = [torch.zeros_like(v) if g is None else g for v, g in zip(variables, gradients)]
            # Bound each row's combined gradient; independent of generation batch size.
            norm = torch.sqrt(sum(g.square().sum(-1) for g in grads)).clamp(min=1.)
            adjusted = [v - strength * g / norm[:, None] for v, g in zip(variables, grads)]
        return adjusted[0].detach(), [v.detach() for v in adjusted[1:]]

    def sample(self, labels, seed=0, batch_size=256):
        seed_all(seed)
        labels = np.asarray(labels, dtype=np.int64)
        if not np.isin(labels, [0, 1]).all() or len(labels) == 0:
            raise ValueError("Provide a nonempty vector of binary encoded labels")
        self.model.eval()
        cfg = self.config
        all_num, all_cat = [], []
        for start in range(0, len(labels), batch_size):
            y = torch.as_tensor(labels[start:start + batch_size], device=self.device)
            n = len(y)
            numeric = torch.randn(n, len(self.preprocessor.schema.numeric), device=self.device)
            categorical = torch.stack([torch.full((n,), k, device=self.device, dtype=torch.long) for k in self.preprocessor.cardinalities], dim=1)
            for t in range(cfg.steps, 0, -1):
                with torch.no_grad():
                    clean, logits = self.model(numeric, categorical, torch.full((n,), t / cfg.steps, device=self.device), y)
                    logits = clean_categorical_logits(logits, categorical, self.preprocessor.cardinalities)
                clean, logits = self.guide(clean, logits, cfg.guidance * self.schedule[t])
                with torch.no_grad():
                    prev_abar, abar = self.abar[t - 1], self.abar[t]
                    c1 = self.beta[t] * prev_abar.sqrt() / (1 - abar)
                    c2 = (1 - prev_abar) * self.alpha[t].sqrt() / (1 - abar)
                    variance = self.beta[t] * (1 - prev_abar) / (1 - abar)
                    numeric = c1 * clean + c2 * numeric
                    if t > 1:
                        numeric += variance.sqrt() * torch.randn_like(numeric)
                    for j, k in enumerate(self.preprocessor.cardinalities):
                        probs = categorical_reverse_probs(logits[j], categorical[:, j], k, t)
                        categorical[:, j] = torch.multinomial(probs, 1).squeeze(1)
            all_num.append(numeric.cpu().numpy())
            all_cat.append(categorical.cpu().numpy())
        return self.preprocessor.inverse(np.concatenate(all_num), np.concatenate(all_cat))

    def save(self, path):
        torch.save({"state_dict": self.model.state_dict(), "config": asdict(self.config),
                    "schema": asdict(self.preprocessor.schema),
                    "preprocessing": self.preprocessor.to_dict(), "constraints": self.constraints.to_dict(),
                    "best_epoch": self.best_epoch, "history": self.history}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        from .data import Schema
        from .preprocessing import Preprocessor
        from .constraints import ConstraintSet, Rule
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint["schema"]
        for name in ("numeric", "categorical", "exclude"):
            state[name] = tuple(state[name])
        schema = Schema(**state)
        prep = Preprocessor.from_dict(schema, checkpoint["preprocessing"])
        constraints = ConstraintSet([Rule(**r) for r in checkpoint["constraints"]])
        cfg = DiffusionConfig(**checkpoint["config"])
        cfg.device = device
        result = cls(prep, checkpoint["state_dict"]["graph"].numpy(), constraints, cfg)
        result.model.load_state_dict(checkpoint["state_dict"])
        result.best_epoch = checkpoint["best_epoch"]
        result.history = checkpoint["history"]
        return result
