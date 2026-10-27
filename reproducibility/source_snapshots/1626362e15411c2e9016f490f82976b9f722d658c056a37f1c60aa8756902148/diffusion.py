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
    numeric_prediction: str = "x0"
    gated_structure: bool = False
    dense_structure: bool = False
    paired_initialization: bool = False
    # "epochs" is the original validation-selected loop and stays the default for
    # archived runs. "steps" trains a fixed optimizer-step budget with linear learning-rate
    # decay and an exponential moving average, then samples from the final EMA weights;
    # validation data are monitored but never used to choose a checkpoint.
    training: str = "epochs"
    train_steps: int = 0
    ema_decay: float = .999
    observation_prior: bool = False
    weight_decay: float = 1e-4
    # Optional explicit MLP widths for the official TabDDPM denoiser; empty keeps width x layers.
    mlp_layers: tuple = ()
    # Decomposition options for the fixed-budget loop (iteration 15). The defaults reproduce the corrected recipe:
    # linear learning-rate decay, weight averaging and the final averaged weights. "validation" instead returns the
    # checkpoint with the lowest validation loss among the checks (one every `check_every` steps, default
    # train_steps // 50), with that loss averaged over `validation_draws` fixed corruption draws; a positive
    # `selection_patience` stops training after that many checks without improvement.
    checkpoint: str = "final"
    validation_draws: int = 1
    selection_patience: int = 0
    check_every: int = 0
    lr_schedule: str = "linear"
    use_ema: bool = True


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


def numeric_target(clean, noise, abar, prediction):
    if prediction == "x0":
        return clean
    if prediction == "v":
        return abar.sqrt() * noise - (1 - abar).sqrt() * clean
    raise ValueError(f"Unknown numeric prediction {prediction}")


def clean_from_prediction(predicted, noisy, abar, prediction):
    if prediction == "x0":
        return predicted
    if prediction == "v":
        return abar.sqrt() * noisy - (1 - abar).sqrt() * predicted
    raise ValueError(f"Unknown numeric prediction {prediction}")


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
        if cfg.numeric_prediction not in ("x0", "v"):
            raise ValueError("numeric_prediction must be x0 or v")
        if cfg.training not in ("epochs", "steps") or (cfg.training == "steps" and cfg.train_steps < 1):
            raise ValueError("training must be 'epochs' or 'steps' with a positive train_steps budget")
        if not 0 <= cfg.ema_decay < 1:
            raise ValueError("ema_decay must lie in [0, 1)")
        seed_all(cfg.seed)
        self.device = torch.device(cfg.device)
        graph = np.asarray(graph) if cfg.graph_enabled else np.zeros_like(graph)
        edges = constraints.hyperedges(preprocessor.token_names) if cfg.hypergraph_enabled else []
        self.model = Denoiser(len(preprocessor.schema.numeric), preprocessor.cardinalities,
            graph, edges, preprocessor.allowed, cfg.width, cfg.layers, cfg.heads,
            cfg.gated_structure, cfg.dense_structure, cfg.paired_initialization).to(self.device)
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
        predicted, logits = self.model(noisy_num, noisy_cat, t / cfg.steps, labels)
        clean = clean_from_prediction(predicted, noisy_num, abar, cfg.numeric_prediction)
        target = numeric_target(numeric, noise, abar, cfg.numeric_prediction)
        observed = categorical[:, len(self.preprocessor.cat_names):].float()
        numeric_loss = ((predicted - target).square() * observed).sum() / observed.sum().clamp(min=1)
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

    def initialize_observation_heads(self, train):
        """Start binary observation heads at fit frequencies (iteration-5 method).

        Heads whose two states both occur get zero weights and log-frequency biases;
        all other parameters keep their seeded initialization.
        """
        _, categorical = self.preprocessor.transform(train)
        offset = len(self.preprocessor.cat_names)
        self.observation_initialization = {}
        with torch.no_grad():
            for i, feature in enumerate(self.preprocessor.schema.numeric):
                counts = np.bincount(categorical[:, offset + i], minlength=2)
                if (counts > 0).all():
                    head = self.model.category_heads[offset + i]
                    head.weight.zero_()
                    head.bias.copy_(torch.as_tensor(np.log(counts / counts.sum()), dtype=head.bias.dtype))
                    self.observation_initialization[feature] = counts.tolist()

    def fit(self, train, validation):
        if self.config.observation_prior:
            self.initialize_observation_heads(train)
        if self.config.training == "steps":
            return self.fit_steps(train, validation)
        numeric, categorical, labels = self.tensors(train)
        val_tensors = self.tensors(validation)
        cfg = self.config
        if cfg.paired_initialization:
            seed_all(cfg.seed + 200003)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
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

    def fit_steps(self, train, validation=None):
        """Fixed-budget training: linear learning-rate decay to zero and EMA weights.

        A fixed-noise validation loss of the EMA model is logged for monitoring only.
        Its generator is separate, so monitoring cannot change the training trajectory,
        and no checkpoint is selected with validation records.
        """
        numeric, categorical, labels = self.tensors(train)
        val_tensors = self.tensors(validation) if validation is not None and len(validation) else None
        cfg = self.config
        if cfg.checkpoint not in ("final", "validation") or cfg.lr_schedule not in ("linear", "constant"):
            raise ValueError("Unknown checkpoint or learning-rate schedule option")
        if cfg.checkpoint == "validation" and val_tensors is None:
            raise ValueError("Validation checkpoint selection needs validation records")
        # Training randomness is independent of architecture size (paired comparisons).
        seed_all(cfg.seed + 200003)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        ema = copy.deepcopy(self.model).eval()
        for parameter in ema.parameters():
            parameter.requires_grad_(False)
        n = len(numeric)
        batch = min(cfg.batch_size, n)
        order, position = torch.randperm(n, device=self.device), 0
        history, running, count = [], 0., 0
        log_every = cfg.check_every or max(1, cfg.train_steps // 50)
        best, best_state, best_step, stale = float("inf"), None, None, 0
        for step in range(cfg.train_steps):
            learning_rate = cfg.learning_rate * (1 - step / cfg.train_steps) if cfg.lr_schedule == "linear" else cfg.learning_rate
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            if position + batch > n:
                order, position = torch.randperm(n, device=self.device), 0
            idx = order[position:position + batch]
            position += batch
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            loss, _ = self.loss(numeric[idx], categorical[idx], labels[idx])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite diffusion training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            optimizer.step()
            with torch.no_grad():
                if cfg.use_ema:
                    for averaged, current in zip(ema.parameters(), self.model.parameters()):
                        averaged.lerp_(current, 1 - cfg.ema_decay)
                else:
                    ema.load_state_dict(self.model.state_dict())
            running += float(loss.detach()) * len(idx)
            count += len(idx)
            if (step + 1) % log_every == 0 or step + 1 == cfg.train_steps:
                record = {"step": step + 1, "learning_rate": learning_rate, "train_loss": running / max(count, 1)}
                if val_tensors is not None:
                    live, self.model = self.model, ema
                    values = []
                    for draw in range(max(1, cfg.validation_draws)):
                        # Draw 0 is the fixed corruption used since iteration 11; further draws average it out.
                        generator = torch.Generator(device=self.device).manual_seed(cfg.seed + 100003 + 7919 * draw)
                        with torch.no_grad():
                            value, details = self.loss(*val_tensors, generator=generator)
                        values.append(float(value))
                    self.model = live
                    record.update({"ema_validation_loss": float(np.mean(values)), **details})
                    if cfg.checkpoint == "validation":
                        if record["ema_validation_loss"] < best:
                            best, best_state, best_step, stale = record["ema_validation_loss"], copy.deepcopy(ema.state_dict()), step + 1, 0
                        else:
                            stale += 1
                history.append(record)
                running, count = 0., 0
                if cfg.checkpoint == "validation" and cfg.selection_patience and stale >= cfg.selection_patience:
                    break
        if cfg.checkpoint == "validation":
            if best_state is None:
                raise FloatingPointError("No finite validation checkpoint")
            self.model.load_state_dict(best_state)
        else:
            self.model.load_state_dict(ema.state_dict())
        self.model.eval()
        self.history, self.best_epoch = history, None
        self.selected_step, self.executed_steps = (best_step if cfg.checkpoint == "validation" else None), step + 1
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
                    predicted, logits = self.model(numeric, categorical, torch.full((n,), t / cfg.steps, device=self.device), y)
                    clean = clean_from_prediction(predicted, numeric, self.abar[t], cfg.numeric_prediction)
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
