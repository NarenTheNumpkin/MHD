"""Experimental train-only observation-head initialization; core sampler unchanged."""
import numpy as np
import torch
from torch.nn import functional as F

from medhyperdiff.diffusion import MixedDiffusion, numeric_target, clean_from_prediction, clean_categorical_logits


class ObservationInitializedDiffusion(MixedDiffusion):
    def __init__(self, *args, observation_weight=1., **kwargs):
        super().__init__(*args, **kwargs)
        if not np.isfinite(observation_weight) or observation_weight <= 0:
            raise ValueError("observation_weight must be finite and positive")
        self.observation_weight = float(observation_weight)
        self.observation_initialization = {}

    def initialize_observation_heads(self, train):
        _, categorical = self.preprocessor.transform(train)
        offset = len(self.preprocessor.cat_names)
        with torch.no_grad():
            for i, feature in enumerate(self.preprocessor.schema.numeric):
                counts = np.bincount(categorical[:, offset+i], minlength=2)
                if (counts > 0).all():
                    probabilities = counts / counts.sum()
                    head = self.model.category_heads[offset+i]
                    head.weight.zero_()
                    head.bias.copy_(torch.as_tensor(np.log(probabilities), dtype=head.bias.dtype, device=head.bias.device))
                    self.observation_initialization[feature] = dict(counts=counts.tolist(), probabilities=probabilities.tolist())

    def fit(self, train, validation):
        self.initialize_observation_heads(train)
        return super().fit(train, validation)

    def loss(self, numeric, categorical, labels, generator=None):
        # The weight-1 branch is exactly the original loss, avoiding any numerical drift.
        if self.observation_weight == 1:
            return super().loss(numeric, categorical, labels, generator)
        cfg, n = self.config, len(numeric)
        t = torch.randint(1, cfg.steps+1, (n,), device=self.device, generator=generator)
        noise = torch.randn(numeric.shape, device=self.device, generator=generator)
        abar = self.abar[t, None]
        noisy_num = abar.sqrt()*numeric + (1-abar).sqrt()*noise
        mask = torch.rand(categorical.shape, device=self.device, generator=generator) < (t[:, None]/cfg.steps)
        noisy_cat = categorical.clone()
        for j, k in enumerate(self.preprocessor.cardinalities):
            noisy_cat[:, j] = torch.where(mask[:, j], k, categorical[:, j])
        predicted, logits = self.model(noisy_num, noisy_cat, t/cfg.steps, labels)
        clean = clean_from_prediction(predicted, noisy_num, abar, cfg.numeric_prediction)
        target = numeric_target(numeric, noise, abar, cfg.numeric_prediction)
        observed = categorical[:, len(self.preprocessor.cat_names):].float()
        numeric_loss = ((predicted-target).square()*observed).sum()/observed.sum().clamp(min=1)
        terms = [F.cross_entropy(logit, categorical[:, j], reduction="none")*mask[:, j]*(cfg.steps/t)
                 for j, logit in enumerate(logits)]
        offset = len(self.preprocessor.cat_names)
        categorical_loss = torch.stack([term*(self.observation_weight if j >= offset else 1.)
                                        for j, term in enumerate(terms)], 1).mean() if terms else numeric_loss*0
        energy_logits = clean_categorical_logits(logits, noisy_cat, self.preprocessor.cardinalities)
        clinical = self.constraints.energy(clean, energy_logits, self.preprocessor, observed)
        clinical_loss = (clinical*(1-(t-1)/cfg.steps).square()).mean()
        total = numeric_loss+categorical_loss+cfg.clinical_weight*clinical_loss
        return total, dict(numeric=float(numeric_loss.detach()), categorical=float(categorical_loss.detach()),
                           clinical=float(clinical_loss.detach()))

    def save(self, path):
        super().save(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        checkpoint["experimental_training"] = dict(method="observation_prior_initialization_v1",
            observation_weight=self.observation_weight, initialization=self.observation_initialization)
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path, device="cpu"):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        metadata = checkpoint["experimental_training"]
        assert metadata["method"] == "observation_prior_initialization_v1"
        result = super().load(path, device)
        weight = metadata["observation_weight"]
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("Invalid saved observation loss weight")
        result.observation_weight = weight
        result.observation_initialization = metadata["initialization"]
        return result
