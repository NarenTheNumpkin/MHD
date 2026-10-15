"""Transparent reference generators and adapters to the official ctgan package."""
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from .preprocessing import Preprocessor


class ReferenceGenerator:
    """Conditional independent marginals, bootstrap, or empirical Gaussian copula.

    The copula uses a stated ordinal encoding for categories and is only a simple
    reference, not SDV's GaussianCopulaSynthesizer or a named paper reproduction.
    """
    def __init__(self, schema, kind="copula", seed=0, decode_precision=False):
        if kind not in ("copula", "independent", "bootstrap"):
            raise ValueError(kind)
        self.schema, self.kind, self.seed = schema, kind, seed
        self.decode_precision = decode_precision

    def fit(self, train):
        self.train = train.copy()
        self.preprocessor = Preprocessor(self.schema, decode_precision=self.decode_precision).fit(train)
        num, cat = self.preprocessor.transform(train)
        self.values = np.concatenate([num, cat], axis=1)
        self.y = (train[self.schema.target].to_numpy() == self.schema.positive).astype(int)
        return self

    def sample(self, labels, seed=None):
        rng = np.random.default_rng(self.seed if seed is None else seed)
        labels = np.asarray(labels)
        output = np.empty((len(labels), self.values.shape[1]))
        for label in np.unique(labels):
            where = np.flatnonzero(labels == label)
            values = self.values[self.y == label]
            n, d = values.shape
            if self.kind == "bootstrap":
                output[where] = values[rng.integers(0, n, len(where))]
            elif self.kind == "independent":
                for j in range(d):
                    output[where, j] = values[rng.integers(0, n, len(where)), j]
            else:
                latent = np.column_stack([norm.ppf((rankdata(values[:, j]) - .5) / n) for j in range(d)])
                # Tied categorical normal scores need centering before correlation.
                latent -= latent.mean(0)
                std = latent.std(0)
                latent /= np.where(std > 1e-6, std, 1)
                covariance = latent.T @ latent / max(n, 1)
                covariance = .95 * covariance + .05 * np.eye(d)
                z = rng.multivariate_normal(np.zeros(d), covariance, len(where))
                quantiles = norm.cdf(z)
                for j in range(d):
                    ordered = np.sort(values[:, j])
                    index = np.minimum((quantiles[:, j] * n).astype(int), n - 1)
                    output[where, j] = ordered[index]
        cut = len(self.schema.numeric)
        return self.preprocessor.inverse(output[:, :cut], output[:, cut:].astype(int))


class CTGANAdapter:
    """Native CTGAN/TVAE models, fitted per class for common p(X|Y) semantics.

    Both use the same explicit numerical missingness representation as diffusion.
    Per-class fitting is a documented adaptation, not published benchmark replication.
    """
    def __init__(self, schema, kind, epochs=100, seed=0, params=None, decode_precision=False):
        """`params` are native constructor arguments; None keeps the original compact preset."""
        self.schema, self.kind, self.epochs, self.seed = schema, kind, epochs, seed
        self.params, self.decode_precision = params, decode_precision

    def fit(self, train):
        from ctgan import CTGAN, TVAE
        self.preprocessor = Preprocessor(self.schema, decode_precision=self.decode_precision).fit(train)
        num, cat = self.preprocessor.transform(train)
        self.num_names = [f"n{i}" for i in range(num.shape[1])]
        self.cat_names = [f"c{i}" for i in range(cat.shape[1])]
        table = pd.DataFrame(num, columns=self.num_names)
        for j, name in enumerate(self.cat_names):
            table[name] = cat[:, j].astype(str)
        labels = (train[self.schema.target].to_numpy() == self.schema.positive).astype(int)
        self.models = {}
        for label in (0, 1):
            if self.params is not None:
                model = (CTGAN if self.kind == "ctgan" else TVAE)(**{**self.params, "epochs": self.epochs,
                    "enable_gpu": False, "verbose": False})
            elif self.kind == "ctgan":
                model = CTGAN(epochs=self.epochs, batch_size=100, pac=10,
                    generator_dim=(64, 64), discriminator_dim=(64, 64), embedding_dim=32,
                    enable_gpu=False, verbose=False)
            else:
                model = TVAE(epochs=self.epochs, batch_size=64, embedding_dim=32,
                    compress_dims=(64, 64), decompress_dims=(64, 64), enable_gpu=False)
            model.set_random_state(self.seed + label)
            model.fit(table.loc[labels == label].reset_index(drop=True), discrete_columns=self.cat_names)
            self.models[label] = model
        return self

    def sample(self, labels, seed=0):
        labels = np.asarray(labels)
        frames = []
        for label in np.unique(labels):
            indexes = np.flatnonzero(labels == label)
            model = self.models[int(label)]
            model.set_random_state(seed + int(label))
            sampled = model.sample(len(indexes))
            numeric = sampled[self.num_names].to_numpy()
            categorical = sampled[self.cat_names].to_numpy(dtype=int)
            decoded = self.preprocessor.inverse(numeric, categorical)
            decoded.index = indexes
            frames.append(decoded)
        return pd.concat(frames).sort_index().reset_index(drop=True)
