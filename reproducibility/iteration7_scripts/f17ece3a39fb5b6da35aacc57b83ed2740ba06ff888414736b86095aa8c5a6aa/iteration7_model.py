"""Per-label native CTGAN/TVAE with explicit, frozen capacity presets."""
import numpy as np
import pandas as pd
from ctgan import CTGAN, TVAE

from medhyperdiff.baselines import CTGANAdapter
from medhyperdiff.preprocessing import Preprocessor


def settings(kind, preset):
    if preset == "default300":
        return dict(epochs=300, enable_gpu=False, verbose=False)
    if preset != "compact100":
        raise ValueError(preset)
    common = dict(epochs=100, embedding_dim=32, enable_gpu=False, verbose=False)
    if kind == "ctgan":
        return dict(**common, batch_size=100, pac=10, generator_dim=(64, 64), discriminator_dim=(64, 64))
    return dict(**common, batch_size=64, compress_dims=(64, 64), decompress_dims=(64, 64))


class NativeAdapter(CTGANAdapter):
    def __init__(self, schema, kind, preset, seed):
        super().__init__(schema, kind, settings(kind, preset)["epochs"], seed)
        self.preset = preset

    def fit(self, train):
        self.preprocessor = Preprocessor(self.schema).fit(train)
        num, cat = self.preprocessor.transform(train)
        self.num_names = [f"n{i}" for i in range(num.shape[1])]
        self.cat_names = [f"c{i}" for i in range(cat.shape[1])]
        table = pd.DataFrame(num, columns=self.num_names)
        for j, name in enumerate(self.cat_names):
            table[name] = cat[:, j].astype(str)
        labels = (train[self.schema.target].to_numpy() == self.schema.positive).astype(int)
        self.models = {}
        for label in (0, 1):
            model = (CTGAN if self.kind == "ctgan" else TVAE)(**settings(self.kind, self.preset))
            model.set_random_state(self.seed + label)
            model.fit(table.loc[labels == label].reset_index(drop=True), discrete_columns=self.cat_names)
            self.models[label] = model
        return self
