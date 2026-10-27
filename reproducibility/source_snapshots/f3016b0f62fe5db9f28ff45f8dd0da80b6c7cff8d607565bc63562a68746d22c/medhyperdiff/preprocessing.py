"""Train-only affine numeric transforms and explicit missingness tokens."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Schema, canonical_category


class Preprocessor:
    missing = "__MISSING__"
    unknown = "__UNKNOWN__"

    def __init__(self, schema: Schema):
        self.schema = schema

    def fit(self, frame):
        if frame.empty:
            raise ValueError("Cannot fit an empty training partition")
        numeric = frame[list(self.schema.numeric)].astype(float)
        # All-missing columns use explicit neutral values; masks remain informative.
        self.medians = numeric.median().fillna(0).to_numpy()
        self.means = numeric.mean().fillna(0).to_numpy()
        self.scales = numeric.std(ddof=0).fillna(1).to_numpy()
        self.scales[self.scales < 1e-6] = 1
        self.cat_names = list(self.schema.categorical)
        self.mask_names = [f"{col}__observed" for col in self.schema.numeric]
        if set(self.mask_names) & set(self.schema.columns):
            raise ValueError("Missingness token names conflict with feature names")
        self.levels = []
        for col in self.cat_names:
            vals = frame[col].map(canonical_category).dropna().unique().tolist()
            if self.missing in vals or self.unknown in vals:
                raise ValueError("Reserved categorical token in source data")
            self.levels.append(sorted(vals) + [self.missing, self.unknown])
        self.levels += [["0", "1"] for _ in self.mask_names]
        # Forbid generation of absent categories/unknown/missing states, based ONLY on fit.
        _, cat = self.transform(frame)
        self.allowed = [np.bincount(cat[:, i], minlength=len(level)) > 0 for i, level in enumerate(self.levels)]
        self.token_names = list(self.schema.numeric) + self.cat_names + self.mask_names
        self.fitted_rows = len(frame)
        return self

    @property
    def cardinalities(self):
        return [len(v) for v in self.levels]

    def transform(self, frame):
        numeric = frame[list(self.schema.numeric)].to_numpy(dtype=float)
        observed = np.isfinite(numeric)
        numeric = np.where(observed, numeric, self.medians)
        numeric = ((numeric - self.means) / self.scales).astype(np.float32)
        codes = []
        for col, levels in zip(self.cat_names, self.levels):
            lookup = {v: i for i, v in enumerate(levels)}
            values = frame[col].map(canonical_category).fillna(self.missing)
            codes.append(values.map(lambda v: lookup.get(v, lookup[self.unknown])).to_numpy(dtype=np.int64))
        codes.extend(observed[:, j].astype(np.int64) for j in range(observed.shape[1]))
        cat = np.stack(codes, axis=1) if codes else np.empty((len(frame), 0), dtype=np.int64)
        return numeric, cat

    def inverse(self, numeric, categorical):
        numeric = np.asarray(numeric) * self.scales + self.means
        categorical = np.asarray(categorical)
        if not np.isfinite(numeric).all():
            raise ValueError("Nonfinite generated numerical values")
        data = {col: numeric[:, i].copy() for i, col in enumerate(self.schema.numeric)}
        for i, col in enumerate(self.cat_names):
            values = np.array(self.levels[i], dtype=object)[categorical[:, i]]
            data[col] = pd.Series(values).replace({self.missing: np.nan, self.unknown: np.nan})
        for i, col in enumerate(self.schema.numeric):
            observed = categorical[:, len(self.cat_names) + i].astype(bool)
            data[col][~observed] = np.nan
        return pd.DataFrame(data)[self.schema.features]

    def to_dict(self):
        return {"means": self.means.tolist(), "scales": self.scales.tolist(),
                "medians": self.medians.tolist(), "levels": self.levels,
                "allowed": [v.tolist() for v in self.allowed], "fitted_rows": self.fitted_rows,
                "token_names": self.token_names}

    @classmethod
    def from_dict(cls, schema, state):
        prep = cls(schema)
        for name in ("means", "scales", "medians"):
            setattr(prep, name, np.asarray(state[name], dtype=float))
        prep.levels = state["levels"]
        prep.allowed = [np.asarray(v, dtype=bool) for v in state["allowed"]]
        prep.fitted_rows = state["fitted_rows"]
        prep.token_names = state["token_names"]
        prep.cat_names = list(schema.categorical)
        prep.mask_names = [f"{c}__observed" for c in schema.numeric]
        return prep
