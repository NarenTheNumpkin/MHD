"""Train-only numeric transforms and explicit missingness tokens.

`numeric_transform="affine"` is the original representation and remains the default,
so archived checkpoints and runs reproduce exactly. `"quantile"` maps each numerical
feature through fit-only mid-rank empirical normal scores. Its inverse is piecewise
linear and bounded by the observed fit range, so generated values cannot leave the
fitted support (a disclosed bounded-support choice, unlike unbounded asinh decoding).
`decode_precision=True` rounds decoded values to the recording resolution detected
from the fit partition (for example 0.1 mg/dL bilirubin); it is applied identically to
every continuous-output generator in comparisons that enable it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .data import Schema, canonical_category

NUMERIC_TRANSFORMS = ("affine", "quantile")


def recorded_decimals(values, max_decimals=6):
    """Smallest decimal count reproducing every finite fit value; None if none does."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    for decimals in range(max_decimals + 1):
        scaled = values * 10.0 ** decimals
        # Only floating-point representation error is tolerated, never a fraction of the grid.
        if np.all(np.abs(scaled - np.round(scaled)) <= 1e-6 + 1e-11 * np.abs(scaled)):
            return decimals
    return None


class Preprocessor:
    missing = "__MISSING__"
    unknown = "__UNKNOWN__"

    def __init__(self, schema: Schema, numeric_transform="affine", decode_precision=False):
        if numeric_transform not in NUMERIC_TRANSFORMS:
            raise ValueError(f"numeric_transform must be one of {NUMERIC_TRANSFORMS}")
        self.schema = schema
        self.numeric_transform = numeric_transform
        self.decode_precision = bool(decode_precision)

    def fit(self, frame):
        if frame.empty:
            raise ValueError("Cannot fit an empty training partition")
        numeric = frame[list(self.schema.numeric)].astype(float)
        # All-missing columns use explicit neutral values; masks remain informative.
        self.medians = numeric.median().fillna(0).to_numpy()
        self.means = numeric.mean().fillna(0).to_numpy()
        self.scales = numeric.std(ddof=0).fillna(1).to_numpy()
        self.scales[self.scales < 1e-6] = 1
        self.knots_x, self.knots_z = [], []
        if self.numeric_transform == "quantile":
            for i, col in enumerate(self.schema.numeric):
                observed = numeric[col].dropna().to_numpy()
                if not len(observed):
                    self.knots_x.append(np.array([self.medians[i]]))
                    self.knots_z.append(np.zeros(1))
                    continue
                values, counts = np.unique(observed, return_counts=True)
                # Mid-rank empirical CDF keeps ties together and avoids infinite end scores.
                midpoint = (np.cumsum(counts) - counts / 2) / counts.sum()
                self.knots_x.append(values.astype(float))
                self.knots_z.append(norm.ppf(midpoint) if len(values) > 1 else np.zeros(1))
        self.decimals = [recorded_decimals(numeric[col]) for col in self.schema.numeric] if self.decode_precision else []
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

    def _encode_numeric(self, numeric):
        if self.numeric_transform == "affine":
            return (numeric - self.means) / self.scales
        return np.column_stack([np.interp(numeric[:, i], self.knots_x[i], self.knots_z[i])
                                for i in range(numeric.shape[1])]) if numeric.shape[1] else numeric

    def physical_numeric(self, numeric):
        """NumPy decode to source units, before precision rounding and missingness."""
        numeric = np.asarray(numeric, dtype=float)
        if self.numeric_transform == "affine":
            return numeric * self.scales + self.means
        return np.column_stack([np.interp(numeric[:, i], self.knots_z[i], self.knots_x[i])
                                for i in range(numeric.shape[1])]) if numeric.shape[1] else numeric

    def physical_torch(self, numeric):
        """Differentiable decode to source units, returned as one tensor per feature.

        The affine branch reproduces the original inline expression exactly. The
        quantile branch is piecewise linear with zero gradient beyond the fitted range.
        """
        import torch
        columns = []
        for i in range(len(self.schema.numeric)):
            if self.numeric_transform == "affine":
                columns.append(numeric[:, i] * float(self.scales[i]) + float(self.means[i]))
                continue
            knots_z = torch.as_tensor(self.knots_z[i], dtype=numeric.dtype, device=numeric.device)
            knots_x = torch.as_tensor(self.knots_x[i], dtype=numeric.dtype, device=numeric.device)
            if len(knots_z) == 1:
                columns.append(knots_x.expand(numeric.shape[0]) + 0 * numeric[:, i])
                continue
            z = numeric[:, i].clamp(knots_z[0], knots_z[-1])
            right = torch.searchsorted(knots_z, z.detach().contiguous()).clamp(1, len(knots_z) - 1)
            z0, z1, x0, x1 = knots_z[right - 1], knots_z[right], knots_x[right - 1], knots_x[right]
            columns.append(x0 + (z - z0) * (x1 - x0) / (z1 - z0))
        return columns

    def transform(self, frame):
        numeric = frame[list(self.schema.numeric)].to_numpy(dtype=float)
        observed = np.isfinite(numeric)
        numeric = np.where(observed, numeric, self.medians)
        numeric = self._encode_numeric(numeric).astype(np.float32)
        codes = []
        for col, levels in zip(self.cat_names, self.levels):
            lookup = {v: i for i, v in enumerate(levels)}
            values = frame[col].map(canonical_category).fillna(self.missing)
            codes.append(values.map(lambda v: lookup.get(v, lookup[self.unknown])).to_numpy(dtype=np.int64))
        codes.extend(observed[:, j].astype(np.int64) for j in range(observed.shape[1]))
        cat = np.stack(codes, axis=1) if codes else np.empty((len(frame), 0), dtype=np.int64)
        return numeric, cat

    def inverse(self, numeric, categorical):
        numeric = self.physical_numeric(numeric)
        categorical = np.asarray(categorical)
        if not np.isfinite(numeric).all():
            raise ValueError("Nonfinite generated numerical values")
        for i, decimals in enumerate(self.decimals):
            if decimals is not None:
                numeric[:, i] = np.round(numeric[:, i], decimals)
        data = {col: numeric[:, i].copy() for i, col in enumerate(self.schema.numeric)}
        for i, col in enumerate(self.cat_names):
            values = np.array(self.levels[i], dtype=object)[categorical[:, i]]
            data[col] = pd.Series(values).replace({self.missing: np.nan, self.unknown: np.nan})
        for i, col in enumerate(self.schema.numeric):
            observed = categorical[:, len(self.cat_names) + i].astype(bool)
            data[col][~observed] = np.nan
        return pd.DataFrame(data)[self.schema.features]

    def to_dict(self):
        state = {"means": self.means.tolist(), "scales": self.scales.tolist(),
                 "medians": self.medians.tolist(), "levels": self.levels,
                 "allowed": [v.tolist() for v in self.allowed], "fitted_rows": self.fitted_rows,
                 "token_names": self.token_names}
        # Legacy affine checkpoints keep their original serialized form.
        if self.numeric_transform != "affine":
            state.update(numeric_transform=self.numeric_transform,
                         knots_x=[v.tolist() for v in self.knots_x], knots_z=[v.tolist() for v in self.knots_z])
        if self.decode_precision:
            state.update(decode_precision=True, decimals=self.decimals)
        return state

    @classmethod
    def from_dict(cls, schema, state):
        prep = cls(schema, state.get("numeric_transform", "affine"), state.get("decode_precision", False))
        for name in ("means", "scales", "medians"):
            setattr(prep, name, np.asarray(state[name], dtype=float))
        prep.knots_x = [np.asarray(v, dtype=float) for v in state.get("knots_x", [])]
        prep.knots_z = [np.asarray(v, dtype=float) for v in state.get("knots_z", [])]
        prep.decimals = list(state.get("decimals", []))
        prep.levels = state["levels"]
        prep.allowed = [np.asarray(v, dtype=bool) for v in state["allowed"]]
        prep.fitted_rows = state["fitted_rows"]
        prep.token_names = state["token_names"]
        prep.cat_names = list(schema.categorical)
        prep.mask_names = [f"{c}__observed" for c in schema.numeric]
        return prep
