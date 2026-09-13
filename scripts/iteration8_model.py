"""Fit-only robust numerical representation for the unchanged diffusion backbone."""
import copy

import numpy as np
import pandas as pd
import torch

from medhyperdiff.constraints import ConstraintSet
from medhyperdiff.preprocessing import Preprocessor
from iteration5_model import ObservationInitializedDiffusion


class AsinhMADPreprocessor(Preprocessor):
    """Monotone asinh transform centered/scaled from the fit partition only.

    The model sees a standardised asinh coordinate. Decoding applies the exact
    inverse, so evaluation and saved synthetic tables remain in source units.
    """

    transform_name = "asinh_mad_v1"

    def fit(self, frame):
        if frame.empty:
            raise ValueError("Cannot fit an empty training partition")
        numeric = frame[list(self.schema.numeric)].astype(float)
        observed = np.isfinite(numeric.to_numpy())
        raw = numeric.to_numpy(dtype=float)
        self.medians = numeric.median().fillna(0).to_numpy(dtype=float)
        filled = np.where(observed, raw, self.medians[None, :])
        self.robust_centers = np.nanmedian(np.where(observed, raw, np.nan), axis=0)
        self.robust_centers = np.where(np.isfinite(self.robust_centers), self.robust_centers, self.medians)
        deviations = np.where(observed, np.abs(raw - self.robust_centers[None, :]), np.nan)
        mad = np.nanmedian(deviations, axis=0)
        self.robust_scales = np.maximum(1.4826 * np.where(np.isfinite(mad), mad, 0.), 1e-6)
        transformed = np.arcsinh((filled - self.robust_centers[None, :]) / self.robust_scales[None, :])
        self.means = transformed.mean(axis=0)
        self.scales = transformed.std(axis=0, ddof=0)
        self.scales[self.scales < 1e-6] = 1
        self.cat_names = list(self.schema.categorical)
        self.mask_names = [f"{col}__observed" for col in self.schema.numeric]
        if set(self.mask_names) & set(self.schema.columns):
            raise ValueError("Missingness token names conflict with feature names")
        self.levels = []
        for col in self.cat_names:
            vals = frame[col].dropna().astype(str).unique().tolist()
            if self.missing in vals or self.unknown in vals:
                raise ValueError("Reserved categorical token in source data")
            self.levels.append(sorted(vals) + [self.missing, self.unknown])
        self.levels += [["0", "1"] for _ in self.mask_names]
        _, cat = self.transform(frame)
        self.allowed = [np.bincount(cat[:, i], minlength=len(level)) > 0 for i, level in enumerate(self.levels)]
        self.token_names = list(self.schema.numeric) + self.cat_names + self.mask_names
        self.fitted_rows = len(frame)
        return self

    def transform(self, frame):
        raw = frame[list(self.schema.numeric)].to_numpy(dtype=float)
        observed = np.isfinite(raw)
        filled = np.where(observed, raw, self.medians[None, :])
        transformed = np.arcsinh((filled - self.robust_centers[None, :]) / self.robust_scales[None, :])
        numeric = ((transformed - self.means[None, :]) / self.scales[None, :]).astype(np.float32)
        codes = []
        for col, levels in zip(self.cat_names, self.levels):
            lookup = {v: i for i, v in enumerate(levels)}
            values = frame[col].astype(object).where(frame[col].notna(), self.missing).astype(str)
            codes.append(values.map(lambda v: lookup.get(v, lookup[self.unknown])).to_numpy(dtype=np.int64))
        codes.extend(observed[:, j].astype(np.int64) for j in range(observed.shape[1]))
        cat = np.stack(codes, axis=1) if codes else np.empty((len(frame), 0), dtype=np.int64)
        return numeric, cat

    def inverse_numeric(self, numeric):
        numeric = np.asarray(numeric, dtype=float)
        transformed = numeric * self.scales[None, :] + self.means[None, :]
        raw = self.robust_centers[None, :] + self.robust_scales[None, :] * np.sinh(transformed)
        if not np.isfinite(raw).all():
            raise ValueError("Nonfinite generated numerical values after asinh inversion")
        return raw

    def inverse_numeric_tensor(self, numeric):
        center = torch.as_tensor(self.robust_centers, dtype=numeric.dtype, device=numeric.device)
        scale = torch.as_tensor(self.robust_scales, dtype=numeric.dtype, device=numeric.device)
        mean = torch.as_tensor(self.means, dtype=numeric.dtype, device=numeric.device)
        std = torch.as_tensor(self.scales, dtype=numeric.dtype, device=numeric.device)
        return center + scale * torch.sinh(numeric * std + mean)

    def inverse(self, numeric, categorical):
        raw = self.inverse_numeric(numeric)
        categorical = np.asarray(categorical)
        data = {col: raw[:, i].copy() for i, col in enumerate(self.schema.numeric)}
        for i, col in enumerate(self.cat_names):
            values = np.array(self.levels[i], dtype=object)[categorical[:, i]]
            data[col] = pd.Series(values).replace({self.missing: np.nan, self.unknown: np.nan})
        for i, col in enumerate(self.schema.numeric):
            observed = categorical[:, len(self.cat_names) + i].astype(bool)
            data[col][~observed] = np.nan
        return pd.DataFrame(data)[self.schema.features]

    def to_dict(self):
        return {"transform": self.transform_name, "robust_centers": self.robust_centers.tolist(),
                "robust_scales": self.robust_scales.tolist(), **super().to_dict()}


class TransformedConstraintSet(ConstraintSet):
    """Evaluate physical-unit rules while the denoiser uses transformed values."""

    transform_name = AsinhMADPreprocessor.transform_name

    def energy(self, numeric, logits, preprocessor, observed=None, detach_masks=True):
        physical = preprocessor.inverse_numeric_tensor(numeric)
        proxy = copy.copy(preprocessor)
        proxy.means = np.zeros_like(preprocessor.means)
        proxy.scales = np.ones_like(preprocessor.scales)
        return super().energy(physical, logits, proxy, observed, detach_masks)


class RobustObservationDiffusion(ObservationInitializedDiffusion):
    """Observation-initialised model using fit-only asinh-MAD coordinates."""

    transform_name = AsinhMADPreprocessor.transform_name
