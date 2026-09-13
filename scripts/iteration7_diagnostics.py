"""Supplemental numerical-tolerance diagnostics; original metrics stay unchanged."""
import numpy as np
import pandas as pd


def tolerances(train, schema):
    # 1 ppm of fit SD, with a one-source-unit floor. Numerical, not clinical, tolerance.
    return {c: 1e-6*max(float(train[c].std()) if np.isfinite(train[c].std()) else 0., 1.) for c in schema.numeric}


def near_copy_rate(synthetic, reference, train, schema):
    eps, matched = tolerances(train, schema), 0
    for start in range(0, len(synthetic), 128):
        sample = synthetic.iloc[start:start+128]
        equal = np.ones((len(sample), len(reference)), dtype=bool)
        for col in schema.features:
            a, b = sample[col].to_numpy()[:, None], reference[col].to_numpy()[None, :]
            missing_a, missing_b = pd.isna(a), pd.isna(b)
            if col in schema.numeric:
                valid = np.isfinite(a) & np.isfinite(b) & (np.abs(a-b) <= eps[col])
            else:
                valid = (~missing_a) & (~missing_b) & (a == b)
            equal &= valid | (missing_a & missing_b)
        matched += equal.any(axis=1).sum()
    return float(matched/len(synthetic))


def boundary_rates(train, validation, synthetic, schema):
    result, eps = {}, tolerances(train, schema)
    for c in schema.numeric:
        fit = train[c].dropna()
        if len(fit) == 0:
            continue
        lo, hi, q95 = fit.min(), fit.max(), fit.quantile(.95)
        row = dict(tolerance=eps[c])
        for label, table in (("fit", train), ("validation", validation), ("synthetic", synthetic)):
            values = table[c].dropna()
            if len(values) == 0:
                continue
            row[label+"_outside_fit_range_strict"] = float(((values < lo) | (values > hi)).mean())
            row[label+"_outside_fit_range_tolerant"] = float(((values < lo-eps[c]) | (values > hi+eps[c])).mean())
            row[label+"_above_fit_q95_strict"] = float((values > q95).mean())
            row[label+"_above_fit_q95_tolerant"] = float((values > q95+eps[c]).mean())
        result[c] = row
    return result
