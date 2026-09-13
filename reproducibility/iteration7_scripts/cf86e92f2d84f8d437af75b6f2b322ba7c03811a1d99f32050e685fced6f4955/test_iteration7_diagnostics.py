"""Positive controls for numerical copy and boundary diagnostics."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from iteration7_diagnostics import near_copy_rate, boundary_rates, tolerances
from medhyperdiff.data import SCHEMAS, toy_data


def test_roundoff_copies_detected_but_real_numeric_changes_rejected():
    schema, fit = SCHEMAS["toy"], toy_data(40, 184)
    generated = fit.copy()
    eps = tolerances(fit, schema)
    for c in schema.numeric:
        generated[c] += eps[c]/2
    assert near_copy_rate(generated, fit, fit, schema) == 1.
    generated["age"] = fit.age.max()+100
    assert near_copy_rate(generated, fit, fit, schema) == 0.


def test_copy_requires_matching_categories_and_missingness():
    schema, fit = SCHEMAS["toy"], toy_data(1, 19)
    generated = fit.copy()
    generated["symptom_a"] = "unseen"
    assert near_copy_rate(generated, fit, fit, schema) == 0.
    generated = fit.copy()
    generated["age"] = np.nan
    assert near_copy_rate(generated, fit, fit, schema) == 0.
    fit["age"] = np.nan
    assert near_copy_rate(generated, fit, fit, schema) == 1.


def test_roundoff_threshold_crossing_is_not_extrapolation():
    schema, fit = SCHEMAS["toy"], toy_data(40, 23)
    generated = fit.copy()
    eps = tolerances(fit, schema)
    generated["age"] = fit.age.max()+eps["age"]*.5
    result = boundary_rates(fit, fit, generated, schema)["age"]
    assert result["synthetic_outside_fit_range_strict"] == 1.
    assert result["synthetic_outside_fit_range_tolerant"] == 0.
    generated["age"] = fit.age.max()+eps["age"]*2
    assert boundary_rates(fit, fit, generated, schema)["age"]["synthetic_outside_fit_range_tolerant"] == 1.
