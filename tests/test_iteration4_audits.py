"""Missing-data bounds must not turn unknown relationships into validity claims."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from medhyperdiff.constraints import ConstraintSet, Rule

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from iteration4 import rule_bounds, missingness_audit
from medhyperdiff.data import SCHEMAS


def test_unknown_rule_union_not_double_counted():
    frame = pd.DataFrame({"a": [-1., np.nan, 1., np.nan], "b": [np.nan, np.nan, 2., 2.]})
    constraints = ConstraintSet([Rule("a_nonnegative", "lower", ("a",)),
                                 Rule("a_le_b", "le", ("a", "b"))])
    result = rule_bounds(frame, constraints)
    assert result["rules"]["a_nonnegative"]["violation_lower"] == .25
    assert result["rules"]["a_nonnegative"]["violation_upper"] == .75
    # Row 0 is both known-invalid and unknown on another rule; count once.
    assert result["any_rule"]["violation_lower"] == .25
    assert result["any_rule"]["violation_upper"] == .75
    assert result["any_rule"]["rows_with_unknown_rule"] == 3


def test_all_unknown_rule_has_zero_to_one_bound():
    frame = pd.DataFrame({"a": [np.nan, np.nan]})
    result = rule_bounds(frame, ConstraintSet([Rule("nonnegative", "lower", ("a",))]))
    assert result["rules"]["nonnegative"]["eligible"] == 0
    assert result["any_rule"]["violation_lower"] == 0
    assert result["any_rule"]["violation_upper"] == 1


def test_concentrated_missingness_not_hidden_by_feature_average():
    schema = SCHEMAS["ilpd"]
    fit = pd.DataFrame({c: np.ones(100) for c in schema.columns})
    validation, synthetic = fit.copy(), fit.copy()
    synthetic.loc[:9, "ag_ratio"] = np.nan
    result = missingness_audit(fit, validation, synthetic, schema)
    assert result["ag_ratio"]["error_vs_fit"] == .1
    assert result["ag_ratio"]["error_vs_validation"] == .1
    assert result["albumin"]["error_vs_fit"] == 0
