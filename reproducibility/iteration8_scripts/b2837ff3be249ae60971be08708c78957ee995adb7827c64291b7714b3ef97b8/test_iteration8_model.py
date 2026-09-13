"""Tests for the fit-only robust representation and physical-unit rule energy."""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from iteration8_model import AsinhMADPreprocessor, TransformedConstraintSet
from medhyperdiff.constraints import build_constraints
from medhyperdiff.data import SCHEMAS, toy_data
from medhyperdiff.preprocessing import Preprocessor


def test_asinh_mad_roundtrip_and_missingness():
    schema = SCHEMAS["toy"]
    frame = toy_data(120, 73)
    frame.loc[:4, "total_protein"] = np.nan
    prep = AsinhMADPreprocessor(schema).fit(frame)
    numeric, categorical = prep.transform(frame)
    restored = prep.inverse(numeric, categorical)
    for col in schema.numeric:
        observed = frame[col].notna().to_numpy()
        np.testing.assert_allclose(restored.loc[observed, col], frame.loc[observed, col], rtol=2e-5, atol=2e-5)
        assert restored[col].isna().tolist() == frame[col].isna().tolist()


def test_transformed_constraint_energy_matches_affine_physical_energy():
    schema = SCHEMAS["toy"]
    frame = toy_data(120, 74)
    affine = Preprocessor(schema).fit(frame)
    robust = AsinhMADPreprocessor(schema).fit(frame)
    base = build_constraints(schema, frame)
    transformed = TransformedConstraintSet(base.rules)
    affine_num, affine_cat = affine.transform(frame.iloc[:16])
    robust_num, robust_cat = robust.transform(frame.iloc[:16])
    torch.manual_seed(19)
    base_logits = [torch.randn(16, k) for k in affine.cardinalities]
    transformed_logits = [x.clone() for x in base_logits]
    base_value = base.energy(torch.tensor(affine_num), [x.clone() for x in base_logits], affine,
                             torch.tensor(affine_cat[:, len(affine.cat_names):], dtype=torch.float32))
    transformed_value = transformed.energy(torch.tensor(robust_num), transformed_logits, robust,
                                           torch.tensor(robust_cat[:, len(robust.cat_names):], dtype=torch.float32))
    torch.testing.assert_close(base_value, transformed_value, rtol=2e-5, atol=2e-5)
