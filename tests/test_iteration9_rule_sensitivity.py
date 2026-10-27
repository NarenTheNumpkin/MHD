"""Unit checks for explicit wrong-rule sensitivity variants."""
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from iteration9 import variant_constraints
from medhyperdiff.constraints import build_constraints
from medhyperdiff.data import SCHEMAS, normalize


def ilpd_fit():
    frame = normalize(pd.read_csv(Path(__file__).resolve().parents[1]/"data/processed/ilpd.csv"), SCHEMAS["ilpd"])
    return frame.iloc[:349]


def test_omission_removes_only_exploratory_identity():
    frame = ilpd_fit(); correct=variant_constraints(SCHEMAS["ilpd"],frame,"correct"); omitted=variant_constraints(SCHEMAS["ilpd"],frame,"omit_protein_identity")
    assert {r.name for r in correct.rules}-{r.name for r in omitted.rules}=={"protein_identity"}


def test_reversed_variants_are_explicit_and_direction_changes():
    frame=ilpd_fit(); correct=build_constraints(SCHEMAS["ilpd"],frame); wrong=variant_constraints(SCHEMAS["ilpd"],frame,"reverse_both")
    assert all("REVERSED" not in r.name for r in correct.rules)
    assert {r.name for r in wrong.rules if "REVERSED" in r.name}=={"direct_le_total_bilirubin_REVERSED","albumin_le_total_REVERSED"}
    assert all(r.status=="sensitivity_intentionally_wrong" for r in wrong.rules if "REVERSED" in r.name)
    assert wrong.evaluate(frame)["any_rule_violation_rate"] > .9
