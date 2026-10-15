"""Explicit schemas, auditable UCI ingestion, and group-safe nested splits."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


@dataclass(frozen=True)
class Schema:
    name: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    target: str
    positive: str
    uci_id: int | None = None
    exclude: tuple[str, ...] = ()

    @property
    def features(self):
        return list(self.numeric + self.categorical)

    @property
    def columns(self):
        return self.features + [self.target]


SCHEMAS = {
    "heart_failure": Schema("heart_failure", (
        "age", "creatinine_phosphokinase", "ejection_fraction", "platelets",
        "serum_creatinine", "serum_sodium"), (
        "anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"),
        "DEATH_EVENT", "1", 519, ("time",)),
    "ilpd": Schema("ilpd", ("age", "total_bilirubin", "direct_bilirubin",
        "alkaline_phosphatase", "alt", "ast", "total_protein", "albumin", "ag_ratio"),
        ("sex",), "disease", "1", 225),
    "ckd": Schema("ckd", ("age", "bp", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"),
        ("sg", "al", "su", "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"),
        "class", "ckd", 336),
    "diabetes": Schema("diabetes", ("Age",), (
        "Gender", "Polyuria", "Polydipsia", "sudden weight loss", "weakness", "Polyphagia",
        "Genital thrush", "visual blurring", "Itching", "Irritability", "delayed healing",
        "partial paresis", "muscle stiffness", "Alopecia", "Obesity"), "class", "Positive", 529),
    # Iteration 12 populations with definitional laboratory/anthropometric identities.
    "nhanes": Schema("nhanes", ("age", "height", "weight", "bmi", "albumin", "total_protein", "globulin", "alt",
        "ast", "alp", "ggt", "total_bilirubin", "creatinine", "bun", "glucose", "sodium", "potassium", "calcium",
        "uric_acid", "triglycerides", "cholesterol"), ("sex",), "diabetes", "1"),
    "nhanes_cbc": Schema("nhanes_cbc", ("age", "rbc", "hgb", "hct", "mcv", "mch", "mchc", "rdw", "wbc", "lymph_pct",
        "mono_pct", "neut_pct", "eos_pct", "baso_pct", "platelets", "mpv"), ("sex",), "diabetes", "1"),
    "thyroid": Schema("thyroid", ("age", "tsh", "t3", "tt4", "t4u", "fti"), (
        "sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication", "sick", "pregnant",
        "thyroid_surgery", "i131_treatment", "query_hypothyroid", "query_hyperthyroid", "lithium", "goitre",
        "tumor", "hypopituitary", "psych", "referral_source"), "condition", "1"),
    "toy": Schema("toy", ("total_protein", "albumin", "ag_ratio", "age"),
        ("symptom_a", "symptom_b", "symptom_c"), "disease", "1"),
}


def canonical_category(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text in ("", "?", "nan", "NaN"):
        return np.nan
    # Numeric binary labels must not vary between CSV and dataframe representations.
    try:
        number = float(text)
        if np.isfinite(number):
            return format(number, ".12g")
    except ValueError:
        pass
    return text


def normalize(frame: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = frame.columns.str.strip()
    if schema.name == "heart_failure":
        frame = frame.rename(columns={"death_event": "DEATH_EVENT"})
    if schema.name == "ckd":
        frame = frame.rename(columns={"wbcc": "wc", "rbcc": "rc"})
    if schema.name == "diabetes":
        aliases = {col.lower().replace(" ", "_"): col for col in schema.columns}
        frame = frame.rename(columns=aliases)
    if schema.name == "ilpd":
        # UCI exposes these positional names in its standardized CSV.
        aliases = dict(zip(["Age", "Gender", "TB", "DB", "Alkphos", "Sgpt", "Sgot", "TP", "ALB", "A/G Ratio", "Selector"],
                           ["age", "sex", "total_bilirubin", "direct_bilirubin", "alkaline_phosphatase", "alt", "ast", "total_protein", "albumin", "ag_ratio", "disease"]))
        frame = frame.rename(columns=aliases)
    missing = set(schema.columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{schema.name}: missing schema columns {sorted(missing)}; found {list(frame.columns)}")
    frame = frame[schema.columns].copy()
    for col in schema.numeric:
        raw = frame[col].map(canonical_category)
        converted = pd.to_numeric(raw, errors="coerce")
        invalid = raw.notna() & converted.isna()
        if invalid.any():
            raise ValueError(f"Unrecognized numeric values in {col}: {raw[invalid].unique()}")
        frame[col] = converted.astype(float)
        if np.isinf(frame[col]).any():
            raise ValueError(f"Infinite values in {col}")
    for col in (*schema.categorical, schema.target):
        frame[col] = frame[col].map(canonical_category)
    if frame[schema.target].isna().any():
        raise ValueError("Missing targets require an explicit study policy")
    if frame[schema.target].nunique() != 2 or schema.positive not in set(frame[schema.target]):
        raise ValueError("This benchmark runner currently requires a binary target with a known positive class")
    return frame.reset_index(drop=True)


def download_uci(name: str, root: str | Path) -> Path:
    schema = SCHEMAS[name]
    if schema.uci_id is None:
        raise ValueError("Toy data are generated locally")
    root = Path(root)
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "processed").mkdir(parents=True, exist_ok=True)
    api = f"https://archive.ics.uci.edu/api/dataset?id={schema.uci_id}"
    def read(url):
        request = urllib.request.Request(url, headers={"User-Agent": "MedHyperDiff research/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    raw_path = root / "raw" / f"{name}.csv"
    metadata_path = root / "raw" / f"{name}.metadata.json"
    cached = raw_path.exists() and metadata_path.exists()
    metadata = json.loads(metadata_path.read_text()) if cached else json.loads(read(api))
    url = metadata["data"]["data_url"]
    if not url.startswith("https://archive.ics.uci.edu/"):
        raise ValueError(f"Unexpected UCI data host: {url}")
    raw = raw_path.read_bytes() if cached else read(url)
    if not cached:
        raw_path.write_bytes(raw)
        metadata_path.write_text(json.dumps(metadata, indent=2))
    frame = normalize(pd.read_csv(raw_path), schema)
    path = root / "processed" / f"{name}.csv"
    frame.to_csv(path, index=False)
    manifest = {
        "processed_at": datetime.now(timezone.utc).isoformat(), "raw_cache_reused": cached, "api": api, "url": url,
        "sha256_raw": hashlib.sha256(raw).hexdigest(), "schema": asdict(schema),
        "rows": len(frame), "missing_by_column": frame.isna().sum().to_dict(),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_predictors": int(frame.duplicated(schema.features).sum()),
        "excluded_columns": list(schema.exclude), "metadata": metadata,
    }
    (root / "raw" / f"{name}.manifest.json").write_text(json.dumps(manifest, indent=2))
    return path


NHANES_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/"
# SHA-256 of the 2017-2018 (cycle J) public files as downloaded on 7 October 2026.
NHANES_FILES = {"DEMO_J": "c0b46e0345ea19404928656277c8b0d10b0cca348a9b2fe4fc3c67e8b7ee73ec",
                "BMX_J": "8d675e42d8826ac98714b2c3dd4c5138a5e353fb4424f7eff5e6db4a01ce838a",
                "BIOPRO_J": "5bcd5722c1892883b96a9d7fed0befadabacae313f800fadc0133cd5dd00c4c6",
                "DIQ_J": "1ecbf5360dfc331d1efbf32198553dc30e9a1f4cff0a907ed30ca72bac797f89"}
NHANES_COLUMNS = {"RIDAGEYR": "age", "BMXHT": "height", "BMXWT": "weight", "BMXBMI": "bmi", "LBXSAL": "albumin",
    "LBXSTP": "total_protein", "LBXSGB": "globulin", "LBXSATSI": "alt", "LBXSASSI": "ast", "LBXSAPSI": "alp",
    "LBXSGTSI": "ggt", "LBXSTB": "total_bilirubin", "LBXSCR": "creatinine", "LBXSBU": "bun", "LBXSGL": "glucose",
    "LBXSNASI": "sodium", "LBXSKSI": "potassium", "LBXSCA": "calcium", "LBXSUA": "uric_acid",
    "LBXSTR": "triglycerides", "LBXSCH": "cholesterol"}
CBC_SHA256 = "00f964098dc91272e415344554cbf1b627ecaeb2eda201518527446a7d81e742"
CBC_COLUMNS = {"RIDAGEYR": "age", "LBXRBCSI": "rbc", "LBXHGB": "hgb", "LBXHCT": "hct", "LBXMCVSI": "mcv",
    "LBXMCHSI": "mch", "LBXMC": "mchc", "LBXRDW": "rdw", "LBXWBCSI": "wbc", "LBXLYPCT": "lymph_pct",
    "LBXMOPCT": "mono_pct", "LBXNEPCT": "neut_pct", "LBXEOPCT": "eos_pct", "LBXBAPCT": "baso_pct",
    "LBXPLTSI": "platelets", "LBXMPSI": "mpv"}
THYROID_URL = "https://archive.ics.uci.edu/static/public/102/thyroid+disease.zip"
THYROID_ZIP_SHA256 = None  # recorded in the manifest at first download; UCI does not publish a hash
THYROID_COLUMNS = ["age", "sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication", "sick", "pregnant",
    "thyroid_surgery", "i131_treatment", "query_hypothyroid", "query_hyperthyroid", "lithium", "goitre", "tumor",
    "hypopituitary", "psych", "tsh_measured", "tsh", "t3_measured", "t3", "tt4_measured", "tt4", "t4u_measured", "t4u",
    "fti_measured", "fti", "tbg_measured", "tbg", "referral_source", "diagnosis"]


def _fetch(url, path, sha256=None):
    if not path.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "MedHyperDiff research/0.1"})
        with urllib.request.urlopen(request, timeout=120) as response:
            path.write_bytes(response.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha256 is not None and digest != sha256:
        raise ValueError(f"{path.name}: SHA-256 {digest} differs from the pinned {sha256}; the source changed")
    return digest


def build_nhanes(root):
    """NHANES 2017-2018 adults (>=20 y) with a standard biochemistry record; label = diagnosed diabetes.

    Globulin is computed by NHANES as total protein minus albumin and BMI as weight/height^2 rounded to
    0.1, so both are definitional identities. Borderline (DIQ010=3) and refused/unknown answers are
    excluded rather than recoded. Age is top-coded at 80 by NHANES. Survey weights are not used: this is
    a convenience population for method evaluation, not a nationally representative estimate.
    """
    root = Path(root)
    raw = root / "raw" / "nhanes"
    raw.mkdir(parents=True, exist_ok=True)
    hashes = {name: _fetch(NHANES_BASE + f"{name}.xpt", raw / f"{name}.xpt", sha) for name, sha in NHANES_FILES.items()}
    read = lambda name: pd.read_sas(raw / f"{name}.xpt", format="xport")
    demo, bmx, bio, diq = (read(n) for n in ("DEMO_J", "BMX_J", "BIOPRO_J", "DIQ_J"))
    frame = (demo[["SEQN", "RIDAGEYR", "RIAGENDR"]].merge(bio[["SEQN"] + [c for c in NHANES_COLUMNS if c.startswith("LB")]],
             on="SEQN").merge(bmx[["SEQN", "BMXHT", "BMXWT", "BMXBMI"]], on="SEQN", how="left")
             .merge(diq[["SEQN", "DIQ010"]], on="SEQN"))
    counts = {"biochemistry_records": int(len(bio))}
    frame = frame[frame.RIDAGEYR >= 20]
    counts["adults_with_biochemistry_and_questionnaire"] = int(len(frame))
    frame = frame[frame.DIQ010.isin([1, 2])]
    counts["after_excluding_borderline_or_unknown_diabetes"] = int(len(frame))
    labs = [c for c in NHANES_COLUMNS if c.startswith("LB")]
    frame = frame[frame[labs].notna().any(axis=1)]
    counts["with_any_biochemistry_value"] = int(len(frame))
    out = frame.rename(columns=NHANES_COLUMNS)
    out["sex"] = out.RIAGENDR.map({1.: "male", 2.: "female"})
    out["diabetes"] = (out.DIQ010 == 1).astype(int).astype(str)
    out = normalize(out, SCHEMAS["nhanes"])
    path = root / "processed" / "nhanes.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    (root / "raw" / "nhanes.manifest.json").write_text(json.dumps({"processed_at": datetime.now(timezone.utc).isoformat(),
        "source": NHANES_BASE, "files_sha256": hashes, "counts": counts, "rows": len(out),
        "missing_by_column": out.isna().sum().to_dict(), "schema": asdict(SCHEMAS["nhanes"]),
        "terms": "NHANES public-use data, U.S. public domain; cite NCHS. Not survey-weighted."}, indent=2))
    return path


def build_nhanes_cbc(root):
    """NHANES 2017-2018 adults (>=20 y) with a complete blood count; label = diagnosed diabetes.

    The analyser measures RBC, haemoglobin and MCV and computes haematocrit, MCH and MCHC; the five
    differential percentages are reported to 0.1 and sum to about 100. These relations therefore hold
    up to reporting rounding. Participants whose CBC panel is entirely missing are excluded.
    """
    root = Path(root)
    raw = root / "raw" / "nhanes"
    raw.mkdir(parents=True, exist_ok=True)
    hashes = {name: _fetch(NHANES_BASE + f"{name}.xpt", raw / f"{name}.xpt", NHANES_FILES[name]) for name in ("DEMO_J", "DIQ_J")}
    hashes["CBC_J"] = _fetch(NHANES_BASE + "CBC_J.xpt", raw / "CBC_J.xpt", CBC_SHA256)
    read = lambda name: pd.read_sas(raw / f"{name}.xpt", format="xport")
    cbc = read("CBC_J")
    frame = (read("DEMO_J")[["SEQN", "RIDAGEYR", "RIAGENDR"]].merge(cbc[["SEQN"] + [c for c in CBC_COLUMNS if c.startswith("LB")]], on="SEQN")
             .merge(read("DIQ_J")[["SEQN", "DIQ010"]], on="SEQN"))
    counts = {"cbc_records": int(len(cbc))}
    frame = frame[(frame.RIDAGEYR >= 20) & frame.DIQ010.isin([1, 2])]
    counts["adults_with_definite_diabetes_answer"] = int(len(frame))
    labs = [c for c in CBC_COLUMNS if c.startswith("LB")]
    frame = frame[frame[labs].notna().any(axis=1)]
    counts["with_any_cbc_value"] = int(len(frame))
    out = frame.rename(columns=CBC_COLUMNS)
    out["sex"] = out.RIAGENDR.map({1.: "male", 2.: "female"})
    out["diabetes"] = (out.DIQ010 == 1).astype(int).astype(str)
    out = normalize(out, SCHEMAS["nhanes_cbc"])
    path = root / "processed" / "nhanes_cbc.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    (root / "raw" / "nhanes_cbc.manifest.json").write_text(json.dumps({"processed_at": datetime.now(timezone.utc).isoformat(),
        "source": NHANES_BASE, "files_sha256": hashes, "counts": counts, "rows": len(out),
        "missing_by_column": out.isna().sum().to_dict(), "schema": asdict(SCHEMAS["nhanes_cbc"]),
        "terms": "NHANES public-use data, U.S. public domain; cite NCHS. Not survey-weighted."}, indent=2))
    return path


def build_thyroid(root):
    """UCI Thyroid Disease, thyroid0387 file (Garvan Institute; Quinlan). Label = any diagnosis code.

    Ages above 120 are source errors and set missing. `*_measured` flags duplicate missingness and
    are dropped; TBG (96% missing) is dropped. FTI is reported by the laboratory as a function of
    TT4 and T4U, so FTI = TT4/T4U holds up to rounding and rare entry errors.
    """
    import io
    import zipfile
    root = Path(root)
    raw = root / "raw" / "thyroid"
    raw.mkdir(parents=True, exist_ok=True)
    digest = _fetch(THYROID_URL, raw / "thyroid+disease.zip", THYROID_ZIP_SHA256)
    with zipfile.ZipFile(raw / "thyroid+disease.zip") as archive:
        text = archive.read("thyroid0387.data").decode()
    frame = pd.read_csv(io.StringIO(text), header=None, names=THYROID_COLUMNS, na_values="?")
    counts = {"records": int(len(frame))}
    frame["diagnosis"] = frame["diagnosis"].astype(str).str.split("[").str[0].str.strip()
    frame["condition"] = (frame["diagnosis"] != "-").astype(int).astype(str)
    frame["age"] = pd.to_numeric(frame["age"], errors="coerce")
    # The source file writes some values with float artifacts (9.599999 for 9.6); no value has more
    # than three genuine decimals, so four-decimal rounding restores the reported numbers.
    for col in ("tsh", "t3", "tt4", "t4u", "fti"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce").round(4)
    counts["age_above_120_set_missing"] = int((frame["age"] > 120).sum())
    frame.loc[frame["age"] > 120, "age"] = np.nan
    out = normalize(frame, SCHEMAS["thyroid"])
    path = root / "processed" / "thyroid.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    (root / "raw" / "thyroid.manifest.json").write_text(json.dumps({"processed_at": datetime.now(timezone.utc).isoformat(),
        "source": THYROID_URL, "zip_sha256": digest, "counts": counts, "rows": len(out),
        "missing_by_column": out.isna().sum().to_dict(), "schema": asdict(SCHEMAS["thyroid"]),
        "terms": "UCI Machine Learning Repository, CC BY 4.0; cite Quinlan, Garvan Institute."}, indent=2))
    return path


def toy_data(n=512, seed=0):
    """Controlled mechanism test; these are simulated values, not patients."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2, n)
    b = rng.integers(0, 2, n)
    c = a ^ b  # Pairwise-independent parity relation is a higher-order negative control.
    albumin = rng.uniform(2, 5, n)
    globulin = rng.uniform(1, 4, n)
    age = rng.uniform(25, 80, n)
    y = rng.binomial(1, 1 / (1 + np.exp(-((age - 50) / 15 + c - .5))))
    return normalize(pd.DataFrame({"total_protein": albumin + globulin, "albumin": albumin,
        "ag_ratio": albumin / globulin, "age": age, "symptom_a": a,
        "symptom_b": b, "symptom_c": c, "disease": y}), SCHEMAS["toy"])


def nested_splits(frame, schema, folds=5, seed=2026, inner_seed=None):
    """Keep identical predictor profiles together, including conflicting labels.

    Original patient IDs, when available, should replace these conservative groups.
    The same fixed split seed must be used for every model and training seed.
    """
    if folds < 2:
        raise ValueError("At least two outer folds are required")
    groups = pd.util.hash_pandas_object(frame[schema.features], index=False).to_numpy()
    y = frame[schema.target].to_numpy()
    outer = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
    for fold, (development, test) in enumerate(outer.split(frame, y, groups)):
        inner = StratifiedGroupKFold(4, shuffle=True,
            random_state=seed + fold + 1 if inner_seed is None else inner_seed + fold)
        train_local, val_local = next(inner.split(frame.iloc[development], y[development], groups[development]))
        train, val = development[train_local], development[val_local]
        sets = [set(groups[i]) for i in (train, val, test)]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        for idx in (train, val, test):
            if len(np.unique(y[idx])) != 2:
                raise ValueError("A split lacks a target class; reduce folds or use a larger dataset")
        yield {"fold": fold, "train": train.tolist(), "validation": val.tolist(), "test": test.tolist()}
