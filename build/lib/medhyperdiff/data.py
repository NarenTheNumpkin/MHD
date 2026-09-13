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
