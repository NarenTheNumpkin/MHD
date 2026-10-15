"""Separated fidelity, utility, constraint coverage, and empirical privacy audits."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
    balanced_accuracy_score, confusion_matrix, brier_score_loss)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .structure import association_matrix


def mean_finite(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def fidelity(train, real, synthetic, schema):
    per_feature = {}
    for col in schema.columns:
        r, s = real[col].dropna(), synthetic[col].dropna()
        metric = {"missing_rate_error": abs(float(real[col].isna().mean() - synthetic[col].isna().mean())),
                  "real_observed": len(r), "synthetic_observed": len(s)}
        if col in schema.numeric:
            spread = float(train[col].std())
            spread = spread if np.isfinite(spread) and spread > 1e-6 else 1
            metric.update({"wasserstein_train_sd": float(wasserstein_distance(r, s) / spread) if len(r) and len(s) else None,
                           "std_ratio_synthetic_to_real": float(s.std() / r.std()) if len(r) > 1 and len(s) > 1 and r.std() > 1e-8 else None,
                           "ks": float(ks_2samp(r, s).statistic) if len(r) and len(s) else None})
        else:
            rc = real[col].fillna("__MISSING__").astype(str).value_counts(normalize=True)
            sc = synthetic[col].fillna("__MISSING__").astype(str).value_counts(normalize=True)
            levels = rc.index.union(sc.index)
            p, q = rc.reindex(levels, fill_value=0).to_numpy(), sc.reindex(levels, fill_value=0).to_numpy()
            metric.update({"total_variation": float(abs(p-q).sum()/2),
                           "js_divergence_base2": float(jensenshannon(p, q, base=2)**2),
                           "category_coverage": float(len(set(rc.index) & set(sc.index))/len(rc))})
        per_feature[col] = metric
    ar, overlap_r = association_matrix(real[schema.columns], schema.numeric)
    ass, overlap_s = association_matrix(synthetic[schema.columns], schema.numeric)
    upper = np.triu(np.ones_like(ar, dtype=bool), 1) & (overlap_r >= 10) & (overlap_s >= 10)
    rmse = float(np.sqrt(np.mean((ar[upper] - ass[upper])**2))) if upper.any() else None
    # Numeric signed correlations catch dependency sign reversal missed by magnitudes.
    rcor = real[list(schema.numeric)].corr(method="spearman", min_periods=10).to_numpy()
    scor = synthetic[list(schema.numeric)].corr(method="spearman", min_periods=10).to_numpy()
    valid = np.triu(np.ones_like(rcor, dtype=bool), 1) & np.isfinite(rcor) & np.isfinite(scor)
    signed = float(np.sqrt(np.mean((rcor[valid] - scor[valid])**2))) if valid.any() else None
    return {"per_feature": per_feature,
        "mean_numeric_wasserstein_train_sd": mean_finite(v.get("wasserstein_train_sd") for v in per_feature.values()),
        "mean_categorical_tv": mean_finite(v.get("total_variation") for v in per_feature.values()),
        "mean_missing_rate_error": mean_finite(v["missing_rate_error"] for v in per_feature.values()),
        "association_rmse": rmse, "association_pairs_evaluated": int(upper.sum()), "signed_numeric_spearman_rmse": signed}


def utility_pipeline(schema, kind, seed):
    encoder = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()), list(schema.numeric)),
        ("categorical", make_pipeline(SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True),
                                      OneHotEncoder(handle_unknown="ignore", sparse_output=False)), list(schema.categorical)),
    ], sparse_threshold=0)
    classifier = (LogisticRegression(max_iter=1000, random_state=seed) if kind == "logistic" else
                  RandomForestClassifier(n_estimators=150, min_samples_leaf=3, n_jobs=1, random_state=seed))
    return make_pipeline(encoder, classifier)


def classification_metrics(y, prob):
    pred = prob >= .5
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    two_classes = len(np.unique(y)) == 2
    return {"auroc": float(roc_auc_score(y, prob)) if two_classes else None,
            "average_precision": float(average_precision_score(y, prob)) if two_classes else None,
            "f1": float(f1_score(y, pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "sensitivity": float(tp/(tp+fn)) if tp+fn else None,
            "specificity": float(tn/(tn+fp)) if tn+fp else None,
            "brier_score": float(brier_score_loss(y, prob))}


def utility(train, real, synthetic, schema, seed=0, predictions=None):
    result = {}
    real_y = (real[schema.target].to_numpy() == schema.positive).astype(int)
    for kind in ("logistic", "forest"):
        result[kind] = {}
        # Each downstream preprocessing pipeline fits ONLY its own training table.
        for name, data in (("TRTR", train), ("TSTR", synthetic), ("AUGMENT_1x", pd.concat([train, synthetic], ignore_index=True))):
            y = (data[schema.target].to_numpy() == schema.positive).astype(int)
            if len(np.unique(y)) < 2:
                result[kind][name] = {"error": "training table lacks a class"}
                continue
            classifier = utility_pipeline(schema, kind, seed)
            classifier.fit(data[schema.features], y)
            prob = classifier.predict_proba(real[schema.features])[:, 1]
            result[kind][name] = classification_metrics(real_y, prob)
            if predictions is not None:
                predictions[f"{kind}_{name}"] = prob
    if predictions is not None:
        predictions["row_index"] = real.index.to_numpy()
        predictions["label"] = real_y
    return result


def c2st(real, synthetic, schema, seed=0):
    """Balanced RF discriminator with its own train/test folds and preprocessing."""
    n = min(len(real), len(synthetic))
    if n < 12:
        return {"auc": None, "n_per_source": n}
    x = pd.concat([real.sample(n=n, random_state=seed), synthetic.sample(n=n, random_state=seed)], ignore_index=True)
    # Include target as a feature to test the full joint table.
    from dataclasses import replace
    joint = replace(schema, categorical=(*schema.categorical, schema.target))
    y = np.r_[np.zeros(n), np.ones(n)]
    scores = np.zeros(2*n)
    groups = pd.util.hash_pandas_object(x[schema.columns], index=False).to_numpy()
    if len(np.unique(groups)) < 3:
        return {"auc": None, "n_per_source": n, "reason": "fewer than three distinct records"}
    for train_idx, test_idx in StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(x, y, groups):
        if len(np.unique(y[train_idx])) < 2:
            return {"auc": None, "n_per_source": n, "reason": "grouped discriminator training fold lacks a source"}
        clf = utility_pipeline(joint, "forest", seed)
        clf.fit(x.iloc[train_idx], y[train_idx])
        scores[test_idx] = clf.predict_proba(x.iloc[test_idx])[:, 1]
    return {"auc": float(roc_auc_score(y, scores)), "n_per_source": n,
            "duplicate_grouped": True,
            "interpretation": "descriptive discriminator result; 0.5 does not prove equality"}


def gower_distances(a, b, train, schema):
    """Bounded training-range Gower-like distance, including missingness mismatch.

    Both missing has distance zero; one missing has distance one. This is an
    explicitly defined variant rather than pairwise deletion of missing features.
    """
    distance = np.zeros((len(a), len(b)), dtype=np.float64)
    for col in schema.features:
        if col in schema.numeric:
            scale = float(train[col].max() - train[col].min())
            scale = scale if np.isfinite(scale) and scale > 1e-6 else 1
            av, bv = a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float)
            am, bm = np.isnan(av)[:, None], np.isnan(bv)[None, :]
            delta = np.minimum(abs(np.nan_to_num(av)[:, None] - np.nan_to_num(bv)[None, :]) / scale, 1)
            delta = np.where(am | bm, (am != bm).astype(float), delta)
        else:
            av, bv = a[col].fillna("__MISSING__").astype(str).to_numpy(), b[col].fillna("__MISSING__").astype(str).to_numpy()
            delta = (av[:, None] != bv[None, :]).astype(float)
        distance += delta
    return distance / len(schema.features)


def nearest(a, b, train, schema, k=1):
    if len(b) < k:
        raise ValueError("Too few reference records for requested neighbors")
    result = []
    for start in range(0, len(a), 128):
        d = gower_distances(a.iloc[start:start+128], b, train, schema)
        result.append(np.sort(np.partition(d, k-1, axis=1)[:, :k], axis=1))
    return np.concatenate(result)


def privacy(train, real, synthetic, schema, seed=0):
    rng = np.random.default_rng(seed)
    # Equal-size reference sets remove the nearest-neighbor reference-size advantage.
    n = min(len(train), len(real))
    if n < 4:
        return {"status": "insufficient candidate records"}
    members = train.iloc[rng.choice(len(train), n, replace=False)].reset_index(drop=True)
    nonmembers = real.iloc[rng.choice(len(real), n, replace=False)].reset_index(drop=True)
    dmem = nearest(synthetic, members, train, schema, k=2)
    dnon = nearest(synthetic, nonmembers, train, schema, k=2)
    def exact(reference):
        # Count generated rows, including repeated copies, not unique set cardinality.
        hashes = pd.util.hash_pandas_object(reference[schema.features].fillna("__MISSING__"), index=False)
        sh = pd.util.hash_pandas_object(synthetic[schema.features].fillna("__MISSING__"), index=False)
        return float(sh.isin(set(hashes)).mean())
    member_score = -nearest(members, synthetic, train, schema)[:, 0]
    nonmember_score = -nearest(nonmembers, synthetic, train, schema)[:, 0]
    cut = n // 2
    threshold = np.quantile(nonmember_score[:cut], .95, method="higher")
    # Orientation and threshold fixed before scoring the remaining candidates.
    test_scores = np.r_[member_score[cut:], nonmember_score[cut:]]
    y = np.r_[np.ones(n-cut), np.zeros(n-cut)]
    valid_ratio = dmem[:, 1] > 0
    return {"status": "empirical diagnostics, not a privacy guarantee", "equal_reference_rows": n,
        "exact_predictor_match_train": exact(train), "exact_predictor_match_holdout": exact(real),
        "dcr_member_median": float(np.median(dmem[:, 0])), "dcr_holdout_median": float(np.median(dnon[:, 0])),
        "dcr_member_q05": float(np.quantile(dmem[:, 0], .05)),
        "nndr_median": float(np.median(dmem[valid_ratio, 0] / dmem[valid_ratio, 1])) if valid_ratio.any() else None,
        "nndr_zero_second_neighbor_count": int((~valid_ratio).sum()),
        "nearest_synthetic_membership_auc": float(roc_auc_score(y, test_scores)),
        "membership_tpr_at_calibrated_threshold": float((member_score[cut:] > threshold).mean()),
        "membership_observed_fpr": float((nonmember_score[cut:] > threshold).mean()),
        "membership_calibration_nonmembers": cut, "membership_test_members": n-cut,
        "attack_limitations": "single nearest-distance attack; no shadow-model, attribute, singling-out or linkability evaluation"}


HARD_RULE_STATUSES = ("schema", "soft_biochemical", "controlled")


def rule_residuals(frame, rule, constraints):
    values = {c: pd.to_numeric(frame[c], errors="coerce").to_numpy() for c in rule.columns}
    raw = np.asarray(constraints.residual(rule, values), dtype=float)
    valid = frame[list(rule.columns)].notna().all(axis=1).to_numpy() & np.isfinite(raw)
    return raw[valid]


def rule_realism(train, real, synthetic, constraints):
    """Compare synthetic rule behaviour with REAL held-out records, not with zero.

    A fit-quantile envelope is violated by about 5% of real records by construction, so
    fewer synthetic violations than real ones is over-regularization, not extra
    validity. For each relational rule we report the absolute violation-rate gap and
    the Wasserstein distance between synthetic and real residual distributions, scaled
    by the fit residual SD. Hard rules (schema/biochemical/controlled) also get a
    pooled any-violation rate, where lower is genuinely better.
    """
    evaluated = {name: constraints.evaluate(frame) for name, frame in (("real", real), ("synthetic", synthetic))}
    per_rule = {}
    for rule in constraints.rules:
        real_rate = evaluated["real"]["rules"][rule.name]["violation_rate"]
        synthetic_rate = evaluated["synthetic"]["rules"][rule.name]["violation_rate"]
        entry = {"status": rule.status, "real_violation_rate": real_rate, "synthetic_violation_rate": synthetic_rate,
                 "violation_gap": abs(synthetic_rate - real_rate) if None not in (real_rate, synthetic_rate) else None}
        if len(rule.columns) >= 2:
            fit_res, real_res, syn_res = (rule_residuals(f, rule, constraints) for f in (train, real, synthetic))
            spread = float(np.std(fit_res)) if len(fit_res) > 1 else 1.
            spread = spread if np.isfinite(spread) and spread > 1e-8 else 1.
            distance = float(wasserstein_distance(real_res, syn_res)) if len(real_res) and len(syn_res) else None
            entry.update(residual_w1=distance, residual_w1_fit_sd=None if distance is None else distance / spread)
        per_rule[rule.name] = entry
    hard = [r for r in constraints.rules if r.status in HARD_RULE_STATUSES]
    def any_violation(frame):
        if not hard:
            return None
        eligible = np.zeros(len(frame), bool)
        violated = np.zeros(len(frame), bool)
        for rule in hard:
            values = {c: pd.to_numeric(frame[c], errors="coerce").to_numpy() for c in rule.columns}
            raw = np.asarray(constraints.residual(rule, values), dtype=float)
            valid = frame[list(rule.columns)].notna().all(axis=1).to_numpy() & np.isfinite(raw)
            eligible |= valid
            violated |= valid & (np.maximum(raw - rule.tolerance, 0) / rule.scale > 1e-6)
        return float(violated.sum() / eligible.sum()) if eligible.any() else None
    gaps = [v["violation_gap"] for v in per_rule.values() if v["violation_gap"] is not None]
    scaled = [v["residual_w1_fit_sd"] for v in per_rule.values() if v.get("residual_w1_fit_sd") is not None]
    return {"rules": per_rule, "hard_rule_violation_rate": {"real": any_violation(real), "synthetic": any_violation(synthetic)},
            "mean_violation_gap": float(np.mean(gaps)) if gaps else None,
            "max_violation_gap": float(np.max(gaps)) if gaps else None,
            "mean_relational_residual_w1_fit_sd": float(np.mean(scaled)) if scaled else None}


def evaluate(train, real, synthetic, schema, constraints, seed=0, predictions=None):
    return {"fidelity": fidelity(train, real, synthetic, schema),
            "constraints": {"train": constraints.evaluate(train), "holdout": constraints.evaluate(real), "synthetic": constraints.evaluate(synthetic),
                            "realism": rule_realism(train, real, synthetic, constraints)},
            "utility": utility(train, real, synthetic, schema, seed, predictions), "c2st": c2st(real, synthetic, schema, seed),
            "privacy": privacy(train, real, synthetic, schema, seed)}
