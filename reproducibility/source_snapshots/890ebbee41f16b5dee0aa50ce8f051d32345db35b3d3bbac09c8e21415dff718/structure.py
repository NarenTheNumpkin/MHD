"""Training-partition association graph. Association is not causation."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import normalized_mutual_info_score


def association_matrix(frame, numeric, min_overlap=10, bins=5):
    """Spearman magnitude for numeric pairs; binned NMI otherwise.

    Bins are local to this estimation sample. Pairwise deletion and overlap counts
    make missingness explicit. These estimator types are not calibrated onto a
    universal strength scale; bootstrap stability and graph ablations are required.
    """
    names = list(frame.columns)
    d = len(names)
    assoc, overlap = np.zeros((d, d)), np.zeros((d, d), dtype=int)
    num_set = set(numeric)
    for i, a in enumerate(names):
        for j in range(i + 1, d):
            b = names[j]
            pair = frame[[a, b]].dropna()
            n = len(pair)
            overlap[i, j] = overlap[j, i] = n
            if n < min_overlap or min(pair[a].nunique(), pair[b].nunique()) < 2:
                continue
            if a in num_set and b in num_set:
                value = abs(float(spearmanr(pair[a], pair[b]).statistic))
            else:
                labels = []
                for col in (a, b):
                    if col in num_set:
                        labels.append(pd.qcut(pair[col], q=min(bins, pair[col].nunique()), labels=False, duplicates="drop").to_numpy())
                    else:
                        labels.append(pair[col].astype(str).to_numpy())
                value = normalized_mutual_info_score(*labels)
            assoc[i, j] = assoc[j, i] = np.nan_to_num(value)
    return assoc, overlap


def dependency_graph(frame, preprocessor, seed=0, bootstraps=20, threshold=.15, stability=.6):
    features = preprocessor.schema.features
    working = frame[features].copy()
    for col in preprocessor.schema.numeric:
        working[f"{col}__observed"] = frame[col].notna().astype(str)
    working = working[preprocessor.token_names]
    full, overlap = association_matrix(working, preprocessor.schema.numeric)
    rng = np.random.default_rng(seed)
    votes = np.zeros_like(full)
    for _ in range(bootstraps):
        sample = working.iloc[rng.integers(0, len(working), len(working))]
        values, _ = association_matrix(sample, preprocessor.schema.numeric)
        votes += values >= threshold
    support = votes / max(bootstraps, 1)
    graph = np.where((full >= threshold) & (support >= stability), full, 0)
    # Missingness mask + own-value association can be an encoding artifact: exclude it.
    for i, col in enumerate(preprocessor.schema.numeric):
        k = preprocessor.token_names.index(f"{col}__observed")
        graph[i, k] = graph[k, i] = 0
    return graph.astype(np.float32), {"names": preprocessor.token_names, "association": full.tolist(),
        "overlap": overlap.tolist(), "bootstrap_support": support.tolist(),
        "bootstraps": bootstraps, "threshold": threshold, "stability": stability}
