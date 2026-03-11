from __future__ import annotations

import argparse
import math
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TARGET_CANDIDATES = ("completed", "target", "label", "y")
NUMERIC_CORR_THRESHOLD = 0.7
CATEGORICAL_ASSOC_THRESHOLD = 0.5
MAX_TARGET_ASSOC_PRINT = 10
MAX_NUMERIC_PAIR_PRINT = 20
MAX_CATEGORICAL_PAIR_PRINT = 10
MAX_CAT_CARDINALITY_FOR_PAIR = 50
MAX_CAT_COLUMNS_FOR_PAIR = 18


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _is_id_like(col: str, series: pd.Series, n_rows: int) -> bool:
    lowered = col.lower()
    if lowered == "id" or lowered.endswith("_id"):
        return True
    if n_rows <= 0:
        return False
    nunique = int(series.nunique(dropna=False))
    return nunique >= max(2, int(n_rows * 0.98))


def _infer_target_col(df: pd.DataFrame) -> str:
    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate

    binary_like: List[str] = []
    for col in df.columns:
        nunique = int(df[col].nunique(dropna=True))
        if 2 <= nunique <= 5:
            binary_like.append(col)
    if binary_like:
        return binary_like[-1]
    return str(df.columns[-1])


def _safe_cramers_v(x: pd.Series, y: pd.Series) -> float:
    joined = pd.DataFrame({"x": x, "y": y}).dropna()
    if joined.empty:
        return float("nan")
    if joined["x"].nunique(dropna=True) < 2 or joined["y"].nunique(dropna=True) < 2:
        return float("nan")

    contingency = pd.crosstab(joined["x"], joined["y"])
    if contingency.empty:
        return float("nan")

    observed = contingency.to_numpy(dtype=float)
    n = float(observed.sum())
    if n <= 0:
        return float("nan")

    row_sum = observed.sum(axis=1, keepdims=True)
    col_sum = observed.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / n
    valid = expected > 0
    if not np.any(valid):
        return float("nan")

    chi2 = np.where(valid, (observed - expected) ** 2 / expected, 0.0).sum()
    phi2 = chi2 / n
    r, k = observed.shape
    denom = min(r - 1, k - 1)
    if denom <= 0:
        return float("nan")

    return float(math.sqrt(max(phi2 / denom, 0.0)))


def _build_multicollinear_groups(
    features: List[str],
    high_pairs: List[Tuple[str, str, float]],
) -> List[List[str]]:
    if not features:
        return []

    adjacency: Dict[str, set[str]] = {feature: set() for feature in features}
    for left, right, _ in high_pairs:
        if left in adjacency and right in adjacency:
            adjacency[left].add(right)
            adjacency[right].add(left)

    groups: List[List[str]] = []
    visited: set[str] = set()

    for start in features:
        if start in visited:
            continue
        stack = [start]
        component: List[str] = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    stack.append(neighbor)
        if len(component) > 1:
            groups.append(sorted(component))

    groups.sort(key=lambda cols: (-len(cols), cols))
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic correlation profiling script.")
    parser.add_argument("--train-path", type=str, required=True, help="Path to train CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.train_path, encoding="utf-8-sig")
    if df.empty:
        print("TARGET_ASSOCIATION_TOP")
        print("Dataset is empty.")
        print("\nNUMERIC_HIGH_CORR_PAIRS")
        print("Dataset is empty.")
        print("\nMULTICOLLINEAR_GROUPS")
        print("Dataset is empty.")
        print("\nCATEGORICAL_ASSOCIATION_TOP")
        print("Dataset is empty.")
        print("\nMETHOD_NOTES")
        print("Deterministic correlation stage executed on empty dataset.")
        return

    target_col = _infer_target_col(df)
    n_rows = int(len(df))

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for col in df.columns:
        if col == target_col:
            continue
        if _is_id_like(str(col), df[col], n_rows):
            continue
        if pd.api.types.is_bool_dtype(df[col].dtype):
            numeric_cols.append(str(col))
        elif pd.api.types.is_numeric_dtype(df[col].dtype):
            numeric_cols.append(str(col))
        else:
            categorical_cols.append(str(col))

    numeric_target_associations: List[Tuple[str, float]] = []
    target_numeric = _safe_numeric(df[target_col]) if target_col in df.columns else pd.Series(dtype=float)
    target_numeric_is_valid = (
        int(target_numeric.notna().sum()) >= 3
        and int(target_numeric.nunique(dropna=True)) >= 2
    )
    if target_numeric_is_valid:
        for col in numeric_cols:
            s = _safe_numeric(df[col])
            joined = pd.concat([s, target_numeric], axis=1).dropna()
            if len(joined) < 3:
                continue
            if joined.iloc[:, 0].nunique() < 2:
                continue
            corr = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
            if np.isfinite(corr):
                numeric_target_associations.append((col, abs(corr)))
    numeric_target_associations.sort(key=lambda row: row[1], reverse=True)

    target_categorical = df[target_col].astype("string") if target_col in df.columns else pd.Series(dtype="string")
    cat_assoc_candidates = categorical_cols + [
        col for col in numeric_cols if int(df[col].nunique(dropna=True)) <= 20
    ]
    seen_cat = set()
    cat_assoc_candidates = [c for c in cat_assoc_candidates if not (c in seen_cat or seen_cat.add(c))]

    categorical_target_associations: List[Tuple[str, float]] = []
    for col in cat_assoc_candidates:
        score = _safe_cramers_v(df[col].astype("string"), target_categorical)
        if np.isfinite(score):
            categorical_target_associations.append((col, float(score)))
    categorical_target_associations.sort(key=lambda row: row[1], reverse=True)

    numeric_df = pd.DataFrame(index=df.index)
    for col in numeric_cols:
        s = _safe_numeric(df[col])
        if int(s.notna().sum()) < 3:
            continue
        if int(s.nunique(dropna=True)) < 2:
            continue
        numeric_df[col] = s

    high_numeric_pairs: List[Tuple[str, str, float]] = []
    if numeric_df.shape[1] >= 2:
        corr_matrix = numeric_df.corr().abs()
        columns = list(corr_matrix.columns)
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                score = float(corr_matrix.iloc[i, j])
                if np.isfinite(score) and score >= NUMERIC_CORR_THRESHOLD:
                    high_numeric_pairs.append((columns[i], columns[j], score))
    high_numeric_pairs.sort(key=lambda row: row[2], reverse=True)

    multicollinear_groups = _build_multicollinear_groups(
        features=list(numeric_df.columns),
        high_pairs=high_numeric_pairs,
    )

    candidate_for_pairs = [
        col
        for col in categorical_cols
        if 2 <= int(df[col].nunique(dropna=True)) <= MAX_CAT_CARDINALITY_FOR_PAIR
    ]
    score_map = {col: score for col, score in categorical_target_associations}
    candidate_for_pairs.sort(key=lambda col: score_map.get(col, 0.0), reverse=True)
    candidate_for_pairs = candidate_for_pairs[:MAX_CAT_COLUMNS_FOR_PAIR]

    categorical_pairs: List[Tuple[str, str, float]] = []
    for left, right in combinations(candidate_for_pairs, 2):
        score = _safe_cramers_v(df[left].astype("string"), df[right].astype("string"))
        if np.isfinite(score) and score >= CATEGORICAL_ASSOC_THRESHOLD:
            categorical_pairs.append((left, right, float(score)))
    categorical_pairs.sort(key=lambda row: row[2], reverse=True)

    print("TARGET_ASSOCIATION_TOP")
    print("----------------------")
    if numeric_target_associations:
        print("Numeric (|Pearson| with target):")
        for col, score in numeric_target_associations[:MAX_TARGET_ASSOC_PRINT]:
            print(f"- {col}: {score:.4f}")
    else:
        print("Numeric (|Pearson| with target): none")
    if categorical_target_associations:
        print("Categorical (Cramer's V with target):")
        for col, score in categorical_target_associations[:MAX_TARGET_ASSOC_PRINT]:
            print(f"- {col}: {score:.4f}")
    else:
        print("Categorical (Cramer's V with target): none")

    print("\nNUMERIC_HIGH_CORR_PAIRS")
    print("-----------------------")
    if high_numeric_pairs:
        for left, right, score in high_numeric_pairs[:MAX_NUMERIC_PAIR_PRINT]:
            print(f"- {left} <-> {right}: {score:.4f}")
    else:
        print(f"No numeric pairs above threshold ({NUMERIC_CORR_THRESHOLD:.2f}).")

    print("\nMULTICOLLINEAR_GROUPS")
    print("---------------------")
    if multicollinear_groups:
        for idx, group in enumerate(multicollinear_groups, start=1):
            print(f"- Group {idx}: {', '.join(group)}")
    else:
        print("No multicollinear groups detected.")

    print("\nCATEGORICAL_ASSOCIATION_TOP")
    print("---------------------------")
    if categorical_pairs:
        for left, right, score in categorical_pairs[:MAX_CATEGORICAL_PAIR_PRINT]:
            print(f"- {left} <-> {right}: {score:.4f}")
    else:
        print(f"No categorical pairs above threshold ({CATEGORICAL_ASSOC_THRESHOLD:.2f}).")

    print("\nMETHOD_NOTES")
    print("------------")
    print(f"- target_column: {target_col}")
    print("- deterministic correlation script (no LLM code generation).")
    print(f"- numeric_pair_threshold: {NUMERIC_CORR_THRESHOLD:.2f}")
    print(f"- categorical_pair_threshold: {CATEGORICAL_ASSOC_THRESHOLD:.2f}")
    print(f"- id_like_columns_excluded: yes (name/id-like high uniqueness rule)")
    print(f"- numeric_features_used: {numeric_df.shape[1]}")
    print(f"- categorical_features_used: {len(categorical_cols)}")


if __name__ == "__main__":
    main()
