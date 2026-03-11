from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

try:
    from category_encoders import TargetEncoder
except Exception:  # noqa: BLE001
    TargetEncoder = None

from src.modules.validator import run_cross_validation

SAFE_FEATURE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


def _safe_name(name: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        return "feature"
    return text


def _dedupe_names(names: Iterable[Any]) -> List[str]:
    out: List[str] = []
    used: set[str] = set()
    for idx, raw in enumerate(names):
        base = _safe_name(raw)
        if not base:
            base = f"feature_{idx}"
        candidate = base
        suffix = 1
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        out.append(candidate)
    return out


def _ensure_dataframe(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return pd.DataFrame(value)


class GeneratedPreprocessor:
    def fit_preprocessor(self, train_df: pd.DataFrame, label_col: str, config: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: implement preprocessing state fitting.
        # Must only use train_df.
        # Must preserve label column handling.
        raise NotImplementedError("TODO: fit_preprocessor")

    def transform_preprocessor(self, df: pd.DataFrame, prep_state: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
        # TODO: implement preprocessing transform.
        # If label_col exists in input, preserve it.
        # Keep stable output schema.
        raise NotImplementedError("TODO: transform_preprocessor")


class GeneratedFeatureEngineering:
    FEATURE_BLOCKS = {
        "generated": {
            "description": "Generated feature blocks from hypotheses",
            "enabled_by_default": True,
        }
    }

    def fit_feature_engineering(
        self,
        train_df: pd.DataFrame,
        label_col: str,
        config: Dict[str, Any],
        enabled_blocks: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        # TODO: implement feature engineering state fitting.
        # Must only use train_df and exclude label from feature schema.
        raise NotImplementedError("TODO: fit_feature_engineering")

    def transform_feature_engineering(self, df: pd.DataFrame, fe_state: Dict[str, Any], config: Dict[str, Any]) -> pd.DataFrame:
        # TODO: implement feature engineering transform.
        # Output must not contain label column.
        # Keep stable/unique/safe feature names.
        raise NotImplementedError("TODO: transform_feature_engineering")

    def feature_registry_from_state(self, fe_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        cols = [str(col) for col in fe_state.get("feature_cols", [])]
        return [{"feature": col, "block": "generated"} for col in cols]


def _load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _parse_enabled_blocks(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None or str(raw).strip() == "":
        return None
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generated E2E pipeline script")
    parser.add_argument("--config", type=str, default="config/dacon.json")
    parser.add_argument("--train-path", type=str, default=None)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--enabled-blocks", type=str, default=None)
    args = parser.parse_args()

    config = _load_config(args.config)
    data_cfg = config.get("data", {}) if isinstance(config, dict) else {}
    train_path = str(args.train_path or data_cfg.get("train_path", "data/dacon/train.csv"))
    train_df = pd.read_csv(train_path, encoding="utf-8-sig")

    preprocessor_module = GeneratedPreprocessor()
    feature_module = GeneratedFeatureEngineering()
    enabled_blocks = _parse_enabled_blocks(args.enabled_blocks)

    cv_result = run_cross_validation(
        config=config,
        train_df=train_df,
        preprocessor_module=preprocessor_module,
        feature_module=feature_module,
        enabled_blocks=enabled_blocks,
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(cv_result, file, ensure_ascii=False, indent=2)

    print(f"Metric: {cv_result.get('metric')}")
    print(f"Mean CV: {float(cv_result.get('mean_cv', 0.0)):.6f}")
    print(f"Std CV : {float(cv_result.get('std_cv', 0.0)):.6f}")
    print(f"Saved CV result to: {args.output_json}")


if __name__ == "__main__":
    main()
