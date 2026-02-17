from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.modules.validator import (
    fit_full_and_predict,
    load_feature_engineering_module,
    load_preprocessor_module,
    run_cross_validation,
)
from src.utils import write_json


def _parse_enabled_blocks(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None or raw.strip() == "":
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def _load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation harness (preprocessor + feature engineering)")
    parser.add_argument("--config", type=str, default="config/dacon.json")
    parser.add_argument("--preprocessor-path", type=str, default=None)
    parser.add_argument("--feature-engineering-path", type=str, default=None)
    parser.add_argument("--cv-type", type=str, default=None, choices=["auto", "stratified", "kfold"])
    parser.add_argument("--n-splits", type=int, default=None)
    parser.add_argument("--model-type", type=str, default=None, choices=["lightgbm", "xgboost", "random_forest"])
    parser.add_argument("--metric", type=str, default=None)
    parser.add_argument("--enabled-blocks", type=str, default=None, help="Comma-separated feature block names.")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--predict-test", action="store_true")
    parser.add_argument("--submission-out", type=str, default=None)
    return parser.parse_args()


def _apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    modeling = config.setdefault("modeling", {})
    validation = modeling.setdefault("validation", {})
    model = modeling.setdefault("model", {})

    if args.cv_type is not None:
        validation["cv_type"] = args.cv_type
    if args.n_splits is not None:
        validation["n_splits"] = int(args.n_splits)
    if args.model_type is not None:
        model["type"] = args.model_type
    if args.metric is not None:
        modeling["metric"] = args.metric
    return config


def main() -> None:
    args = parse_args()
    config = _apply_overrides(_load_config(args.config), args)

    data_cfg = config.get("data", {})
    train_path = str(data_cfg.get("train_path", "data/dacon/train.csv"))
    test_path = str(data_cfg.get("test_path", "data/dacon/test.csv"))
    submission_path = str(data_cfg.get("submission_path", "data/dacon/sample_submission.csv"))

    preprocessor_module = load_preprocessor_module(args.preprocessor_path)
    feature_module = load_feature_engineering_module(args.feature_engineering_path)
    enabled_blocks = _parse_enabled_blocks(args.enabled_blocks)

    train_df = pd.read_csv(train_path, encoding="utf-8-sig")
    cv_result = run_cross_validation(
        config=config,
        train_df=train_df,
        preprocessor_module=preprocessor_module,
        feature_module=feature_module,
        enabled_blocks=enabled_blocks,
    )

    print(f"Metric: {cv_result['metric']}")
    print(f"CV Type: {cv_result['cv_type_used']} ({cv_result['n_splits']} folds)")
    print(f"Model Used: {cv_result['model_type_used']}")
    print(f"Mean CV: {cv_result['mean_cv']:.6f}")
    print(f"Std CV : {cv_result['std_cv']:.6f}")
    if args.preprocessor_path:
        print(f"Preprocessor: {args.preprocessor_path}")
    if args.feature_engineering_path:
        print(f"Feature Engineering: {args.feature_engineering_path}")

    if args.output_json:
        write_json(args.output_json, cv_result)
        print(f"Saved CV result to: {args.output_json}")

    if args.predict_test:
        test_df = pd.read_csv(test_path, encoding="utf-8-sig")
        sample_submission_df = pd.read_csv(submission_path, encoding="utf-8-sig")
        full_result = fit_full_and_predict(
            config=config,
            train_df=train_df,
            test_df=test_df,
            sample_submission_df=sample_submission_df,
            preprocessor_module=preprocessor_module,
            feature_module=feature_module,
            enabled_blocks=enabled_blocks,
        )

        submission_out = args.submission_out
        if submission_out is None:
            submission_out = str(Path("runs") / "validation_submission.csv")
        Path(submission_out).parent.mkdir(parents=True, exist_ok=True)
        full_result["submission_df"].to_csv(submission_out, index=False, encoding="utf-8-sig")
        print(f"Saved submission to: {submission_out}")


if __name__ == "__main__":
    main()
