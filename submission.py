import importlib.util
import inspect
import json
import os
import re
from argparse import ArgumentParser
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from autogluon.tabular import TabularPredictor


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def infer_label_col(train_df: pd.DataFrame, sample_submission_df: pd.DataFrame, id_col: str) -> str:
    submission_cols = [col for col in sample_submission_df.columns if col != id_col]
    if len(submission_cols) == 1:
        return submission_cols[0]
    if len(train_df.columns) >= 1:
        return str(train_df.columns[-1])
    raise ValueError("Unable to infer label column. Please set submission.data.label_col in config.")


def split_holdout(train_df: pd.DataFrame, holdout_frac: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < holdout_frac < 1:
        raise ValueError("submission.validation.holdout_frac must be between 0 and 1.")
    if len(train_df) < 2:
        raise ValueError("Train data must have at least 2 rows for holdout split.")

    tuning_df = train_df.sample(frac=holdout_frac, random_state=random_state)
    train_only_df = train_df.drop(index=tuning_df.index)

    if tuning_df.empty or train_only_df.empty:
        tuning_size = int(round(len(train_df) * holdout_frac))
        tuning_size = max(1, min(len(train_df) - 1, tuning_size))
        shuffled = train_df.sample(frac=1.0, random_state=random_state)
        tuning_df = shuffled.iloc[:tuning_size]
        train_only_df = shuffled.iloc[tuning_size:]

    return train_only_df.reset_index(drop=True), tuning_df.reset_index(drop=True)


def build_fit_kwargs(model_cfg: Dict[str, Any], fit_cfg: Dict[str, Any]) -> Dict[str, Any]:
    fit_kwargs: Dict[str, Any] = {}

    presets = model_cfg.get("presets", "best_quality")
    if presets is not None:
        fit_kwargs["presets"] = presets

    time_limit = model_cfg.get("time_limit")
    if time_limit is not None:
        fit_kwargs["time_limit"] = time_limit

    num_gpus = model_cfg.get("num_gpus")
    if num_gpus is not None:
        fit_kwargs["num_gpus"] = num_gpus

    fit_kwargs["num_bag_folds"] = 0
    fit_kwargs["num_stack_levels"] = 0
    fit_kwargs.update(fit_cfg)

    fit_signature = inspect.signature(TabularPredictor.fit)
    accepts_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in fit_signature.parameters.values()
    )
    if accepts_var_kwargs:
        return fit_kwargs

    valid_fit_params = set(fit_signature.parameters.keys())
    filtered_kwargs: Dict[str, Any] = {}
    ignored_keys: List[str] = []
    for key, value in fit_kwargs.items():
        if key in valid_fit_params:
            filtered_kwargs[key] = value
        else:
            ignored_keys.append(key)

    if ignored_keys:
        print(f"[WARN] Ignored unsupported fit keys: {ignored_keys}")

    return filtered_kwargs


def load_module_from_path(module_path: str, module_name: str) -> ModuleType:
    path = Path(module_path)
    if not path.exists():
        raise FileNotFoundError(f"Module path not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_module_functions(module: Any, required_functions: List[str], module_label: str) -> None:
    missing = [name for name in required_functions if not hasattr(module, name)]
    if missing:
        raise ValueError(f"{module_label} module missing required function(s): {missing}")


def ensure_dataframe(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, pd.Series):
        return value.to_frame()
    return pd.DataFrame(value)


def deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col) for col in out.columns]
    return out.loc[:, ~out.columns.duplicated()].copy()


def sanitize_feature_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", str(name))


def sanitize_feature_columns(
    train_df: pd.DataFrame,
    other_dfs: List[pd.DataFrame],
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    sanitized = [sanitize_feature_name(col) for col in train_df.columns]
    seen: Dict[str, int] = {}
    final_cols: List[str] = []
    for col in sanitized:
        if col not in seen:
            seen[col] = 0
            final_cols.append(col)
        else:
            seen[col] += 1
            final_cols.append(f"{col}_{seen[col]}")

    out_train = train_df.copy()
    out_train.columns = final_cols

    out_others: List[pd.DataFrame] = []
    for other in other_dfs:
        out = other.copy()
        out.columns = final_cols
        out_others.append(out)
    return out_train, out_others


def parse_enabled_blocks(raw: Any) -> Optional[List[str]]:
    if raw is None:
        return None
    if isinstance(raw, list):
        blocks = [str(item).strip() for item in raw if str(item).strip()]
        return blocks if blocks else None
    text = str(raw).strip()
    if text == "":
        return None
    return [item.strip() for item in text.split(",") if item.strip()]


def build_runtime_config(config: Dict[str, Any], submission_data_cfg: Dict[str, Any], label_col: str, enabled_blocks: Optional[Iterable[str]]) -> Dict[str, Any]:
    runtime = dict(config)
    runtime["label_col"] = label_col
    data_cfg = dict(runtime.get("data", {}) or {})
    data_cfg.update(submission_data_cfg)
    data_cfg["label_col"] = label_col
    runtime["data"] = data_cfg
    if enabled_blocks is not None:
        runtime["enabled_blocks"] = [str(block) for block in enabled_blocks]
    return runtime


def resolve_module_paths(
    run_id: str,
    iteration: int,
    submission_cfg: Dict[str, Any],
    preprocessor_override: Optional[str],
    feature_override: Optional[str],
) -> Tuple[str, str]:
    impl_dir = Path("runs") / run_id / f"iteration_{iteration}" / "implement"

    preprocessor_path = preprocessor_override or submission_cfg.get("preprocessor_path")
    if not preprocessor_path:
        preprocessor_path = str(impl_dir / "preprocessor.py")

    feature_path = feature_override or submission_cfg.get("feature_engineering_path")
    if not feature_path:
        feature_path = str(impl_dir / "feature_engineering.py")
    if not os.path.exists(feature_path):
        fallback = submission_cfg.get("feature_generator_path")
        if fallback:
            feature_path = str(fallback)

    return str(preprocessor_path), str(feature_path)


def apply_generated_feature_pipeline(
    config: Dict[str, Any],
    submission_data_cfg: Dict[str, Any],
    submission_cfg: Dict[str, Any],
    label_col: str,
    train_only_df: pd.DataFrame,
    tuning_df: pd.DataFrame,
    test_df: pd.DataFrame,
    preprocessor_module: Any,
    feature_module: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    enabled_blocks = parse_enabled_blocks(submission_cfg.get("enabled_blocks"))
    runtime_config = build_runtime_config(
        config=config,
        submission_data_cfg=submission_data_cfg,
        label_col=label_col,
        enabled_blocks=enabled_blocks,
    )

    prep_state = preprocessor_module.fit_preprocessor(train_only_df.copy(), label_col, runtime_config)
    train_pre = ensure_dataframe(preprocessor_module.transform_preprocessor(train_only_df.copy(), prep_state, runtime_config))
    tuning_pre = ensure_dataframe(preprocessor_module.transform_preprocessor(tuning_df.copy(), prep_state, runtime_config))
    test_pre = ensure_dataframe(preprocessor_module.transform_preprocessor(test_df.copy(), prep_state, runtime_config))

    if label_col not in train_pre.columns:
        raise ValueError(f"train preprocessor output must contain label column '{label_col}'")
    if label_col not in tuning_pre.columns:
        raise ValueError(f"tuning preprocessor output must contain label column '{label_col}'")

    fe_state = feature_module.fit_feature_engineering(
        train_df=train_pre.copy(),
        label_col=label_col,
        config=runtime_config,
        enabled_blocks=enabled_blocks,
    )

    x_train = ensure_dataframe(feature_module.transform_feature_engineering(train_pre.copy(), fe_state, runtime_config))
    x_tuning = ensure_dataframe(feature_module.transform_feature_engineering(tuning_pre.copy(), fe_state, runtime_config))
    x_test = ensure_dataframe(feature_module.transform_feature_engineering(test_pre.copy(), fe_state, runtime_config))

    x_train = deduplicate_columns(x_train)
    x_tuning = deduplicate_columns(x_tuning)
    x_test = deduplicate_columns(x_test)

    if label_col in x_train.columns:
        x_train = x_train.drop(columns=[label_col])
    if label_col in x_tuning.columns:
        x_tuning = x_tuning.drop(columns=[label_col])
    if label_col in x_test.columns:
        x_test = x_test.drop(columns=[label_col])

    x_tuning = x_tuning.reindex(columns=x_train.columns)
    x_test = x_test.reindex(columns=x_train.columns)

    if bool(submission_cfg.get("sanitize_feature_names", True)):
        x_train, [x_tuning, x_test] = sanitize_feature_columns(x_train, [x_tuning, x_test])

    x_train = x_train.fillna(0)
    x_tuning = x_tuning.fillna(0)
    x_test = x_test.fillna(0)

    train_ag = x_train.copy()
    train_ag[label_col] = train_pre[label_col].values
    tuning_ag = x_tuning.copy()
    tuning_ag[label_col] = tuning_pre[label_col].values
    return train_ag, tuning_ag, x_test


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config/dacon.json")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--preprocessor_path", type=str, default=None)
    parser.add_argument("--feature_path", type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    config = read_json(args.config)
    submission_cfg = dict(config.get("submission", {}) or {})

    run_id = args.run_id or submission_cfg.get("run_id")
    if not run_id:
        raise ValueError("run_id is required. Set --run_id or submission.run_id in config.")
    iteration = args.iteration if args.iteration is not None else int(submission_cfg.get("iteration", 1))

    submission_data_cfg = dict(submission_cfg.get("data", {}) or {})
    submission_model_cfg = dict(submission_cfg.get("model", {}) or {})
    submission_validation_cfg = dict(submission_cfg.get("validation", {}) or {})
    submission_fit_cfg = dict(submission_cfg.get("fit", {}) or {})

    train_path = submission_data_cfg.get("train_path", "data/dacon/train.csv")
    test_path = submission_data_cfg.get("test_path", "data/dacon/test.csv")
    sample_submission_path = submission_data_cfg.get(
        "sample_submission_path",
        submission_data_cfg.get("submission_path", "data/dacon/sample_submission.csv"),
    )
    id_col = str(submission_data_cfg.get("id_col", "ID"))

    output_path = args.output_path or submission_data_cfg.get("output_path") or submission_cfg.get("output_path")
    if not output_path:
        output_path = "submissions/dacon/submission.csv"
    output_path = str(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    model_path = submission_cfg.get("model_path")
    if not model_path:
        model_path = os.path.join(os.path.dirname(output_path) or ".", f"model_{run_id}_iter_{iteration}")
    model_path = str(model_path)

    if os.path.exists(model_path):
        raise FileExistsError(f"Model output directory already exists: {model_path}")

    allow_overwrite_output = bool(submission_cfg.get("allow_overwrite_output", True))
    if os.path.exists(output_path) and not allow_overwrite_output:
        raise FileExistsError(f"Submission file already exists: {output_path}")

    preprocessor_path, feature_path = resolve_module_paths(
        run_id=run_id,
        iteration=iteration,
        submission_cfg=submission_cfg,
        preprocessor_override=args.preprocessor_path,
        feature_override=args.feature_path,
    )
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor module not found: {preprocessor_path}")
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Feature engineering module not found: {feature_path}")

    preprocessor_module = load_module_from_path(preprocessor_path, "submission_preprocessor_module")
    feature_module = load_module_from_path(feature_path, "submission_feature_engineering_module")
    assert_module_functions(preprocessor_module, ["fit_preprocessor", "transform_preprocessor"], "preprocessor")
    assert_module_functions(feature_module, ["fit_feature_engineering", "transform_feature_engineering"], "feature_engineering")

    train_df = load_csv(train_path)
    test_df = load_csv(test_path)
    sample_submission_df = load_csv(sample_submission_path)

    label_col = submission_data_cfg.get("label_col")
    if label_col is None:
        label_col = infer_label_col(train_df, sample_submission_df, id_col)
        print(f"[INFO] label_col inferred as '{label_col}'")
    label_col = str(label_col)
    if label_col not in train_df.columns:
        raise ValueError(f"label_col '{label_col}' not found in train columns")

    holdout_frac = float(submission_validation_cfg.get("holdout_frac", 0.2))
    random_state = int(submission_validation_cfg.get("random_state", 42))
    train_only_df, tuning_df = split_holdout(train_df=train_df, holdout_frac=holdout_frac, random_state=random_state)

    train_ag, tuning_ag, test_ag = apply_generated_feature_pipeline(
        config=config,
        submission_data_cfg=submission_data_cfg,
        submission_cfg=submission_cfg,
        label_col=label_col,
        train_only_df=train_only_df,
        tuning_df=tuning_df,
        test_df=test_df,
        preprocessor_module=preprocessor_module,
        feature_module=feature_module,
    )

    fit_kwargs = build_fit_kwargs(model_cfg=submission_model_cfg, fit_cfg=submission_fit_cfg)
    predictor = TabularPredictor(
        label=label_col,
        eval_metric=submission_model_cfg.get("eval_metric"),
        problem_type=submission_model_cfg.get("problem_type"),
        path=model_path,
    )
    predictor.fit(train_data=train_ag, tuning_data=tuning_ag, **fit_kwargs)

    predictions = predictor.predict(test_ag)
    submission_df = sample_submission_df.copy()
    if id_col in test_df.columns and id_col in submission_df.columns:
        submission_df[id_col] = test_df[id_col].values
    submission_df[label_col] = predictions.values
    submission_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"[INFO] run_id: {run_id}, iteration: {iteration}")
    print(f"[INFO] preprocessor: {preprocessor_path}")
    print(f"[INFO] feature_engineering: {feature_path}")
    print(f"[INFO] model saved to: {model_path}")
    print(f"[INFO] submission saved to: {output_path}")


if __name__ == "__main__":
    main()
