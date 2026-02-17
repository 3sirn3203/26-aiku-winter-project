import inspect
import json
import os
from argparse import ArgumentParser

import pandas as pd
from autogluon.tabular import TabularPredictor


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def infer_label_col(train_df, sample_submission_df, id_col):
    submission_cols = [col for col in sample_submission_df.columns if col != id_col]
    if len(submission_cols) == 1:
        return submission_cols[0]
    if len(train_df.columns) >= 1:
        return train_df.columns[-1]
    raise ValueError("Unable to infer label column. Please set data.label_col in config.")


def split_holdout(train_df, holdout_frac, random_state):
    if not 0 < holdout_frac < 1:
        raise ValueError("validation.holdout_frac must be between 0 and 1.")
    if len(train_df) < 2:
        raise ValueError("Train data must have at least 2 rows for holdout split.")

    tuning_df = train_df.sample(frac=holdout_frac, random_state=random_state)
    train_only_df = train_df.drop(index=tuning_df.index)

    # Ensure both partitions are non-empty even on edge-case sizes.
    if tuning_df.empty or train_only_df.empty:
        tuning_size = int(round(len(train_df) * holdout_frac))
        tuning_size = max(1, min(len(train_df) - 1, tuning_size))
        shuffled = train_df.sample(frac=1.0, random_state=random_state)
        tuning_df = shuffled.iloc[:tuning_size]
        train_only_df = shuffled.iloc[tuning_size:]

    return train_only_df.reset_index(drop=True), tuning_df.reset_index(drop=True)


def build_fit_kwargs(config):
    model_cfg = config.get("model", {})
    fit_cfg = config.get("fit", {})

    fit_kwargs = {}
    presets = model_cfg.get("presets", "best_quality")
    if presets is not None:
        fit_kwargs["presets"] = presets

    time_limit = model_cfg.get("time_limit")
    if time_limit is not None:
        fit_kwargs["time_limit"] = time_limit

    num_gpus = model_cfg.get("num_gpus")
    if num_gpus is not None:
        fit_kwargs["num_gpus"] = num_gpus

    # Force holdout-only flow by default (no bagging/cv stack).
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
    filtered_kwargs = {}
    ignored_keys = []
    for key, value in fit_kwargs.items():
        if key in valid_fit_params:
            filtered_kwargs[key] = value
        else:
            ignored_keys.append(key)

    if ignored_keys:
        print(f"[WARN] Ignored unsupported fit keys: {ignored_keys}")

    return filtered_kwargs


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="baseline/config/dacon.json")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    config = read_json(args.config)

    data_cfg = config.get("data", {})
    train_path = data_cfg.get("train_path", "data/dacon/train.csv")
    test_path = data_cfg.get("test_path", "data/dacon/test.csv")
    sample_submission_path = data_cfg.get(
        "sample_submission_path",
        data_cfg.get("submission_path", "data/dacon/sample_submission.csv"),
    )
    output_path = data_cfg.get("output_path", "baseline/generated/dacon_ver1")
    id_col = data_cfg.get("id_col", "ID")
    validation_cfg = config.get("validation", {})
    holdout_frac = validation_cfg.get("holdout_frac", 0.2)
    random_state = validation_cfg.get("random_state", 42)

    if os.path.exists(output_path):
        raise FileExistsError(f"Output directory already exists: {output_path}")
    os.makedirs(output_path, exist_ok=False)
    model_path = os.path.join(output_path, "model")
    submission_path = os.path.join(output_path, "submission.csv")

    train_df = load_csv(train_path)
    test_df = load_csv(test_path)
    sample_submission_df = load_csv(sample_submission_path)

    label_col = data_cfg.get("label_col")
    if label_col is None:
        label_col = infer_label_col(train_df, sample_submission_df, id_col)
        print(f"[INFO] label_col inferred as '{label_col}'")

    if label_col not in train_df.columns:
        raise ValueError(f"label_col '{label_col}' not found in train columns")

    model_cfg = config.get("model", {})
    eval_metric = model_cfg.get("eval_metric")
    problem_type = model_cfg.get("problem_type")
    train_only_df, tuning_df = split_holdout(
        train_df=train_df,
        holdout_frac=holdout_frac,
        random_state=random_state,
    )

    fit_kwargs = build_fit_kwargs(config)

    predictor = TabularPredictor(
        label=label_col,
        eval_metric=eval_metric,
        problem_type=problem_type,
        path=model_path,
    )
    predictor.fit(train_data=train_only_df, tuning_data=tuning_df, **fit_kwargs)

    predictions = predictor.predict(test_df)

    submission_df = sample_submission_df.copy()
    if id_col in test_df.columns:
        submission_df[id_col] = test_df[id_col].values
    submission_df[label_col] = predictions.values

    submission_df.to_csv(submission_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Model saved to: {model_path}")
    print(f"[INFO] Submission saved to: {submission_path}")


if __name__ == "__main__":
    main()
