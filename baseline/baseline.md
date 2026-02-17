# Baseline Implementation TODO

## 1) Extend Config Schema
- [ ] Define `data` fields: `label_col`, `sample_submission_path`, `output_path`, `id_col`
- [ ] Define `model` fields: `eval_metric`, `problem_type`, `presets`, `time_limit`, `num_gpus`
- [ ] Define `validation` fields:
  - [ ] `holdout_frac`: holdout split ratio
  - [ ] `random_state`: split reproducibility
- [ ] Define `fit` fields for AutoGluon `fit()` hyperparameters

## 2) Implement Runner Script (`baseline.py`)
- [ ] Load `--config` and validate inputs
- [ ] Load train/test/sample-submission CSV files
- [ ] Initialize `TabularPredictor` from config
- [ ] Apply manual holdout split (`train_data`, `tuning_data`)
- [ ] Run `fit()` from config (`presets`, `time_limit`, `hyperparameters`, etc.)
- [ ] Predict on test and save in submission format
- [ ] Treat `output_path` as an output directory:
  - [ ] Raise an error if the directory already exists
  - [ ] Save `model/` and `submission.csv` inside it

## 3) Update Dataset Configs
- [ ] Complete missing fields in `baseline/config/dacon.json`
- [ ] Create and complete `baseline/config/kaggle.json`

## 4) Documentation (`baseline/Readme.md`)
- [ ] Add installation guide (`baseline/requirements.txt`)
- [ ] Add execution examples for dacon/kaggle
- [ ] Document config schema and example
- [ ] Document holdout-only validation flow

## 5) Validation
- [ ] Run once with dacon config
- [ ] Run once with kaggle config
- [ ] Confirm `submission.csv` is generated under `output_path`
