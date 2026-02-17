# Baseline (AutoGluon)

`baseline` 디렉토리는 LLM Agent 없이 AutoGluon 기반의 탭уляр 예측 baseline을 생성합니다.  
데이터셋은 config 파일로 분리하여 `dacon`, `kaggle`을 각각 독립 실행합니다.

## 1. 설치

프로젝트 루트에서:

```bash
pip install -r baseline/requirements.txt
```

## 2. 실행

### Dacon

```bash
python baseline/baseline_autogluon.py --config baseline/config/dacon.json
```

결과 파일:
- `baseline/generated/submission_dacon.csv`

### Kaggle

```bash
python baseline/baseline_autogluon.py --config baseline/config/kaggle.json
```

결과 파일:
- `baseline/generated/submission_kaggle.csv`

## 3. Config 구조

예시(`baseline/config/dacon.json`):

```json
{
  "data": {
    "train_path": "data/dacon/train.csv",
    "test_path": "data/dacon/test.csv",
    "sample_submission_path": "data/dacon/sample_submission.csv",
    "output_path": "baseline/generated/submission_dacon.csv",
    "id_col": "ID",
    "label_col": "completed"
  },
  "model": {
    "eval_metric": "f1",
    "problem_type": "binary",
    "presets": "best_quality",
    "time_limit": 1800
  },
  "validation": {
    "method": "holdout",
    "holdout_frac": 0.2,
    "num_bag_folds": 5,
    "num_bag_sets": 1,
    "num_stack_levels": 0
  },
  "fit": {
    "hyperparameters": "default"
  }
}
```

## 4. Holdout / CV 전환

- `validation.method = "holdout"`
  - `holdout_frac` 사용
- `validation.method = "cv"`
  - AutoGluon bagging 기반 CV 사용
  - `num_bag_folds`, `num_bag_sets`, `num_stack_levels` 사용

## 5. 하이퍼파라미터 관리

- 공통 학습 설정: `model` 섹션 (`presets`, `time_limit`, `eval_metric`, `problem_type`)
- 세부 `fit()` 인자: `fit` 섹션
  - 예: `hyperparameters`, `included_model_types`, `excluded_model_types` 등
  - `TabularPredictor.fit()`에서 지원하지 않는 키는 경고 후 무시됩니다.
