# Baseline (AutoGluon)

`baseline` 디렉토리는 LLM Agent 없이 AutoGluon 기반의 tabular 예측 baseline을 생성합니다.  
데이터셋은 config 파일로 분리하여 `dacon`, `kaggle`을 각각 독립 실행합니다.

## 1. 설치

프로젝트 루트에서:

```bash
pip install -r baseline/requirements.txt
```

## 2. 실행

### Dacon

```bash
python baseline/baseline.py --config baseline/config/dacon.json
```

결과 파일:
- `baseline/generated/.dacon/submission_YYYYMMDD_HHMMSS.csv`

### Kaggle

```bash
python baseline/baseline.py --config baseline/config/kaggle.json
```

결과 파일:
- `baseline/generated/.kaggle/submission_YYYYMMDD_HHMMSS.csv`

## 3. Config 구조

예시(`baseline/config/dacon.json`):

```json
{
  "data": {
    "train_path": "data/dacon/train.csv",
    "test_path": "data/dacon/test.csv",
    "sample_submission_path": "data/dacon/sample_submission.csv",
    "output_path": "baseline/generated",
    "id_col": "ID",
    "label_col": "completed"
  },
  "output": {
    "save_model": false,
    "dataset_dir": "dacon",
    "hidden_dataset_dir": true,
    "submission_prefix": "submission"
  },
  "model": {
    "eval_metric": "f1",
    "problem_type": "binary",
    "presets": "best_quality",
    "time_limit": 1800,
    "num_gpus": 0
  },
  "validation": {
    "holdout_frac": 0.2,
    "random_state": 42
  },
  "fit": {
    "hyperparameters": "default",
    "fit_strategy": "sequential"
  }
}
```

## 4. Holdout 검증

- 현재 baseline은 holdout만 사용합니다.
- `validation.holdout_frac` 비율로 train을 직접 분할하여
  - `train_data`
  - `tuning_data`
  로 `fit()`에 전달합니다.
- `validation.random_state`로 split 재현성을 제어합니다.

## 5. 하이퍼파라미터 관리

- 공통 학습 설정: `model` 섹션 (`presets`, `time_limit`, `eval_metric`, `problem_type`, `num_gpus`)
- 세부 `fit()` 인자: `fit` 섹션
  - 예: `hyperparameters`, `included_model_types`, `excluded_model_types` 등
  - `TabularPredictor.fit()`에서 지원하지 않는 키는 경고 후 무시됩니다.
- GPU 사용 시 `model.num_gpus`를 `1` 이상으로 설정합니다.
- `output.save_model=false`이면 학습 후 모델 디렉토리를 삭제하고 submission 파일만 남깁니다.
- `output_path`는 파일이 아니라 실행 결과 디렉토리입니다.
  - `output.dataset_dir`(예: `dacon`, `kaggle`) 하위 디렉토리를 자동으로 생성합니다.
  - `output.hidden_dataset_dir=true`이면 `.dacon`, `.kaggle`처럼 숨김 디렉토리로 저장됩니다.
  - 파일명은 `submission_타임스탬프.csv` 형식으로 누적 저장됩니다.
