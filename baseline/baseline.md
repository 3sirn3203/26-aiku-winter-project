# Baseline 구현 TODO

## 1) Config 스키마 확장
- [ ] `data` 섹션에 `label_col`, `sample_submission_path`, `output_path`, `id_col` 명시
- [ ] `model` 섹션에 `eval_metric`, `problem_type`, `presets`, `time_limit` 추가
- [ ] `validation` 섹션 추가
  - [ ] `method`: `holdout` 또는 `cv`
  - [ ] `holdout_frac`: holdout 비율
  - [ ] `num_bag_folds`: CV(=bagging folds) 수
  - [ ] `num_bag_sets`, `num_stack_levels` (선택)
- [ ] `fit` 섹션 추가: AutoGluon `fit()`에 전달할 하이퍼파라미터 관리

## 2) 실행 스크립트 구현 (`baseline.py`)
- [ ] `--config` 인자 로딩 및 유효성 검증
- [ ] train/test/sample_submission CSV 로드
- [ ] config 기반으로 `TabularPredictor` 초기화
- [ ] validation 설정 반영
  - [ ] `holdout`: `holdout_frac` 기반 학습
  - [ ] `cv`: `num_bag_folds` 기반 학습
- [ ] config 기반으로 `fit()` 실행 (`presets`, `time_limit`, `hyperparameters` 등)
- [ ] test 예측 후 sample submission 포맷에 맞춰 저장
- [ ] `output_path` 폴더 자동 생성

## 3) Config 파일 보완
- [ ] `baseline/config/dacon.json`에 누락 필드 보완
- [ ] `baseline/config/kaggle.json` 신규 작성

## 4) 문서화 (`baseline/Readme.md`)
- [ ] 설치 방법(`baseline/requirements.txt`)
- [ ] 실행 방법(dacon/kaggle 각각)
- [ ] config 필드 설명 및 예시
- [ ] holdout/cv 전환 방법 설명

## 5) 검증
- [ ] dacon config로 1회 실행 확인
- [ ] kaggle config로 1회 실행 확인
- [ ] `output_path`에 submission 파일 생성 확인
