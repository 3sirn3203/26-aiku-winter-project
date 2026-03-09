# 26-Winter Project: LLM 기반 Tabular Feature Engineering

## 소개
이 프로젝트는 LLM이 tabular 데이터의 전처리/피처 엔지니어링 코드를 반복적으로 개선하고, CV 결과를 기반으로 다음 시도를 자동으로 조정하는 실험 파이프라인입니다.

핵심 특징:
- Iteration 기반 자동 개선 루프 (Profile -> Hypothesis -> Implement -> Execute -> Diagnose)
- 3단계에서 E2E 스켈레톤 코드의 TODO를 LLM이 완성
- 4단계에서 생성된 `implement_pipeline.py`를 직접 실행해 성능 검증
- 실행 산출물(`runs/<run_id>/...`) 자동 저장 및 HTML/JSON 리포트 생성

## 방법론
파이프라인은 iteration마다 아래 단계를 수행합니다.

1. `Step 1: Profiling`  
   - 데이터 기초 통계/상관 특성/리스크를 분석해 프로파일 컨텍스트를 생성합니다.

2. `Step 2: Hypothesis`  
   - 프로파일 결과(선택적으로 웹 리서치 포함)를 바탕으로 전처리/피처 가설을 생성합니다.

3. `Step 3: Implement`  
   - 고정된 E2E 스켈레톤(`src/prompt/3_implement_e2e_skeleton.py`)을 프롬프트에 포함해,
     TODO 구간만 가설에 맞게 채운 전체 실행 스크립트(`implement_pipeline.py`)를 생성합니다.
   - 문법 체크 실패 시 재시도합니다.

4. `Step 4: Execute`  
   - 생성된 `implement_pipeline.py`를 실행해 교차검증을 수행합니다.
   - `mean_cv`, `std_cv`, `metric`, `objective_mean`를 포함한 결과 JSON을 저장합니다.

5. `Step 5: Diagnose`  
   - 실행 결과/로그를 분석해 실패 원인 및 다음 iteration 개선 피드백을 생성합니다.

6. `Final Report`  
   - 전체 iteration 결과를 취합해 `report.html`, `report.json`을 만듭니다.

## 환경 설정
1. Python 가상환경 생성/활성화

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. 패키지 설치

```bash
pip install -r requirements.txt
```

3. API Key 설정

```bash
cp .env.example .env
```

`.env` 파일에 아래 값을 설정합니다.

```bash
GEMINI_API_KEY=YOUR_API_KEY
```

## 사용 방법
### 1) 파이프라인 실행

```bash
python3 main.py --config config/dacon.json
```

다른 데이터셋 설정 예시:

```bash
python3 main.py --config config/kaggle.json
```

실행 후 산출물:
- `runs/<run_id>/config.json`
- `runs/<run_id>/iteration_<n>/profile/`
- `runs/<run_id>/iteration_<n>/hypothesis/`
- `runs/<run_id>/iteration_<n>/implement/implement_pipeline.py`
- `runs/<run_id>/iteration_<n>/execute/cv_result.json`
- `runs/<run_id>/iteration_<n>/diagnose/`
- `runs/<run_id>/report.html`, `runs/<run_id>/report.json`

### 2) 제출 파일 생성

```bash
python3 submission.py --config config/dacon.json --run_id <RUN_ID> --iteration <ITER>
```

동작 방식:
- 기본적으로 해당 iteration의 `implement_pipeline.py`를 사용합니다.
- 필요 시 직접 스크립트 지정 가능:

```bash
python3 submission.py --config config/dacon.json --run_id <RUN_ID> --iteration <ITER> --pipeline_script_path runs/<RUN_ID>/iteration_<ITER>/implement/implement_pipeline.py
```
