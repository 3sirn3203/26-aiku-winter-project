# LLM 기반 Tabular Data Analysis Agent

## 개요
이 프로젝트는 CSV/Excel 형태의 Tabular 데이터에 대해 **계획 → 코드 생성 → 실행 → 리뷰**를 반복 수행하는 자율 에이전트를 구현합니다. 실행 결과로 전처리/학습/추론 스크립트와 실행 요약 리포트를 생성합니다.

## 에이전트 역할 및 실행 흐름
### 역할
- **Planner**: 데이터 분석 계획과 평가 전략 수립
- **Coder**: 전처리/학습/추론 코드 생성
- **Reviewer**: 실행 결과를 검토하고 다음 반복 여부 결정

### 역할 간 상호작용 방식
각 역할은 직접 대화하지 않고, **공유 상태(`AgentState`)를 통해 간접적으로 상호작용**합니다.
- Planner가 생성한 `plan`이 state에 저장됨 → Coder가 해당 컨텍스트를 참고
- Coder가 생성한 코드가 `generated_sections`/`generated_code`로 저장됨 → Executor가 실행
- Executor가 기록한 `execution_result`/`last_run`을 Reviewer가 보고 다음 반복 여부 결정

### 실행 흐름 (LangGraph)
LangGraph 기반의 상태 그래프를 통해 다음 순서로 반복 실행됩니다.
1. `plan_step` → Planner LLM이 분석 계획 생성
2. `code_gen_step` → Coder LLM이 섹션별 코드 생성
3. `execute_step` → 생성된 `pipeline.py` 로컬 실행
4. `review_step` → Reviewer LLM이 다음 반복 여부 판단

`max_iters` 또는 stop 조건에 도달하면 종료됩니다.

## 사용 도구 및 프레임워크
### LLM
- **google-genai** (`google.genai`)
- 설정 위치: `configs/config.yaml`, `configs/agents.yaml`

### 오케스트레이션
- **LangGraph**: 상태 기반 반복 에이전트 흐름
- **LangChain**: 의존성으로 포함 (확장 가능)

### 실행 환경 (Sandbox)
- **Local Runner** (`src/sandbox/local_runner.py`)
  - 생성된 코드를 subprocess로 실행하며 timeout을 적용

### 유틸리티
- YAML 로더: `src/utils/config.py`
- 코드 추출: `src/utils/codegen.py`

## 생성되는 결과물 (Outputs)
### 코드 산출물
- `generated_*/scripts/pipeline.py`
  - 전처리/피처 엔지니어링 파이프라인
- `generated_*/scripts/train_advanced.py`
  - 고성능 모델 학습 스크립트
- `generated_*/scripts/inference.py`
  - 추론 스크립트

### 리포트
- `generated_*/reports/summary.md`
  - 실행 요약 (exit code, runtime, stdout/stderr)
  - `execute_step` 종료 시 기록됨

### 임시 파일
- `generated_*/scripts/tmp*.py`
  - 실행 시 생성되는 임시 스크립트

## 프로젝트 구조 (핵심 컴포넌트)
```
configs/
  agents.yaml         # 역할별 LLM 설정
  config.yaml         # 경로/에이전트/모델 설정
  prompts.yaml        # 프롬프트 템플릿

data_dacon/           # 예시 데이터

generated_dacon/
  scripts/            # 생성된 코드 파일
  reports/            # 실행 요약 리포트

src/
  agent/              # LangGraph 워크플로우 (state/nodes/graph)
  llm/                # Gemini 래퍼
  sandbox/            # 로컬 실행기
  utils/              # config/codegen 유틸

main.py               # 엔트리 포인트
```

## 실행 방법
```bash
python main.py \
  --input-file data_dacon/train.csv \
  --target-column completed \
  --problem-type classification \
  --max-iters 3 \
  --execute
```

`.env` 파일을 통해 API 키를 로드합니다 (python-dotenv 사용).

## 현재 동작 특성
- Coder는 **섹션별**로 LLM을 호출해 코드 잘림 문제를 완화합니다.
- 데이터 경로 오류 시 실행이 즉시 실패하며, 오류는 `summary.md`에 기록됩니다.
- 반복은 `max_iters` 기준으로 종료됩니다.

## 향후 개선 포인트
- 섹션 누락/잘림 시 재요청(리트라이) 로직 추가
- 생성 코드 정합성 검사 및 테스트 강화
- 리포트에 메트릭/아티팩트 정보 확장
