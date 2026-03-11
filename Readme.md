# 데이터 분석 agent 만들어서 공모전 Ssalmuck하기

📢 2026년 겨울학기 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다

## 소개

(프로젝트를 소개해주세요)

## 방법론

(문제를 정의하고 이를 해결한 방법을 가독성 있게 설명해주세요)

## 환경 설정

1. Python 가상환경 생성 및 의존성 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. 환경변수 설정 (`GEMINI_API_KEY`)

```bash
cp .env.example .env
```

`.env`에 아래 값을 설정합니다.

```bash
GEMINI_API_KEY=YOUR_API_KEY
```

3. 데이터/설정 파일 확인
- Dacon: `config/dacon.json`, `data/dacon/*`
- Kaggle: `config/kaggle.json`, `data/kaggle/*`

## 사용 방법

1. 전체 파이프라인 실행 + submission 생성

```bash
# 기본: config/dacon.json
./scripts.sh

# Kaggle 설정
./scripts.sh config/kaggle.json
```

2. 파이프라인을 생성한 후 특정 run으로 submission 생성

```bash
python -m main --config config/dacon.json
python -m main --config config/kaggle.json
```

```bash
# best iteration 자동 선택(기본)
python -m submission --config config/kaggle.json --run_id <RUN_ID>

# iteration 수동 지정
python -m submission --config config/kaggle.json --run_id <RUN_ID> --iteration <ITERATION>
```

결과 파일은 기본적으로 `submissions/*/submission_<RUN_ID>_iter_<ITERATION>.csv` 형식으로 저장됩니다.

## 예시 결과

(사용 방법을 실행했을 때 나타나는 결과나 시각화 이미지를 보여주세요)

## 팀원

<table>
  <tr>
    <td align="center" valign="top">
      <img src="https://github.com/3sirn3203.png?size=200" width="120" alt="김우진" /><br/>
      <b>김우진</b><br/>
      <a href="https://github.com/3sirn3203">@3sirn3203</a><br/>
      팀장
    </td>
    <td align="center" valign="top">
      <img src="https://github.com/JuHyeon1222.png?size=200" width="120" alt="박주현" /><br/>
      <b>박주현</b><br/>
      <a href="https://github.com/JuHyeon1222">@JuHyeon1222</a>
    </td>
    <td align="center" valign="top">
      <img src="https://github.com/tigris-ignea.png?size=200" width="120" alt="박서연" /><br/>
      <b>박서연</b><br/>
      <a href="https://github.com/tigris-ignea">@tigris-ignea</a>
    </td>
    <td align="center" valign="top">
      <img src="https://github.com/hansimboy.png?size=200" width="120" alt="장국영" /><br/>
      <b>장국영</b><br/>
      <a href="https://github.com/hansimboy">@hansimboy</a>
    </td>
    <td align="center" valign="top">
      <img src="https://github.com/studipu.png?size=200" width="120" alt="백승우" /><br/>
      <b>백승우</b><br/>
      <a href="https://github.com/studipu">@studipu</a>
    </td>
  </tr>
</table>
