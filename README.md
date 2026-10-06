# 로그 분석을 통한 비정상 행위 탐지

시스템 로그(full_log)를 분석해 **위험도(level 0~6)를 분류**하고 비정상 행위를 탐지하는 프로젝트.
정보보안 동아리 **Ping!** 학기 프로젝트로 시작했고, 동덕여자대학교 교내 SW 경진대회(2024.11.06)에서 **장려상**을 받았다.

> ### 솔직한 배경 — 왜 다시 완성했나
> 대회 당시에는 **기간 내 모델을 완성하지 못했다.** 정규식으로 로그 필드를 추출하고
> 규칙 기반 탐지를 시도하는 단계에서 멈췄는데, 원인은 두 가지였다.
> 1. 학습 데이터에 **여러 유형의 로그(kibana·logstash·wazuh 등)가 섞여 있다는 걸 뒤늦게 발견**했고, 유형별로 다른 전처리가 필요했다.
> 2. **"비정상"의 기준을 명확히 정의하지 못해** 머신러닝 적용 단계까지 가지 못했다.
>
> 발표 때는 완성 대신 *실패 원인 분석 + 참고 프로젝트(DistilBERT) 학습 + 개선 방향*을 정직하게 발표했고 그것으로 수상했다.
> 이 저장소는 **그때 못 끝낸 것을 지금 제대로 완성한 결과물**이다. 아래 성능 수치는 모두 실제로 학습·평가해 얻은 값이다.

---

## 데이터셋

- **DACON 시스템 로그 분석 경진대회** — <https://dacon.io/competitions/official/235717/data>
- `train.csv` : `id, level, full_log` (472,972행, 라벨 있음)
- `test.csv`  : `id, full_log` (라벨 없음 — 대회 제출용이라 자체 평가에는 쓰지 않음)
- 용량이 커서 저장소에는 포함하지 않는다(`.gitignore`). 위 링크에서 받아 `data/`에 둔다.

**클래스 분포(train 전체) — 극심한 불균형:**

| level | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| 개수 | 334,065 | 132,517 | **12** | 4,141 | **10** | 2,219 | **8** |

level 2·4·6은 전체에서 8~12개뿐이다. (참고 프로젝트가 이 클래스들을 oversampling한 이유.)

## 접근

라벨이 없는 `test.csv`로는 정량 평가가 불가능하므로, 정직한 성능 측정을 위해 **`train.csv`를
stratified 분할**(train/val/test)해 평가한다.

1. **로그 파싱(`src/features.py`)** — 포맷이 섞여 있어 "하나의 정규식"은 깨지기 쉽다는 과거 교훈을 반영.
   - 구조적 파싱: timestamp/application/level 키워드/IP 개수/길이를 *관대하게* 추출(실패 시 None).
   - 텍스트 정규화: 타임스탬프·IP·숫자를 `<TS>/<IP>/<NUM>` 토큰으로 치환해 일반화.
2. **Baseline (실제 실행)** — TF-IDF(word 1·2-gram) + LogisticRegression(`class_weight='balanced'`).
   포맷 혼재에 견고하고 CPU에서 빠르다.
3. **고급 모델 (코드 제공)** — DistilBERT fine-tuning + **Focal Loss** + 소수 클래스 oversampling + `max_len=512`.
   참고 프로젝트의 아이디어를 올바른 코드로 재구성. (아래 "실행 범위" 참고)

## 결과 (Baseline — 실제 측정값)

`train.csv` 전체 472,972행, held-out test로 평가. 학습+평가 약 **150초**(CPU).

| 지표 | 값 |
|---|---|
| accuracy | **0.9961** |
| weighted-F1 | **0.9962** |
| macro-F1 | **0.8952** |

| level | precision | recall | f1 | test 표본 수 |
|---|---|---|---|---|
| 0 | 0.999 | 0.996 | 0.998 | 66,813 |
| 1 | 0.995 | 0.996 | 0.996 | 26,504 |
| 2 | 0.667 | 1.000 | 0.800 | 2 |
| 3 | 0.975 | 0.994 | 0.984 | 828 |
| 4 | 1.000 | 1.000 | 1.000 | 2 |
| 5 | 0.709 | 0.977 | 0.822 | 444 |
| 6 | 1.000 | 0.500 | 0.667 | 2 |

> ⚠️ **해석 주의:** 로그가 상당히 정형적(템플릿)이어서 다수 클래스(0·1·3·5)는 매우 높은 성능이 나온다.
> 반면 level 2·4·6은 **test 표본이 2개뿐**이라 수치(F1 0.667~1.0)는 통계적으로 신뢰하기 어렵다.
> macro-F1(0.895)이 accuracy(0.996)보다 낮은 것이 이 소수 클래스 문제를 그대로 보여준다.
> 전체 지표: [`results/baseline_metrics.json`](results/baseline_metrics.json)

## 실행 방법

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) 데이터: DACON 235717에서 받아 data/train.csv, data/test.csv 로 저장

# 2) baseline 학습·평가 (전체)
python scripts/run_baseline.py --data data/train.csv
#    빠른 실험
python scripts/run_baseline.py --nrows 20000

# 3) (선택) DistilBERT — GPU 권장
pip install -r requirements-nlp.txt
python scripts/run_distilbert.py --nrows 20000 --epochs 3
```

## 프로젝트 구조

```
src/
  data.py            # 로딩 + stratified 분할(소수 클래스 폴백 처리)
  features.py        # 로그 파싱 + TF-IDF 벡터화
  evaluate.py        # per-class P/R/F1, macro-F1, confusion matrix
  models/
    baseline.py      # TF-IDF + LogisticRegression(balanced)  ← 실행됨
    distilbert.py    # DistilBERT + Focal Loss + oversampling ← 코드 제공
scripts/
  run_baseline.py    # baseline CLI
  run_distilbert.py  # DistilBERT CLI
notebooks/
  legacy_preprocess.ipynb  # 대회 당시 원본 전처리 노트북(기록 보존)
results/
  baseline_metrics.json    # 실제 측정 지표
```

## 실행 범위 (정직하게)

- ✅ **Baseline(TF-IDF + LogisticRegression)**: 전체 데이터로 실제 학습·평가함. 위 수치가 그 결과.
- 🧩 **DistilBERT + Focal Loss**: 재현 가능한 코드로 제공하되, CPU에서 512 토큰·12 epoch 학습은
  비현실적으로 느려 이 환경에서는 전체 학습을 돌리지 않았다. GPU 환경에서 `run_distilbert.py`로 실행 가능.

## 한계 및 개선 방향

- **소수 클래스(2·4·6) 데이터 자체가 극소수** — oversampling/합성(SMOTE 변형)·능동학습으로 표본 확보 필요.
- 현재는 로그를 독립적으로 분류 — 실제 이상탐지는 **시퀀스/시간 패턴**(로그 급증, 세션 단위)이 중요하므로
  시계열·세션 집계 피처와 LogBERT 류 접근을 더할 여지가 있다.
- 평가가 단일 분할 기준 — **k-fold 교차검증**으로 안정성을 확인할 것.
- 다수 클래스 성능이 과도하게 높아 **데이터 누수(로그 템플릿 암기)** 가능성 점검 필요.
