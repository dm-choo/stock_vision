# Stock Vision

다음 분기 주가 상승 가능성이 높은 종목을 스코어링하고, 분기 리밸런싱 백테스트로 모델 정확도를 검증하며, 분위 회귀로 수익률 크기를 예측하는 Python 분석 프로젝트.

- **대상 시장**: 미국 S&P 500, 한국 KOSPI 200
- **데이터 소스**: yfinance, FinanceDataReader (무료 API)
- **백테스트 결과** (S&P 500, 2022-2026, top_n=20): CAGR 27.5% vs 벤치마크 18.2%, Alpha +9.3%

---

## 설치

```bash
uv sync --dev
```

Python 3.12+, [uv](https://github.com/astral-sh/uv) 필요.

---

## 사용법

### 스코어링

```bash
uv run python main.py --market us              # S&P 500 스코어링 (상위 20개 출력)
uv run python main.py --market kr              # KOSPI 스코어링
uv run python main.py --market us --top 30     # 상위 30개 출력
uv run python main.py --market us --no-cache   # 캐시 무시하고 재수집
```

결과는 `data/us/scores.csv` / `data/kr/scores.csv`로 저장.

### 백테스트

5년치 가격 데이터(`data/us/prices_5y.parquet`)가 필요합니다.

```bash
uv run python main.py --market us --backtest                          # 기술적 모드
uv run python main.py --market us --backtest --backtest-mode hybrid   # hybrid 모드 (look-ahead 있음)
uv run python main.py --market us --backtest --top-n 10               # 포트폴리오 10종목
```

### 통계 검증

```bash
uv run python main.py --market us --validate   # Walk-Forward + Monte Carlo 실행
```

### 수익률 예측

5년치 가격 데이터와 `scores.csv`가 필요합니다 (`--market us` 먼저 실행).

```bash
uv run python main.py --market us --predict                           # clean 모드 (기술적 신호만)
uv run python main.py --market us --predict --predict-mode research   # +현재 펀더멘탈 (look-ahead 주의)
uv run python main.py --market us --predict --validate                # 예측 + Walk-Forward CV 성능 평가
uv run python main.py --market us --predict --model-type gbm          # GBM 모델 사용
uv run python main.py --market us --predict --top-n 30                # 상위 30개 출력
```

결과는 `data/us/predictions_clean.csv` / `data/us/predictions_research.csv`로 저장.

### 테스트

```bash
uv run pytest tests/                           # 전체 테스트 (99개)
uv run pytest tests/test_technical.py -v       # 단일 파일
uv run pytest tests/ -k "test_rsi"             # 키워드 매칭
```

---

## 스코어링 구조

### 데이터 흐름

```
Collector ──► fundamental.py ──► composite.py ──► scores.csv
              (섹터 상대 비교)
Collector ──► technical.py ──┘
              (절대값 기준)
```

### 펀더멘탈 팩터 (60%)

섹터 평균 대비 상대 비율 → 전체 percentile rank (0~100)

| 지표 | 방향 | 가중치 | 설명 |
|------|------|--------|------|
| ROE | 높을수록 | 20% | 자기자본이익률 |
| PER | 낮을수록 | 20% | 주가수익비율 (음수=NaN) |
| PBR | 낮을수록 | 10% | 주가순자산비율 (음수=NaN) |
| revenue_growth | 높을수록 | 20% | 매출성장률 |
| debt_to_equity | 낮을수록 | 10% | 부채비율 |
| fcf_yield | 높을수록 | 10% | FCF 수익률 |
| op_margin | 높을수록 | 5% | 영업이익률 |
| ev_ebitda | 낮을수록 | 5% | EV/EBITDA (음수=NaN) |

- **섹터 상대 비교**: 동일 섹터 내 평균 대비 비율로 정규화
- **Unknown 섹터**: 전체 유니버스 평균 기준으로 비교
- **클리핑**: 5~95th percentile (revenue_growth는 5~90th)

### 기술적 팩터 (40%)

| 지표 | 가중치 | 설명 |
|------|--------|------|
| mom_12_1 | 25% | 12-1 모멘텀 (최근 1달 제외, Jegadeesh & Titman 1993) |
| mom_3m_adj | 20% | 변동성 조정 3개월 모멘텀 |
| mom_6m | 15% | 6개월 모멘텀 |
| consistency | 15% | 모멘텀 일관성 (63일 중 양수 수익률 비율) |
| rsi | 10% | RSI — 55 근처 최적, 양 극단 패널티 |
| ma_signal | 15% | 50일MA / 200일MA 골든크로스 신호 |

- 200거래일 미만 데이터 종목 제외
- 전체 유니버스 percentile rank (RSI는 절대 공식 적용)

### 합산

`composite_score = percentile(fundamental_score) × 0.60 + percentile(technical_score) × 0.40`

두 점수의 분산 차이를 percentile 재정규화로 보정한 뒤 가중합.

---

## 백테스트 엔진

### 작동 방식

1. 분기 말 거래일에 리밸런싱 (QE 스냅)
2. 리밸런싱 시점까지의 가격 데이터로 기술적 점수 계산 (look-ahead 없음)
3. 상위 `top_n`개 종목 equal-weight 매수
4. 다음 분기까지 보유 후 수익률 기록

### 모드

| 모드 | 설명 | look-ahead bias |
|------|------|-----------------|
| `technical` | 가격 데이터만 사용 | 없음 |
| `hybrid` | 기술적 + 현재 펀더멘탈 스크린 | 있음 (명시적 경고) |

### 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `top_n` | 20 | 분기 보유 종목 수 |
| `min_history_days` | 252 | 최소 거래일 (신규 상장 이상치 방지) |
| `transaction_cost` | 0.001 | 단방향 거래비용 (0.1%) |
| `lookback_days` | 300 | 기술적 지표 계산 창 |

### 성과 지표

CAGR, Alpha, Sharpe, Sortino, Calmar, Information Ratio, Max Drawdown, Win Rate, Turnover

---

## 검증 결과 (S&P 500, 2022-2026)

| | 전략 (top_n=20) | 벤치마크 |
|--|--|--|
| CAGR | 27.5% | 18.2% |
| Alpha | +9.3% | — |
| Sharpe | 1.15 | 1.12 |
| Sortino | 1.59 | 1.67 |
| Calmar | 1.00 | — |
| IR | 0.52 | — |
| Max MDD | -27.6% | -17.8% |
| Win Rate | 53.3% | — |

**Monte Carlo (n=5,000)**: 전략 CAGR이 무작위 포트폴리오 대비 93.7th percentile (p=0.063)

**주의사항**:
- 15개 분기(3.75년)는 통계적 유의성 확보에 부족한 기간
- Walk-Forward 분석에서 시장 국면 의존성 확인 (모멘텀 약세 구간에서 Alpha 음수)
- `hybrid` 모드는 현재 펀더멘탈 사용으로 look-ahead bias 존재

---

## 수익률 예측 모델 (Phase 3)

### 개요

복합 점수로 종목을 랭킹하는 것을 넘어, **다음 분기에 얼마나 오를지** 수익률 크기를 예측합니다.

- **예측 형태**: q10 / q50 / q90 분위 회귀 → 80% 신뢰구간 제공
- **학습 데이터**: 5년치 가격 × 500+ 종목 × ~14 분기 ≈ 7,000 샘플
- **예측 대상**: 다음 분기 개별 종목 수익률

### 모드

| 모드 | 피처 | look-ahead bias | 용도 |
|------|------|-----------------|------|
| `clean` | 기술적 신호 7개 + 횡단면 순위 | 없음 | 실전 예측 |
| `research` | clean + 현재 펀더멘탈 10개 | 있음 | 연구/상한선 추정 |

### 피처셋 (CLEAN)

`technical_score`, `mom_12_1_score`, `mom_3m_adj_score`, `mom_6m_score`, `consistency_score`, `rsi_score`, `ma_score` + 각 피처의 분기 내 횡단면 순위 (총 14개)

### 모델

| 모델 | 설명 |
|------|------|
| `ridge` (기본) | q50: Ridge(α=1.0), q10/q90: QuantileRegressor(α=0.1) |
| `gbm` | HistGradientBoostingRegressor(loss="quantile", max_depth=3) |

### 검증 지표 (Purged Walk-Forward CV)

IC (Information Coefficient) = 분기 내 예측 순위 vs 실제 순위의 스피어만 상관

| IC 범위 | 해석 |
|---------|------|
| < 0.02 | 예측력 없음 |
| 0.02 ~ 0.05 | 약한 예측력 |
| 0.05 ~ 0.10 | 유의미한 예측력 |
| > 0.10 | 과적합/look-ahead 의심 |

### 알려진 한계

- `QuantileRegressor(alpha=0.1)` 정규화가 강해 q10/q90 CI가 훈련 데이터 전체 분위값으로 수렴하는 경향 있음 (alpha 축소 필요)
- 분기 수익률 std ≈ 16% — 노이즈가 커서 q50 절대값보다 종목 간 상대 순위에 의미를 두는 것이 적절
- 투자 조언 아님. 과거 패턴 기반 통계적 추정치.

---

## 프로젝트 구조

```
stock_vision/
├── main.py                    # CLI 진입점
├── src/
│   ├── collectors/
│   │   ├── us_collector.py    # S&P 500 데이터 수집 (yfinance)
│   │   └── kr_collector.py    # KOSPI 데이터 수집 (FinanceDataReader)
│   ├── factors/
│   │   ├── fundamental.py     # 섹터 상대 펀더멘탈 스코어
│   │   └── technical.py       # 모멘텀/RSI/MA 기술적 스코어
│   ├── scoring/
│   │   └── composite.py       # 복합 점수 합산 및 순위
│   ├── backtest/
│   │   ├── engine.py          # 분기 리밸런싱 백테스트 엔진
│   │   ├── metrics.py         # 성과 지표 (CAGR, Sharpe, Sortino 등)
│   │   └── validation.py      # Walk-Forward + Monte Carlo 검증
│   └── predictor/
│       ├── data_collector.py  # (점수, 다음 분기 수익률) 학습 쌍 수집
│       ├── feature_builder.py # CLEAN/RESEARCH 피처셋 + 횡단면 순위
│       ├── model.py           # ReturnPredictor: q10/q50/q90 분위 회귀
│       ├── validator.py       # PurgedWalkForwardCV + IC 평가
│       ├── predictor.py       # QuarterlyReturnPredictor 통합 API
│       └── display.py         # 예측 결과 테이블 출력
├── tests/                     # 단위 테스트 (99개)
├── data/
│   ├── us/                    # S&P 500 캐시 (parquet, gitignore)
│   └── kr/                    # KOSPI 캐시 (parquet, gitignore)
└── scripts/
    └── pre_commit_test.sh     # 커밋 전 pytest 자동 실행 훅
```

---

## 개발 환경

- `git commit` 시 pytest가 자동으로 실행됩니다. 테스트 실패 시 커밋이 차단됩니다.
- 캐시 파일(`data/`)은 `.gitignore`에 포함되어 있습니다. `--no-cache` 옵션으로 재수집할 수 있습니다.
