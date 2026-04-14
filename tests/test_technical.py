"""
src/factors/technical.py 단위 테스트

검증 항목:
  1. 200거래일 미만 종목(SHORT) 제외
  2. RSI 점수 공식 — 55 근처 최고점, 양 극단 패널티
  3. 모멘텀 계산 정확성 — 알려진 가격으로 수동 검증
  4. MA 시그널 — 50일MA > 200일MA 이면 signal > 1
  5. 기술적 점수 범위 0~100
"""

import numpy as np
import pandas as pd
import pytest

from src.factors.technical import _rsi, _rsi_score, compute_technical_factors, TECH_WEIGHTS


# ── RSI 공식 단위 테스트 ──────────────────────────────────────────────

class TestRsiScore:
    def test_optimal_rsi_scores_highest(self):
        """RSI 55는 최고점(100)에 가까워야 한다."""
        assert _rsi_score(55) == pytest.approx(100.0)

    def test_extreme_high_rsi_penalized(self):
        """RSI 80 이상은 과매수로 최적값(RSI 55)보다 점수가 절반 이하여야 한다."""
        assert _rsi_score(85) < _rsi_score(55) / 2

    def test_extreme_low_rsi_penalized(self):
        """RSI 25 이하는 과매도로 낮은 점수를 받아야 한다."""
        assert _rsi_score(20) < 30

    def test_nan_rsi_returns_nan(self):
        """NaN RSI는 NaN 점수를 반환해야 한다."""
        assert np.isnan(_rsi_score(float("nan")))

    def test_score_clipped_to_0_100(self):
        """점수는 0~100 범위를 벗어나지 않아야 한다."""
        for val in [0, 10, 50, 55, 90, 100]:
            score = _rsi_score(val)
            assert 0 <= score <= 100, f"RSI {val} → score {score} 범위 초과"


# ── 200일 미만 종목 제외 ──────────────────────────────────────────────

def test_short_ticker_excluded(sample_prices):
    """100거래일 데이터만 있는 SHORT 종목은 결과에서 제외되어야 한다."""
    result = compute_technical_factors(sample_prices)
    assert "SHORT" not in result["ticker"].values, \
        "200거래일 미만 종목은 기술적 분석에서 제외되어야 함"


def test_full_tickers_included(sample_prices):
    """250거래일 데이터가 있는 종목들은 모두 포함되어야 한다."""
    result = compute_technical_factors(sample_prices)
    expected = {"TECH1", "TECH2", "ENRG1", "ENRG2"}
    assert expected.issubset(set(result["ticker"].values))


# ── 모멘텀 계산 정확성 ────────────────────────────────────────────────

def test_momentum_calculation():
    """
    선형 상승 가격으로 6개월 모멘텀 수동 검증.
    250거래일 동안 100 → 200으로 상승: 마지막 값=200, 126일전=149.8 → mom_6m ≈ 0.334
    """
    dates = pd.bdate_range(end="2025-12-31", periods=250)
    prices_linear = np.linspace(100, 200, 250)
    prices = pd.DataFrame({"LINEAR": prices_linear}, index=dates)
    result = compute_technical_factors(prices)
    assert not result.empty
    mom = result.loc[result["ticker"] == "LINEAR", "mom_6m"].values[0]
    expected = (prices_linear[-1] / prices_linear[-126]) - 1
    assert mom == pytest.approx(expected, rel=1e-3)


# ── MA 시그널 ─────────────────────────────────────────────────────────

def test_ma_signal_uptrend():
    """
    지속 상승 주가에서는 50일MA > 200일MA → ma_signal > 1 이어야 한다.
    """
    dates = pd.bdate_range(end="2025-12-31", periods=250)
    uptrend = np.linspace(50, 200, 250)
    prices = pd.DataFrame({"UP": uptrend}, index=dates)
    result = compute_technical_factors(prices)
    assert result.loc[result["ticker"] == "UP", "ma_signal"].values[0] > 1.0


def test_ma_signal_downtrend():
    """
    지속 하락 주가에서는 50일MA < 200일MA → ma_signal < 1 이어야 한다.
    """
    dates = pd.bdate_range(end="2025-12-31", periods=250)
    downtrend = np.linspace(200, 50, 250)
    prices = pd.DataFrame({"DOWN": downtrend}, index=dates)
    result = compute_technical_factors(prices)
    assert result.loc[result["ticker"] == "DOWN", "ma_signal"].values[0] < 1.0


# ── 점수 범위 ─────────────────────────────────────────────────────────

def test_technical_score_range(sample_prices):
    """technical_score는 0~100 범위 내에 있어야 한다."""
    result = compute_technical_factors(sample_prices)
    scores = result["technical_score"].dropna()
    assert (scores >= 0).all() and (scores <= 100).all(), \
        f"점수 범위 초과: min={scores.min():.2f}, max={scores.max():.2f}"


# ── 신규 지표 테스트 ──────────────────────────────────────────────────

def test_mom_12_1_positive_on_uptrend():
    """꾸준히 상승하는 가격에서 mom_12_1은 양수여야 한다."""
    dates = pd.bdate_range(end="2025-12-31", periods=260)
    prices = pd.DataFrame({"UP": np.linspace(100, 200, 260)}, index=dates)
    result = compute_technical_factors(prices)
    mom = result.loc[result["ticker"] == "UP", "mom_12_1"].values[0]
    assert mom > 0, f"상승 추세에서 mom_12_1은 양수여야 함: {mom}"


def test_mom_12_1_negative_on_downtrend():
    """꾸준히 하락하는 가격에서 mom_12_1은 음수여야 한다."""
    dates = pd.bdate_range(end="2025-12-31", periods=260)
    prices = pd.DataFrame({"DOWN": np.linspace(200, 100, 260)}, index=dates)
    result = compute_technical_factors(prices)
    mom = result.loc[result["ticker"] == "DOWN", "mom_12_1"].values[0]
    assert mom < 0, f"하락 추세에서 mom_12_1은 음수여야 함: {mom}"


def test_consistency_high_on_all_up():
    """매일 상승하는 가격에서 consistency는 1.0에 가까워야 한다."""
    dates = pd.bdate_range(end="2025-12-31", periods=250)
    prices = pd.DataFrame({"ALLUP": np.linspace(100, 200, 250)}, index=dates)
    result = compute_technical_factors(prices)
    cons = result.loc[result["ticker"] == "ALLUP", "consistency"].values[0]
    assert cons > 0.9, f"매일 상승 시 consistency는 0.9 이상이어야 함: {cons}"


def test_volatility_adjusted_mom_nan_on_zero_vol():
    """변동성이 0이면 mom_3m_adj는 NaN이어야 한다 (0으로 나누기 방지)."""
    dates = pd.bdate_range(end="2025-12-31", periods=250)
    # 완전히 flat한 가격 → 변동성 0
    prices = pd.DataFrame({"FLAT": np.ones(250) * 100.0}, index=dates)
    result = compute_technical_factors(prices)
    adj = result.loc[result["ticker"] == "FLAT", "mom_3m_adj"].values[0]
    assert np.isnan(adj), "변동성이 0이면 mom_3m_adj는 NaN이어야 함"


def test_tech_weights_sum_to_one():
    """TECH_WEIGHTS 가중치 합계는 1.0이어야 한다."""
    total = sum(TECH_WEIGHTS.values())
    assert total == pytest.approx(1.0, rel=1e-6)
