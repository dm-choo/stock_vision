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

from src.factors.technical import _rsi, _rsi_score, compute_technical_factors


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
    선형 상승 가격으로 3개월 모멘텀 수동 검증.
    250거래일 동안 100 → 200으로 상승: 마지막 값=200, 63일전=약 147 → mom_3m ≈ 0.36
    """
    dates = pd.bdate_range(end="2025-12-31", periods=250)
    prices = pd.DataFrame(
        {"LINEAR": np.linspace(100, 200, 250)},
        index=dates,
    )
    result = compute_technical_factors(prices)
    assert not result.empty
    mom = result.loc[result["ticker"] == "LINEAR", "mom_3m"].values[0]
    expected = (200 / np.linspace(100, 200, 250)[-63]) - 1
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
