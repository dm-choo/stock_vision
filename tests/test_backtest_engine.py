"""
src/backtest/engine.py 단위 테스트

결정적 가격 데이터(선형 상승/하락 트렌드)로 엔진 동작 검증.
외부 API 미사용.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestResult, _rebalance_dates, run_backtest


# ── 픽스처 ────────────────────────────────────────────────────────────

@pytest.fixture
def uptrend_prices() -> pd.DataFrame:
    """
    10개 종목, 800거래일 선형 상승 가격.
    기간이 충분해 여러 분기 리밸런싱이 가능.
    """
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2021-01-04", periods=800)
    data = {}
    for i in range(10):
        base = np.linspace(100, 200, 800)
        noise = rng.normal(0, 1, 800)
        data[f"T{i:02d}"] = np.maximum(base + noise, 1.0)
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def downtrend_prices() -> pd.DataFrame:
    """10개 종목, 800거래일 선형 하락 가격."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2021-01-04", periods=800)
    data = {}
    for i in range(10):
        base = np.linspace(200, 100, 800)
        noise = rng.normal(0, 1, 800)
        data[f"D{i:02d}"] = np.maximum(base + noise, 1.0)
    return pd.DataFrame(data, index=dates)


# ── 리밸런싱 날짜 생성 ────────────────────────────────────────────────

def test_rebalance_dates_all_trading_days(uptrend_prices):
    """생성된 리밸런싱 날짜는 모두 실제 거래일이어야 한다."""
    trading_days = set(uptrend_prices.index)
    dates = _rebalance_dates(uptrend_prices, "2022-01-01", "2023-12-31")
    for d in dates:
        assert d in trading_days, f"{d}는 거래일이 아님"


def test_rebalance_dates_at_least_4_per_year(uptrend_prices):
    """2년 기간에 최소 7개 이상의 분기 리밸런싱 날짜가 있어야 한다."""
    dates = _rebalance_dates(uptrend_prices, "2022-01-01", "2023-12-31")
    assert len(dates) >= 7


# ── 백테스트 실행 ─────────────────────────────────────────────────────

def test_run_backtest_returns_result_type(uptrend_prices):
    """run_backtest는 BacktestResult를 반환해야 한다."""
    result = run_backtest(uptrend_prices, top_n=3, mode="technical")
    assert isinstance(result, BacktestResult)


def test_backtest_uptrend_positive_cagr(uptrend_prices):
    """상승 추세 시장에서 포트폴리오 CAGR은 양수여야 한다."""
    result = run_backtest(uptrend_prices, top_n=3, mode="technical")
    assert result.metrics["cagr"] > 0, "상승 추세에서 CAGR은 양수여야 함"


def test_backtest_downtrend_negative_cagr(downtrend_prices):
    """하락 추세 시장에서 포트폴리오 CAGR은 음수여야 한다."""
    result = run_backtest(downtrend_prices, top_n=3, mode="technical")
    assert result.metrics["cagr"] < 0, "하락 추세에서 CAGR은 음수여야 함"


def test_backtest_mdd_nonpositive(uptrend_prices):
    """Max Drawdown은 항상 0 이하여야 한다."""
    result = run_backtest(uptrend_prices, top_n=3, mode="technical")
    assert result.metrics["max_drawdown"] <= 0


def test_backtest_win_rate_range(uptrend_prices):
    """Win Rate는 0~1 사이여야 한다."""
    result = run_backtest(uptrend_prices, top_n=3, mode="technical")
    wr = result.metrics["win_rate"]
    assert 0.0 <= wr <= 1.0


def test_backtest_holdings_count(uptrend_prices):
    """holdings 리스트 길이 = quarterly_portfolio 길이."""
    result = run_backtest(uptrend_prices, top_n=3, mode="technical")
    assert len(result.holdings) == len(result.quarterly_portfolio)


def test_backtest_top_n_respected(uptrend_prices):
    """각 분기 보유 종목 수는 top_n 이하여야 한다 (데이터 부족 시 줄어들 수 있음)."""
    top_n = 4
    result = run_backtest(uptrend_prices, top_n=top_n, mode="technical")
    for h in result.holdings:
        assert len(h["tickers"]) <= top_n


def test_portfolio_value_starts_at_one(uptrend_prices):
    """포트폴리오 시작 값은 1.0이어야 한다."""
    result = run_backtest(uptrend_prices, top_n=3, mode="technical")
    assert result.portfolio_value.iloc[0] == pytest.approx(1.0)


def test_benchmark_value_starts_at_one(uptrend_prices):
    """벤치마크 시작 값은 1.0이어야 한다."""
    result = run_backtest(uptrend_prices, top_n=3, mode="technical")
    assert result.benchmark_value.iloc[0] == pytest.approx(1.0)


def test_portfolio_and_benchmark_different(uptrend_prices):
    """포트폴리오와 벤치마크 가치는 서로 달라야 한다 (집중 포트폴리오 vs 전체 평균)."""
    result = run_backtest(uptrend_prices, top_n=3, mode="technical")
    assert not result.portfolio_value.equals(result.benchmark_value)


# ── 유효성 검사 ───────────────────────────────────────────────────────

def test_invalid_lookback_raises(uptrend_prices):
    """lookback_days < 200이면 ValueError가 발생해야 한다."""
    with pytest.raises(ValueError, match="lookback_days"):
        run_backtest(uptrend_prices, top_n=3, mode="technical", lookback_days=100)


def test_hybrid_without_fundamentals_raises(uptrend_prices):
    """hybrid 모드에서 fundamentals=None이면 ValueError가 발생해야 한다."""
    with pytest.raises(ValueError, match="fundamentals"):
        run_backtest(uptrend_prices, top_n=3, mode="hybrid", fundamentals=None)
