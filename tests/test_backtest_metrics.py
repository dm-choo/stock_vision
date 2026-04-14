"""
src/backtest/metrics.py 단위 테스트

결정적 시리즈로 각 지표의 수식을 직접 검증.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import (
    cagr,
    calmar_ratio,
    compute_all_metrics,
    information_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)


def _make_value(daily_returns: list[float], start: str = "2021-01-04") -> pd.Series:
    """일별 수익률 리스트 → 누적 가치 시리즈"""
    dates = pd.bdate_range(start=start, periods=len(daily_returns) + 1)
    values = [1.0]
    for r in daily_returns:
        values.append(values[-1] * (1 + r))
    return pd.Series(values, index=dates)


# ── CAGR ──────────────────────────────────────────────────────────────

def test_cagr_double_in_one_year():
    """1년 만에 2배 → CAGR ≈ 100%"""
    dates = pd.date_range("2021-01-01", "2022-01-01", periods=2)
    s = pd.Series([1.0, 2.0], index=dates)
    assert cagr(s) == pytest.approx(1.0, rel=0.01)


def test_cagr_flat():
    """수익 없음 → CAGR = 0%"""
    dates = pd.date_range("2021-01-01", "2022-01-01", periods=2)
    s = pd.Series([1.0, 1.0], index=dates)
    assert cagr(s) == pytest.approx(0.0, abs=1e-6)


def test_cagr_loss():
    """1년 50% 손실 → CAGR = -50%"""
    dates = pd.date_range("2021-01-01", "2022-01-01", periods=2)
    s = pd.Series([1.0, 0.5], index=dates)
    assert cagr(s) == pytest.approx(-0.5, rel=0.01)


def test_cagr_single_point_returns_nan():
    """데이터 1개 → NaN"""
    s = pd.Series([1.0], index=pd.date_range("2021-01-01", periods=1))
    assert np.isnan(cagr(s))


# ── Max Drawdown ──────────────────────────────────────────────────────

def test_mdd_no_drawdown():
    """단조 상승 → MDD = 0"""
    s = _make_value([0.01] * 100)
    assert max_drawdown(s) == pytest.approx(0.0, abs=1e-6)


def test_mdd_known_drawdown():
    """
    100 → 50 → 80: 낙폭 = (100-50)/100 = 50%
    """
    dates = pd.bdate_range("2021-01-01", periods=3)
    s = pd.Series([1.0, 0.5, 0.8], index=dates)
    assert max_drawdown(s) == pytest.approx(-0.5, rel=0.01)


def test_mdd_always_negative_or_zero():
    """MDD는 항상 0 이하여야 한다."""
    rng = np.random.default_rng(99)
    returns = rng.normal(0.001, 0.02, 500)
    s = _make_value(returns.tolist())
    assert max_drawdown(s) <= 0


# ── Sharpe Ratio ─────────────────────────────────────────────────────

def test_sharpe_zero_variance_returns_nan():
    """수익률 분산 0 → NaN"""
    s = _make_value([0.001] * 252)
    # 수익률이 고정이면 std=0 → NaN
    assert np.isnan(sharpe_ratio(s))


def test_sharpe_positive_drift_is_positive():
    """
    고정된 양의 수익률 + 작은 노이즈 → Sharpe > 0.
    순수 랜덤 시드는 누적 손실 가능성이 있으므로 결정적 시리즈 사용.
    """
    rng = np.random.default_rng(0)
    # 일 +0.3% 고정 드리프트 + 작은 노이즈 → 누적 수익 보장
    noise = rng.normal(0.0, 0.005, 252)
    returns = [0.003 + n for n in noise]
    s = _make_value(returns)
    assert sharpe_ratio(s) > 0


def test_sharpe_negative_drift_is_negative():
    """음의 평균 수익률 → Sharpe < 0"""
    rng = np.random.default_rng(8)
    returns = rng.normal(-0.001, 0.01, 252)
    s = _make_value(returns.tolist())
    assert sharpe_ratio(s) < 0


# ── Win Rate ─────────────────────────────────────────────────────────

def test_win_rate_all_outperform():
    """모든 분기 초과 수익 → win rate = 1.0"""
    idx = pd.date_range("2021-01-01", periods=4, freq="QE")
    port = pd.Series([0.05, 0.07, 0.04, 0.06], index=idx)
    bench = pd.Series([0.03, 0.05, 0.02, 0.04], index=idx)
    assert win_rate(port, bench) == pytest.approx(1.0)


def test_win_rate_half():
    """절반 초과 → win rate = 0.5"""
    idx = pd.date_range("2021-01-01", periods=4, freq="QE")
    port = pd.Series([0.05, 0.01, 0.04, 0.01], index=idx)
    bench = pd.Series([0.03, 0.05, 0.02, 0.04], index=idx)
    assert win_rate(port, bench) == pytest.approx(0.5)


# ── compute_all_metrics ───────────────────────────────────────────────

def test_compute_all_metrics_keys():
    """compute_all_metrics가 필요한 모든 키를 반환해야 한다."""
    rng = np.random.default_rng(42)
    port_v = _make_value(rng.normal(0.001, 0.015, 504).tolist())
    bench_v = _make_value(rng.normal(0.0008, 0.015, 504).tolist())

    q_idx = pd.date_range("2021-01-01", periods=8, freq="QE")
    q_port = pd.Series(rng.normal(0.03, 0.05, 8), index=q_idx)
    q_bench = pd.Series(rng.normal(0.025, 0.05, 8), index=q_idx)

    result = compute_all_metrics(port_v, bench_v, q_port, q_bench)
    expected_keys = {
        "cagr", "benchmark_cagr", "alpha",
        "sharpe", "benchmark_sharpe",
        "sortino", "benchmark_sortino",
        "calmar", "information_ratio",
        "max_drawdown", "benchmark_mdd",
        "win_rate", "total_quarters",
    }
    assert expected_keys == set(result.keys())


# ── Sortino Ratio ────────────────────────────────────────────────────

def test_sortino_positive_drift_is_positive():
    """양의 드리프트 → Sortino > 0."""
    rng = np.random.default_rng(1)
    returns = [0.003 + n for n in rng.normal(0.0, 0.005, 252)]
    s = _make_value(returns)
    assert sortino_ratio(s) > 0


def test_sortino_zero_variance_returns_nan():
    """하방 수익률이 없으면 (모두 양수) Sortino는 NaN이어야 한다."""
    # 매일 정확히 같은 양의 수익률 → 하방 편차 = 0
    s = _make_value([0.001] * 252)
    result = sortino_ratio(s)
    assert np.isnan(result)


# ── Calmar Ratio ──────────────────────────────────────────────────────

def test_calmar_no_mdd_returns_nan():
    """MDD = 0이면 Calmar는 NaN이어야 한다."""
    s = _make_value([0.01] * 100)  # 단조 상승 → MDD = 0
    assert np.isnan(calmar_ratio(s))


def test_calmar_positive_for_uptrend_with_drawdown():
    """상승 추세 + 낙폭 존재 시 Calmar > 0."""
    rng = np.random.default_rng(99)
    returns = rng.normal(0.002, 0.02, 504)
    s = _make_value(returns.tolist())
    if max_drawdown(s) < 0:  # 낙폭이 실제로 발생한 경우만 검증
        assert calmar_ratio(s) > 0


# ── Information Ratio ─────────────────────────────────────────────────

def test_ir_all_outperform():
    """초과수익이 항상 양수인 경우 IR > 0."""
    idx = pd.date_range("2021-01-01", periods=8, freq="QE")
    # 초과수익이 변동하지만 항상 양수 → std > 0, mean > 0 → IR > 0
    port  = pd.Series([0.05, 0.07, 0.04, 0.06, 0.08, 0.03, 0.09, 0.05], index=idx)
    bench = pd.Series([0.03, 0.05, 0.02, 0.04, 0.06, 0.01, 0.07, 0.03], index=idx)
    assert information_ratio(port, bench) > 0


def test_ir_insufficient_quarters_returns_nan():
    """분기 수 < 4이면 IR은 NaN이어야 한다."""
    idx = pd.date_range("2021-01-01", periods=3, freq="QE")
    port = pd.Series([0.05, 0.04, 0.06], index=idx)
    bench = pd.Series([0.03, 0.02, 0.04], index=idx)
    assert np.isnan(information_ratio(port, bench))
