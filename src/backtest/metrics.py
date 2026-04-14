"""
백테스트 성과 지표

입력: 포트폴리오 누적 가치 시리즈 (index=date, value=누적 배수, 시작=1.0)

지표:
  CAGR              연환산 복리 수익률
  Sharpe Ratio      연환산 위험 조정 수익률 (무위험이자율 0% 가정)
  Sortino Ratio     하방 편차 기준 위험 조정 수익률
  Calmar Ratio      CAGR / |MDD| — 드로우다운 대비 수익 효율
  Information Ratio 분기 초과수익 일관성 (연환산)
  Max Drawdown      최대 낙폭 (음수)
  Win Rate          분기별 벤치마크 대비 초과 수익 비율
  Alpha             연환산 초과 수익 (vs 벤치마크 CAGR)
"""

import numpy as np
import pandas as pd


def cagr(value_series: pd.Series) -> float:
    """
    CAGR = (최종가치 / 초기가치) ^ (1 / 연수) - 1
    value_series: 날짜 인덱스, 시작값 1.0 기준 누적 배수
    """
    if len(value_series) < 2:
        return float("nan")
    years = (value_series.index[-1] - value_series.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return float((value_series.iloc[-1] / value_series.iloc[0]) ** (1 / years) - 1)


def max_drawdown(value_series: pd.Series) -> float:
    """
    MDD = max(1 - value / cumulative_max)
    반환값은 음수 (e.g. -0.25 = 25% 낙폭)
    """
    if value_series.empty:
        return float("nan")
    roll_max = value_series.cummax()
    drawdown = value_series / roll_max - 1
    return float(drawdown.min())


def sharpe_ratio(value_series: pd.Series, periods_per_year: int = 252) -> float:
    """
    Sharpe = mean(일별수익률) / std(일별수익률) × sqrt(periods_per_year)
    무위험이자율 0% 가정.
    """
    daily_returns = value_series.pct_change().dropna()
    if daily_returns.std() == 0 or len(daily_returns) < 2:
        return float("nan")
    return float(daily_returns.mean() / daily_returns.std() * np.sqrt(periods_per_year))


def sortino_ratio(value_series: pd.Series, periods_per_year: int = 252) -> float:
    """
    Sortino = mean(일별수익률) / std(하방 수익률) × sqrt(periods_per_year)
    하방 편차: 음수 수익률만의 표준편차. Sharpe보다 손실 리스크를 정확히 반영.
    """
    daily_returns = value_series.pct_change().dropna()
    downside = daily_returns[daily_returns < 0]
    if len(downside) < 2 or downside.std() == 0:
        return float("nan")
    return float(daily_returns.mean() / downside.std() * np.sqrt(periods_per_year))


def calmar_ratio(value_series: pd.Series) -> float:
    """
    Calmar = CAGR / |MDD|
    드로우다운 대비 수익 효율을 나타냄. 값이 클수록 낙폭 대비 수익이 우수.
    """
    mdd = max_drawdown(value_series)
    if mdd == 0:
        return float("nan")
    return float(cagr(value_series) / abs(mdd))


def information_ratio(
    quarterly_port: pd.Series,
    quarterly_bench: pd.Series,
) -> float:
    """
    IR = mean(분기 초과수익) / std(분기 초과수익) × sqrt(4)
    Alpha의 일관성을 측정. IR > 0.5이면 우수한 일관성으로 간주.
    """
    common = quarterly_port.index.intersection(quarterly_bench.index)
    if len(common) < 4:
        return float("nan")
    excess = quarterly_port[common] - quarterly_bench[common]
    if excess.std() == 0:
        return float("nan")
    return float(excess.mean() / excess.std() * np.sqrt(4))


def win_rate(
    portfolio_quarterly: pd.Series,
    benchmark_quarterly: pd.Series,
) -> float:
    """
    분기별 포트폴리오 수익률이 벤치마크를 초과한 비율.
    두 시리즈는 동일한 분기 인덱스를 가져야 함.
    """
    common = portfolio_quarterly.index.intersection(benchmark_quarterly.index)
    if len(common) == 0:
        return float("nan")
    outperform = (portfolio_quarterly[common] > benchmark_quarterly[common]).sum()
    return float(outperform / len(common))


def compute_all_metrics(
    portfolio_value: pd.Series,
    benchmark_value: pd.Series,
    quarterly_port: pd.Series,
    quarterly_bench: pd.Series,
) -> dict:
    """
    전체 성과 지표 딕셔너리 반환.

    Args:
        portfolio_value:   일별 포트폴리오 누적 가치 (시작=1.0)
        benchmark_value:   일별 벤치마크 누적 가치 (시작=1.0)
        quarterly_port:    분기별 포트폴리오 수익률
        quarterly_bench:   분기별 벤치마크 수익률
    """
    port_cagr = cagr(portfolio_value)
    bench_cagr = cagr(benchmark_value)

    return {
        "cagr": port_cagr,
        "benchmark_cagr": bench_cagr,
        "alpha": port_cagr - bench_cagr,
        "sharpe": sharpe_ratio(portfolio_value),
        "benchmark_sharpe": sharpe_ratio(benchmark_value),
        "sortino": sortino_ratio(portfolio_value),
        "benchmark_sortino": sortino_ratio(benchmark_value),
        "calmar": calmar_ratio(portfolio_value),
        "information_ratio": information_ratio(quarterly_port, quarterly_bench),
        "max_drawdown": max_drawdown(portfolio_value),
        "benchmark_mdd": max_drawdown(benchmark_value),
        "win_rate": win_rate(quarterly_port, quarterly_bench),
        "total_quarters": len(quarterly_port),
    }
