"""
분기 리밸런싱 백테스트 엔진

두 가지 모드:
  technical  가격 데이터만 사용 → look-ahead bias 없음, 기술적 신호 순수 검증
  hybrid     기술적 신호 + 현재 펀더멘탈 정적 스크린 (look-ahead 한계 명시)

분기마다:
  1. 해당 시점까지의 가격으로 기술적 점수 계산 (과거 창만 사용)
  2. (hybrid) 펀더멘탈 필터로 유니버스 축소
  3. 상위 top_n 종목 equal-weight 매수
  4. 다음 분기까지 보유 → 수익률 기록
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.backtest.metrics import compute_all_metrics
from src.factors.fundamental import compute_fundamental_scores
from src.factors.technical import compute_technical_factors
from src.scoring.composite import MIN_VALID_FACTORS, compute_composite_score


@dataclass
class BacktestResult:
    """백테스트 실행 결과"""

    mode: str                                    # 'technical' | 'hybrid'
    portfolio_value: pd.Series                   # 일별 포트폴리오 누적 가치 (시작=1.0)
    benchmark_value: pd.Series                   # 일별 벤치마크 누적 가치
    quarterly_portfolio: pd.Series               # 분기별 포트폴리오 수익률
    quarterly_benchmark: pd.Series               # 분기별 벤치마크 수익률
    holdings: list[dict] = field(default_factory=list)  # 분기별 보유 종목
    metrics: dict = field(default_factory=dict)  # compute_all_metrics 결과

    def summary(self) -> str:
        m = self.metrics
        lines = [
            f"\n{'='*55}",
            f"  백테스트 결과 [{self.mode.upper()} 모드]",
            f"{'='*55}",
            f"  기간       : {self.portfolio_value.index[0].date()} ~ {self.portfolio_value.index[-1].date()}",
            f"  분기 수    : {m.get('total_quarters', '-')}",
            f"{'─'*55}",
            f"  CAGR       : {m.get('cagr', float('nan')):.1%}  (벤치마크 {m.get('benchmark_cagr', float('nan')):.1%})",
            f"  Alpha      : {m.get('alpha', float('nan')):.1%}",
            f"  Sharpe     : {m.get('sharpe', float('nan')):.2f}  (벤치마크 {m.get('benchmark_sharpe', float('nan')):.2f})",
            f"  Sortino    : {m.get('sortino', float('nan')):.2f}  (벤치마크 {m.get('benchmark_sortino', float('nan')):.2f})",
            f"  Calmar     : {m.get('calmar', float('nan')):.2f}",
            f"  IR         : {m.get('information_ratio', float('nan')):.2f}",
            f"  Max MDD    : {m.get('max_drawdown', float('nan')):.1%}  (벤치마크 {m.get('benchmark_mdd', float('nan')):.1%})",
            f"  Win Rate   : {m.get('win_rate', float('nan')):.1%} 분기 초과",
            f"  Avg 턴오버  : {m.get('avg_turnover', float('nan')):.1%}  (비용 차감 {m.get('total_cost_drag', 0.0):.2%})",
        ]
        if self.mode == "hybrid":
            lines.append(f"  ⚠  펀더멘탈은 현재 데이터 사용 → look-ahead bias 존재")
        lines.append(f"{'='*55}\n")
        return "\n".join(lines)


def _rebalance_dates(
    prices: pd.DataFrame,
    start: str,
    end: str,
    freq: str = "QE",
) -> pd.DatetimeIndex:
    """
    백테스트 리밸런싱 날짜 생성.
    실제 거래일로 스냅: 해당 분기 말 이전 마지막 거래일 사용.
    """
    ideal = pd.date_range(start=start, end=end, freq=freq)
    trading_days = prices.index
    snapped = []
    for d in ideal:
        prior = trading_days[trading_days <= d]
        if len(prior) > 0:
            snapped.append(prior[-1])
    return pd.DatetimeIndex(sorted(set(snapped)))


def _period_return(prices: pd.DataFrame, tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> float:
    """
    [start, end] 구간에서 equal-weight 포트폴리오 수익률.
    데이터 없는 종목은 제외.
    """
    cols = [t for t in tickers if t in prices.columns]
    if not cols:
        return 0.0
    sub = prices.loc[start:end, cols].dropna(axis=1, how="any")
    if sub.empty or len(sub) < 2:
        return 0.0
    stock_returns = sub.iloc[-1] / sub.iloc[0] - 1
    return float(stock_returns.mean())


def run_backtest(
    prices: pd.DataFrame,
    top_n: int = 20,
    start: str | None = None,
    end: str | None = None,
    mode: str = "technical",
    fundamentals: pd.DataFrame | None = None,
    fund_weight: float = 0.60,
    tech_weight: float = 0.40,
    lookback_days: int = 300,
    min_history_days: int = 252,
    transaction_cost: float = 0.001,
    collect_universe_scores: bool = False,
) -> BacktestResult:
    """
    분기 리밸런싱 백테스트 실행.

    Args:
        prices:                  일별 종가 DataFrame (index=date, col=ticker)
        top_n:                   분기마다 매수할 상위 종목 수
        start:                   백테스트 시작일 (None이면 가격 데이터 시작 + lookback_days)
        end:                     백테스트 종료일 (None이면 가격 데이터 마지막 날)
        mode:                    'technical' | 'hybrid'
        fundamentals:            hybrid 모드 시 필요한 현재 펀더멘탈 DataFrame
        fund_weight:             hybrid 모드 시 펀더멘탈 가중치
        tech_weight:             hybrid 모드 시 기술적 가중치
        lookback_days:           기술적 지표 계산에 사용할 과거 창 크기 (최소 200 필요)
        min_history_days:        리밸런싱 시점 기준 최소 거래일 수 (신규 상장 이상치 방지)
        transaction_cost:        단방향 리밸런싱 거래비용 비율 (기본 0.1%)
        collect_universe_scores: True이면 각 분기 전체 유니버스 기술적 점수를
                                 holdings[i]["universe_scores"]에 저장 (예측 학습 데이터 수집용)

    Returns:
        BacktestResult
    """
    if mode == "hybrid" and fundamentals is None:
        raise ValueError("hybrid 모드는 fundamentals DataFrame이 필요합니다.")
    if lookback_days < 200:
        raise ValueError("lookback_days는 최소 200 이상이어야 합니다 (200일 MA 요건).")

    # 백테스트 구간 설정
    first_possible = prices.index[lookback_days]
    bt_start = pd.Timestamp(start) if start else first_possible
    bt_end = pd.Timestamp(end) if end else prices.index[-1]

    rebal_dates = _rebalance_dates(prices, str(bt_start.date()), str(bt_end.date()))
    if len(rebal_dates) < 2:
        raise ValueError("리밸런싱 날짜가 2개 미만입니다. 기간을 늘려주세요.")

    quarterly_port: dict[pd.Timestamp, float] = {}
    quarterly_bench: dict[pd.Timestamp, float] = {}
    holdings: list[dict] = []
    turnover_ratios: list[float] = []

    # 벤치마크: 전체 유니버스 equal-weight
    universe_tickers = prices.columns.tolist()

    for i in range(len(rebal_dates) - 1):
        rebal = rebal_dates[i]
        next_rebal = rebal_dates[i + 1]

        # 과거 창 (look-ahead 없이 rebal 시점까지만)
        window_start = rebal - pd.Timedelta(days=lookback_days * 2)
        price_window = prices.loc[window_start:rebal]

        # IPO 필터: 리밸런싱 시점 기준 min_history_days 이상 거래일이 있는 종목만 허용
        full_history = prices.loc[:rebal]
        eligible = [
            c for c in prices.columns
            if full_history[c].dropna().shape[0] >= min_history_days
        ]
        price_window = price_window[eligible]

        # 기술적 점수 계산
        tech_scores = compute_technical_factors(price_window)

        if mode == "technical":
            if tech_scores.empty:
                continue
            selected = tech_scores.nlargest(top_n, "technical_score")["ticker"].tolist()

        else:  # hybrid
            fund_scored = compute_fundamental_scores(fundamentals)
            if tech_scores.empty or fund_scored.empty:
                continue
            result = compute_composite_score(
                fund_scored, tech_scores,
                fund_weight=fund_weight,
                tech_weight=tech_weight,
            )
            selected = result.head(top_n)["ticker"].tolist()

        if not selected:
            continue

        # 거래비용 계산: 직전 분기 대비 변경된 종목 비율 × transaction_cost
        prev_tickers = holdings[-1]["tickers"] if holdings else []
        turnover = len(set(selected).symmetric_difference(set(prev_tickers)))
        cost_drag = (turnover / max(len(selected), 1)) * transaction_cost
        turnover_ratios.append(turnover / max(len(selected), 1))

        # 분기 수익률 계산 (거래비용 차감)
        port_ret = _period_return(prices, selected, rebal, next_rebal) - cost_drag
        bench_ret = _period_return(prices, universe_tickers, rebal, next_rebal)

        quarterly_port[next_rebal] = port_ret
        quarterly_bench[next_rebal] = bench_ret

        holding_entry: dict = {
            "date": rebal,
            "tickers": selected,
            "period_return": port_ret,
            "cost_drag": cost_drag,
        }
        # 예측 학습 데이터 수집 모드: 전체 유니버스 기술적 점수 저장
        if collect_universe_scores and not tech_scores.empty:
            score_cols = ["technical_score", "mom_12_1_score", "mom_3m_adj_score",
                          "mom_6m_score", "consistency_score", "rsi_score", "ma_score"]
            available = [c for c in score_cols if c in tech_scores.columns]
            holding_entry["universe_scores"] = (
                tech_scores.set_index("ticker")[available].to_dict(orient="index")
            )
        holdings.append(holding_entry)

    if not quarterly_port:
        raise ValueError("유효한 분기 수익률이 없습니다. 데이터와 기간을 확인하세요.")

    q_port = pd.Series(quarterly_port)
    q_bench = pd.Series(quarterly_bench)

    # 일별 포트폴리오 가치 재구성 (분기 수익률 → 일별 보간)
    port_value, bench_value = _build_daily_value(prices, holdings, universe_tickers, rebal_dates)

    metrics = compute_all_metrics(port_value, bench_value, q_port, q_bench)
    metrics["avg_turnover"] = float(sum(turnover_ratios) / len(turnover_ratios)) if turnover_ratios else float("nan")
    metrics["total_cost_drag"] = float(sum(turnover_ratios) * transaction_cost) if turnover_ratios else 0.0

    return BacktestResult(
        mode=mode,
        portfolio_value=port_value,
        benchmark_value=bench_value,
        quarterly_portfolio=q_port,
        quarterly_benchmark=q_bench,
        holdings=holdings,
        metrics=metrics,
    )


def _build_daily_value(
    prices: pd.DataFrame,
    holdings: list[dict],
    universe_tickers: list[str],
    rebal_dates: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series]:
    """
    분기별 보유 종목으로 일별 포트폴리오/벤치마크 가치 시리즈 생성.
    포트폴리오와 벤치마크를 독립된 딕셔너리로 추적해 덮어쓰기 버그 방지.
    """
    port_map: dict[pd.Timestamp, float] = {rebal_dates[0]: 1.0}
    bench_map: dict[pd.Timestamp, float] = {rebal_dates[0]: 1.0}

    prev_port = 1.0
    prev_bench = 1.0

    bench_cols = [t for t in universe_tickers if t in prices.columns]

    for h in holdings:
        period_start = h["date"]
        next_idx = rebal_dates.searchsorted(period_start) + 1
        if next_idx >= len(rebal_dates):
            break
        period_end = rebal_dates[next_idx]

        tickers = h["tickers"]
        port_cols = [t for t in tickers if t in prices.columns]
        period_prices = prices.loc[period_start:period_end]
        if len(period_prices) < 2:
            continue

        # 포트폴리오 일별 가치 (기간 시작에 거래비용 차감)
        port_sub = period_prices[port_cols].dropna(axis=1, how="any")
        if not port_sub.empty:
            cost = h.get("cost_drag", 0.0)
            daily_returns = port_sub.pct_change().dropna(how="all").mean(axis=1)
            first = True
            for d, r in daily_returns.items():
                if first:
                    # 첫 거래일에 비용 차감 (리밸런싱 실행일)
                    prev_port *= (1 + r) * (1 - cost)
                    first = False
                else:
                    prev_port *= (1 + r)
                port_map[d] = prev_port

        # 벤치마크 일별 가치
        bench_sub = period_prices[bench_cols].dropna(axis=1, how="any")
        if not bench_sub.empty:
            for d, r in bench_sub.pct_change().dropna(how="all").mean(axis=1).items():
                prev_bench *= (1 + r)
                bench_map[d] = prev_bench

    def to_series(m: dict) -> pd.Series:
        return pd.Series(m).sort_index()

    return to_series(port_map), to_series(bench_map)
