"""
백테스트 통계 검증 모듈

두 가지 검증 기법:
  1. Monte Carlo 유의성 검정
     - 귀무가설: 전략 수익률 = 동일 기간 무작위 분기 순열
     - 분기별 수익률을 섞어 n_simulations번 반복 → CAGR/Sharpe 분포 추정
     - 실제 전략의 p-value(단측) 반환

  2. Walk-Forward 검증
     - 슬라이딩 윈도우로 학습/검증 구간을 순차 분리
     - 각 구간에서 독립적으로 run_backtest 실행
     - 기간 과적합 여부 확인
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.engine import _rebalance_dates, run_backtest
from src.backtest.metrics import cagr, sharpe_ratio


def monte_carlo_significance(
    quarterly_returns: pd.Series,
    n_simulations: int = 10_000,
    random_seed: int = 42,
) -> dict:
    """
    분기 수익률 순열 검정으로 전략의 통계적 유의성 평가.

    귀무가설: 관측된 CAGR/Sharpe는 동일 기간 무작위 포트폴리오와 차이 없음.

    Args:
        quarterly_returns: 분기별 포트폴리오 수익률 (pd.Series, index=date)
        n_simulations:     순열 반복 횟수
        random_seed:       재현성을 위한 시드

    Returns:
        dict:
          sim_cagr_mean, sim_cagr_std   — 시뮬레이션 CAGR 분포
          sim_sharpe_mean, sim_sharpe_std
          actual_cagr, actual_sharpe    — 실제 전략 값
          p_value_cagr, p_value_sharpe  — 단측 p-value (낮을수록 유의)
          percentile_cagr               — 실제 전략의 분위수 (높을수록 우수)
    """
    rng = np.random.default_rng(random_seed)
    returns = quarterly_returns.values

    # 실제 전략 지표
    actual_cagr = _quarterly_cagr(returns)
    actual_sharpe = _quarterly_sharpe(returns)

    sim_cagrs: list[float] = []
    sim_sharpes: list[float] = []

    for _ in range(n_simulations):
        shuffled = rng.permutation(returns)
        sim_cagrs.append(_quarterly_cagr(shuffled))
        sim_sharpes.append(_quarterly_sharpe(shuffled))

    sim_cagr_arr = np.array(sim_cagrs)
    sim_sharpe_arr = np.array(sim_sharpes)

    p_val_cagr = float((sim_cagr_arr >= actual_cagr).mean())
    p_val_sharpe = float((sim_sharpe_arr >= actual_sharpe).mean())

    return {
        "actual_cagr":       actual_cagr,
        "actual_sharpe":     actual_sharpe,
        "sim_cagr_mean":     float(sim_cagr_arr.mean()),
        "sim_cagr_std":      float(sim_cagr_arr.std()),
        "sim_sharpe_mean":   float(sim_sharpe_arr.mean()),
        "sim_sharpe_std":    float(sim_sharpe_arr.std()),
        "p_value_cagr":      p_val_cagr,
        "p_value_sharpe":    p_val_sharpe,
        "percentile_cagr":   float((sim_cagr_arr < actual_cagr).mean() * 100),
    }


def walk_forward_backtest(
    prices: pd.DataFrame,
    train_quarters: int = 8,
    test_quarters: int = 4,
    step_quarters: int = 2,
    **backtest_kwargs,
) -> pd.DataFrame:
    """
    Walk-Forward 검증: 슬라이딩 윈도우로 과적합 여부를 확인.

    학습 구간(train_quarters)으로 모델을 선택하지 않고,
    매 검증 구간(test_quarters)을 독립적으로 백테스트해
    시장 국면 변화에 따른 성과 안정성을 평가함.

    Args:
        prices:          일별 종가 DataFrame
        train_quarters:  학습 창 분기 수 (현재는 미사용, 향후 파라미터 최적화 연동용)
        test_quarters:   각 검증 구간의 분기 수
        step_quarters:   슬라이드 간격 (분기 수)
        **backtest_kwargs: run_backtest에 전달할 추가 파라미터

    Returns:
        DataFrame:
          test_start, test_end,
          test_cagr, test_benchmark_cagr, test_alpha,
          test_sharpe, test_mdd, test_win_rate, test_quarters_count
    """
    # 전체 기간의 분기 날짜 생성
    first_valid = prices.index[backtest_kwargs.get("lookback_days", 300)]
    all_rebal = _rebalance_dates(prices, str(first_valid.date()), str(prices.index[-1].date()))

    if len(all_rebal) < train_quarters + test_quarters:
        raise ValueError(
            f"분기 날짜가 부족합니다. 필요: {train_quarters + test_quarters}, "
            f"가용: {len(all_rebal)}"
        )

    rows = []
    # 검증 구간 슬라이드
    test_start_idx = train_quarters  # 학습 창 이후부터 검증 시작
    while test_start_idx + test_quarters <= len(all_rebal):
        test_start = all_rebal[test_start_idx]
        test_end_idx = min(test_start_idx + test_quarters, len(all_rebal) - 1)
        test_end = all_rebal[test_end_idx]

        try:
            result = run_backtest(
                prices=prices,
                start=str(test_start.date()),
                end=str(test_end.date()),
                **backtest_kwargs,
            )
            m = result.metrics
            rows.append({
                "test_start":          test_start.date(),
                "test_end":            test_end.date(),
                "test_cagr":           m.get("cagr", float("nan")),
                "test_benchmark_cagr": m.get("benchmark_cagr", float("nan")),
                "test_alpha":          m.get("alpha", float("nan")),
                "test_sharpe":         m.get("sharpe", float("nan")),
                "test_mdd":            m.get("max_drawdown", float("nan")),
                "test_win_rate":       m.get("win_rate", float("nan")),
                "test_quarters_count": m.get("total_quarters", 0),
            })
        except ValueError:
            pass  # 구간 내 유효한 분기가 없는 경우 스킵

        test_start_idx += step_quarters

    return pd.DataFrame(rows)


def print_walk_forward_summary(wf_df: pd.DataFrame) -> None:
    """Walk-Forward 결과 요약 출력."""
    if wf_df.empty:
        print("Walk-Forward 결과가 없습니다.")
        return

    print(f"\n{'='*70}")
    print(f"  Walk-Forward 검증 결과 ({len(wf_df)}개 구간)")
    print(f"{'='*70}")
    print(f"{'구간':>22}  {'CAGR':>7}  {'Alpha':>7}  {'Sharpe':>7}  {'MDD':>7}  {'WinRate':>8}")
    print(f"{'─'*70}")
    for _, row in wf_df.iterrows():
        period = f"{row['test_start']} ~ {row['test_end']}"
        print(
            f"{period:>22}  "
            f"{row['test_cagr']:>7.1%}  "
            f"{row['test_alpha']:>7.1%}  "
            f"{row['test_sharpe']:>7.2f}  "
            f"{row['test_mdd']:>7.1%}  "
            f"{row['test_win_rate']:>8.1%}"
        )
    print(f"{'─'*70}")
    print(
        f"{'평균':>22}  "
        f"{wf_df['test_cagr'].mean():>7.1%}  "
        f"{wf_df['test_alpha'].mean():>7.1%}  "
        f"{wf_df['test_sharpe'].mean():>7.2f}  "
        f"{wf_df['test_mdd'].mean():>7.1%}  "
        f"{wf_df['test_win_rate'].mean():>8.1%}"
    )
    print(f"{'='*70}\n")


def print_monte_carlo_summary(mc: dict) -> None:
    """Monte Carlo 검정 결과 요약 출력."""
    sig_cagr = "유의 (p<0.05)" if mc["p_value_cagr"] < 0.05 else "비유의"
    sig_sharpe = "유의 (p<0.05)" if mc["p_value_sharpe"] < 0.05 else "비유의"
    print(f"\n{'='*55}")
    print(f"  Monte Carlo 유의성 검정 (n={10_000:,})")
    print(f"{'='*55}")
    print(f"  실제 CAGR    : {mc['actual_cagr']:.1%}  (분위수 {mc['percentile_cagr']:.1f}th)")
    print(f"  시뮬 CAGR    : {mc['sim_cagr_mean']:.1%} ± {mc['sim_cagr_std']:.1%}")
    print(f"  p-value(CAGR): {mc['p_value_cagr']:.4f}  → {sig_cagr}")
    print(f"{'─'*55}")
    print(f"  실제 Sharpe  : {mc['actual_sharpe']:.2f}")
    print(f"  시뮬 Sharpe  : {mc['sim_sharpe_mean']:.2f} ± {mc['sim_sharpe_std']:.2f}")
    print(f"  p-value(Sharpe): {mc['p_value_sharpe']:.4f}  → {sig_sharpe}")
    print(f"{'='*55}\n")


# ── 내부 헬퍼 ──────────────────────────────────────────────────────────

def _quarterly_cagr(returns: np.ndarray) -> float:
    """분기 수익률 배열 → 연환산 CAGR."""
    cum = float(np.prod(1 + returns))
    years = len(returns) / 4.0
    if years <= 0 or cum <= 0:
        return float("nan")
    return float(cum ** (1 / years) - 1)


def _quarterly_sharpe(returns: np.ndarray) -> float:
    """분기 수익률 배열 → 연환산 Sharpe (무위험이자율 0%)."""
    if len(returns) < 2 or np.std(returns) == 0:
        return float("nan")
    return float(np.mean(returns) / np.std(returns) * np.sqrt(4))
