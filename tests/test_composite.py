"""
src/scoring/composite.py 단위 테스트

검증 항목:
  1. MIN_VALID_FACTORS 필터 — valid_factor_count < 3인 종목 제외
  2. percentile 재정규화 — 합산 전 두 점수의 분산이 동일해야 함
  3. composite_score 범위 — 0~100
  4. 순위 일관성 — composite_score 높을수록 rank 낮아야 함
  5. 전체 파이프라인 통합 — fixtures 데이터로 end-to-end 실행
"""

import numpy as np
import pandas as pd
import pytest  # noqa: F401

from src.factors.fundamental import compute_fundamental_scores
from src.factors.technical import compute_technical_factors
from src.scoring.composite import (
    MIN_VALID_FACTORS,
    _percentile_rank,
    compute_composite_score,
)


# ── percentile 재정규화 ───────────────────────────────────────────────

def test_percentile_rank_uniform_distribution():
    """
    percentile rank 후 두 시리즈의 std가 동일해야 한다 (분산 보정 효과).
    """
    rng = np.random.default_rng(0)
    narrow = pd.Series(rng.normal(50, 5, 200))   # std ≈ 5
    wide = pd.Series(rng.normal(50, 25, 200))    # std ≈ 25

    narrow_norm = _percentile_rank(narrow)
    wide_norm = _percentile_rank(wide)

    assert narrow_norm.std() == pytest.approx(wide_norm.std(), rel=0.05), \
        "percentile 정규화 후 두 시리즈의 분산이 동일해야 함"


def test_percentile_rank_preserves_order():
    """percentile rank는 원본 순서를 보존해야 한다."""
    s = pd.Series([10.0, 30.0, 20.0, 40.0])
    ranked = _percentile_rank(s)
    assert ranked.iloc[0] < ranked.iloc[2] < ranked.iloc[1] < ranked.iloc[3]


# ── MIN_VALID_FACTORS 필터 ────────────────────────────────────────────

def test_min_valid_factors_filter(sample_fundamentals, sample_prices):
    """valid_factor_count < MIN_VALID_FACTORS인 종목은 최종 결과에 없어야 한다."""
    fund = compute_fundamental_scores(sample_fundamentals)
    tech = compute_technical_factors(sample_prices)
    result = compute_composite_score(fund, tech)

    low_factor_tickers = fund.loc[
        fund["valid_factor_count"] < MIN_VALID_FACTORS, "ticker"
    ].tolist()

    for ticker in low_factor_tickers:
        assert ticker not in result["ticker"].values, \
            f"{ticker}(유효팩터 부족)이 최종 결과에 포함되면 안 됨"


def test_sufficient_factor_tickers_included(sample_fundamentals, sample_prices):
    """valid_factor_count >= MIN_VALID_FACTORS이고 가격 데이터가 있는 종목은 결과에 포함되어야 한다."""
    fund = compute_fundamental_scores(sample_fundamentals)
    tech = compute_technical_factors(sample_prices)
    result = compute_composite_score(fund, tech)

    # TECH1, TECH2, ENRG1, ENRG2 — 모두 팩터 5개, 가격 250일 보유
    for ticker in ["TECH1", "TECH2", "ENRG1", "ENRG2"]:
        assert ticker in result["ticker"].values, \
            f"{ticker}는 최종 결과에 포함되어야 함"


# ── 점수 범위 및 순위 일관성 ─────────────────────────────────────────

def test_composite_score_range(sample_fundamentals, sample_prices):
    """composite_score는 0~100 범위 내에 있어야 한다."""
    fund = compute_fundamental_scores(sample_fundamentals)
    tech = compute_technical_factors(sample_prices)
    result = compute_composite_score(fund, tech)

    scores = result["composite_score"].dropna()
    assert (scores >= 0).all() and (scores <= 100).all(), \
        f"점수 범위 초과: min={scores.min():.2f}, max={scores.max():.2f}"


def test_rank_consistent_with_score(sample_fundamentals, sample_prices):
    """composite_score가 높을수록 rank 값이 작아야 한다 (1등이 최고점)."""
    fund = compute_fundamental_scores(sample_fundamentals)
    tech = compute_technical_factors(sample_prices)
    result = compute_composite_score(fund, tech)

    sorted_by_score = result.sort_values("composite_score", ascending=False)
    assert list(sorted_by_score["rank"]) == sorted(result["rank"].tolist()), \
        "rank는 composite_score 내림차순과 일치해야 함"


def test_rank_bounds(sample_fundamentals, sample_prices):
    """rank는 1 이상 n 이하 범위 내에 있어야 한다 (동점 허용)."""
    fund = compute_fundamental_scores(sample_fundamentals)
    tech = compute_technical_factors(sample_prices)
    result = compute_composite_score(fund, tech)

    n = len(result)
    assert result["rank"].min() == 1, "최솟값은 1이어야 함"
    assert result["rank"].max() <= n, f"최댓값은 {n} 이하여야 함"


# ── 가중치 파라미터 변경 ─────────────────────────────────────────────

def test_custom_weights_change_scores():
    """
    fundamental과 technical 순위가 서로 반대인 데이터에서
    가중치 변경 시 composite_score 순서가 바뀌어야 한다.
    """
    # 펀더멘탈: A > B, 기술적: B > A 가 되도록 점수를 직접 주입
    fund_df = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "name": ["Alpha", "Beta"],
            "sector": ["Tech", "Tech"],
            "fundamental_score": [90.0, 10.0],
            "valid_factor_count": [5, 5],
        }
    )
    tech_df = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "technical_score": [10.0, 90.0],
        }
    )

    fund_only = compute_composite_score(fund_df, tech_df, fund_weight=1.0, tech_weight=0.0)
    tech_only = compute_composite_score(fund_df, tech_df, fund_weight=0.0, tech_weight=1.0)

    top_fund = fund_only.iloc[0]["ticker"]
    top_tech = tech_only.iloc[0]["ticker"]
    assert top_fund == "A", "펀더멘탈 100% 가중치면 A가 1위여야 함"
    assert top_tech == "B", "기술적 100% 가중치면 B가 1위여야 함"
