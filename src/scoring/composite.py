"""
최종 복합 스코어 산출

fundamental_score와 technical_score는 각각 분산이 다르기 때문에
단순 가중합 시 분산이 큰 쪽이 실제 영향력을 더 갖게 됨.
합산 전 두 점수를 각각 percentile rank(0~100)로 재정규화해
동일한 분산 구조를 보장한 뒤 가중합.

기본 가중치: 펀더멘탈 60%, 기술적 40%
"""

import pandas as pd

MIN_VALID_FACTORS = 3  # 이 수 미만의 유효 팩터를 가진 종목은 결과에서 제외

_OUTPUT_COLS = [
    "rank",
    "ticker",
    "name",
    "sector",
    "composite_score",
    "fundamental_score",
    "technical_score",
    "valid_factor_count",
    # 펀더멘탈 세부
    "roe_score",
    "per_score",
    "pbr_score",
    "revenue_growth_score",
    "debt_to_equity_score",
    # 기술적 세부
    "mom_3m_score",
    "mom_6m_score",
    "rsi_score",
    "ma_score",
    # 원본 지표값
    "per",
    "pbr",
    "roe",
    "revenue_growth",
    "debt_to_equity",
    "mom_3m",
    "mom_6m",
    "rsi",
    "ma_signal",
    "market_cap",
]


def _percentile_rank(series: pd.Series) -> pd.Series:
    """0~100 percentile rank로 재정규화 (분포를 균일하게 만들어 분산 통일)"""
    return series.rank(pct=True, na_option="keep") * 100


def compute_composite_score(
    fundamental_df: pd.DataFrame,
    technical_df: pd.DataFrame,
    fund_weight: float = 0.60,
    tech_weight: float = 0.40,
) -> pd.DataFrame:
    """
    입력:
      fundamental_df  compute_fundamental_scores() 결과
      technical_df    compute_technical_factors() 결과
    출력:
      composite_score 기준 내림차순 정렬된 DataFrame

    fundamental_score, technical_score 원본은 그대로 유지하고,
    합산 시에는 percentile 재정규화된 값을 사용해 분산 차이를 보정.
    """
    merged = fundamental_df.merge(technical_df, on="ticker", how="inner")

    # 유효 팩터 수가 기준 미만인 종목 제외 (데이터 부족으로 신뢰도 낮음)
    if "valid_factor_count" in merged.columns:
        excluded = merged[merged["valid_factor_count"] < MIN_VALID_FACTORS]["ticker"].tolist()
        if excluded:
            print(f"[필터] 유효 팩터 {MIN_VALID_FACTORS}개 미만 제외 ({len(excluded)}개): {excluded}")
        merged = merged[merged["valid_factor_count"] >= MIN_VALID_FACTORS].copy()

    # 분산 보정: 각 점수를 percentile rank로 재정규화 후 가중합
    # (원본 점수는 참고용으로 보존)
    fund_norm = _percentile_rank(merged["fundamental_score"])
    tech_norm = _percentile_rank(merged["technical_score"])
    merged["composite_score"] = fund_norm * fund_weight + tech_norm * tech_weight

    merged["rank"] = (
        merged["composite_score"].rank(ascending=False, method="min").astype(int)
    )

    out_cols = [c for c in _OUTPUT_COLS if c in merged.columns]
    return merged[out_cols].sort_values("rank").reset_index(drop=True)


def print_top_n(result: pd.DataFrame, n: int = 20) -> None:
    """상위 N개 종목 요약 출력"""
    display_cols = ["rank", "ticker", "name", "sector",
                    "composite_score", "fundamental_score", "technical_score"]
    display_cols = [c for c in display_cols if c in result.columns]

    top = result.head(n)[display_cols].copy()
    for col in ["composite_score", "fundamental_score", "technical_score"]:
        if col in top.columns:
            top[col] = top[col].round(1)

    print(f"\n{'=' * 70}")
    print(f"  Top {n} 종목 (composite_score 기준)")
    print(f"{'=' * 70}")
    print(top.to_string(index=False))
    print(f"{'=' * 70}\n")
