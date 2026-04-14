"""
피처 빌더

두 가지 피처셋:
  CLEAN    기술적 신호만 (look-ahead bias 없음) — 기본값
  RESEARCH 기술적 + 현재 펀더멘탈 (look-ahead bias 있음, 연구 목적)

횡단면 순위(cross-sectional rank) 추가:
  분기별로 각 피처의 상대 순위를 추가 피처로 포함.
  절대 점수보다 상대 순위가 시장 국면 변화에 더 robust함.
"""

from __future__ import annotations

import pandas as pd

# CLEAN 피처셋: 기술적 신호만 (look-ahead 없음)
CLEAN_FEATURES = [
    "technical_score",
    "mom_12_1_score",
    "mom_3m_adj_score",
    "mom_6m_score",
    "consistency_score",
    "rsi_score",
    "ma_score",
]

# RESEARCH 피처셋: 현재 펀더멘탈 포함 (look-ahead bias 주의)
_RESEARCH_EXTRA = [
    "composite_score",
    "fundamental_score",
    "roe_score",
    "per_score",
    "pbr_score",
    "revenue_growth_score",
    "fcf_yield_score",
    "op_margin_score",
    "ev_ebitda_score",
    "debt_to_equity_score",
]
RESEARCH_FEATURES = CLEAN_FEATURES + _RESEARCH_EXTRA


class FeatureBuilder:
    """
    학습 데이터 DataFrame에서 피처 행렬 X와 라벨 y를 구성.
    """

    def build(
        self,
        df: pd.DataFrame,
        mode: str = "clean",
        add_cross_sectional_rank: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Args:
            df:   TrainingDataCollector.collect() 결과
            mode: "clean" | "research"
            add_cross_sectional_rank: 분기 내 상대 순위 피처 추가

        Returns:
            X: 피처 DataFrame
            y: 다음 분기 수익률 Series
        """
        base_cols = CLEAN_FEATURES if mode == "clean" else RESEARCH_FEATURES
        available = [c for c in base_cols if c in df.columns]

        X = df[available].copy()

        if add_cross_sectional_rank and "quarter_date" in df.columns:
            for col in available:
                rank_col = f"{col}_rank"
                X[rank_col] = df.groupby("quarter_date")[col].rank(pct=True)

        y = df["next_quarter_return"].copy()
        return X, y

    def build_inference(
        self,
        scores_df: pd.DataFrame,
        mode: str = "clean",
        add_cross_sectional_rank: bool = True,
    ) -> tuple[pd.DataFrame, None]:
        """
        현재 스코어 DataFrame을 예측용 피처 행렬로 변환.
        (라벨 없음, y=None 반환)

        Args:
            scores_df: compute_technical_factors() 또는
                       compute_composite_score() 결과
        """
        base_cols = CLEAN_FEATURES if mode == "clean" else RESEARCH_FEATURES
        available = [c for c in base_cols if c in scores_df.columns]

        X = scores_df[available].copy()

        if add_cross_sectional_rank:
            for col in available:
                rank_col = f"{col}_rank"
                X[rank_col] = X[col].rank(pct=True)

        return X, None
