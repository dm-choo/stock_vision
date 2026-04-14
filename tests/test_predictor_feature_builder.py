"""
FeatureBuilder 단위 테스트
"""

import numpy as np
import pandas as pd
import pytest

from src.predictor.feature_builder import (
    CLEAN_FEATURES,
    RESEARCH_FEATURES,
    FeatureBuilder,
)


@pytest.fixture
def training_df():
    """결정적 학습 DataFrame (3개 분기 × 10종목)."""
    rng = np.random.default_rng(7)
    dates = pd.to_datetime(["2024-01-01", "2024-04-01", "2024-07-01"])
    rows = []
    for d in dates:
        for i in range(10):
            row = {"quarter_date": d, "ticker": f"T{i:02d}", "next_quarter_return": rng.normal(0.05, 0.15)}
            for col in RESEARCH_FEATURES:
                row[col] = rng.uniform(0, 100)
            rows.append(row)
    return pd.DataFrame(rows)


class TestFeatureBuilderBuild:
    def test_clean_mode_columns(self, training_df):
        fb = FeatureBuilder()
        X, y = fb.build(training_df, mode="clean")
        # 기본 컬럼 + rank 컬럼
        base = [c for c in CLEAN_FEATURES if c in training_df.columns]
        assert all(c in X.columns for c in base)

    def test_research_mode_has_extra_columns(self, training_df):
        fb = FeatureBuilder()
        X_clean, _ = fb.build(training_df, mode="clean")
        X_research, _ = fb.build(training_df, mode="research")
        assert X_research.shape[1] > X_clean.shape[1]

    def test_cross_sectional_rank_added(self, training_df):
        fb = FeatureBuilder()
        X, _ = fb.build(training_df, mode="clean", add_cross_sectional_rank=True)
        rank_cols = [c for c in X.columns if c.endswith("_rank")]
        assert len(rank_cols) > 0

    def test_no_rank_when_disabled(self, training_df):
        fb = FeatureBuilder()
        X, _ = fb.build(training_df, mode="clean", add_cross_sectional_rank=False)
        rank_cols = [c for c in X.columns if c.endswith("_rank")]
        assert len(rank_cols) == 0

    def test_rank_values_in_0_1(self, training_df):
        """rank(pct=True) 결과는 (0, 1] 범위여야 함."""
        fb = FeatureBuilder()
        X, _ = fb.build(training_df, mode="clean", add_cross_sectional_rank=True)
        rank_cols = [c for c in X.columns if c.endswith("_rank")]
        for col in rank_cols:
            assert X[col].between(0, 1, inclusive="right").all(), f"{col} 범위 오류"

    def test_label_shape(self, training_df):
        fb = FeatureBuilder()
        X, y = fb.build(training_df, mode="clean")
        assert len(X) == len(y) == len(training_df)

    def test_missing_columns_skipped(self):
        """RESEARCH_FEATURES 중 없는 컬럼은 조용히 무시해야 함."""
        df = pd.DataFrame({
            "quarter_date": pd.to_datetime(["2024-01-01"] * 5),
            "ticker": [f"T{i}" for i in range(5)],
            "technical_score": [60, 70, 55, 80, 65],
            "next_quarter_return": [0.05, 0.10, -0.02, 0.08, 0.03],
        })
        fb = FeatureBuilder()
        X, y = fb.build(df, mode="clean")
        assert "technical_score" in X.columns
        assert len(X) == 5


class TestFeatureBuilderBuildInference:
    def test_inference_returns_none_label(self, training_df):
        fb = FeatureBuilder()
        X, y = fb.build_inference(training_df, mode="clean")
        assert y is None

    def test_inference_shape_matches(self, training_df):
        fb = FeatureBuilder()
        X_train, _ = fb.build(training_df, mode="clean")
        X_infer, _ = fb.build_inference(training_df, mode="clean")
        # 같은 피처셋 (rank 방식만 다름: 분기 내 vs 전체)
        base_cols = [c for c in X_train.columns if not c.endswith("_rank")]
        for col in base_cols:
            assert col in X_infer.columns
