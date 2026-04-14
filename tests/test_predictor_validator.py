"""
PurgedWalkForwardCV 단위 테스트
"""

import numpy as np
import pandas as pd
import pytest

from src.predictor.feature_builder import CLEAN_FEATURES, FeatureBuilder
from src.predictor.model import ReturnPredictor
from src.predictor.validator import PurgedWalkForwardCV


@pytest.fixture
def multi_quarter_data():
    """10개 분기 × 50종목 결정적 데이터."""
    rng = np.random.default_rng(13)
    dates = pd.date_range("2020-01-01", periods=10, freq="QS")
    rows = []
    for d in dates:
        for i in range(50):
            row = {
                "quarter_date": d,
                "ticker": f"T{i:02d}",
                "next_quarter_return": rng.normal(0.04, 0.12),
            }
            for col in CLEAN_FEATURES:
                row[col] = rng.uniform(0, 100)
            rows.append(row)
    return pd.DataFrame(rows)


class TestPurgedWalkForwardCVSplit:
    def test_split_produces_folds(self, multi_quarter_data):
        cv = PurgedWalkForwardCV(min_train_quarters=4, test_quarters=2, purge_gap=1)
        splits = cv.split(multi_quarter_data)
        assert len(splits) > 0

    def test_no_overlap_train_test(self, multi_quarter_data):
        """학습 인덱스와 테스트 인덱스가 겹치지 않아야 함."""
        cv = PurgedWalkForwardCV(min_train_quarters=4, test_quarters=2, purge_gap=1)
        splits = cv.split(multi_quarter_data)
        for train_idx, test_idx in splits:
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0

    def test_train_grows_each_fold(self, multi_quarter_data):
        """expanding window: 각 fold마다 학습 데이터가 증가해야 함."""
        cv = PurgedWalkForwardCV(min_train_quarters=4, test_quarters=2, purge_gap=1)
        splits = cv.split(multi_quarter_data)
        train_sizes = [len(tr) for tr, _ in splits]
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] >= train_sizes[i - 1]

    def test_purge_gap_enforced(self, multi_quarter_data):
        """purge_gap=1: 학습 마지막 분기와 테스트 첫 분기 사이 1분기 공백 존재 검증."""
        cv = PurgedWalkForwardCV(min_train_quarters=4, test_quarters=2, purge_gap=1)
        splits = cv.split(multi_quarter_data)
        quarters = sorted(multi_quarter_data["quarter_date"].unique())

        for train_idx, test_idx in splits:
            train_dates = multi_quarter_data.loc[train_idx, "quarter_date"]
            test_dates  = multi_quarter_data.loc[test_idx,  "quarter_date"]
            train_last_q = quarters.index(train_dates.max())
            test_first_q = quarters.index(test_dates.min())
            assert test_first_q - train_last_q > 1  # purge_gap=1 → 최소 2 간격

    def test_too_few_quarters_raises(self):
        """분기 수 부족 시 evaluate()에서 ValueError."""
        rng = np.random.default_rng(0)
        dates = pd.date_range("2020-01-01", periods=3, freq="QS")
        rows = [{"quarter_date": d, "ticker": "T0", "next_quarter_return": 0.05,
                 "technical_score": 50.0} for d in dates]
        df = pd.DataFrame(rows)

        fb = FeatureBuilder()
        X, y = fb.build(df, mode="clean", add_cross_sectional_rank=False)
        model = ReturnPredictor(model_type="ridge")

        cv = PurgedWalkForwardCV(min_train_quarters=6, test_quarters=2, purge_gap=1)
        with pytest.raises(ValueError, match="교차검증"):
            cv.evaluate(model, X, y, df)


class TestPurgedWalkForwardCVEvaluate:
    def test_evaluate_returns_dataframe(self, multi_quarter_data):
        fb = FeatureBuilder()
        X, y = fb.build(multi_quarter_data, mode="clean")
        model = ReturnPredictor(model_type="ridge")

        cv = PurgedWalkForwardCV(min_train_quarters=4, test_quarters=2, purge_gap=1)
        result = cv.evaluate(model, X, y, multi_quarter_data)

        assert isinstance(result, pd.DataFrame)
        assert "ic_mean" in result.columns
        assert "mae" in result.columns
        assert "direction_acc" in result.columns

    def test_direction_acc_in_range(self, multi_quarter_data):
        fb = FeatureBuilder()
        X, y = fb.build(multi_quarter_data, mode="clean")
        model = ReturnPredictor(model_type="ridge")

        cv = PurgedWalkForwardCV(min_train_quarters=4, test_quarters=2, purge_gap=1)
        result = cv.evaluate(model, X, y, multi_quarter_data)

        assert result["direction_acc"].between(0, 1).all()

    def test_mae_nonnegative(self, multi_quarter_data):
        fb = FeatureBuilder()
        X, y = fb.build(multi_quarter_data, mode="clean")
        model = ReturnPredictor(model_type="ridge")

        cv = PurgedWalkForwardCV(min_train_quarters=4, test_quarters=2, purge_gap=1)
        result = cv.evaluate(model, X, y, multi_quarter_data)

        assert (result["mae"] >= 0).all()
