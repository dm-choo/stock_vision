"""
ReturnPredictor 단위 테스트

외부 API 없이 결정적 데이터로 모델 로직만 검증.
"""

import numpy as np
import pandas as pd
import pytest

from src.predictor.model import QUANTILES, ReturnPredictor


@pytest.fixture
def toy_data():
    """결정적 학습 데이터 (200 샘플 × 4 피처)."""
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({
        "f1": rng.normal(0, 1, n),
        "f2": rng.normal(0, 1, n),
        "f3": rng.uniform(0, 1, n),
        "f4": rng.uniform(0, 1, n),
    })
    y = pd.Series(0.3 * X["f1"] - 0.2 * X["f2"] + rng.normal(0, 0.1, n))
    return X, y


class TestReturnPredictorRidge:
    def test_fit_returns_self(self, toy_data):
        X, y = toy_data
        model = ReturnPredictor(model_type="ridge")
        result = model.fit(X, y)
        assert result is model

    def test_predict_shape(self, toy_data):
        X, y = toy_data
        model = ReturnPredictor(model_type="ridge").fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X), 3)
        assert list(preds.columns) == ["pred_q10", "pred_q50", "pred_q90"]

    def test_quantile_ordering(self, toy_data):
        """q10 ≤ q50 ≤ q90 이 전체 샘플의 80% 이상에서 성립해야 함."""
        X, y = toy_data
        model = ReturnPredictor(model_type="ridge").fit(X, y)
        preds = model.predict(X)
        ordered = (preds["pred_q10"] <= preds["pred_q50"]) & (preds["pred_q50"] <= preds["pred_q90"])
        assert ordered.mean() >= 0.80

    def test_predict_before_fit_raises(self):
        X = pd.DataFrame({"f1": [1.0, 2.0]})
        model = ReturnPredictor(model_type="ridge")
        with pytest.raises(AssertionError):
            model.predict(X)

    def test_feature_names_stored(self, toy_data):
        X, y = toy_data
        model = ReturnPredictor(model_type="ridge").fit(X, y)
        assert model.feature_names == list(X.columns)

    def test_nan_input_handled(self, toy_data):
        """NaN을 중앙값으로 대체하므로 예측이 실패하지 않아야 함."""
        X, y = toy_data
        model = ReturnPredictor(model_type="ridge").fit(X, y)
        X_nan = X.copy()
        X_nan.iloc[0, 0] = np.nan
        preds = model.predict(X_nan)
        assert not preds.isna().any().any()

    def test_feature_importance_returns_series(self, toy_data):
        X, y = toy_data
        model = ReturnPredictor(model_type="ridge").fit(X, y)
        imp = model.get_feature_importance()
        assert isinstance(imp, pd.Series)
        assert len(imp) == X.shape[1]
        assert (imp >= 0).all()

    def test_unknown_column_in_inference(self, toy_data):
        """추론 시 학습에 없던 컬럼은 NaN으로 처리되어야 함."""
        X, y = toy_data
        model = ReturnPredictor(model_type="ridge").fit(X, y)
        X_extra = X.copy()
        X_extra["unknown_col"] = 999.0
        # reindex drops extra column, so predict should still work
        preds = model.predict(X_extra)
        assert preds.shape[0] == len(X_extra)


class TestReturnPredictorGBM:
    def test_fit_and_predict(self, toy_data):
        X, y = toy_data
        model = ReturnPredictor(model_type="gbm").fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X), 3)

    def test_feature_importance_gbm(self, toy_data):
        """GBM은 feature_importances_ 미지원 — 균등 분포 반환."""
        X, y = toy_data
        model = ReturnPredictor(model_type="gbm").fit(X, y)
        imp = model.get_feature_importance()
        assert len(imp) == X.shape[1]
        assert abs(imp.sum() - 1.0) < 1e-6
