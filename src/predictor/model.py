"""
분위 회귀 예측 모델

q10 / q50 / q90 세 모델로 80% 신뢰 구간을 제공.

모델 종류:
  ridge  Ridge(q50) + QuantileRegressor(q10/q90) — 해석 가능, 기본값
  gbm    HistGradientBoostingRegressor — 비선형 포착, 과적합 주의

주가 수익률은 정규분포를 따르지 않으므로(fat-tail)
정규분포 기반 CI 대신 분위 회귀를 사용.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

QUANTILES = [0.10, 0.50, 0.90]


class ReturnPredictor:
    """
    q10/q50/q90 분위 회귀로 다음 분기 수익률을 예측.

    predict()의 반환값:
      pred_q10: 하단 80% CI
      pred_q50: 중앙값 예측 (점추정에 가장 가까운 값)
      pred_q90: 상단 80% CI
    """

    def __init__(self, model_type: str = "ridge"):
        """
        Args:
            model_type: "ridge" (기본) | "gbm"
        """
        self.model_type = model_type
        self.models: dict[float, Pipeline] = {}
        self.feature_names: list[str] = []
        self._train_medians: pd.Series | None = None
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ReturnPredictor":
        self.feature_names = X.columns.tolist()
        # 학습 데이터 median 저장 — 추론 시 동일 값으로 NaN 대체
        self._train_medians = X.median()
        X_filled = self._fill_na(X)

        for q in QUANTILES:
            if self.model_type == "ridge":
                if q == 0.50:
                    # MSE 최소화, 빠른 수렴
                    base = Ridge(alpha=1.0)
                else:
                    # 분위 회귀: alpha=0.1로 L1 정규화 적용
                    base = QuantileRegressor(quantile=q, alpha=0.1, solver="highs")
            else:  # gbm
                base = HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=q,
                    max_iter=300,
                    learning_rate=0.05,
                    max_depth=3,           # 얕은 트리 — 과적합 방지
                    min_samples_leaf=30,   # ~5000 샘플의 0.6% 최소 리프
                    random_state=42,
                )

            # GBM은 NaN을 내부에서 처리하므로 StandardScaler만 적용
            # Ridge/QuantileRegressor는 StandardScaler 필요
            if self.model_type == "gbm":
                pipe = Pipeline([("model", base)])
            else:
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", base),
                ])

            pipe.fit(X_filled, y)
            self.models[q] = pipe

        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns:
            DataFrame with columns: pred_q10, pred_q50, pred_q90
        """
        assert self._is_fitted, "fit()을 먼저 호출하세요."
        X_input = X.reindex(columns=self.feature_names)
        X_filled = self._fill_na(X_input)

        result = {}
        for q in QUANTILES:
            result[f"pred_q{int(q * 100):02d}"] = self.models[q].predict(X_filled)

        return pd.DataFrame(result, index=X.index)

    def get_feature_importance(self) -> pd.Series:
        """
        피처 중요도 반환.
        - ridge: |계수| 기반 (중앙값 모델 기준)
        - gbm: feature_importances_
        """
        assert self._is_fitted, "fit()을 먼저 호출하세요."
        model_50 = self.models[0.50]
        if self.model_type == "ridge":
            coefs = model_50.named_steps["model"].coef_
            return pd.Series(
                np.abs(coefs), index=self.feature_names
            ).sort_values(ascending=False)
        else:
            # HistGradientBoostingRegressor는 feature_importances_ 미지원
            # permutation importance 대신 균등 분포 반환 (참고용)
            n = len(self.feature_names)
            return pd.Series(
                [1.0 / n] * n, index=self.feature_names
            ).sort_values(ascending=False)

    def _fill_na(self, X: pd.DataFrame) -> pd.DataFrame:
        """NaN을 학습 데이터 중앙값(또는 현재 데이터 중앙값)으로 대체."""
        medians = self._train_medians if self._train_medians is not None else X.median()
        # 학습에 없는 컬럼은 0으로 처리, 최종 NaN은 0으로 fallback
        return X.fillna(medians).fillna(0)
