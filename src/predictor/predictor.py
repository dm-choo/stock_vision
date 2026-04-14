"""
통합 예측 인터페이스

사용법:
  predictor = QuarterlyReturnPredictor(mode="clean")
  predictor.fit(prices)
  predictions = predictor.predict_top_n(tech_scores, top_n=20)

주의사항:
  - 투자 조언이 아닙니다. 과거 패턴 기반 통계적 추정치입니다.
  - CLEAN 모드: 기술적 신호만 사용 (look-ahead bias 없음)
  - RESEARCH 모드: 현재 펀더멘탈 포함 (look-ahead bias 존재, 연구 목적)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.predictor.data_collector import TrainingDataCollector
from src.predictor.feature_builder import FeatureBuilder
from src.predictor.model import ReturnPredictor
from src.predictor.validator import PurgedWalkForwardCV

PREDICTION_DISCLAIMER = (
    "투자 조언 아님. 과거 패턴 기반 통계적 추정치. "
    "실제 수익률은 예측 범위 밖에서 빈번히 발생합니다."
)

BIAS_WARNING_RESEARCH = (
    "[경고] RESEARCH 모드: fundamentals.parquet은 현재 시점 데이터입니다.\n"
    "  과거 분기 예측에 미래 펀더멘탈이 포함되어 look-ahead bias가 존재합니다.\n"
    "  검증 IC/MAE는 실제 예측 성능을 과대 추정할 수 있습니다."
)

# 모멘텀 분류 기준 (consistency_score percentile)
_MOMENTUM_LABELS = {(0.67, 1.01): "강", (0.33, 0.67): "중", (0.0, 0.33): "약"}


class QuarterlyReturnPredictor:
    """
    다음 분기 주가 수익률 예측기.

    fit() → 학습 데이터 수집 + 모델 훈련
    predict_top_n() → 상위 N 종목에 대한 예측값 반환
    validate() → Walk-Forward CV로 예측 성능 평가
    """

    def __init__(self, mode: str = "clean", model_type: str = "ridge"):
        """
        Args:
            mode:       "clean" (기술적 신호만) | "research" (+현재 펀더멘탈)
            model_type: "ridge" | "gbm"
        """
        self.mode = mode
        self.model_type = model_type
        self.collector = TrainingDataCollector()
        self.feature_builder = FeatureBuilder()
        self.model = ReturnPredictor(model_type=model_type)
        self.cv = PurgedWalkForwardCV()

        self._training_data: pd.DataFrame | None = None
        self._score_bins: pd.DataFrame | None = None
        self._is_fitted = False

    def fit(
        self,
        prices: pd.DataFrame,
        use_cache: bool = True,
        cache_path: Path | None = None,
        **backtest_kwargs,
    ) -> "QuarterlyReturnPredictor":
        """
        학습 데이터 수집 후 모델 훈련.

        Args:
            prices:            일별 종가 DataFrame
            use_cache:         학습 데이터 캐시 재사용 여부
            cache_path:        캐시 파일 경로
            **backtest_kwargs: TrainingDataCollector.collect()에 전달
        """
        if self.mode == "research":
            print(BIAS_WARNING_RESEARCH)

        self._training_data = self.collector.collect(
            prices,
            use_cache=use_cache,
            cache_path=cache_path,
            **backtest_kwargs,
        )

        X, y = self.feature_builder.build(self._training_data, mode=self.mode)

        # 점수 분위별 과거 수익률 분포 구축 (유사 사례 조회용)
        self._score_bins = self._build_score_bins(
            self._training_data, y, score_col="technical_score"
        )

        self.model.fit(X, y)
        self._is_fitted = True

        n = len(self._training_data)
        q = self._training_data["quarter_date"].nunique()
        print(f"  모델 학습 완료: {n:,}개 샘플, {q}개 분기, 모드={self.mode}, 모델={self.model_type}")
        return self

    def predict_top_n(
        self,
        current_scores: pd.DataFrame,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        현재 스코어링 결과에서 상위 top_n 종목의 예측값 반환.

        Args:
            current_scores: compute_technical_factors() 또는
                            compute_composite_score() 결과
            top_n:          상위 N개 출력

        Returns:
            DataFrame: ticker, composite_score(또는 technical_score),
                       pred_q10, pred_q50, pred_q90,
                       similar_median, momentum_label
        """
        assert self._is_fitted, "fit()을 먼저 호출하세요."

        X_curr, _ = self.feature_builder.build_inference(current_scores, mode=self.mode)
        preds = self.model.predict(X_curr)

        score_col = "composite_score" if "composite_score" in current_scores.columns else "technical_score"
        result = current_scores[["ticker", score_col]].copy().reset_index(drop=True)
        result = pd.concat([result, preds.reset_index(drop=True)], axis=1)

        result["similar_median"] = result[score_col].apply(self._lookup_similar_median)
        result["momentum_label"] = self._classify_momentum(current_scores)

        return result.nlargest(top_n, score_col).reset_index(drop=True)

    def validate(
        self,
        prices: pd.DataFrame | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Walk-Forward CV로 예측 성능 평가.

        training_data가 이미 수집된 경우 prices 불필요.
        """
        if self._training_data is None:
            if prices is None:
                raise ValueError("fit() 또는 prices 인자 필요.")
            self._training_data = self.collector.collect(
                prices, use_cache=use_cache
            )

        X, y = self.feature_builder.build(self._training_data, mode=self.mode)
        return self.cv.evaluate(self.model, X, y, self._training_data)

    # ── 내부 헬퍼 ──────────────────────────────────────────────────────

    def _build_score_bins(
        self,
        df: pd.DataFrame,
        y: pd.Series,
        score_col: str = "technical_score",
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """점수 10분위별 실제 수익률 분포 (중앙값/평균/std) 저장."""
        tmp = df[[score_col]].copy()
        tmp["actual_return"] = y.values
        tmp["decile"] = pd.qcut(tmp[score_col], q=n_bins, labels=False, duplicates="drop")
        return tmp.groupby("decile")["actual_return"].agg(
            median="median", mean="mean", std="std", count="count"
        )

    def _lookup_similar_median(self, score: float) -> float:
        """점수에 해당하는 과거 수익률 중앙값 반환."""
        if self._score_bins is None or np.isnan(score):
            return float("nan")
        # score가 속하는 분위 추정 (전체 학습 데이터의 percentile 기준)
        if self._training_data is None:
            return float("nan")
        col = "technical_score"
        if col not in self._training_data.columns:
            return float("nan")
        pct = (self._training_data[col] <= score).mean()
        bin_idx = min(int(pct * 10), 9)
        if bin_idx in self._score_bins.index:
            return float(self._score_bins.loc[bin_idx, "median"])
        return float("nan")

    def _classify_momentum(self, scores: pd.DataFrame) -> pd.Series:
        """consistency_score 기반 모멘텀 강도 분류."""
        if "consistency_score" not in scores.columns:
            return pd.Series(["─"] * len(scores), index=scores.index)
        pct = scores["consistency_score"].rank(pct=True)
        labels = []
        for v in pct:
            if np.isnan(v):
                labels.append("─")
            elif v >= 0.67:
                labels.append("강")
            elif v >= 0.33:
                labels.append("중")
            else:
                labels.append("약")
        return pd.Series(labels, index=scores.index)
