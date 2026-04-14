"""
Purged Walk-Forward 교차검증

일반 k-fold는 미래 데이터가 학습에 섞이므로 시계열에 절대 사용 불가.
Purged Walk-Forward: expanding 학습 창 + purge_gap + 미래 전용 테스트.

평가 지표:
  IC (Information Coefficient): 분기 내 예측 순위 vs 실제 수익률 순위의 스피어만 상관
    < 0.02  : 예측력 없음
    0.02~0.05: 약한 예측력
    0.05~0.10: 유의미한 예측력 (목표)
    > 0.10  : 강한 예측력 → 과적합/look-ahead 의심 (경고 출력)

  MAE: 수익률 절대 오차 평균 (참고용, 주식 수익률의 노이즈가 커서 낮기 어려움)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

IC_THRESHOLDS = {
    "negligible": 0.02,
    "weak":       0.05,
    "moderate":   0.10,  # IC > 이 값이면 과적합 경고
}


class PurgedWalkForwardCV:
    """
    Expanding-window walk-forward 교차검증.

    Args:
        min_train_quarters: 최소 학습 분기 수 (기본 6 = 약 3,000 샘플)
        test_quarters:      검증 구간 분기 수
        purge_gap:          학습/검증 사이 제거할 분기 수 (라벨 누수 방지)
    """

    def __init__(
        self,
        min_train_quarters: int = 6,
        test_quarters: int = 2,
        purge_gap: int = 1,
    ):
        self.min_train_quarters = min_train_quarters
        self.test_quarters = test_quarters
        self.purge_gap = purge_gap

    def split(
        self, df: pd.DataFrame, date_col: str = "quarter_date"
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        (train_indices, test_indices) 쌍 목록 반환.
        """
        quarters = sorted(df[date_col].unique())
        splits = []

        train_end_idx = self.min_train_quarters - 1
        while True:
            test_start = train_end_idx + 1 + self.purge_gap
            test_end = test_start + self.test_quarters
            if test_end > len(quarters):
                break

            train_quarters = quarters[: train_end_idx + 1]
            test_quarters_range = quarters[test_start:test_end]

            train_mask = df[date_col].isin(train_quarters)
            test_mask = df[date_col].isin(test_quarters_range)

            splits.append((
                df.index[train_mask].to_numpy(),
                df.index[test_mask].to_numpy(),
            ))
            train_end_idx += self.test_quarters

        return splits

    def evaluate(
        self,
        predictor,
        X: pd.DataFrame,
        y: pd.Series,
        df: pd.DataFrame,
        date_col: str = "quarter_date",
    ) -> pd.DataFrame:
        """
        각 fold의 IC, MAE, 방향 정확도를 계산해 DataFrame으로 반환.

        Args:
            predictor: ReturnPredictor 인스턴스 (fit/predict 인터페이스)
            X:         피처 DataFrame (predictor.feature_names 기준)
            y:         라벨 Series (next_quarter_return)
            df:        원본 DataFrame (quarter_date 컬럼 포함)
        """
        splits = self.split(df, date_col)
        if not splits:
            raise ValueError(
                f"교차검증 fold를 만들 수 없습니다. "
                f"최소 {self.min_train_quarters + self.purge_gap + self.test_quarters}개 분기 필요."
            )

        rows = []
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train = X.loc[train_idx]
            y_train = y.loc[train_idx]
            X_test  = X.loc[test_idx]
            y_test  = y.loc[test_idx]

            # 새 인스턴스에 학습 (fold 간 독립)
            from src.predictor.model import ReturnPredictor
            fold_model = ReturnPredictor(model_type=predictor.model_type)
            fold_model.fit(X_train, y_train)
            preds = fold_model.predict(X_test)

            pred_q50 = preds["pred_q50"].values
            actual   = y_test.values

            mae = float(np.abs(pred_q50 - actual).mean())

            # IC: 분기별 스피어만 상관의 평균
            test_df = df.loc[test_idx].copy()
            test_df["pred_q50"] = pred_q50
            test_df["actual"]   = actual

            ic_per_quarter = []
            for _, grp in test_df.groupby(date_col):
                if len(grp) < 5:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rho, _ = spearmanr(grp["pred_q50"], grp["actual"])
                if not np.isnan(rho):
                    ic_per_quarter.append(rho)

            mean_ic = float(np.mean(ic_per_quarter)) if ic_per_quarter else float("nan")
            ic_std  = float(np.std(ic_per_quarter))  if ic_per_quarter else float("nan")

            # 방향 정확도: 예측 부호 vs 실제 부호
            direction_acc = float((np.sign(pred_q50) == np.sign(actual)).mean())

            train_quarters = sorted(df.loc[train_idx, date_col].unique())
            test_quarters  = sorted(df.loc[test_idx,  date_col].unique())

            rows.append({
                "fold":              fold_idx + 1,
                "train_end":         str(train_quarters[-1].date()) if hasattr(train_quarters[-1], "date") else str(train_quarters[-1]),
                "test_start":        str(test_quarters[0].date())  if hasattr(test_quarters[0], "date")  else str(test_quarters[0]),
                "test_end":          str(test_quarters[-1].date()) if hasattr(test_quarters[-1], "date") else str(test_quarters[-1]),
                "n_train":           len(train_idx),
                "n_test":            len(test_idx),
                "ic_mean":           mean_ic,
                "ic_std":            ic_std,
                "mae":               mae,
                "direction_acc":     direction_acc,
            })

        cv_df = pd.DataFrame(rows)

        # IC > 0.10이면 과적합/look-ahead 경고
        mean_ic_overall = cv_df["ic_mean"].mean()
        if mean_ic_overall > IC_THRESHOLDS["moderate"]:
            print(
                f"\n  [경고] 평균 IC={mean_ic_overall:.3f} > {IC_THRESHOLDS['moderate']} — "
                "과적합 또는 look-ahead bias 가능성을 점검하세요."
            )

        return cv_df


def print_cv_summary(cv_df: pd.DataFrame) -> None:
    """Walk-Forward CV 결과 요약 출력."""
    if cv_df.empty:
        print("교차검증 결과가 없습니다.")
        return

    mean_ic  = cv_df["ic_mean"].mean()
    mean_mae = cv_df["mae"].mean()
    mean_dir = cv_df["direction_acc"].mean()

    # IC 등급 판정
    if mean_ic < IC_THRESHOLDS["negligible"]:
        grade = "예측력 없음"
    elif mean_ic < IC_THRESHOLDS["weak"]:
        grade = "약한 예측력"
    elif mean_ic < IC_THRESHOLDS["moderate"]:
        grade = "유의미한 예측력"
    else:
        grade = "강한 예측력 (과적합 점검 필요)"

    print(f"\n{'='*65}")
    print(f"  Walk-Forward CV 결과 ({len(cv_df)}개 fold)")
    print(f"{'='*65}")
    print(f"  {'Fold':>5}  {'학습종료':>12}  {'검증구간':>23}  {'IC':>7}  {'Dir%':>6}")
    print(f"{'─'*65}")
    for _, row in cv_df.iterrows():
        print(
            f"  {int(row['fold']):>5}  "
            f"{row['train_end']:>12}  "
            f"{row['test_start']} ~ {row['test_end']}  "
            f"{row['ic_mean']:>7.3f}  "
            f"{row['direction_acc']:>6.1%}"
        )
    print(f"{'─'*65}")
    print(f"  평균 IC : {mean_ic:.3f}  → {grade}")
    print(f"  평균 MAE: {mean_mae:.3f}  (분기 수익률 절대 오차)")
    print(f"  방향 정확도: {mean_dir:.1%}")
    print(f"{'='*65}\n")
