"""
예측 결과 포맷 출력

목표 출력 형태:
  종목    점수   예측수익률  신뢰구간(80%)       유사사례중앙값  모멘텀
  AAPL    88.2    +12.3%    [+4.1%, +20.5%]      +11.8%         강

한계 경고도 함께 출력.
"""

from __future__ import annotations

import pandas as pd

from src.predictor.predictor import PREDICTION_DISCLAIMER
from src.predictor.validator import IC_THRESHOLDS, print_cv_summary

_COL_WIDTHS = {
    "ticker":        8,
    "score":        7,
    "pred_q50":    10,
    "ci":          22,
    "similar":     14,
    "momentum":     6,
}

_HEADER = (
    f"{'종목':<8}  {'점수':>7}  {'예측수익률':>10}  "
    f"{'신뢰구간(80%)':^22}  {'유사사례중앙값':>14}  {'모멘텀':>6}"
)
_SEP = "─" * 75


def _pct(v: float, width: int = 7) -> str:
    """float → '+12.3%' 형태 문자열."""
    if v != v:  # NaN
        return f"{'─':>{width}}"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.1f}%".rjust(width)


def print_prediction_table(
    result: pd.DataFrame,
    mode: str = "clean",
    cv_df: pd.DataFrame | None = None,
) -> None:
    """
    predict_top_n() 결과를 테이블로 출력.

    Args:
        result:  QuarterlyReturnPredictor.predict_top_n() 반환값
        mode:    "clean" | "research"
        cv_df:   validate() 반환값 (있으면 요약 출력)
    """
    score_col = "composite_score" if "composite_score" in result.columns else "technical_score"

    print(f"\n{'='*75}")
    print(f"  다음 분기 수익률 예측  (모드: {mode.upper()})")
    print(f"{'='*75}")
    print(f"  {_HEADER}")
    print(f"  {_SEP}")

    for _, row in result.iterrows():
        ticker  = str(row["ticker"])
        score   = row[score_col]
        q50     = row.get("pred_q50", float("nan"))
        q10     = row.get("pred_q10", float("nan"))
        q90     = row.get("pred_q90", float("nan"))
        similar = row.get("similar_median", float("nan"))
        mom_lbl = row.get("momentum_label", "─")

        score_str   = f"{score:.1f}" if score == score else "─"
        q50_str     = _pct(q50, width=8)
        ci_str      = f"[{_pct(q10, width=6)}, {_pct(q90, width=6)}]"
        similar_str = _pct(similar, width=12)

        print(
            f"  {ticker:<8}  {score_str:>7}  {q50_str:>10}  "
            f"{ci_str:^22}  {similar_str:>14}  {mom_lbl:>6}"
        )

    print(f"  {_SEP}")
    print(f"\n  * {PREDICTION_DISCLAIMER}\n")

    if cv_df is not None and not cv_df.empty:
        print_cv_summary(cv_df)


def print_feature_importance(predictor) -> None:
    """모델 피처 중요도 출력."""
    try:
        imp = predictor.model.get_feature_importance()
    except AssertionError:
        print("  [피처 중요도] 모델이 학습되지 않았습니다.")
        return

    print(f"\n{'─'*45}")
    print("  피처 중요도 (상위 10개)")
    print(f"{'─'*45}")
    for feat, val in imp.head(10).items():
        bar = "█" * int(val / imp.max() * 20)
        print(f"  {feat:<30}  {bar:<20}  {val:.4f}")
    print()
