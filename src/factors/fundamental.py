"""
섹터 상대 펀더멘탈 스코어 계산

각 지표를 절대값이 아닌 '섹터 평균 대비 비율'로 정규화한 뒤,
전체 유니버스 내 percentile rank(0~100)로 변환한다.

지표별 방향성:
  ROE          높을수록 좋음 → stock / sector_avg
  PER          낮을수록 좋음(저평가) → sector_avg / stock
  PBR          낮을수록 좋음 → sector_avg / stock
  revenue_growth  높을수록 좋음 → stock / sector_avg
  debt_to_equity  낮을수록 좋음 → sector_avg / stock
"""

import numpy as np
import pandas as pd

# 지표별 (컬럼명, 방향, 최종점수 가중치)
FACTOR_CONFIG = [
    ("roe", "higher", 0.25),
    ("per", "lower", 0.25),
    ("pbr", "lower", 0.15),
    ("revenue_growth", "higher", 0.25),
    ("debt_to_equity", "lower", 0.10),
]


def _sector_relative_ratio(df: pd.DataFrame, col: str, direction: str) -> pd.Series:
    """
    섹터 평균 대비 비율 계산.
    - 음수 PER/PBR(적자 종목) 등 의미 없는 값은 NaN 처리.
    - direction='higher': stock / sector_avg
    - direction='lower' : sector_avg / stock
    """
    series = df[col].copy()

    # PER, PBR은 음수면 무의미 → NaN
    if col in ("per", "pbr"):
        series = series.where(series > 0, other=np.nan)

    sector_avg = df.groupby("sector")[col].transform(
        lambda x: x[x > 0].mean() if col in ("per", "pbr") else x.mean()
    )

    # 섹터 평균이 0이거나 NaN이면 비교 불가 → NaN
    sector_avg = sector_avg.replace(0, np.nan)

    if direction == "higher":
        ratio = series / sector_avg
    else:
        ratio = sector_avg / series.replace(0, np.nan)

    # 극단적 아웃라이어 제거 (1~99 percentile 클리핑)
    lo, hi = ratio.quantile(0.01), ratio.quantile(0.99)
    return ratio.clip(lo, hi)


def compute_fundamental_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    입력: 펀더멘탈 DataFrame (ticker, sector, per, pbr, roe, revenue_growth, debt_to_equity, ...)
    출력: 원본 컬럼 + 각 지표 점수(_score) + fundamental_score(가중합, 0~100)

    섹터 정보가 없는 종목은 'Unknown' 섹터로 분류해 처리.
    """
    result = df.copy()
    result["sector"] = result["sector"].fillna("Unknown")

    score_cols = []
    weights = []

    for col, direction, weight in FACTOR_CONFIG:
        if col not in result.columns:
            continue

        ratio = _sector_relative_ratio(result, col, direction)
        score_col = f"{col}_score"

        # percentile rank → 0~100 점수
        result[score_col] = ratio.rank(pct=True, na_option="keep") * 100
        score_cols.append(score_col)
        weights.append(weight)

    if not score_cols:
        result["fundamental_score"] = np.nan
        return result

    # 가중치 정규화 (일부 컬럼 누락 시 재조정)
    total_w = sum(weights)
    norm_weights = [w / total_w for w in weights]

    # NaN이 있는 행은 유효한 점수만으로 가중평균
    score_matrix = result[score_cols]
    weight_matrix = pd.DataFrame(
        {col: w for col, w in zip(score_cols, norm_weights)},
        index=result.index,
    )
    # NaN 위치의 가중치를 0으로 → 유효한 팩터만 합산 후 재정규화
    valid_mask = score_matrix.notna()
    adjusted_weights = weight_matrix.where(valid_mask, 0)
    weight_sum = adjusted_weights.sum(axis=1).replace(0, np.nan)
    weighted_scores = (score_matrix.fillna(0) * adjusted_weights).sum(axis=1)
    result["fundamental_score"] = weighted_scores / weight_sum

    return result
