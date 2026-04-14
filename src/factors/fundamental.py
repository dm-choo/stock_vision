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

# 지표별 (컬럼명, 방향, 최종점수 가중치, 클리핑 percentile)
FACTOR_CONFIG: list[tuple[str, str, float, tuple[float, float]]] = [
    ("roe",            "higher", 0.20, (0.05, 0.95)),
    ("per",            "lower",  0.20, (0.05, 0.95)),
    ("pbr",            "lower",  0.10, (0.05, 0.95)),
    ("revenue_growth", "higher", 0.20, (0.05, 0.90)),  # 성장률 상단 더 강하게 클리핑
    ("debt_to_equity", "lower",  0.10, (0.05, 0.95)),
    ("fcf_yield",      "higher", 0.10, (0.05, 0.95)),  # FCF 수익률 (실제 현금창출 능력)
    ("op_margin",      "higher", 0.05, (0.05, 0.95)),  # 영업이익률 (비즈니스 수익성)
    ("ev_ebitda",      "lower",  0.05, (0.05, 0.95)),  # EV/EBITDA (부채 중립 밸류에이션)
]

# 음수 값이 무의미한 지표 목록 (음수 → NaN 처리)
_NEGATIVE_INVALID_COLS = ("per", "pbr", "ev_ebitda")


def _sector_relative_ratio(
    df: pd.DataFrame,
    col: str,
    direction: str,
    clip_pct: tuple[float, float] = (0.05, 0.95),
) -> pd.Series:
    """
    섹터 평균 대비 비율 계산.
    - 음수 PER/PBR(적자 종목) 등 의미 없는 값은 NaN 처리.
    - direction='higher': stock / sector_avg
    - direction='lower' : sector_avg / stock
    - 'Unknown' 섹터 종목은 전체 유니버스 평균을 기준으로 비교.
    - clip_pct: 아웃라이어 제거 percentile 범위 (기본 5~95%)
    """
    series = df[col].copy()

    # 음수 값이 무의미한 지표 처리 → NaN
    if col in _NEGATIVE_INVALID_COLS:
        series = series.where(series > 0, other=np.nan)

    sector_avg = df.groupby("sector")[col].transform(
        lambda x: x[x > 0].mean() if col in _NEGATIVE_INVALID_COLS else x.mean()
    )

    # Unknown 섹터: 섹터 평균 대신 전체 유니버스 평균으로 대체
    # (단일 섹터로 묶여 서로만 비교되는 왜곡 방지)
    unknown_mask = df["sector"] == "Unknown"
    if unknown_mask.any():
        if col in _NEGATIVE_INVALID_COLS:
            global_avg = series[series > 0].mean()
        else:
            global_avg = series.mean()
        sector_avg = sector_avg.copy()
        sector_avg[unknown_mask] = global_avg

    # 섹터 평균이 0이거나 NaN이면 비교 불가 → NaN
    sector_avg = sector_avg.replace(0, np.nan)

    if direction == "higher":
        ratio = series / sector_avg
    else:
        ratio = sector_avg / series.replace(0, np.nan)

    # 아웃라이어 제거 (clip_pct 범위 클리핑)
    lo, hi = ratio.quantile(clip_pct[0]), ratio.quantile(clip_pct[1])
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

    for col, direction, weight, clip_pct in FACTOR_CONFIG:
        if col not in result.columns:
            continue

        ratio = _sector_relative_ratio(result, col, direction, clip_pct=clip_pct)
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
    result["valid_factor_count"] = valid_mask.sum(axis=1)

    adjusted_weights = weight_matrix.where(valid_mask, 0)
    weight_sum = adjusted_weights.sum(axis=1).replace(0, np.nan)
    weighted_scores = (score_matrix.fillna(0) * adjusted_weights).sum(axis=1)
    result["fundamental_score"] = weighted_scores / weight_sum

    return result
