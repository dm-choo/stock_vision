"""
src/factors/fundamental.py 단위 테스트

검증 항목:
  1. 음수 PER/PBR → per_score / pbr_score NaN 처리
  2. Unknown 섹터 → 전체 유니버스 평균으로 비교 (섹터 내 단독 비교 금지)
  3. valid_factor_count → 실제 NaN이 아닌 팩터 수 정확히 집계
  4. NaN이 많은 종목(NODATA)의 가중 평균 → 유효 팩터만으로 재정규화
  5. 섹터 상대 방향성 → ROE 높은 종목이 같은 섹터 내에서 높은 점수
"""

import numpy as np
import pandas as pd
import pytest

from src.factors.fundamental import FACTOR_CONFIG, compute_fundamental_scores


def test_negative_per_yields_nan_score(sample_fundamentals):
    """음수 PER 종목(BADPE)은 per_score가 NaN이어야 한다."""
    result = compute_fundamental_scores(sample_fundamentals)
    badpe = result[result["ticker"] == "BADPE"]
    assert badpe["per_score"].isna().all(), "음수 PER은 per_score NaN 처리되어야 함"


def test_unknown_sector_gets_score(sample_fundamentals):
    """Unknown 섹터 종목(UNKNWN)은 NaN이 아닌 fundamental_score를 가져야 한다."""
    result = compute_fundamental_scores(sample_fundamentals)
    unknwn = result[result["ticker"] == "UNKNWN"]
    assert unknwn["fundamental_score"].notna().all(), \
        "Unknown 섹터 종목도 전체 평균 기준으로 점수가 산출되어야 함"


def test_unknown_sector_differs_from_isolated_group():
    """
    Unknown 섹터가 2개일 때 서로만 비교하면 점수가 50 근처에 고정됨.
    전체 평균 기준 비교 시에는 상대 성과에 따라 점수가 분산되어야 함.
    """
    df = pd.DataFrame(
        {
            "ticker": ["U1", "U2", "ANCHOR"],
            "sector": ["Unknown", "Unknown", "Technology"],
            "per": [10.0, 40.0, 20.0],
            "pbr": [1.0, 4.0, 2.0],
            "roe": [0.30, 0.05, 0.15],
            "revenue_growth": [0.20, 0.02, 0.10],
            "debt_to_equity": [10.0, 100.0, 50.0],
        }
    )
    result = compute_fundamental_scores(df)
    u1_score = result.loc[result["ticker"] == "U1", "fundamental_score"].values[0]
    u2_score = result.loc[result["ticker"] == "U2", "fundamental_score"].values[0]
    # U1이 모든 지표에서 우세하므로 U2보다 점수가 높아야 함
    assert u1_score > u2_score, \
        "전체 평균 기준 비교 시 지표가 우세한 Unknown 종목이 더 높은 점수를 받아야 함"


def test_valid_factor_count_nodata(sample_fundamentals):
    """
    pbr·roe·revenue_growth·debt_to_equity 모두 NaN인 NODATA 종목은
    유효 팩터 수가 1 (per 만 유효) 이어야 한다.
    conftest 픽스처에서 NODATA.pbr = NaN으로 설정됨.
    """
    result = compute_fundamental_scores(sample_fundamentals)
    nodata = result[result["ticker"] == "NODATA"]
    assert nodata["valid_factor_count"].values[0] == 1


def test_valid_factor_count_full_data(sample_fundamentals):
    """데이터가 완전한 TECH1 종목은 valid_factor_count가 5여야 한다."""
    result = compute_fundamental_scores(sample_fundamentals)
    tech1 = result[result["ticker"] == "TECH1"]
    assert tech1["valid_factor_count"].values[0] == 5


def test_higher_roe_gets_higher_roe_score(sample_fundamentals):
    """같은 섹터(Technology)에서 ROE가 더 높은 TECH1이 TECH2보다 roe_score가 높아야 한다."""
    result = compute_fundamental_scores(sample_fundamentals)
    roe_tech1 = result.loc[result["ticker"] == "TECH1", "roe_score"].values[0]
    roe_tech2 = result.loc[result["ticker"] == "TECH2", "roe_score"].values[0]
    assert roe_tech1 > roe_tech2, "ROE가 높은 종목이 같은 섹터 내에서 더 높은 roe_score를 가져야 함"


def test_lower_per_gets_higher_per_score(sample_fundamentals):
    """같은 섹터(Energy)에서 PER이 더 낮은 ENRG1이 ENRG2보다 per_score가 높아야 한다."""
    result = compute_fundamental_scores(sample_fundamentals)
    per_enrg1 = result.loc[result["ticker"] == "ENRG1", "per_score"].values[0]
    per_enrg2 = result.loc[result["ticker"] == "ENRG2", "per_score"].values[0]
    assert per_enrg1 > per_enrg2, "PER이 낮은 종목(저평가)이 더 높은 per_score를 가져야 함"


def test_fundamental_score_range(sample_fundamentals):
    """fundamental_score는 0~100 범위 내에 있어야 한다."""
    result = compute_fundamental_scores(sample_fundamentals)
    scores = result["fundamental_score"].dropna()
    assert (scores >= 0).all() and (scores <= 100).all(), \
        f"점수 범위 초과: min={scores.min():.2f}, max={scores.max():.2f}"


def test_sector_fillna_unknown(sample_fundamentals):
    """sector가 NaN인 종목은 'Unknown'으로 처리되어야 한다."""
    df = sample_fundamentals.copy()
    df.loc[0, "sector"] = np.nan
    result = compute_fundamental_scores(df)
    assert result.loc[0, "sector"] == "Unknown"
