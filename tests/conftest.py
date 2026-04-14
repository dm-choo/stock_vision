"""
공통 테스트 픽스처

외부 API(yfinance, FinanceDataReader)를 호출하지 않고
로직을 검증하기 위한 결정적(deterministic) 샘플 데이터.

픽스처 설계 원칙:
- 섹터가 2개(Technology, Energy) + Unknown 1개로 섹터 상대 비교 로직을 검증 가능
- 일부 종목에 의도적으로 NaN / 음수 PER 삽입 → 경계값 처리 검증
- 가격 데이터는 250거래일 보장 → 200일 최소 조건 충족
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_fundamentals() -> pd.DataFrame:
    """
    6개 종목 펀더멘탈 픽스처.

    의도된 이상값:
      - BADPE : PER 음수 (적자 기업) → per_score NaN 처리 대상
      - NODATA: roe·revenue_growth·debt_to_equity 모두 NaN → valid_factor_count < 3 제외 대상
      - UNKNWN: sector 'Unknown' → 전체 평균 기준 비교 대상
    """
    return pd.DataFrame(
        {
            "ticker": ["TECH1", "TECH2", "ENRG1", "ENRG2", "BADPE", "NODATA", "UNKNWN"],
            "name": [
                "TechAlpha", "TechBeta", "EnergyAlpha", "EnergyBeta",
                "BadPE Corp", "NoData Corp", "Unknown Corp",
            ],
            "sector": [
                "Technology", "Technology", "Energy", "Energy",
                "Technology", "Energy", "Unknown",
            ],
            "per": [20.0, 30.0, 10.0, 15.0, -5.0, 12.0, 18.0],
            "pbr": [5.0, 8.0, 1.5, 2.0, 3.0, np.nan, 2.5],
            "roe": [0.20, 0.15, 0.12, 0.10, 0.05, np.nan, 0.14],
            "revenue_growth": [0.15, 0.10, 0.08, 0.05, 0.20, np.nan, 0.09],
            "debt_to_equity": [30.0, 50.0, 20.0, 25.0, 80.0, np.nan, 35.0],
            "market_cap": [5e11, 3e11, 2e11, 1e11, 5e10, 8e10, 9e10],
        }
    )


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """
    5개 종목 250거래일 종가 픽스처 (seed 고정으로 결정적).

    SHORT: 100거래일만 존재 → 200일 최소 조건 미달로 기술적 분석에서 제외되어야 함.
    """
    rng = np.random.default_rng(42)
    dates_full = pd.bdate_range(end="2025-12-31", periods=250)
    dates_short = pd.bdate_range(end="2025-12-31", periods=100)

    tickers_full = ["TECH1", "TECH2", "ENRG1", "ENRG2"]
    data = {
        t: 100 * np.cumprod(1 + rng.normal(0.0005, 0.018, 250))
        for t in tickers_full
    }
    prices = pd.DataFrame(data, index=dates_full)

    # SHORT 종목: 100일치만
    short_series = pd.Series(
        100 * np.cumprod(1 + rng.normal(0.0005, 0.018, 100)),
        index=dates_short,
        name="SHORT",
    )
    return prices.join(short_series, how="outer")
