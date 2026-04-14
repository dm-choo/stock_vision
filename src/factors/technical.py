"""
기술적 지표 스코어 계산

기술적 지표는 섹터 무관 절대값 기준 → 전체 유니버스 percentile rank(0~100).

지표:
  mom_3m   3개월 가격 모멘텀 (상위 percentile 선호)
  mom_6m   6개월 가격 모멘텀 (상위 percentile 선호)
  rsi      RSI 14일 — 40~65 구간을 선호 (과매수/과매도 제외)
  ma_signal  50일MA / 200일MA — 골든크로스(>1) 선호
"""

import numpy as np
import pandas as pd

# 기술적 지표 가중치
TECH_WEIGHTS = {
    "mom_3m_score": 0.30,
    "mom_6m_score": 0.30,
    "rsi_score": 0.20,
    "ma_score": 0.20,
}

# 거래일 기준 근사값
_DAYS_3M = 63
_DAYS_6M = 126
_DAYS_200 = 200
_DAYS_50 = 50
_RSI_PERIOD = 14


def _rsi(series: pd.Series, period: int = _RSI_PERIOD) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).iloc[-1]


def _rsi_score(rsi_val: float) -> float:
    """
    RSI 점수: 55 근처를 최고점(100)으로, 멀어질수록 감소.
    30 이하 / 80 이상은 0에 가깝게.
    """
    if pd.isna(rsi_val):
        return np.nan
    return float(np.clip(100 - abs(rsi_val - 55) * 2.2, 0, 100))


def compute_technical_factors(prices: pd.DataFrame) -> pd.DataFrame:
    """
    입력: prices DataFrame (index=date, columns=tickers)
    출력: DataFrame (ticker, mom_3m, mom_6m, rsi, ma_signal,
                     mom_3m_score, mom_6m_score, rsi_score, ma_score,
                     technical_score)

    200일 미만 데이터인 종목은 제외.
    """
    records = []

    for ticker in prices.columns:
        series = prices[ticker].dropna()
        if len(series) < _DAYS_200:
            continue

        try:
            current = series.iloc[-1]

            mom_3m = (current / series.iloc[-_DAYS_3M] - 1) if len(series) >= _DAYS_3M else np.nan
            mom_6m = (current / series.iloc[-_DAYS_6M] - 1) if len(series) >= _DAYS_6M else np.nan

            rsi_val = _rsi(series)

            ma50 = series.rolling(_DAYS_50).mean().iloc[-1]
            ma200 = series.rolling(_DAYS_200).mean().iloc[-1]
            ma_signal = (ma50 / ma200) if (ma200 and not np.isnan(ma200)) else np.nan

            records.append(
                {
                    "ticker": str(ticker),
                    "mom_3m": mom_3m,
                    "mom_6m": mom_6m,
                    "rsi": rsi_val,
                    "ma_signal": ma_signal,
                }
            )
        except Exception:
            pass

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # RSI 점수: 절대 기준 변환
    df["rsi_score"] = df["rsi"].apply(_rsi_score)

    # 모멘텀·MA 점수: 전체 percentile rank
    df["mom_3m_score"] = df["mom_3m"].rank(pct=True, na_option="keep") * 100
    df["mom_6m_score"] = df["mom_6m"].rank(pct=True, na_option="keep") * 100
    df["ma_score"] = df["ma_signal"].rank(pct=True, na_option="keep") * 100

    score_cols = list(TECH_WEIGHTS.keys())
    score_matrix = df[score_cols]
    weight_matrix = pd.DataFrame(TECH_WEIGHTS, index=df.index)

    valid_mask = score_matrix.notna()
    adjusted_weights = weight_matrix.where(valid_mask, 0)
    weight_sum = adjusted_weights.sum(axis=1).replace(0, np.nan)
    df["technical_score"] = (score_matrix.fillna(0) * adjusted_weights).sum(axis=1) / weight_sum

    return df[
        ["ticker", "mom_3m", "mom_6m", "rsi", "ma_signal",
         "mom_3m_score", "mom_6m_score", "rsi_score", "ma_score", "technical_score"]
    ]
