"""
미국 주식 데이터 수집 (yfinance 기반)
- 유니버스: S&P 500
- 펀더멘탈: PER, PBR, ROE, 매출성장률, 부채비율
- 가격: Close 기준 1년치
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

CACHE_DIR = Path("data/us")


def get_sp500_tickers() -> list[str]:
    """S&P 500 종목 리스트 (Wikipedia). User-Agent 헤더로 403 우회."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    df = pd.read_html(io.StringIO(resp.text))[0]
    # BRK.B → BRK-B 처럼 yfinance 형식으로 변환
    return df["Symbol"].str.replace(".", "-", regex=False).tolist()


def fetch_fundamentals(tickers: list[str], use_cache: bool = True) -> pd.DataFrame:
    """
    yfinance로 S&P 500 펀더멘탈 수집.
    캐시가 있으면 재사용 (data/us/fundamentals.parquet).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "fundamentals.parquet"

    if use_cache and cache_path.exists():
        print("[cache] US fundamentals loaded from cache")
        return pd.read_parquet(cache_path)

    records = []
    for ticker in tqdm(tickers, desc="Fetching US fundamentals"):
        try:
            info = yf.Ticker(ticker).info
            fcf = info.get("freeCashflow")
            mktcap = info.get("marketCap")
            fcf_yield = (fcf / mktcap) if (fcf and mktcap and mktcap > 0) else np.nan

            records.append(
                {
                    "ticker": ticker,
                    "name": info.get("longName"),
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "per": info.get("trailingPE"),
                    "pbr": info.get("priceToBook"),
                    "roe": info.get("returnOnEquity"),
                    "revenue_growth": info.get("revenueGrowth"),
                    "debt_to_equity": info.get("debtToEquity"),
                    "market_cap": mktcap,
                    "fcf_yield": fcf_yield,
                    "op_margin": info.get("operatingMargins"),
                    "ev_ebitda": info.get("enterpriseToEbitda"),
                }
            )
        except Exception as e:
            print(f"[WARN] {ticker}: {e}")

    df = pd.DataFrame(records)
    df.to_parquet(cache_path, index=False)
    return df


def fetch_price_history(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    """
    yfinance 배치 다운로드로 종가 히스토리 수집.
    캐시가 있으면 재사용 (data/us/prices_{period}.parquet).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"prices_{period}.parquet"

    if cache_path.exists():
        print("[cache] US prices loaded from cache")
        return pd.read_parquet(cache_path)

    raw = yf.download(tickers, period=period, auto_adjust=True, progress=True)
    prices = raw["Close"] if "Close" in raw.columns else raw.xs("Close", axis=1, level=0)
    prices.to_parquet(cache_path)
    return prices
