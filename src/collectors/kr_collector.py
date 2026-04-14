"""
한국 주식 데이터 수집
- 유니버스: KOSPI 시가총액 상위 200
- 펀더멘탈: yfinance (.KS 접미사)
- 가격: FinanceDataReader (KRX 직접)
- 섹터: FinanceDataReader StockListing
"""

from datetime import datetime, timedelta
from pathlib import Path

import FinanceDataReader as fdr
import pandas as pd
import yfinance as yf
from tqdm import tqdm

CACHE_DIR = Path("data/kr")


def get_kospi_universe(top_n: int = 200) -> pd.DataFrame:
    """
    KOSPI 종목 리스트 + 섹터.
    시가총액 기준 상위 top_n 종목 반환.
    columns: ticker, name, sector, industry, market
    """
    listing = fdr.StockListing("KOSPI")

    # FinanceDataReader 컬럼명 정규화 (버전마다 다를 수 있음)
    listing.columns = [c.strip() for c in listing.columns]
    col_map = {
        "Symbol": "ticker",
        "Name": "name",
        "Sector": "sector",
        "Industry": "industry",
        "Market": "market",
        "Marcap": "market_cap",
    }
    listing = listing.rename(columns={k: v for k, v in col_map.items() if k in listing.columns})

    # 시가총액 기준 정렬 (컬럼이 없으면 순서 그대로 사용)
    if "market_cap" in listing.columns:
        listing = listing.dropna(subset=["market_cap"])
        listing = listing.sort_values("market_cap", ascending=False)

    listing = listing.head(top_n).reset_index(drop=True)

    required = ["ticker", "name", "sector"]
    for col in required:
        if col not in listing.columns:
            listing[col] = None

    return listing[["ticker", "name", "sector", "industry"] if "industry" in listing.columns else ["ticker", "name", "sector"]]


def fetch_fundamentals(universe: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
    """
    yfinance로 한국 주식 펀더멘탈 수집 (.KS 접미사).
    섹터가 없는 종목은 yfinance 섹터로 보완.
    캐시: data/kr/fundamentals.parquet
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "fundamentals.parquet"

    if use_cache and cache_path.exists():
        print("[cache] KR fundamentals loaded from cache")
        return pd.read_parquet(cache_path)

    records = []
    for _, row in tqdm(universe.iterrows(), total=len(universe), desc="Fetching KR fundamentals"):
        yf_ticker = str(row["ticker"]).zfill(6) + ".KS"
        try:
            info = yf.Ticker(yf_ticker).info
            # 섹터: FinanceDataReader 값 우선, 없으면 yfinance 값 사용
            sector = row.get("sector") or info.get("sector")
            records.append(
                {
                    "ticker": row["ticker"],
                    "name": row.get("name") or info.get("longName"),
                    "sector": sector,
                    "industry": row.get("industry") or info.get("industry"),
                    "per": info.get("trailingPE"),
                    "pbr": info.get("priceToBook"),
                    "roe": info.get("returnOnEquity"),
                    "revenue_growth": info.get("revenueGrowth"),
                    "debt_to_equity": info.get("debtToEquity"),
                    "market_cap": info.get("marketCap"),
                }
            )
        except Exception as e:
            print(f"[WARN] {row['ticker']}: {e}")

    df = pd.DataFrame(records)
    df.to_parquet(cache_path, index=False)
    return df


def fetch_price_history(tickers: list[str], period_days: int = 365) -> pd.DataFrame:
    """
    FinanceDataReader로 종가 히스토리 수집.
    캐시: data/kr/prices_{period_days}d.parquet
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"prices_{period_days}d.parquet"

    if cache_path.exists():
        print("[cache] KR prices loaded from cache")
        return pd.read_parquet(cache_path)

    end = datetime.today()
    start = end - timedelta(days=period_days)

    closes = {}
    for code in tqdm(tickers, desc="Fetching KR prices"):
        try:
            df = fdr.DataReader(str(code).zfill(6), start, end)
            if "Close" in df.columns:
                closes[code] = df["Close"]
        except Exception:
            pass

    prices = pd.DataFrame(closes)
    prices.to_parquet(cache_path)
    return prices
