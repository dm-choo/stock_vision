"""
Stock Vision — 메인 실행 진입점

사용법:
  uv run python main.py --market us          # 미국 S&P 500 분석
  uv run python main.py --market kr          # 한국 KOSPI 분석
  uv run python main.py --market us --top 30 # 상위 30개 출력
  uv run python main.py --market us --no-cache  # 캐시 무시하고 새로 수집
"""

import argparse

import pandas as pd

from src.collectors import kr_collector, us_collector
from src.factors.fundamental import compute_fundamental_scores
from src.factors.technical import compute_technical_factors
from src.scoring.composite import compute_composite_score, print_top_n


def run_us(use_cache: bool = True, top_n: int = 20) -> pd.DataFrame:
    print("\n[US] S&P 500 분석 시작")

    tickers = us_collector.get_sp500_tickers()
    print(f"  유니버스: {len(tickers)}개 종목")

    fund_raw = us_collector.fetch_fundamentals(tickers, use_cache=use_cache)
    prices = us_collector.fetch_price_history(tickers, period="1y")

    fund_scored = compute_fundamental_scores(fund_raw)
    tech_scored = compute_technical_factors(prices)

    result = compute_composite_score(fund_scored, tech_scored)
    print_top_n(result, n=top_n)

    out_path = "data/us/scores.csv"
    result.to_csv(out_path, index=False)
    print(f"  전체 결과 저장: {out_path}")

    return result


def run_kr(use_cache: bool = True, top_n: int = 20) -> pd.DataFrame:
    print("\n[KR] KOSPI 분석 시작")

    universe = kr_collector.get_kospi_universe(top_n=200)
    print(f"  유니버스: {len(universe)}개 종목")

    fund_raw = kr_collector.fetch_fundamentals(universe, use_cache=use_cache)
    prices = kr_collector.fetch_price_history(universe["ticker"].tolist(), period_days=365)

    fund_scored = compute_fundamental_scores(fund_raw)
    tech_scored = compute_technical_factors(prices)

    result = compute_composite_score(fund_scored, tech_scored)
    print_top_n(result, n=top_n)

    out_path = "data/kr/scores.csv"
    result.to_csv(out_path, index=False)
    print(f"  전체 결과 저장: {out_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Stock Vision — 주가 상승 가능성 스코어링")
    parser.add_argument("--market", choices=["us", "kr"], required=True, help="분석 대상 시장")
    parser.add_argument("--top", type=int, default=20, help="상위 N개 출력 (기본: 20)")
    parser.add_argument("--no-cache", action="store_true", help="캐시 무시하고 데이터 새로 수집")
    args = parser.parse_args()

    use_cache = not args.no_cache

    if args.market == "us":
        run_us(use_cache=use_cache, top_n=args.top)
    elif args.market == "kr":
        run_kr(use_cache=use_cache, top_n=args.top)


if __name__ == "__main__":
    main()
