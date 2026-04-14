"""
Stock Vision — 메인 실행 진입점

사용법:
  uv run python main.py --market us                    # 미국 S&P 500 스코어링
  uv run python main.py --market kr                    # 한국 KOSPI 스코어링
  uv run python main.py --market us --top 30           # 상위 30개 출력
  uv run python main.py --market us --no-cache         # 캐시 무시하고 새로 수집
  uv run python main.py --market us --backtest         # 백테스트 실행 (technical 모드)
  uv run python main.py --market us --backtest --backtest-mode hybrid  # hybrid 모드
  uv run python main.py --market us --backtest --top-n 30              # 상위 30개로 백테스트
"""

import argparse

import pandas as pd

from src.backtest.engine import run_backtest
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


def run_us_backtest(top_n: int = 20, mode: str = "technical") -> None:
    print(f"\n[US 백테스트] mode={mode}, top_n={top_n}")

    prices_5y_path = "data/us/prices_5y.parquet"
    try:
        prices = pd.read_parquet(prices_5y_path)
        print(f"  가격 데이터: {prices.index[0].date()} ~ {prices.index[-1].date()}")
    except FileNotFoundError:
        print("  5년치 가격 데이터가 없습니다. 먼저 스코어링을 실행해 1년치 데이터를 확보하거나,")
        print("  별도로 data/us/prices_5y.parquet 를 준비하세요.")
        return

    fundamentals = None
    if mode == "hybrid":
        fund_path = "data/us/fundamentals.parquet"
        try:
            fundamentals = pd.read_parquet(fund_path)
            print(f"  ⚠  hybrid 모드: 현재 펀더멘탈 데이터 사용 → look-ahead bias 존재")
        except FileNotFoundError:
            print(f"  펀더멘탈 캐시({fund_path})가 없습니다. --market us 먼저 실행하세요.")
            return

    result = run_backtest(
        prices=prices,
        top_n=top_n,
        mode=mode,
        fundamentals=fundamentals,
    )
    print(result.summary())

    out_path = f"data/us/backtest_{mode}.csv"
    pd.DataFrame({
        "date": result.quarterly_portfolio.index,
        "portfolio": result.quarterly_portfolio.values,
        "benchmark": result.quarterly_benchmark.values,
        "alpha": (result.quarterly_portfolio - result.quarterly_benchmark).values,
    }).to_csv(out_path, index=False)
    print(f"  분기별 수익률 저장: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Stock Vision — 주가 상승 가능성 스코어링 + 백테스트")
    parser.add_argument("--market", choices=["us", "kr"], required=True)
    parser.add_argument("--top", type=int, default=20, help="스코어링 상위 N개 출력")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--backtest", action="store_true", help="백테스트 실행")
    parser.add_argument("--backtest-mode", choices=["technical", "hybrid"], default="technical")
    parser.add_argument("--top-n", type=int, default=20, help="백테스트 포트폴리오 종목 수")
    args = parser.parse_args()

    use_cache = not args.no_cache

    if args.backtest:
        if args.market == "us":
            run_us_backtest(top_n=args.top_n, mode=args.backtest_mode)
        else:
            print("[KR 백테스트] 아직 미구현입니다.")
    else:
        if args.market == "us":
            run_us(use_cache=use_cache, top_n=args.top)
        elif args.market == "kr":
            run_kr(use_cache=use_cache, top_n=args.top)


if __name__ == "__main__":
    main()
