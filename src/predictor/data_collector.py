"""
예측 모델 학습 데이터 수집

run_backtest(collect_universe_scores=True)를 활용해
매 리밸런싱 분기마다 전체 유니버스의 기술적 점수와
다음 분기 개별 종목 수익률 쌍을 수집.

반환 DataFrame 컬럼:
  quarter_date       리밸런싱 날짜 (행의 분기 기준)
  ticker
  technical_score, mom_12_1_score, mom_3m_adj_score,
  mom_6m_score, consistency_score, rsi_score, ma_score  — 피처
  next_quarter_return  — 라벨 (다음 분기까지의 개별 종목 수익률)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.engine import _period_return, run_backtest

CACHE_DIR = Path("data/us")

# 이 미만이면 모델 학습에 충분하지 않음
MIN_VIABLE_SAMPLES = 2_000

_SCORE_COLS = [
    "technical_score",
    "mom_12_1_score",
    "mom_3m_adj_score",
    "mom_6m_score",
    "consistency_score",
    "rsi_score",
    "ma_score",
]


class TrainingDataCollector:
    """
    백테스트 엔진을 재사용해 학습용 (피처, 다음 분기 수익률) 쌍을 수집.

    기존 engine.py를 수정하지 않고 collect_universe_scores=True 파라미터로
    매 분기 전체 유니버스 점수를 holdings에 포함시킨 뒤 후처리.
    """

    def collect(
        self,
        prices: pd.DataFrame,
        use_cache: bool = True,
        cache_path: Path | None = None,
        **backtest_kwargs,
    ) -> pd.DataFrame:
        """
        학습 데이터 수집.

        Args:
            prices:           일별 종가 DataFrame
            use_cache:        캐시 파일이 있으면 재사용
            cache_path:       캐시 파일 경로 (None이면 기본 경로 사용)
            **backtest_kwargs: run_backtest에 전달할 추가 파라미터

        Returns:
            DataFrame: quarter_date, ticker, [features], next_quarter_return
        """
        if cache_path is None:
            cache_path = CACHE_DIR / "training_data_clean.parquet"

        if use_cache and cache_path.exists():
            df = pd.read_parquet(cache_path)
            print(f"[cache] 학습 데이터 로드: {len(df):,}개 샘플 ({cache_path})")
            return df

        print("[수집] 전체 유니버스 점수 + 다음 분기 수익률 수집 중...")

        # top_n을 충분히 크게 설정해 전체 유니버스가 holdings에 들어오도록 함
        n_universe = len(prices.columns)
        result = run_backtest(
            prices=prices,
            top_n=n_universe,
            collect_universe_scores=True,
            transaction_cost=0.0,  # 학습 데이터 수집용 — 비용 불필요
            **backtest_kwargs,
        )

        rows = self._extract_rows(prices, result.holdings)
        df = pd.DataFrame(rows)

        if df.empty:
            raise ValueError("학습 데이터를 수집하지 못했습니다. 가격 데이터를 확인하세요.")

        if len(df) < MIN_VIABLE_SAMPLES:
            raise ValueError(
                f"학습 데이터가 부족합니다 ({len(df):,}개). "
                f"최소 {MIN_VIABLE_SAMPLES:,}개 필요. "
                "5년치 가격 데이터가 있는지 확인하세요."
            )

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        print(f"  수집 완료: {len(df):,}개 샘플 ({df['quarter_date'].nunique()}개 분기)")
        print(f"  캐시 저장: {cache_path}")
        return df

    def _extract_rows(
        self,
        prices: pd.DataFrame,
        holdings: list[dict],
    ) -> list[dict]:
        """
        holdings 목록에서 (분기, 티커, 점수, 다음 분기 수익률) 행을 추출.

        마지막 분기는 다음 분기 수익률이 없으므로 제외.
        """
        rows = []

        for i in range(len(holdings) - 1):
            h = holdings[i]
            h_next = holdings[i + 1]

            universe_scores = h.get("universe_scores")
            if not universe_scores:
                continue  # collect_universe_scores=False 인 경우 스킵

            quarter_date = h["date"]
            next_date = h_next["date"]

            # 다음 분기 개별 종목 수익률 계산
            for ticker, scores in universe_scores.items():
                ret = _period_return(prices, [ticker], quarter_date, next_date)
                if np.isnan(ret):
                    continue

                row = {
                    "quarter_date": quarter_date,
                    "ticker": ticker,
                    "next_quarter_return": ret,
                }
                row.update({k: scores.get(k, np.nan) for k in _SCORE_COLS})
                rows.append(row)

        return rows
