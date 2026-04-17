# python/research/config.py
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Literal


@dataclass
class ScoreWeights:
    w_ret: float = 1.0
    w_prob: float = 1.0
    w_qual: float = 1.0
    w_tech: float = 1.0
    w_risk: float = 1.0
    w_market: float = 0.0  # 시장 보정 가중치 (미사용 시 0)


@dataclass
class BacktestConfig:
    # 기간 / 모델 버전
    start_date: date
    end_date: date
    model_versions: List[str]
    run_id: Optional[int] = None  # Research prediction_history.run_id 사용 시 지정

    # 백테스트 설정
    horizon_days: int = 60
    top_n: int = 20
    score_column: str = "final_score_custom"  # 기본적으로 커스텀 점수 사용

    # 점수 가중치
    weights: ScoreWeights = field(default_factory=ScoreWeights)

    # 데이터 소스 설정
    prediction_source: Literal["db", "csv", "research"] = "csv"
    price_source: Literal["db", "csv"] = "csv"
    predictions_csv: str = "data/predictions.csv"
    prices_csv: str = "data/prices_daily_adjusted.csv"

    # 벤치마크
    benchmark_code: Optional[str] = None
    benchmark_csv: Optional[str] = None

    # 출력 경로
    output_dir: str = "outputs/backtest"
