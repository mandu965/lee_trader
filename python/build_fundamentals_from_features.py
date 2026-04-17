import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
FEATURES_CSV = DATA_DIR / "features.csv"
FUND_CSV = DATA_DIR / "fundamentals.csv"


def deterministic_rng(code: str) -> np.random.Generator:
    h = hashlib.md5(code.encode("utf-8")).hexdigest()[:8]
    seed = int(h, 16) & 0x7FFFFFFF
    return np.random.default_rng(seed)


def main() -> None:
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"features.csv not found: {FEATURES_CSV.resolve()}")
    df = pd.read_csv(FEATURES_CSV, dtype={"code": str})
    if "date" not in df.columns or "code" not in df.columns:
        raise ValueError("features.csv must contain 'date' and 'code'")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "code"])
    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    # 각 종목별 가장 이른 날짜를 fundamentals 기준일로 사용(asof backward merge에 유리)
    min_dates = df.groupby("code")["date"].min().reset_index().rename(columns={"date": "date_min"})

    rows = []
    for _, row in min_dates.iterrows():
        code = str(row["code"])
        d = row["date_min"]
        rng = deterministic_rng(code)
        # 간단한 합리적 범위의 의사-재무지표(결정론적, 재현 가능)
        roe = float(rng.uniform(0.05, 0.25))          # 5% ~ 25%
        op_margin = float(rng.uniform(0.05, 0.20))    # 5% ~ 20%
        debt_ratio = float(rng.uniform(0.20, 1.50))   # 20% ~ 150%
        ocf_to_assets = float(rng.uniform(-0.02, 0.12))  # -2% ~ 12%
        net_margin = float(rng.uniform(0.00, 0.15))   # 0% ~ 15%

        rows.append(
            {
                "date": d,
                "code": code,
                "roe": roe,
                "op_margin": op_margin,
                "debt_ratio": debt_ratio,
                "ocf_to_assets": ocf_to_assets,
                "net_margin": net_margin,
            }
        )

    out = pd.DataFrame(rows).sort_values(["code", "date"]).reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(FUND_CSV, index=False, encoding="utf-8")
    print(f"Saved fundamentals: {FUND_CSV.resolve()} (rows={len(out)})")


if __name__ == "__main__":
    main()
