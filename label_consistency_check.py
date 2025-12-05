import pandas as pd
import numpy as np
from pathlib import Path

# ---- 설정 (경로 필요시 수정) ----
DATA_DIR = Path("data")
LABELS_PATH = DATA_DIR / "labels.csv"
PREDS_PATH = DATA_DIR / "predictions.csv"
PRICES_PATH = DATA_DIR / "prices_daily_adjusted.csv"  # adj_close 기준
PRICE_CLOSE_COL = "adj_close"  # prices 파일의 종가/조정종가 컬럼명
HORIZONS = [60, 90]
RET_TOL = 1e-4
MDD_TOL = 1e-4


def load_csv(path: Path, dtype_code: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    if dtype_code and "code" in df.columns:
        df["code"] = df["code"].astype(str).str.zfill(6)
    return df


def recompute_returns(prices: pd.DataFrame, horizon: int, close_col: str = "close") -> pd.DataFrame:
    prices = prices.sort_values(["code", "date"])
    out = []
    for code, g in prices.groupby("code"):
        g = g.copy()
        closes = g[close_col].values
        dates = g["date"].values
        n = len(g)
        f_ret = np.full(n, np.nan)
        f_mdd = np.full(n, np.nan)
        for i in range(n):
            j = i + horizon
            if j >= n:
                continue
            window = closes[i : j + 1]
            start = window[0]
            end = window[-1]
            if start and not np.isnan(start) and end and not np.isnan(end) and start != 0:
                f_ret[i] = end / start - 1.0
            run_max = np.maximum.accumulate(window)
            dd = window / run_max - 1.0
            f_mdd[i] = dd.min()
        tmp = pd.DataFrame(
            {
                "date": dates,
                "code": code,
                f"recalc_return_{horizon}d": f_ret,
                f"recalc_mdd_{horizon}d": f_mdd,
            }
        )
        out.append(tmp)
    return pd.concat(out, ignore_index=True)


def describe_series(x: pd.Series, name: str):
    x = pd.to_numeric(x, errors="coerce")
    stats = {
        "count": x.notna().sum(),
        "mean": x.mean(),
        "std": x.std(),
        "min": x.min(),
        "1%": x.quantile(0.01),
        "99%": x.quantile(0.99),
        "max": x.max(),
    }
    print(f"[{name}] {stats}")


def main():
    labels = load_csv(LABELS_PATH)
    preds = load_csv(PREDS_PATH)
    prices = load_csv(PRICES_PATH)
    if PRICE_CLOSE_COL not in prices.columns:
        raise ValueError(f"{PRICE_CLOSE_COL} not found in prices file")
    prices = prices.rename(columns={PRICE_CLOSE_COL: "close"})

    print("Labels rows:", len(labels), "Preds rows:", len(preds), "Prices rows:", len(prices))

    for h in HORIZONS:
        rec = recompute_returns(prices[["date", "code", "close"]], horizon=h, close_col="close")
        merged = labels.merge(rec, on=["date", "code"], how="left")

        lab_ret_col = f"realized_return_{h}d" if f"realized_return_{h}d" in labels.columns else f"target_{h}d"
        lab_mdd_col = f"realized_mdd_{h}d" if f"realized_mdd_{h}d" in labels.columns else f"target_mdd_{h}d"

        if lab_ret_col in merged.columns:
            diff = merged[lab_ret_col] - merged[f"recalc_return_{h}d"]
            diff_abs = diff.abs()
            merged["abs_diff_ret"] = diff_abs
            print(f"\n[RET {h}d] mean_abs_err={diff_abs.mean():.6g}, max_abs_err={diff_abs.max():.6g}")
            bad = merged.loc[diff_abs > RET_TOL, ["date", "code", lab_ret_col, f"recalc_return_{h}d", "abs_diff_ret"]]
            print(f"RET {h}d > tol ({RET_TOL}) rows: {len(bad)}/{len(merged)}")
            if len(bad) > 0:
                print(bad.head(5))
        else:
            print(f"[RET {h}d] label column not found: {lab_ret_col}")

        if lab_mdd_col in merged.columns:
            diff = merged[lab_mdd_col] - merged[f"recalc_mdd_{h}d"]
            diff_abs = diff.abs()
            merged["abs_diff_mdd"] = diff_abs
            print(f"[MDD {h}d] mean_abs_err={diff_abs.mean():.6g}, max_abs_err={diff_abs.max():.6g}")
            bad = merged.loc[diff_abs > MDD_TOL, ["date", "code", lab_mdd_col, f"recalc_mdd_{h}d", "abs_diff_mdd"]]
            print(f"MDD {h}d > tol ({MDD_TOL}) rows: {len(bad)}/{len(merged)}")
            if len(bad) > 0:
                print(bad.head(5))
        else:
            print(f"[MDD {h}d] label column not found: {lab_mdd_col}")

    for col in [f"realized_return_{h}d" for h in HORIZONS] + [f"pred_return_{h}d" for h in HORIZONS]:
        if col in labels.columns:
            describe_series(labels[col], f"labels {col}")
        if col in preds.columns:
            describe_series(preds[col], f"preds {col}")

    for src, df, col in [("labels", labels, f"realized_return_{HORIZONS[0]}d"), ("preds", preds, f"pred_return_{HORIZONS[0]}d")]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            mask = (s.abs() > 1.0) | (s < s.quantile(0.01)) | (s > s.quantile(0.99))
            outliers = df.loc[mask, ["date", "code", col]].head(20)
            print(f"\n[{src} {col}] outlier samples (top 20):")
            print(outliers)

    if "date" in labels.columns:
        print("\nLabels date range:", labels["date"].min(), "->", labels["date"].max())
    if "date" in preds.columns:
        print("Preds date range:", preds["date"].min(), "->", preds["date"].max())

    if "date" in labels.columns and "date" in preds.columns:
        label_max = labels["date"].max()
        future_preds = preds.loc[preds["date"] > label_max]
        print(f"Preds after label max date: {len(future_preds)} rows (labels max={label_max.date() if pd.notna(label_max) else None})")

    print("\nDone. Check the printed stats/outlier samples; optionally add plots in a notebook if needed.")


if __name__ == "__main__":
    main()
