from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = BASE_DIR / "data" / "ranking_final.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "experiments" / "theme_weight"
DEFAULT_PRICE_CSV = BASE_DIR / "data" / "prices_daily_adjusted.csv"
DEFAULT_SNAPSHOT_DIR = BASE_DIR / "data" / "history" / "ranking"
BACKTEST_HORIZONS = [20, 60, 90]
HORIZON_PRIORITY = {"60d": 3, "20d": 2, "90d": 1}
NONE_RATIO_PENALTY_THRESHOLD = 0.40
NONE_RATIO_PENALTY = 0.25
THEME_COUNT_PENALTY_THRESHOLD = 5
THEME_COUNT_PENALTY = 0.20
MIN_REGIME_SAMPLE_ROWS = 10


def parse_weights(raw: str) -> list[float]:
    weights = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not weights:
        raise ValueError("at least one weight is required")
    if any(w < 0 or w > 1 for w in weights):
        raise ValueError("weights must be within [0, 1]")
    return weights


def _safe_parse_explain(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _to_float(value: object) -> float | None:
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(value) else float(value)


def _fmt_weight(weight: float) -> str:
    return f"{weight:.2f}".replace(".", "_")


def _is_active_theme(value: object) -> bool:
    return str(value or "").strip() not in {"", "(none)", "nan", "None"}


def _resolve_base_score(row: pd.Series) -> tuple[float, str]:
    base = _to_float(row.get("base_score"))
    if base is not None:
        return base, "base_score_column"
    payload = _safe_parse_explain(row.get("explain"))
    base = _to_float(payload.get("base_score"))
    if base is not None:
        return base, "explain.base_score"
    final_v2 = _to_float(row.get("final_score_v2"))
    final_score = _to_float(row.get("final_score"))
    theme_score = _to_float(row.get("theme_score"))
    theme_weight = _to_float(row.get("theme_weight"))
    if final_v2 is not None and theme_score is not None and theme_weight is not None and 0 <= theme_weight < 1:
        return (final_v2 - theme_weight * theme_score) / (1 - theme_weight), "reconstructed_from_final_score_v2"
    if final_score is not None:
        return final_score, "final_score_fallback"
    if final_v2 is not None:
        return final_v2, "final_score_v2_fallback"
    score = _to_float(row.get("score"))
    return (score or 0.0), "score_fallback" if score is not None else "zero_fallback"


def _resolve_contribution(row: pd.Series, weight: float) -> tuple[float, str]:
    value = _to_float(row.get("theme_contribution"))
    if value is not None:
        return value, "theme_contribution_column"
    payload = _safe_parse_explain(row.get("explain"))
    theme_payload = payload.get("theme")
    if isinstance(theme_payload, dict):
        value = _to_float(theme_payload.get("contribution"))
        if value is not None:
            return value, "explain.theme.contribution"
    theme_score = _to_float(row.get("theme_score")) or 0.0
    return weight * theme_score, "weight_times_theme_score"


def load_input_ranking(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    out = df.copy()
    out["date"] = pd.to_datetime(out.get("date"), errors="coerce")
    out["ticker"] = out.get("ticker", out.get("code", "")).astype(str).str.zfill(6)
    out["name"] = out.get("name", "").fillna("").astype(str)
    out["dominant_theme"] = out.get("dominant_theme", "").fillna("(none)").replace("", "(none)").astype(str)
    out["theme_score"] = pd.to_numeric(out.get("theme_score"), errors="coerce").fillna(0.0)
    out["final_score"] = pd.to_numeric(out.get("final_score"), errors="coerce")
    out["final_score_v2"] = pd.to_numeric(out.get("final_score_v2"), errors="coerce")
    out["score"] = pd.to_numeric(out.get("score"), errors="coerce")
    out["theme_weight"] = pd.to_numeric(out.get("theme_weight"), errors="coerce")
    if "regime" not in out.columns:
        out["regime"] = ""
    return out


def compute_weighted_score(df: pd.DataFrame, weight: float) -> pd.DataFrame:
    out = df.copy()
    resolved = out.apply(_resolve_base_score, axis=1, result_type="expand")
    out["base_score_resolved"] = pd.to_numeric(resolved[0], errors="coerce").fillna(0.0)
    out["base_score_source"] = resolved[1].astype(str)
    out["theme_weight_experiment"] = float(weight)
    out["final_score_weighted"] = (1 - weight) * out["base_score_resolved"] + weight * out["theme_score"]
    contrib = out.apply(lambda r: _resolve_contribution(r, weight), axis=1, result_type="expand")
    out["theme_contribution_experiment"] = pd.to_numeric(contrib[0], errors="coerce").fillna(0.0)
    out["contribution_source_used"] = contrib[1].astype(str)
    return out


def build_weighted_ranking(df: pd.DataFrame, weight: float, topn: int) -> pd.DataFrame:
    out = compute_weighted_score(df, weight)
    if out["date"].notna().any():
        out["rank_weighted"] = out.groupby("date")["final_score_weighted"].rank(method="first", ascending=False).astype("Int64")
    else:
        out["rank_weighted"] = out["final_score_weighted"].rank(method="first", ascending=False).astype("Int64")
    latest = out[out["date"] == out["date"].max()].copy() if out["date"].notna().any() else out.copy()
    latest = latest.sort_values(["final_score_weighted", "ticker"], ascending=[False, True]).head(topn).copy()
    latest["topn_flag"] = True
    out = out.merge(latest[["ticker", "topn_flag"]], on="ticker", how="left")
    out["topn_flag"] = out["topn_flag"].eq(True)
    return out


def summarize_weight_result(df: pd.DataFrame, weight: float, topn: int) -> dict[str, object]:
    latest = df[df["date"] == df["date"].max()].copy() if df["date"].notna().any() else df.copy()
    latest = latest.sort_values(["final_score_weighted", "ticker"], ascending=[False, True]).head(topn).copy()
    active = latest["dominant_theme"].apply(_is_active_theme)
    contrib = pd.to_numeric(latest["theme_contribution_experiment"], errors="coerce").fillna(0.0)
    return {
        "weight": float(weight),
        "row_count": int(len(df)),
        "top20_theme_count": int(active.sum()),
        "top20_none_ratio": float((~active).mean()) if len(active) else 0.0,
        "avg_theme_contribution": float(contrib.mean()) if len(contrib) else 0.0,
        "total_theme_contribution": float(contrib.sum()) if len(contrib) else 0.0,
        "top20_theme_list": ",".join(sorted({t for t in latest.loc[active, "dominant_theme"].astype(str) if t})),
    }


def save_weight_outputs(output_dir: Path, weight: float, df: pd.DataFrame) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"ranking_w_{_fmt_weight(weight)}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _load_prices(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    print(f"BACKTEST_PRICE_SOURCE={path}")
    df = pd.read_csv(path, low_memory=False)
    close_col = "adj_close" if "adj_close" in df.columns else "close"
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["ticker"] = out["code"].astype(str).str.zfill(6)
    out["close"] = pd.to_numeric(out[close_col], errors="coerce")
    return out.dropna(subset=["date", "ticker", "close"])[["date", "ticker", "close"]].sort_values(["ticker", "date"])


def _discover_snapshot_dir(path: Path) -> Path | None:
    if path.exists():
        return path
    for candidate in [DEFAULT_SNAPSHOT_DIR, BASE_DIR / "outputs" / "history" / "ranking", BASE_DIR / "output" / "history" / "ranking"]:
        if candidate.exists():
            return candidate
    return None


def _snapshot_date_from_df(df: pd.DataFrame, path: Path) -> pd.Timestamp | None:
    for col in ("as_of_date", "date"):
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().any():
                return parsed.max()
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return pd.to_datetime(digits[:8], format="%Y%m%d", errors="coerce") if len(digits) >= 8 else None


def _load_snapshot_history(snapshot_dir: Path) -> pd.DataFrame:
    resolved_dir = _discover_snapshot_dir(snapshot_dir)
    if resolved_dir is None:
        return pd.DataFrame(columns=["snapshot_date", "ticker", "regime"])
    print(f"SNAPSHOT_DIR={resolved_dir}")
    frames: list[pd.DataFrame] = []
    for path in sorted(resolved_dir.glob("*_ranking_final.csv")):
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        snap_date = _snapshot_date_from_df(df, path)
        if pd.isna(snap_date):
            continue
        std = load_input_ranking(path)
        std["snapshot_date"] = snap_date
        std["as_of_date"] = snap_date
        frames.append(std)
    if not frames:
        return pd.DataFrame(columns=["snapshot_date", "ticker", "regime"])
    out = pd.concat(frames, ignore_index=True)
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"], errors="coerce")
    return out.dropna(subset=["snapshot_date", "ticker"]).copy()


def _forward_result(prices: pd.DataFrame, ticker: str, entry_date: pd.Timestamp, horizon: int) -> dict[str, object]:
    history = prices[(prices["ticker"] == ticker) & (prices["date"] >= entry_date)].sort_values("date").reset_index(drop=True)
    if history.empty:
        return {"price_source_found": False, "missing_reason": "join_key_mismatch", "maturity_status": "missing_price_data", "included_in_eval": False}
    if len(history) <= horizon:
        return {"price_source_found": True, "missing_reason": "horizon_not_matured", "maturity_status": "not_matured", "included_in_eval": False, "entry_price": float(history.loc[0, "close"])}
    entry = float(history.loc[0, "close"])
    if entry <= 0:
        return {"price_source_found": True, "missing_reason": "missing_price_data", "maturity_status": "missing_price_data", "included_in_eval": False}
    path = history.iloc[: horizon + 1].copy()
    exit_price = float(path.iloc[-1]["close"])
    returns_path = path["close"].astype(float) / entry - 1.0
    drawdown = path["close"].astype(float) / path["close"].astype(float).cummax() - 1.0
    return {
        "price_source_found": True,
        "missing_reason": "",
        "maturity_status": "matured",
        "included_in_eval": True,
        "entry_price": entry,
        "exit_date": path.iloc[-1]["date"],
        "exit_price": exit_price,
        "return": float(returns_path.iloc[-1]),
        "drawdown_like": float(drawdown.min()),
    }


def _evaluate_weight_on_history(history: pd.DataFrame, weight: float, horizon: int, prices: pd.DataFrame, topn: int) -> pd.DataFrame:
    if history.empty or "snapshot_date" not in history.columns:
        return pd.DataFrame(
            columns=[
                "weight", "snapshot_date", "ticker", "rank", "entry_date", "entry_price", "exit_date", "exit_price",
                "return", "drawdown_like", "price_source_found", "missing_reason", "maturity_status",
                "included_in_eval", "dominant_theme", "theme_contribution_experiment", "regime",
            ]
        )
    rows: list[dict[str, object]] = []
    for snap_date, snap_df in history.groupby("snapshot_date"):
        weighted = build_weighted_ranking(snap_df.copy(), weight, topn)
        top = weighted.sort_values(["final_score_weighted", "ticker"], ascending=[False, True]).head(topn).copy()
        for row in top.itertuples(index=False):
            result = _forward_result(prices, str(row.ticker), pd.to_datetime(snap_date), horizon)
            rows.append({
                "weight": float(weight),
                "snapshot_date": pd.to_datetime(snap_date).strftime("%Y-%m-%d"),
                "ticker": str(row.ticker),
                "rank": int(getattr(row, "rank_weighted", 0) or 0),
                "entry_date": pd.to_datetime(snap_date).strftime("%Y-%m-%d"),
                "entry_price": result.get("entry_price"),
                "exit_date": pd.to_datetime(result.get("exit_date")).strftime("%Y-%m-%d") if result.get("exit_date") is not None else "",
                "exit_price": result.get("exit_price"),
                "return": result.get("return"),
                "drawdown_like": result.get("drawdown_like"),
                "price_source_found": bool(result.get("price_source_found", False)),
                "missing_reason": result.get("missing_reason", ""),
                "maturity_status": result.get("maturity_status", "unknown"),
                "included_in_eval": bool(result.get("included_in_eval", False)),
                "dominant_theme": str(getattr(row, "dominant_theme", "(none)") or "(none)"),
                "theme_contribution_experiment": _to_float(getattr(row, "theme_contribution_experiment", None)) or 0.0,
                "regime": str(getattr(row, "regime", "") or "").lower(),
            })
    return pd.DataFrame(rows)


def _diagnostic_status(debug_df: pd.DataFrame) -> str:
    if debug_df.empty:
        return "no_matured_snapshots"
    if debug_df["included_in_eval"].astype(bool).sum() == 0:
        reasons = debug_df["missing_reason"].fillna("").astype(str)
        if (reasons == "horizon_not_matured").all():
            return "no_matured_snapshots"
        if (reasons == "join_key_mismatch").any():
            return "join_key_mismatch"
        if (reasons == "missing_price_data").any():
            return "missing_price_data"
        return "unknown"
    if (~debug_df["included_in_eval"].astype(bool)).any():
        return "partial_maturity_only"
    returns = pd.to_numeric(debug_df["return"], errors="coerce").dropna()
    if not returns.empty and returns.abs().sum() == 0:
        return "unknown"
    return "ok"


def _aggregate_debug(debug_df: pd.DataFrame, weight: float, horizon: int) -> dict[str, object]:
    df = debug_df[debug_df["weight"] == float(weight)].copy()
    inc = df[df["included_in_eval"].astype(bool)].copy()
    returns = pd.to_numeric(inc["return"], errors="coerce").dropna()
    drawdowns = pd.to_numeric(inc["drawdown_like"], errors="coerce").dropna()
    if returns.empty:
        avg_return = median_return = win_rate = sharpe_like = max_dd = 0.0
    else:
        avg_return = float(returns.mean())
        median_return = float(returns.median())
        win_rate = float((returns > 0).mean())
        std = float(returns.std(ddof=0))
        sharpe_like = float(avg_return / std) if std > 0 else 0.0
        max_dd = float(drawdowns.min()) if not drawdowns.empty else 0.0
    active = inc["dominant_theme"].apply(_is_active_theme) if not inc.empty else pd.Series(dtype=bool)
    snap_cnt = int(inc["snapshot_date"].nunique()) if not inc.empty else 0
    return {
        "weight": float(weight),
        "horizon": f"{horizon}d",
        "evaluated_snapshot_count": snap_cnt,
        "evaluated_position_count": int(len(inc)),
        "avg_return": avg_return,
        "median_return": median_return,
        "win_rate": win_rate,
        "sharpe_like": sharpe_like,
        "max_drawdown_like": max_dd,
        "top20_theme_count_avg": float(active.groupby(inc["snapshot_date"]).sum().mean()) if snap_cnt else 0.0,
        "top20_none_ratio_avg": float((~active).groupby(inc["snapshot_date"]).mean().mean()) if snap_cnt else 0.0,
        "valid_return_count": int(returns.shape[0]),
        "missing_price_count": int((~df["price_source_found"].astype(bool)).sum()),
        "diagnostic_status": _diagnostic_status(df),
        "total_theme_contribution": float(pd.to_numeric(inc["theme_contribution_experiment"], errors="coerce").fillna(0.0).sum()),
    }


def _select_best_weight(performance_df: pd.DataFrame) -> dict[str, object]:
    frame = performance_df.copy()
    for col in ["weight", "avg_return", "median_return", "win_rate", "sharpe_like", "max_drawdown_like", "top20_theme_count_avg", "top20_none_ratio_avg"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    frame["horizon_priority"] = frame["horizon"].astype(str).map(HORIZON_PRIORITY).fillna(0)
    frame["score"] = frame["sharpe_like"] * 0.50 + frame["avg_return"] * 0.30 + frame["win_rate"] * 0.15 - frame["max_drawdown_like"] * 0.05
    frame.loc[frame["top20_none_ratio_avg"] > NONE_RATIO_PENALTY_THRESHOLD, "score"] -= NONE_RATIO_PENALTY
    frame.loc[frame["top20_theme_count_avg"] < THEME_COUNT_PENALTY_THRESHOLD, "score"] -= THEME_COUNT_PENALTY
    frame = frame.sort_values(["horizon_priority", "score", "sharpe_like", "avg_return", "weight"], ascending=[False, False, False, False, True]).reset_index(drop=True)
    best = frame.iloc[0]
    return {
        "best_weight": float(best["weight"]),
        "selected_horizon": str(best["horizon"]),
        "composite_score": float(best["score"]),
        "reason": [
            "highest composite score on preferred horizon",
            "acceptable top20 theme coverage" if float(best["top20_theme_count_avg"]) >= THEME_COUNT_PENALTY_THRESHOLD else "theme count penalty applied",
            "none ratio remained controlled" if float(best["top20_none_ratio_avg"]) <= NONE_RATIO_PENALTY_THRESHOLD else "high none ratio penalty applied",
        ],
    }


def _write_best_weight_outputs(output_dir: Path, payload: dict[str, object]) -> tuple[Path, Path]:
    json_path = output_dir / "best_weight.json"
    md_path = output_dir / "best_weight_report.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(["# Best Theme Weight", "", f"- best_weight: {payload.get('best_weight')}", f"- selected_horizon: {payload.get('selected_horizon')}", f"- composite_score: {payload.get('composite_score')}"] + [f"- reason: {x}" for x in payload.get("reason", [])]) + "\n", encoding="utf-8")
    return json_path, md_path


def _select_best_weight_by_regime(perf_regime_df: pd.DataFrame, global_best: dict[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {"global": global_best.get("best_weight"), "global_detail": global_best}
    if perf_regime_df.empty:
        return payload
    for regime in sorted(perf_regime_df["regime"].dropna().astype(str).str.lower().unique().tolist()):
        subset = perf_regime_df[perf_regime_df["regime"].astype(str).str.lower() == regime].copy()
        sample_rows = int(pd.to_numeric(subset["sample_rows"], errors="coerce").fillna(0).max()) if "sample_rows" in subset.columns else 0
        if sample_rows < MIN_REGIME_SAMPLE_ROWS:
            payload[regime] = global_best.get("best_weight")
            payload[f"{regime}_detail"] = {"fallback_applied": True, "fallback_reason": f"insufficient regime sample rows ({sample_rows} < {MIN_REGIME_SAMPLE_ROWS})", "sample_rows": sample_rows}
        else:
            best = _select_best_weight(subset)
            payload[regime] = best.get("best_weight")
            payload[f"{regime}_detail"] = {**best, "fallback_applied": False, "fallback_reason": "(none)", "sample_rows": sample_rows}
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment with alternative theme-score weights.")
    p.add_argument("--weights", required=True)
    p.add_argument("--input", default=str(DEFAULT_INPUT))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--topn", type=int, default=20)
    p.add_argument("--mode", default="operational")
    p.add_argument("--run-backtest", action="store_true")
    p.add_argument("--select-best", action="store_true")
    p.add_argument("--snapshot-dir", default=str(DEFAULT_SNAPSHOT_DIR))
    p.add_argument("--min-snapshots-20d", type=int, default=5)
    p.add_argument("--min-snapshots-60d", type=int, default=3)
    p.add_argument("--min-snapshots-90d", type=int, default=3)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input) if Path(args.input).is_absolute() else BASE_DIR / args.input
    output_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else BASE_DIR / args.output_dir
    snapshot_dir = Path(args.snapshot_dir) if Path(args.snapshot_dir).is_absolute() else BASE_DIR / args.snapshot_dir
    try:
        weights = parse_weights(args.weights)
        base_df = load_input_ranking(input_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"INPUT_ERROR: {exc}")
        return 1

    generated_files: list[Path] = []
    summaries: list[dict[str, object]] = []
    contribution_valid_rows = 0
    for weight in weights:
        weighted = build_weighted_ranking(base_df, weight, args.topn)
        generated_files.append(save_weight_outputs(output_dir, weight, weighted))
        summaries.append(summarize_weight_result(weighted, weight, args.topn))
        contribution_valid_rows += int(pd.to_numeric(weighted["theme_contribution_experiment"], errors="coerce").fillna(0.0).gt(0).sum())
        s = summaries[-1]
        print(f"[WEIGHT={weight:.2f}] ranking generated")
        print(f"[WEIGHT={weight:.2f}] top20_theme_count={s['top20_theme_count']} none_ratio={s['top20_none_ratio']:.2f} total_contribution={s['total_theme_contribution']:.2f}")
    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / "weight_generation_summary.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    generated_files.append(summary_path)

    matured_snapshots_found = 0
    evaluated_positions = 0
    best_allowed = True
    total_snapshots_found = 0
    matured_by_horizon = {20: 0, 60: 0, 90: 0}
    evaluated_positions_by_horizon = {20: 0, 60: 0, 90: 0}
    performance_path = output_dir / "performance_summary.csv"
    perf_regime_path = output_dir / "performance_by_regime.csv"
    if args.run_backtest:
        try:
            prices = _load_prices(DEFAULT_PRICE_CSV)
            history = _load_snapshot_history(snapshot_dir)
        except (FileNotFoundError, ValueError) as exc:
            print(f"BACKTEST_ERROR: {exc}")
            return 1
        total_snapshots_found = int(history["snapshot_date"].nunique()) if not history.empty and "snapshot_date" in history.columns else 0
        perf_rows: list[dict[str, object]] = []
        perf_regime_rows: list[dict[str, object]] = []
        for horizon in BACKTEST_HORIZONS:
            frames: list[pd.DataFrame] = []
            for weight in weights:
                debug_df = _evaluate_weight_on_history(history, weight, horizon, prices, args.topn)
                frames.append(debug_df)
                agg = _aggregate_debug(debug_df, weight, horizon)
                perf_rows.append(agg)
                matured_snapshots_found = max(matured_snapshots_found, int(agg["evaluated_snapshot_count"]))
                evaluated_positions += int(agg["evaluated_position_count"])
                matured_by_horizon[horizon] = max(matured_by_horizon[horizon], int(agg["evaluated_snapshot_count"]))
                evaluated_positions_by_horizon[horizon] += int(agg["evaluated_position_count"])
                print(f"[WEIGHT={weight:.2f}][{horizon}d] avg_return={agg['avg_return']:.4f} win_rate={agg['win_rate']:.2f} sharpe_like={agg['sharpe_like']:.2f}")
                if not history.empty and "regime" in history.columns:
                    for regime in sorted(history["regime"].dropna().astype(str).str.lower().unique().tolist()):
                        reg_hist = history[history["regime"].fillna("").astype(str).str.lower() == regime].copy()
                        reg_debug = _evaluate_weight_on_history(reg_hist, weight, horizon, prices, args.topn)
                        reg_agg = _aggregate_debug(reg_debug, weight, horizon)
                        reg_agg["regime"] = regime
                        reg_agg["sample_rows"] = int(len(reg_hist))
                        perf_regime_rows.append(reg_agg)
            debug_path = output_dir / f"backtest_debug_{horizon}d.csv"
            pd.concat(frames, ignore_index=True).to_csv(debug_path, index=False, encoding="utf-8-sig") if frames else pd.DataFrame().to_csv(debug_path, index=False, encoding="utf-8-sig")
            generated_files.append(debug_path)
        pd.DataFrame(perf_rows).to_csv(performance_path, index=False, encoding="utf-8-sig")
        generated_files.append(performance_path)
        if perf_regime_rows:
            pd.DataFrame(perf_regime_rows).to_csv(perf_regime_path, index=False, encoding="utf-8-sig")
            generated_files.append(perf_regime_path)

    if args.select_best:
        if not performance_path.exists():
            print("SELECT_ERROR: performance_summary.csv not found")
            return 1
        performance_df = pd.read_csv(performance_path, low_memory=False)
        mins = {"20d": args.min_snapshots_20d, "60d": args.min_snapshots_60d, "90d": args.min_snapshots_90d}
        insufficient = []
        for horizon, minimum in mins.items():
            subset = performance_df[performance_df["horizon"].astype(str) == horizon]
            if subset.empty or pd.to_numeric(subset["evaluated_snapshot_count"], errors="coerce").fillna(0).max() < minimum:
                insufficient.append(f"{horizon}<{minimum}")
        if insufficient:
            best_allowed = False
            blocked = {
                "blocked": True,
                "reason": "insufficient matured snapshots",
                "insufficient_horizons": insufficient,
                "diagnostic_status_breakdown": performance_df["diagnostic_status"].value_counts(dropna=False).to_dict() if "diagnostic_status" in performance_df.columns else {},
            }
            blocked_path = output_dir / "best_weight_blocked.json"
            blocked_path.write_text(json.dumps(blocked, ensure_ascii=False, indent=2), encoding="utf-8")
            generated_files.append(blocked_path)
        else:
            best = _select_best_weight(performance_df)
            generated_files.extend(_write_best_weight_outputs(output_dir, best))
            regime_payload = {"global": best.get("best_weight"), "global_detail": best}
            if perf_regime_path.exists():
                regime_payload = _select_best_weight_by_regime(pd.read_csv(perf_regime_path, low_memory=False), best)
            regime_json = output_dir / "best_weight_by_regime.json"
            regime_json.write_text(json.dumps(regime_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            generated_files.append(regime_json)
            print(f"BEST WEIGHT = {float(best['best_weight']):.2f}")

    print(f"contribution valid rows: {contribution_valid_rows}")
    print(f"total snapshots found: {total_snapshots_found}")
    print(f"matured snapshots found: {matured_snapshots_found}")
    print(f"matured snapshots found by horizon: 20d={matured_by_horizon[20]} 60d={matured_by_horizon[60]} 90d={matured_by_horizon[90]}")
    print(f"evaluated positions: {evaluated_positions}")
    print(f"evaluated positions by horizon: 20d={evaluated_positions_by_horizon[20]} 60d={evaluated_positions_by_horizon[60]} 90d={evaluated_positions_by_horizon[90]}")
    print(f"best weight selection allowed: {str(best_allowed).lower()}")
    print("generated files:")
    for path in generated_files:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
