import argparse
import logging
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

try:
    from pykrx import stock as pykrx_stock
except Exception:
    pykrx_stock = None

try:
    from download_etf_master import get_etf_tickers
except Exception:
    get_etf_tickers = None


DATA_DIR = Path("data")
THEME_ETF_MASTER_CSV = DATA_DIR / "theme_etf_master.csv"
OUTPUT_CSV = DATA_DIR / "theme_etf_master_expansion_candidates.csv"
OUTPUT_MD = DATA_DIR / "theme_etf_master_expansion_review.md"
PRICES_RAW_CSV = DATA_DIR / "prices_daily_raw.csv"

DEFAULT_LOOKBACK_DAYS = 14
DEFAULT_TOP_N = 5
DEFAULT_MIN_SCORE = 2.0


@dataclass(frozen=True)
class ThemeCandidateRule:
    theme_id: str
    theme_name: str
    keywords: tuple[str, ...]


THEME_RULES: tuple[ThemeCandidateRule, ...] = (
    ThemeCandidateRule("HBM", "HBM", ("HBM", "반도체", "메모리", "AI반도체", "반도체핵심공정")),
    ThemeCandidateRule("SEMIEQP", "반도체장비", ("반도체", "반도체핵심공정", "소부장", "AI반도체")),
    ThemeCandidateRule("AIPCB", "AI서버기판", ("AI반도체", "반도체", "테크", "AI", "반도체핵심공정")),
    ThemeCandidateRule("POWER", "전력설비", ("전력", "전력설비", "전선", "원전", "전력망")),
    ThemeCandidateRule("DEFENSE", "방산", ("방산", "우주", "국방", "항공우주")),
    ThemeCandidateRule("BATTERY", "2차전지", ("2차전지", "배터리", "리튬", "전기차")),
    ThemeCandidateRule("SHIP", "조선", ("조선", "해운", "조선해양", "해양")),
    ThemeCandidateRule("BIO", "바이오", ("바이오", "헬스케어", "제약", "의료")),
    ThemeCandidateRule("TELCO", "Digital Connectivity", ("통신", "인터넷", "커뮤니케이션", "5G", "통신서비스")),
    ThemeCandidateRule("BROKER", "Brokerage Markets", ("증권", "금융", "코리아밸류업", "배당")),
    ThemeCandidateRule("BANKRET", "Retail Finance Return", ("은행", "금융", "고배당", "밸류업", "배당")),
    ThemeCandidateRule("FINPLAT", "Digital Finance Platform", ("금융", "인터넷", "플랫폼", "핀테크", "테크")),
    ThemeCandidateRule("PLATECO", "Digital Platform Ecosystem", ("인터넷", "플랫폼", "테크", "미디어", "콘텐츠")),
    ThemeCandidateRule("GAMEIP", "Game IP Monetization", ("게임", "콘텐츠", "미디어", "엔터테인먼트")),
    ThemeCandidateRule("AIRMOB", "Air Mobility Recovery", ("항공", "여행", "운송", "모빌리티")),
    ThemeCandidateRule("AISOFT", "Enterprise AI Software", ("AI", "로봇", "테크", "소프트웨어", "반도체")),
)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze theme ETF master expansion candidates.")
    parser.add_argument("--theme-etf-master", type=Path, default=THEME_ETF_MASTER_CSV)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=OUTPUT_MD)
    return parser.parse_args()


def recent_business_dates(lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> list[str]:
    today = date.today()
    out: list[str] = []
    for offset in range(lookback_days + 1):
        current = today - timedelta(days=offset)
        if current.weekday() < 5:
            out.append(current.strftime("%Y%m%d"))
    return out


def fetch_all_etf_names() -> pd.DataFrame:
    if pykrx_stock is None:
        raise RuntimeError("pykrx is not available")
    if get_etf_tickers is not None:
        try:
            tickers = get_etf_tickers()
            rows = []
            for ticker in tickers:
                code = str(ticker).strip().zfill(6)
                if not code:
                    continue
                try:
                    name = str(pykrx_stock.get_etf_ticker_name(code) or "").strip()
                except Exception:
                    continue
                if name:
                    rows.append({"etf_code": code, "etf_name": name, "as_of_date": date.today().strftime("%Y%m%d")})
            if rows:
                df = pd.DataFrame(rows).drop_duplicates(subset=["etf_code"]).reset_index(drop=True)
                logging.info("Loaded ETF universe via download_etf_master fallback rows=%s", len(df))
                return df
        except Exception as exc:
            logging.warning("download_etf_master.get_etf_tickers fallback failed: %s", exc)
    errors: list[str] = []
    for business_date in recent_business_dates():
        try:
            tickers = pykrx_stock.get_etf_ticker_list(business_date)
            rows = []
            for ticker in tickers:
                code = str(ticker).strip().zfill(6)
                if not code:
                    continue
                try:
                    name = str(pykrx_stock.get_etf_ticker_name(code) or "").strip()
                except Exception as exc:
                    errors.append(f"{code}: name_fetch failed: {exc}")
                    continue
                if name:
                    rows.append({"etf_code": code, "etf_name": name, "as_of_date": business_date})
            if rows:
                df = pd.DataFrame(rows).drop_duplicates(subset=["etf_code"]).reset_index(drop=True)
                logging.info("Loaded ETF universe rows=%s as_of_date=%s", len(df), business_date)
                return df
        except Exception as exc:
            errors.append(f"{business_date}: ticker_list failed: {exc}")
    raise RuntimeError("Unable to load ETF universe from pykrx. recent_errors=" + " | ".join(errors[-5:]))


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def compact_text(value: str) -> str:
    return re.sub(r"[\s_\-/&(),]+", "", normalize_text(value).lower())


def load_theme_master(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"theme_etf_master is empty: {path}")
    required = {"theme_id", "theme_name", "etf_code", "etf_name"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"theme_etf_master missing columns: {sorted(missing)}")
    out = df.copy()
    out["theme_id"] = out["theme_id"].astype(str).str.strip().str.upper()
    out["theme_name"] = out["theme_name"].astype(str).map(normalize_text)
    out["etf_code"] = out["etf_code"].astype(str).str.strip().str.zfill(6)
    out["etf_name"] = out["etf_name"].astype(str).map(normalize_text)
    return out


def load_local_etf_universe(theme_master: pd.DataFrame, prices_raw_path: Path = PRICES_RAW_CSV) -> pd.DataFrame:
    if not prices_raw_path.exists():
        return pd.DataFrame(columns=["etf_code", "etf_name", "as_of_date"])
    price_df = pd.read_csv(prices_raw_path, usecols=["date", "code", "asset_type"])
    if "asset_type" not in price_df.columns:
        return pd.DataFrame(columns=["etf_code", "etf_name", "as_of_date"])
    etf_df = price_df[price_df["asset_type"].astype(str).str.lower() == "etf"].copy()
    if etf_df.empty:
        return pd.DataFrame(columns=["etf_code", "etf_name", "as_of_date"])
    latest_date = str(etf_df["date"].max())
    codes = etf_df["code"].astype(str).str.strip().str.zfill(6).drop_duplicates().to_frame(name="etf_code")
    name_map = theme_master[["etf_code", "etf_name"]].drop_duplicates().set_index("etf_code")["etf_name"].to_dict()
    codes["etf_name"] = codes["etf_code"].map(name_map).fillna(codes["etf_code"])
    codes["as_of_date"] = latest_date
    logging.info("Loaded local ETF universe rows=%s latest_date=%s", len(codes), latest_date)
    return codes


def score_candidate(etf_name: str, rule: ThemeCandidateRule) -> tuple[float, list[str]]:
    name = normalize_text(etf_name)
    name_lower = name.lower()
    name_compact = compact_text(name)
    matched: list[str] = []
    score = 0.0
    for keyword in rule.keywords:
        raw = normalize_text(keyword)
        keyword_lower = raw.lower()
        keyword_compact = compact_text(raw)
        if keyword_lower and keyword_lower in name_lower:
            matched.append(raw)
            score += 1.0
            continue
        if keyword_compact and keyword_compact in name_compact:
            matched.append(raw)
            score += 0.8
    if matched and any(token in matched for token in ("AI", "HBM", "반도체", "전력", "방산", "2차전지", "바이오")):
        score += 0.5
    return score, matched


def build_candidate_frame(theme_master: pd.DataFrame, etf_universe: pd.DataFrame, top_n: int, min_score: float) -> pd.DataFrame:
    theme_counts = theme_master.groupby("theme_id")["etf_code"].nunique().rename("current_theme_etf_count")
    existing_by_theme = theme_master.groupby("theme_id")["etf_code"].agg(lambda s: set(s.astype(str))).to_dict()
    rule_by_theme = {rule.theme_id: rule for rule in THEME_RULES}
    rows: list[dict[str, object]] = []

    for theme_id, current_row in theme_master.groupby("theme_id", as_index=False).first().iterrows():
        pass
    for theme_id, current_row in theme_master.groupby("theme_id").first().iterrows():
        rule = rule_by_theme.get(theme_id)
        if rule is None:
            continue
        existing_codes = existing_by_theme.get(theme_id, set())
        scored_rows: list[dict[str, object]] = []
        for item in etf_universe.itertuples(index=False):
            if item.etf_code in existing_codes:
                continue
            score, matched_keywords = score_candidate(item.etf_name, rule)
            if score < min_score:
                continue
            scored_rows.append(
                {
                    "theme_id": theme_id,
                    "theme_name": current_row["theme_name"],
                    "current_theme_etf_count": int(theme_counts.get(theme_id, 0)),
                    "current_etf_codes": ",".join(sorted(existing_codes)),
                    "candidate_etf_code": item.etf_code,
                    "candidate_etf_name": item.etf_name,
                    "candidate_score": round(float(score), 4),
                    "matched_keywords": "|".join(matched_keywords),
                    "universe_as_of_date": item.as_of_date,
                }
            )
        scored_rows.sort(key=lambda x: (-float(x["candidate_score"]), str(x["candidate_etf_code"])))
        for rank, row in enumerate(scored_rows[:top_n], start=1):
            row["candidate_rank"] = rank
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out[
        [
            "theme_id",
            "theme_name",
            "current_theme_etf_count",
            "current_etf_codes",
            "candidate_rank",
            "candidate_etf_code",
            "candidate_etf_name",
            "candidate_score",
            "matched_keywords",
            "universe_as_of_date",
        ]
    ].sort_values(["theme_id", "candidate_rank", "candidate_score"], ascending=[True, True, False]).reset_index(drop=True)


def write_markdown_report(candidate_df: pd.DataFrame, theme_master: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    theme_summary = (
        candidate_df.groupby("theme_id")["candidate_etf_code"].nunique().rename("candidate_count")
        if not candidate_df.empty
        else pd.Series(dtype="int64")
    )
    expandable = int((theme_summary >= 1).sum()) if not theme_summary.empty else 0
    multi_expandable = int((theme_summary >= 2).sum()) if not theme_summary.empty else 0
    total_themes = int(theme_master["theme_id"].nunique())
    lines = [
        "# Theme ETF Master Expansion Review",
        "",
        "## Current Structure",
        "",
        f"- current themes in `theme_etf_master.csv`: {total_themes}",
        f"- all current themes are single-ETF mapped: {theme_master.groupby('theme_id')['etf_code'].nunique().max() == 1}",
        f"- themes with at least 1 candidate from current ETF universe: {expandable}",
        f"- themes with at least 2 candidate ETFs from current ETF universe: {multi_expandable}",
        "",
        "## Interpretation",
        "",
        "- `build_stock_theme_daily.py` is now breadth-aware, but actual `theme_topk_count>1` still requires multiple ETF mappings per theme/date.",
        "- The next bottleneck is no longer the aggregation code itself but `theme_etf_master.csv` input breadth.",
        "- Candidate rows below are heuristic suggestions from the current KRX ETF name universe. They should be manually reviewed before promotion.",
        "",
        "## Theme Candidate Summary",
        "",
    ]
    if candidate_df.empty:
        lines.extend(["- no candidate ETF rows met the current score threshold.", ""])
    else:
        summary_df = (
            candidate_df.groupby(["theme_id", "theme_name"], as_index=False)
            .agg(candidate_count=("candidate_etf_code", "nunique"), top_candidate=("candidate_etf_name", "first"))
            .sort_values(["candidate_count", "theme_id"], ascending=[False, True])
        )
        for row in summary_df.itertuples(index=False):
            lines.append(f"- `{row.theme_id}` {row.theme_name}: candidates={row.candidate_count}, top_candidate={row.top_candidate}")
        lines.append("")
        lines.append("## Recommended Next Step")
        lines.append("")
        lines.append("- Start with themes that have at least 2 candidates and already show ranking relevance: `HBM`, `SEMIEQP`, `AIPCB`, `POWER`, `BATTERY`, `AISOFT` if present in the candidate table.")
        lines.append("- Promote only KRX ETFs with actual price history already available in the project.")
        lines.append("- After updating `theme_etf_master.csv`, rerun `compute_theme_etf_daily.py` and `build_stock_theme_daily.py`, then verify whether `theme_topk_count>1` appears in `theme_level_aggregation_debug.csv`.")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    setup_logging()
    args = parse_args()
    theme_master = load_theme_master(args.theme_etf_master)
    try:
        etf_universe = fetch_all_etf_names()
    except Exception as exc:
        logging.warning("Remote ETF universe load failed -> fallback to local ETF rows: %s", exc)
        etf_universe = load_local_etf_universe(theme_master)
        if etf_universe.empty:
            raise
    candidate_df = build_candidate_frame(
        theme_master=theme_master,
        etf_universe=etf_universe,
        top_n=args.top_n,
        min_score=args.min_score,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    candidate_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    write_markdown_report(candidate_df, theme_master, args.output_md)
    logging.info(
        "Expansion analysis completed themes=%s candidate_rows=%s output_csv=%s output_md=%s",
        theme_master["theme_id"].nunique(),
        len(candidate_df),
        args.output_csv,
        args.output_md,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
