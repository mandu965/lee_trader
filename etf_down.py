"""
Download day-level ETF prices from Naver with a proper User-Agent and pagination.
Output: date, close columns only (UTF-8).
"""

import io
import time
import pandas as pd
import requests
from pathlib import Path

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
}


def download_naver_etf(etf_code: str, out_path: Path, max_pages: int = 120, verify_ssl: bool = True) -> None:
    all_rows = []
    for page in range(1, max_pages + 1):
        url = f"https://finance.naver.com/item/sise_day.naver?code={etf_code}&page={page}"
        resp = requests.get(url, headers=HEADERS, timeout=10, verify=verify_ssl)
        resp.raise_for_status()
        # Naver is EUC-KR; use content to avoid encoding issues
        df_list = pd.read_html(io.BytesIO(resp.content))
        if not df_list:
            break
        df = df_list[0]
        df = df.dropna(subset=["날짜", "종가"])
        if df.empty:
            break
        df = df.rename(columns={"날짜": "date", "종가": "close"})
        df["date"] = pd.to_datetime(df["date"])
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        all_rows.append(df[["date", "close"]])
        # polite delay
        time.sleep(0.2)

    if not all_rows:
        raise RuntimeError(f"No data fetched for {etf_code}")

    out = pd.concat(all_rows, ignore_index=True)
    out = out.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"saved: {out_path} rows={len(out)}")


if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    download_naver_etf("069500", Path("data/benchmarks/kospi_069500.csv"), verify_ssl=False)
    download_naver_etf("229200", Path("data/benchmarks/kosdaq_229200.csv"), verify_ssl=False)
