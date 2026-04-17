# merge_universe.py
"""
universe_base.csv + interest_universe.csv 를 합쳐
최종 universe.csv 를 만드는 스크립트.

- universe_base.csv : 자동으로 선정된 종목들 (시총/유동성 기준)
- interest_universe.csv : 사용자가 수동으로 추가한 관심 종목들
- universe.csv : 이 둘을 합친 최종 유니버스 (파이프라인 전체에서 사용)

이 파일을 run_pipeline.py 안에서
fetch_top_universe 다음에 실행시키면 된다.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def load_csv_if_exists(path: Path) -> pd.DataFrame:
    """파일이 있으면 읽고, 없으면 빈 DataFrame 리턴."""
    if path.exists():
        return pd.read_csv(path)
    else:
        return pd.DataFrame()

def main():
    base_path = DATA_DIR / "universe.csv"
    interest_path = DATA_DIR / "interest_universe.csv"
    final_path = DATA_DIR / "universe.csv"

    # 1) 기본 유니버스 로딩
    df_base = load_csv_if_exists(base_path)
    if df_base.empty:
        print(f"[WARN] {base_path.name} 가 비어있거나 없습니다.")
    
    # 2) 관심 유니버스 로딩
    df_interest = load_csv_if_exists(interest_path)
    if df_interest.empty:
        print(f"[INFO] 관심 종목 파일({interest_path.name})이 없거나 비어있습니다.")
    
    # 관심 종목이 없으면 그냥 base를 universe로 저장하고 끝낼 수도 있음
    if df_interest.empty:
        if not df_base.empty:
            df_base.to_csv(final_path, index=False)
            print(f"[INFO] 관심 종목 없이 {final_path.name} 저장 완료.")
        else:
            print("[ERROR] 기본 유니버스와 관심 종목이 모두 비어 있습니다.")
        return

    # 3) code 컬럼 문자열화 (0으로 시작하는 코드 보호)
    for df in (df_base, df_interest):
        if not df.empty and "code" in df.columns:
            df["code"] = df["code"].astype(str).str.zfill(6)

    # 4) concat + 중복 제거
    df_all = pd.concat([df_base, df_interest], ignore_index=True)
    if "code" not in df_all.columns:
        raise ValueError("universe_base.csv 또는 interest_universe.csv 에 'code' 컬럼이 없습니다.")

    df_all = df_all.drop_duplicates(subset=["code"])

    # 5) 컬럼 정리 (name/market/sector가 없으면 만들어 둔다)
    for col in ["name", "market", "sector"]:
        if col not in df_all.columns:
            df_all[col] = None

    # (선택) 여기서 추가로 market/sector를 채우는 로직을 넣을 수 있음
    # 예: sectors_template.csv, price 메타에서 merge 등

    # 6) 최종 저장
    df_all.to_csv(final_path, index=False)
    print(f"[INFO] 최종 유니버스 {final_path.name} 저장 완료. 종목 수: {len(df_all)}")

if __name__ == "__main__":
    main()
