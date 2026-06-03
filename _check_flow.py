import os
from dotenv import load_dotenv
import psycopg2
load_dotenv()
conn = psycopg2.connect(os.environ['DATABASE_URL'])
cur = conn.cursor()

# features 테이블에 flow 컬럼 존재 여부
cur.execute("""
    SELECT column_name FROM information_schema.columns
    WHERE table_name='features' AND column_name LIKE 'flow%'
    ORDER BY column_name
""")
flow_cols = [r[0] for r in cur.fetchall()]
print('flow columns in features:', flow_cols)

# features 테이블 최근 날짜별 flow 데이터 현황
if flow_cols:
    col = flow_cols[0]
    cur.execute(f"""
        SELECT date, COUNT(*) as total,
               SUM(CASE WHEN {col} IS NOT NULL THEN 1 ELSE 0 END) as not_null
        FROM features
        GROUP BY date ORDER BY date DESC LIMIT 5
    """)
    print(f'\nfeatures {col} by date:')
    for row in cur.fetchall():
        print(row)
else:
    cur.execute("SELECT date, COUNT(*) FROM features GROUP BY date ORDER BY date DESC LIMIT 5")
    print('\nfeatures rows by date:')
    for row in cur.fetchall():
        print(row)

# features.csv 최신 파일 확인
import pandas as pd
csv_path = 'data/features.csv'
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path, dtype={'code': str}, nrows=5)
    flow_csv_cols = [c for c in df.columns if 'flow' in c]
    print(f'\nfeatures.csv flow columns: {flow_csv_cols}')
    if flow_csv_cols:
        print(df[['code'] + flow_csv_cols].head())
else:
    print('\nfeatures.csv not found')

conn.close()
