"""merge_flow 진단 스크립트"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import psycopg2

db_url = os.environ.get('DATABASE_URL', '')
print(f'DATABASE_URL prefix: {db_url[:40]}...')

# 1. flow_daily 최신 날짜 확인
conn = psycopg2.connect(db_url)
cur = conn.cursor()
cur.execute("SELECT date, COUNT(*), SUM(CASE WHEN fetch_status='success' THEN 1 ELSE 0 END) FROM flow_daily GROUP BY date ORDER BY date DESC LIMIT 5")
print('\n[flow_daily 최근 날짜]')
for r in cur.fetchall():
    print(f'  date={r[0]}  total={r[1]}  success={r[2]}')

# 2. features 최신 날짜 확인
cur.execute("SELECT date, COUNT(*) FROM features GROUP BY date ORDER BY date DESC LIMIT 3")
print('\n[features 최근 날짜]')
for r in cur.fetchall():
    print(f'  date={r[0]}  rows={r[1]}')

# 3. features의 date 컬럼 타입
cur.execute("SELECT data_type FROM information_schema.columns WHERE table_name='features' AND column_name='date'")
print('\n[features.date column type]', cur.fetchone())

# 4. flow_daily의 date 컬럼 타입
cur.execute("SELECT data_type FROM information_schema.columns WHERE table_name='flow_daily' AND column_name='date'")
print('[flow_daily.date column type]', cur.fetchone())

# 5. features.csv 날짜 샘플
import pandas as pd
csv = pd.read_csv('data/features.csv', dtype={'code': str}, nrows=3)
print(f'\n[features.csv date sample] type={type(csv["date"].iloc[0])} val={csv["date"].iloc[0]!r}')

# 6. flow_daily에서 features와 매칭 가능한지 직접 확인
latest_feature_date = csv['date'].iloc[0]
cur.execute("SELECT COUNT(*) FROM flow_daily WHERE date::text = %s AND fetch_status='success'", (str(latest_feature_date)[:10],))
print(f'[flow_daily rows matching feature date {str(latest_feature_date)[:10]}]', cur.fetchone())

conn.close()
