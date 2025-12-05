Lee_trader SQLite Schema (현재 DB 기준)
======================================

공통
----
- DB 파일: `data/lee_trader.db`
- 날짜 컬럼은 YYYY-MM-DD 문자열로 저장됨.
- 모든 코드(code)는 문자열(선행 0 유지).

stocks
------
- code TEXT PRIMARY KEY
- name TEXT NOT NULL
- market TEXT
- sector TEXT
- listed_at DATE
- delisted_at DATE

market_status
-------------
- date DATE PRIMARY KEY
- kospi_close REAL
- kospi_ma20 REAL
- volatility_5d REAL
- foreign_net_5d REAL
- market_up INTEGER

prices_raw / prices_clean / prices_adjusted
-------------------------------------------
- date DATE NOT NULL
- code TEXT NOT NULL
- open REAL (raw/clean만)
- high REAL (raw/clean만)
- low REAL (raw/clean만)
- close REAL (raw/clean만)
- adj_open REAL (adjusted만)
- adj_high REAL (adjusted만)
- adj_low REAL (adjusted만)
- adj_close REAL (adjusted만)
- volume REAL
- PRIMARY KEY (date, code)

fundamentals
------------
- date DATE NOT NULL
- code TEXT NOT NULL
- roe REAL
- op_margin REAL
- debt_ratio REAL
- ocf_to_assets REAL
- net_margin REAL
- PRIMARY KEY (date, code)

quality
-------
- date DATE NOT NULL
- code TEXT NOT NULL
- quality_score REAL
- PRIMARY KEY (date, code)

features
--------
- date DATE NOT NULL
- code TEXT NOT NULL
- close REAL
- ret_1d REAL
- ret_5d REAL
- ret_10d REAL
- mom_20 REAL
- ma_5 REAL
- ma_20 REAL
- ma_60 REAL
- close_over_ma20 REAL
- vol_20 REAL
- vol_60 REAL
- rsi_14 REAL
- volume REAL
- vol_ma_20 REAL
- vol_ratio_20 REAL
- quality_score REAL
- PRIMARY KEY (date, code)

labels
------
- date DATE NOT NULL
- code TEXT NOT NULL
- target_60d REAL
- target_90d REAL
- target_log_60d REAL
- target_log_90d REAL
- target_mdd_60d REAL
- target_mdd_90d REAL
- target_60d_top20 REAL
- target_90d_top20 REAL
- realized_price_60d REAL
- realized_price_90d REAL
- PRIMARY KEY (date, code)
- FOREIGN KEY (code) REFERENCES stocks(code)

predictions
-----------
- date DATE NOT NULL
- code TEXT NOT NULL
- pred_return_60d REAL
- pred_return_90d REAL
- pred_mdd_60d REAL
- pred_mdd_90d REAL
- prob_top20_60d REAL
- prob_top20_90d REAL
- score REAL
- PRIMARY KEY (date, code)

daily_scores
------------
- date DATE NOT NULL
- code TEXT NOT NULL
- score REAL
- composite REAL
- PRIMARY KEY (date, code)
- FOREIGN KEY (code) REFERENCES stocks(code)

daily_ranking
-------------
- date DATE NOT NULL
- code TEXT NOT NULL
- close REAL
- pred_return_60d REAL
- pred_return_90d REAL
- pred_mdd_60d REAL
- pred_mdd_90d REAL
- prob_top20_60d REAL
- prob_top20_90d REAL
- score REAL
- score_score REAL
- composite REAL
- quality_score REAL
- name TEXT
- market TEXT
- sector TEXT
- tech_score REAL
- pred_score REAL
- ret_score REAL
- prob_score REAL
- qual_score REAL
- safety_score REAL
- liquidity_score REAL
- final_score REAL
- risk_penalty REAL
- market_up INTEGER
- market_status_date DATE
- market_kospi_close REAL
- market_kospi_ma20 REAL
- market_vol_5d REAL
- market_foreign_5d REAL
- generated_at TEXT
- model_version TEXT
- PRIMARY KEY (date, code)
- FOREIGN KEY (code) REFERENCES stocks(code)

backtest_trades
---------------
- id INTEGER PRIMARY KEY AUTOINCREMENT
- trade_date DATE NOT NULL
- strategy TEXT NOT NULL
- code TEXT NOT NULL
- final_score REAL
- pred_return_60d REAL
- realized_return REAL NOT NULL
- created_at TEXT DEFAULT (datetime('now'))
- FOREIGN KEY (code) REFERENCES stocks(code)

trades
------
- trade_id INTEGER PRIMARY KEY AUTOINCREMENT
- date DATE NOT NULL
- side TEXT NOT NULL
- code TEXT NOT NULL
- name TEXT
- market TEXT
- sector TEXT
- qty REAL
- price REAL
- amount REAL
- fee REAL
- memo TEXT
- created_at TEXT
