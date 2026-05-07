FROM python:3.11-slim

WORKDIR /app

COPY python/requirements.txt ./requirements.txt

# LightGBM runtime dependency (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Fail the image build immediately if required SQL assets were excluded.
RUN test -f /app/postgres/analytics_live_trade_views.sql
RUN test -f /app/postgres/live_trade_ledger_tables.sql

RUN mkdir -p /app/data /app/logs
