@echo off
REM LeeTrader - Daily pipeline runner
REM This script changes to project dir and runs the python pipeline via Docker Compose.

setlocal
cd /d D:\ai\Lee_trader
docker compose run --rm python-pipeline
endlocal
