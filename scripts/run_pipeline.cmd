<<<<<<< HEAD
@echo off
REM LeeTrader - Daily pipeline runner
REM This script changes to project dir and runs the python pipeline via Docker Compose.

setlocal
cd /d D:\ai\Lee_trader
docker compose run --rm python-pipeline
endlocal
=======
@echo off
REM LeeTrader - Daily pipeline runner
REM This script changes to project dir and runs the python pipeline via Docker Compose.

setlocal
cd /d D:\ai\Lee_trader
docker compose run --rm python-pipeline
endlocal
>>>>>>> eac8d622da2de3cb84a3dc38e9c673de512459ae
