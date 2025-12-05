param(
    [string]$ProjectRoot = "."
)

function Write-NoBom {
    param(
        [string]$Path,
        [string]$Content
    )
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Content, $utf8NoBom)
}

$root = Resolve-Path $ProjectRoot

Write-Host "Project root: $root"

# Create folders
$pythonDir = Join-Path $root "python"
$nodeDir   = Join-Path $root "node"
$dataDir   = Join-Path $root "data"

New-Item -ItemType Directory -Force -Path $pythonDir | Out-Null
New-Item -ItemType Directory -Force -Path $nodeDir   | Out-Null
New-Item -ItemType Directory -Force -Path $dataDir   | Out-Null

Write-Host "Created folders: python, node, data"

# --------------------------------------------------
# docker-compose.yml  (UTF-8 no BOM)
# --------------------------------------------------
$compose = @'
version: "3.9"

services:
  python-pipeline:
    build: ./python
    container_name: lee_trader_pipeline
    working_dir: /app
    command: ["python", "run_pipeline.py"]
    volumes:
      - ./data:/app/data
    env_file:
      - .env

  node-api:
    build: ./node
    container_name: lee_trader_api
    working_dir: /app
    command: ["npm", "run", "start"]
    ports:
      - "3000:3000"
    volumes:
      - ./data:/app/data
    env_file:
      - .env
'@

Write-NoBom (Join-Path $root "docker-compose.yml") $compose
Write-Host "docker-compose.yml created"

# --------------------------------------------------
# .env 템플릿
# --------------------------------------------------
$envContent = @'
KIS_BASE_URL=https://openapivts.koreainvestment.com:29443
KIS_APP_KEY=YOUR_APP_KEY
KIS_APP_SECRET=YOUR_APP_SECRET
KIS_APP_ID=YOUR_ACCOUNT_ID
KIS_APP_PASSWORD=YOUR_ACCOUNT_PASSWORD
NODE_ENV=development
'@

Write-NoBom (Join-Path $root ".env") $envContent
Write-Host ".env created"

# --------------------------------------------------
# python/Dockerfile
# --------------------------------------------------
$pyDocker = @'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /app/data
'@

Write-NoBom (Join-Path $pythonDir "Dockerfile") $pyDocker
Write-Host "python/Dockerfile created"

# requirements.txt
$req = @'
requests
pandas
numpy
scikit-learn
xgboost
python-dotenv
'@

Write-NoBom (Join-Path $pythonDir "requirements.txt") $req
Write-Host "python/requirements.txt created"

# --------------------------------------------------
# Python pipeline skeleton
# --------------------------------------------------
$pipelineFiles = @(
    "run_pipeline.py",
    "download_prices_kis.py",
    "clean_prices.py",
    "feature_builder.py",
    "scoring.py",
    "label_builder.py",
    "model_train.py",
    "model_predict.py"
)

foreach ($f in $pipelineFiles) {
    $path = Join-Path $pythonDir $f
    if (-not (Test-Path $path)) {
        Write-NoBom $path "print('$f TODO implement')"
    }
}

# run_pipeline.py 내용 덮어쓰기
$runPipeline = @'
import subprocess
import sys
import logging
from datetime import datetime
from pathlib import Path

STEPS = [
    ("download_prices_kis", "download_prices_kis.py"),
    ("clean_prices", "clean_prices.py"),
    ("feature_builder", "feature_builder.py"),
    ("scoring", "scoring.py"),
    ("label_builder", "label_builder.py"),
    ("model_train", "model_train.py"),
    ("model_predict", "model_predict.py"),
]

def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logging.info("Pipeline started")

def run_step(name, script):
    logging.info(f"Running step: {name} -> {script}")
    script_path = Path(script)
    if not script_path.exists():
        logging.warning(f"Script missing: {script_path.resolve()}")
        return
    result = subprocess.run([sys.executable, str(script_path)], check=False)
    if result.returncode != 0:
        logging.error(f"Step failed: {name} (return code={result.returncode})")
        raise RuntimeError(f"Step {name} failed")
    logging.info(f"Step completed: {name}")

def main():
    setup_logging()
    try:
        for name, script in STEPS:
            run_step(name, script)
    except Exception as e:
        logging.exception(f"Pipeline error: {e}")
        sys.exit(1)
    logging.info("Pipeline finished successfully")

if __name__ == "__main__":
    main()
'@

Write-NoBom (Join-Path $pythonDir "run_pipeline.py") $runPipeline
Write-Host "Python pipeline skeleton created"

# --------------------------------------------------
# node/Dockerfile, package.json, index.js
# --------------------------------------------------
$nodeDocker = @'
FROM node:20
WORKDIR /app
COPY package*.json ./
RUN npm install --legacy-peer-deps
COPY . .
EXPOSE 3000
CMD ["npm", "run", "start"]
'@

Write-NoBom (Join-Path $nodeDir "Dockerfile") $nodeDocker
Write-Host "node/Dockerfile created"

$packageJson = @'
{
  "name": "lee-trader-api",
  "version": "1.0.0",
  "description": "Minimal API",
  "main": "index.js",
  "scripts": {
    "start": "node index.js"
  },
  "dependencies": {
    "express": "^4.19.2",
    "cors": "^2.8.5"
  }
}
'@

Write-NoBom (Join-Path $nodeDir "package.json") $packageJson

$indexJs = @'
const express = require("express");
const cors = require("cors");
const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

app.get("/api/health", (req, res) => {
  res.json({ status: "ok", message: "API running" });
});

app.listen(PORT, () => console.log("Server running on port " + PORT));
'@

Write-NoBom (Join-Path $nodeDir "index.js") $indexJs
Write-Host "Node minimal API created"

Write-Host "=== Bootstrap Finished (UTF-8 no BOM files) ==="
