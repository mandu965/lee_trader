FROM python:3.11-slim

# 프로젝트 전체를 /app 에 둘 거라, workdir 은 /app
WORKDIR /app

# 먼저 requirements 설치 (python 폴더 안에 있음)
COPY python/requirements.txt ./requirements.txt
# LightGBM runtime dependency (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 전체 복사
COPY . .

# data / logs 디렉토리 보장
RUN mkdir -p /app/data /app/logs
