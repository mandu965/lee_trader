# Node LTS 이미지
FROM node:18

# 앱 루트 폴더 설정
WORKDIR /app

# Ensure data directory exists so SQLite DB can be created at runtime
RUN mkdir -p /app/data

# node 폴더의 package.json만 먼저 복사 (캐시 최적화)
COPY node/package*.json ./node/

# node_modules 설치
RUN cd node && npm install

# 전체 프로젝트 복사
COPY . .

# node 서버가 있는 폴더로 이동
WORKDIR /app/node

# Render가 노출할 포트
EXPOSE 3000

# 서버 실행
CMD ["node", "index.js"]
