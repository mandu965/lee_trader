# Node LTS (20) 슬림 이미지 사용
FROM node:20-slim

# 런타임 환경
ENV NODE_ENV=production

# 앱 루트
WORKDIR /app/node

# 의존성만 먼저 복사(캐시 최적화)
COPY node/package*.json ./

# 프로덕션 의존성 설치
RUN npm ci --omit=dev

# 앱 소스 복사 (public, index.js 등)
COPY node/ .

# 데이터 폴백용 빈 디렉터리 생성 (DB 우선 사용, CSV는 백업용)
RUN mkdir -p /app/data

# Render가 할당하는 PORT 사용 (index.js에서 PORT 읽음)
EXPOSE 3000

CMD ["node", "index.js"]
