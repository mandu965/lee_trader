# Node LTS(20) 기반 슬림 이미지 사용
FROM node:20-slim

# 런타임 환경
ENV NODE_ENV=production

# 앱 루트 디렉터리
WORKDIR /app/node

# 1) 의존성 파일만 먼저 복사 (빌드 캐시 최적화)
#    로컬 레포 기준: node/package.json, node/package-lock.json 이 여기로 들어옴
COPY node/package*.json ./

# 2) 프로덕션 의존성 설치
#    - package-lock.json 이 없더라도 동작하도록 npm install 사용
#    - devDependencies 는 설치하지 않도록 --omit=dev 옵션 사용
RUN npm install --omit=dev

# 3) 실제 Node 소스 코드 복사 (index.js, routes 등)
COPY node/ .

# 4) 데이터 폴백용 디렉터리 생성 (DB가 없을 때를 대비한 기본 경로)
RUN mkdir -p /app/data

# 5) Render가 할당하는 PORT 사용 (index.js 에서 process.env.PORT 사용)
EXPOSE 3000

# 6) 앱 시작 커맨드
CMD ["node", "index.js"]
