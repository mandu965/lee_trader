# AdSense 승인 준비 체크리스트

Lee Trader Lab은 애드센스 심사에서 정보형 금융 콘텐츠 사이트로 보이도록 운영한다.

## 공개 사이트 구조

- 홈: `/`
- 소개: `/about`
- 방법론: `/methodology`
- 용어 해설: `/glossary`
- 문의: `/contact`
- 개인정보처리방침: `/privacy`
- 이용약관: `/terms`
- 면책조항: `/disclaimer`
- 블로그: `/blog`
- 시장 해설: `/reports`
- 사이트맵: `/sitemap.xml`
- 광고 파일: `/ads.txt`

## 승인 전 확인 기준

- 공개 첫 화면은 운영 앱이 아니라 설명형 홈이어야 한다.
- 운영성 화면은 `/app` 이하 또는 별도 경로로 분리하고 `noindex` 처리한다.
- 블로그와 리포트 상세 페이지는 정상 한글, 발행일, 요약, 본문을 포함해야 한다.
- 금융 콘텐츠는 수익 보장, 즉시 매수, 무조건 상승 같은 표현을 피한다.
- 개인정보처리방침에는 쿠키, 광고, 분석 도구, 문의 정보 처리 기준을 적는다.
- 문의 페이지에는 실제 연락 가능한 이메일을 노출한다.
- `robots.txt`는 공개 콘텐츠를 허용하고 API와 운영 화면을 제한한다.
- `ads.txt`에는 Google publisher line을 유지한다.

## 로컬 점검

```bash
python python/check_adsense_readiness.py
node --check node/index.js
```

서버 확인이 필요하면 `node` 폴더에서 실행한다.

```bash
cd node
npm install
npm start
```

확인 URL:

- `http://localhost:3000/`
- `http://localhost:3000/api/site-library`
- `http://localhost:3000/sitemap.xml`

## 운영 원칙

애드센스 승인을 위해 광고보다 콘텐츠 신뢰도를 먼저 유지한다. 신규 글을 추가할 때는 최소 800자 이상의 설명형 본문, 명확한 제목, 과장 없는 결론, 면책 문구 또는 유의사항을 포함한다.
