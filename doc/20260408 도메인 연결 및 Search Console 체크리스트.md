# 2026-04-08 도메인 연결 및 Search Console 체크리스트

기준일: 2026-04-08 KST

이 문서는 `Lee_trader` 공개 사이트를 실제 도메인에 연결하고, Search Console 및 애드센스 심사 준비까지 이어가기 위한 실행 체크리스트입니다.

---

## 1. 목표

이번 단계의 목표는 아래 4가지입니다.

- 실제 도메인 연결
- 공개 사이트 canonical / sitemap 기준 URL 확정
- Search Console 등록
- 애드센스 신청 전 기술 조건 점검

현재 코드 기준으로 공개 사이트의 기준 URL은 `SITE_BASE_URL` 환경변수로 제어합니다.

관련 파일:
- [.env.example](D:/ai/Lee_trader/.env.example)
- [node/index.js](D:/ai/Lee_trader/node/index.js)

---

## 2. 도메인 선택 기준

도메인은 짧고 기억하기 쉬워야 하고, 공개 콘텐츠 사이트와 운영 앱을 함께 담을 수 있어야 합니다.

권장 조건:
- 15자 내외
- 영문 소문자 위주
- 하이픈 최소화
- 너무 일반적인 투자 키워드 남용 지양
- 서비스 확장 시에도 어색하지 않을 것

예시 방향:
- `leetraderlab.com`
- `leetrader.ai`
- `leetraderlab.kr`

실무적으로는 `.com` 우선, 없으면 `.ai` 또는 `.kr`을 검토하는 편이 무난합니다.

---

## 3. DNS / 서버 연결 순서

## 3.1 도메인 구입 후 할 일

1. 도메인 구매
2. DNS 관리 화면 접속
3. 서버 공인 IP 확인
4. `A 레코드` 설정
5. 필요하면 `www`용 `CNAME` 추가

예시:
- `@` -> 서버 IP
- `www` -> `@`

## 3.2 서버 측 할 일

도메인이 연결되면 `.env`에 아래 값을 넣습니다.

```env
SITE_BASE_URL=https://www.실제도메인.com
```

그 다음 Node API를 다시 빌드합니다.

```powershell
docker compose up -d --build node-api
```

이 값이 반영되면 아래 항목이 실제 도메인 기준으로 바뀝니다.

- canonical
- og:url
- robots.txt의 sitemap 경로
- sitemap.xml 내부 URL

---

## 4. HTTPS 준비

애드센스와 검색 신뢰도 관점에서 HTTPS는 사실상 필수입니다.

권장 방식:
- Nginx reverse proxy
- Let's Encrypt 인증서

필수 조건:
- `http://도메인` 접속 시 `https://도메인`으로 리다이렉트
- `www` 사용 여부를 정하고 한쪽으로 통일

권장 정규화 예시:
- `http://example.com` -> `https://www.example.com`
- `https://example.com` -> `https://www.example.com`

즉 canonical과 실제 접속 URL이 항상 같아야 합니다.

---

## 5. Search Console 등록 순서

## 5.1 등록 전 확인

- 공개 홈 접속 가능
- `/reports`, `/blog` 접속 가능
- 상세 글 URL 접속 가능
- `robots.txt` 정상 응답
- `sitemap.xml` 정상 응답

## 5.2 등록 절차

1. Search Console 접속
2. 속성 추가
3. `도메인 속성` 또는 `URL 접두어 속성` 선택
4. 소유권 인증
5. `sitemap.xml` 제출

권장:
- 가능하면 `도메인 속성`으로 등록

인증 방식 예시:
- DNS TXT 레코드

등록 후 제출할 값:

```text
https://www.실제도메인.com/sitemap.xml
```

---

## 6. 애드센스 신청 전 기술 체크

## 6.1 필수 수준

- 실제 도메인 연결 완료
- HTTPS 적용 완료
- 공개 페이지 정상 노출
- About / Contact / Privacy / Terms / Disclaimer 존재
- 글 15~20개 이상 게시
- 각 글에 본문과 날짜 존재

## 6.2 권장

- 홈에서 리포트/블로그로 이동 가능
- 목록에서 상세 글로 자연스럽게 이동 가능
- 앱 페이지와 공개 페이지가 탐색상 구분됨
- 404 처리 존재
- 모바일에서도 읽기 가능

---

## 7. 운영 체크리스트

도메인 연결 직후 아래를 반드시 확인합니다.

1. 공개 홈 열기
2. 글 상세 열기
3. 브라우저 소스 보기에서 `canonical` 확인
4. `robots.txt` 확인
5. `sitemap.xml` 확인
6. Search Console 제출

점검 명령 예시:

```powershell
Invoke-WebRequest -UseBasicParsing https://www.실제도메인.com/robots.txt
Invoke-WebRequest -UseBasicParsing https://www.실제도메인.com/sitemap.xml
Invoke-WebRequest -UseBasicParsing https://www.실제도메인.com/blog/why-score-is-not-buy-signal
```

---

## 8. 실제 적용 예시

예를 들어 최종 도메인을 `https://www.leetraderlab.com`으로 정하면:

`.env`

```env
SITE_BASE_URL=https://www.leetraderlab.com
```

재기동:

```powershell
docker compose up -d --build node-api
```

확인:

- `https://www.leetraderlab.com/robots.txt`
- `https://www.leetraderlab.com/sitemap.xml`
- 상세 글 canonical이 `https://www.leetraderlab.com/...` 으로 나오는지 확인

---

## 9. 다음 단계

도메인 연결 이후 바로 이어갈 일은 아래 3개입니다.

1. Analytics 연결
2. Search Console 인덱싱 상태 확인
3. 애드센스 신청

그 다음에는 아래를 진행하면 됩니다.

- 404 페이지 추가
- 공개 글 지속 발행
- 홈/목록/상세 페이지 디자인 고도화

