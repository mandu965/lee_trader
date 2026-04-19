# Markdown Content Guide

게시글은 아래 두 폴더에서 관리합니다.

- `node/content/blog`
- `node/content/reports`

파일 예시:

```md
---
title: 글 제목
slug: seo-friendly-slug
category: 카테고리명
excerpt: 목록과 메타 설명에 사용할 짧은 소개문
date: 2026-04-20
featured: true
---
본문은 마크다운으로 작성합니다.

## 소제목

- 목록
- 목록

[내부 링크](/blog/example-post)
```

주의:

- `slug`는 URL에 그대로 사용됩니다.
- `excerpt`는 목록 요약과 상세 메타 설명에 사용됩니다.
- `featured: true`면 홈과 목록에서 우선 노출됩니다.
- 본문 내부 링크는 상대 경로 대신 `/blog/...`, `/reports/...` 형태를 권장합니다.
