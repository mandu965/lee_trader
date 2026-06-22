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
modified: 2026-04-20
featured: true
indexable: true
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
- `modified`는 실제 본문을 수정한 날에만 갱신합니다. 단순 배포일로 바꾸지 않습니다.
- `indexable: false`면 검수 대기 글로 처리되어 sitemap·홈·목록에서 제외되고,
  상세 URL에는 `noindex, follow`가 적용되며 AdSense 스크립트를 넣지 않습니다.
- 출처, 실제 사례, 내부 링크, 작성자 관점이 충분해진 뒤 `indexable: true`로 승격합니다.
- 본문 내부 링크는 상대 경로 대신 `/blog/...`, `/reports/...` 형태를 권장합니다.
