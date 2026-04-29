const express = require("express");
const cors = require("cors");
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
require("dotenv").config();
const { Pool } = require("pg");
const operatorAccess = require("./operatorAccess");

const app = express();
const PORT = process.env.PORT || 3000;
const SITE_BASE_URL = (process.env.SITE_BASE_URL || "http://localhost:3000").replace(/\/+$/, "");
const GA_MEASUREMENT_ID = (process.env.GA_MEASUREMENT_ID || "G-TSJDVJKDFQ").trim();
const NAVER_ANALYTICS_WA = (process.env.NAVER_ANALYTICS_WA || "1680ec0ed78cdb0").trim();

// ---------------------
// Env / Postgres Pool
// ---------------------
function resolveDataDir() {
  const candidates = [
    "/app/data",
    path.join(__dirname, "data"),
    path.join(__dirname, "..", "data"),
    path.join(process.cwd(), "data"),
  ];
  for (const p of candidates) {
    try {
      if (fs.existsSync(p)) return p;
    } catch {}
  }
  return path.join(__dirname, "data");
}
const DATA_DIR = resolveDataDir();
console.log("[DATA_DIR]", DATA_DIR);

function resolveOutputsDir() {
  const candidates = [
    "/app/outputs",
    path.join(__dirname, "outputs"),
    path.join(__dirname, "..", "outputs"),
    path.join(process.cwd(), "outputs"),
  ];
  for (const p of candidates) {
    try {
      if (fs.existsSync(p)) return p;
    } catch {}
  }
  return path.join(__dirname, "outputs");
}
const OUTPUTS_DIR = resolveOutputsDir();
console.log("[OUTPUTS_DIR]", OUTPUTS_DIR);

function resolveServingDir() {
  const candidates = [
    "/app/serving",
    path.join(__dirname, "serving"),
    path.join(__dirname, "..", "serving"),
    path.join(process.cwd(), "serving"),
  ];
  for (const p of candidates) {
    try {
      if (fs.existsSync(p)) return p;
    } catch {}
  }
  return path.join(__dirname, "serving");
}
const SERVING_DIR = resolveServingDir();
console.log("[SERVING_DIR]", SERVING_DIR);

const { DATABASE_URL } = process.env;
if (!DATABASE_URL) {
  console.error("DATABASE_URL not set. API will fail to reach Postgres.");
}

const pool = new Pool({
  connectionString: DATABASE_URL,
  max: 10,
  idleTimeoutMillis: 0,
  connectionTimeoutMillis: 5000,
});
const VISITOR_COOKIE_NAME = "lt_visitor_id";
const VISITOR_COOKIE_MAX_AGE_SEC = 60 * 60 * 24 * 365;
const ANALYTICS_ROUTE_EXCLUDE_PREFIXES = ["/api", "/assets"];
const ANALYTICS_EXTENSIONS_EXCLUDE = new Set([
  ".js", ".css", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".map", ".txt", ".xml", ".json", ".woff", ".woff2",
]);
let pageViewSchemaReady = null;
let meaningfulnessReviewSchemaReady = null;
let liveTradeReviewSchemaReady = null;

// ---------------------
// Helpers
// ---------------------
const csvCache = new Map();
const jsonCache = new Map();
let siteLibraryCache = { cacheKey: null, items: [] };
const CONTENT_DIR = path.join(__dirname, "content");

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (m) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[m]));
}

function stripHtml(value) {
  return String(value ?? "")
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function slugify(value) {
  return String(value || "")
    .toLowerCase()
    .trim()
    .replace(/['"]/g, "")
    .replace(/[^a-z0-9가-힣]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function parseFrontMatter(raw) {
  const normalized = String(raw || "").replace(/^\uFEFF/, "").replace(/\r\n/g, "\n");
  if (!normalized.startsWith("---\n")) {
    return { attributes: {}, body: normalized };
  }
  const endIndex = normalized.indexOf("\n---\n", 4);
  if (endIndex === -1) {
    return { attributes: {}, body: normalized };
  }
  const header = normalized.slice(4, endIndex).split("\n");
  const attributes = {};
  header.forEach((line) => {
    const index = line.indexOf(":");
    if (index === -1) return;
    const key = line.slice(0, index).trim();
    const rawValue = line.slice(index + 1).trim();
    if (!key) return;
    if (/^(true|false)$/i.test(rawValue)) {
      attributes[key] = /^true$/i.test(rawValue);
      return;
    }
    if (key === "tags" && rawValue.includes(",")) {
      attributes[key] = rawValue.split(",").map((entry) => entry.trim()).filter(Boolean);
      return;
    }
    attributes[key] = rawValue.replace(/^"(.*)"$/, "$1").replace(/^'(.*)'$/, "$1");
  });
  return {
    attributes,
    body: normalized.slice(endIndex + 5).trim(),
  };
}

function renderInlineMarkdown(value) {
  let html = escapeHtml(value || "");
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a class="text-link" href="$2">$1</a>');
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  return html;
}

function renderMarkdown(markdown) {
  const lines = String(markdown || "").replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let paragraph = [];
  let listItems = [];

  function flushParagraph() {
    if (!paragraph.length) return;
    html.push(`<p>${renderInlineMarkdown(paragraph.join(" "))}</p>`);
    paragraph = [];
  }

  function flushList() {
    if (!listItems.length) return;
    html.push(`<ul>${listItems.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</ul>`);
    listItems = [];
  }

  lines.forEach((line) => {
    const text = line.trim();
    if (!text) {
      flushParagraph();
      flushList();
      return;
    }
    const heading = text.match(/^(#{1,3})\s+(.*)$/);
    if (heading) {
      flushParagraph();
      flushList();
      const level = Math.min(heading[1].length + 1, 4);
      html.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
      return;
    }
    const bullet = text.match(/^[-*]\s+(.*)$/);
    if (bullet) {
      flushParagraph();
      listItems.push(bullet[1]);
      return;
    }
    paragraph.push(text);
  });

  flushParagraph();
  flushList();
  return html.join("");
}

function estimateReadingTime(value) {
  const plain = stripHtml(value || "");
  const minutes = Math.max(3, Math.ceil(plain.length / 260));
  return `${minutes}분`;
}

function readJsonFile(filePath, fallback) {
  try {
    if (!fs.existsSync(filePath)) return fallback;
    const stat = fs.statSync(filePath);
    const cacheKey = `${stat.mtimeMs}:${stat.size}`;
    const cached = jsonCache.get(filePath);
    if (cached && cached.cacheKey === cacheKey) return cached.value;
    const raw = fs.readFileSync(filePath, "utf-8").replace(/^\uFEFF/, "");
    const value = JSON.parse(raw);
    jsonCache.set(filePath, { cacheKey, value });
    return value;
  } catch (error) {
    console.error("readJsonFile error", filePath, error);
    return fallback;
  }
}

function readHomepageContent() {
  return readJsonFile(path.join(CONTENT_DIR, "homepage.json"), {
    marketSummary: {},
    picks: [],
  });
}

function readMarkdownEntries(sectionDir, section) {
  if (!fs.existsSync(sectionDir)) return [];
  return fs.readdirSync(sectionDir)
    .filter((name) => name.endsWith(".md"))
    .map((name) => {
      const filePath = path.join(sectionDir, name);
      const raw = fs.readFileSync(filePath, "utf-8");
      const stat = fs.statSync(filePath);
      const { attributes, body } = parseFrontMatter(raw);
      const renderedBody = renderMarkdown(body);
      const title = attributes.title || path.basename(name, ".md");
      const slug = attributes.slug || slugify(path.basename(name, ".md"));
      const excerpt = attributes.excerpt || stripHtml(renderedBody).slice(0, 140);
      return {
        slug,
        section,
        category: attributes.category || "일반",
        title,
        excerpt,
        date: attributes.date || stat.mtime.toISOString().slice(0, 10),
        readingTime: attributes.readingTime || estimateReadingTime(renderedBody),
        featured: Boolean(attributes.featured),
        body: renderedBody,
      };
    });
}

function readSiteLibrary() {
  const sectionDirs = [
    path.join(CONTENT_DIR, "blog"),
    path.join(CONTENT_DIR, "reports"),
  ];
  const cacheKey = sectionDirs
    .filter((dirPath) => fs.existsSync(dirPath))
    .flatMap((dirPath) => fs.readdirSync(dirPath)
      .filter((name) => name.endsWith(".md"))
      .map((name) => {
        const stat = fs.statSync(path.join(dirPath, name));
        return `${dirPath}:${name}:${stat.mtimeMs}:${stat.size}`;
      }))
    .join("|");
  if (siteLibraryCache.cacheKey === cacheKey) return siteLibraryCache.items;

  const items = [
    ...readMarkdownEntries(path.join(CONTENT_DIR, "blog"), "blog"),
    ...readMarkdownEntries(path.join(CONTENT_DIR, "reports"), "report"),
  ].sort((a, b) => String(b.date).localeCompare(String(a.date)));
  siteLibraryCache = { cacheKey, items };
  return items;
}

function buildAbsoluteUrl(pathname) {
  return `${SITE_BASE_URL}${pathname.startsWith("/") ? pathname : `/${pathname}`}`;
}

function renderJsonLd(value) {
  if (!value) return "";
  const payload = Array.isArray(value) ? value : [value];
  return payload
    .filter(Boolean)
    .map((entry) => `<script type="application/ld+json">${JSON.stringify(entry)}</script>`)
    .join("\n");
}

function buildPublicPageMeta(pathname) {
  const normalized = pathname === "/index.html" ? "/" : pathname;
  const organization = {
    "@context": "https://schema.org",
    "@type": "Organization",
    name: "Lee Trader Lab",
    url: SITE_BASE_URL,
    email: "mandu965@naver.com",
  };
  const website = {
    "@context": "https://schema.org",
    "@type": "WebSite",
    name: "Lee Trader Lab",
    url: SITE_BASE_URL,
    inLanguage: "ko-KR",
    description: "국내 주식 데이터와 운영 기준을 설명형 콘텐츠로 제공하는 금융 정보 플랫폼",
  };
  const pages = {
    "/": {
      title: "Lee Trader Lab | 국내 주식 데이터 해설과 투자 판단 가이드",
      description: "국내 주식 시장 해설, 투자 판단 기준, 용어 설명, 운영 메모를 제공하는 금융 정보형 플랫폼입니다.",
      canonicalPath: "/",
      type: "website",
      structuredData: [organization, website],
    },
    "/about": {
      title: "회사 소개 | Lee Trader Lab",
      description: "Lee Trader Lab의 운영 목적, 콘텐츠 원칙, 데이터 해석 기준, 독자 대상 범위를 소개합니다.",
      canonicalPath: "/about",
      structuredData: [
        organization,
        {
          "@context": "https://schema.org",
          "@type": "AboutPage",
          name: "회사 소개",
          url: buildAbsoluteUrl("/about"),
          description: "Lee Trader Lab의 운영 목적과 콘텐츠 원칙 소개",
        },
      ],
    },
    "/methodology": {
      title: "방법론 | Lee Trader Lab",
      description: "점수, 시장 국면, 리스크 관리, 운영 가드, 검증 절차를 설명하는 방법론 페이지입니다.",
      canonicalPath: "/methodology",
    },
    "/glossary": {
      title: "용어 해설 | Lee Trader Lab",
      description: "Lee Trader Lab 공개 페이지에서 사용하는 핵심 투자·운영 용어를 쉬운 말로 풀이합니다.",
      canonicalPath: "/glossary",
    },
    "/operator-note": {
      title: "운영 안내 | Lee Trader Lab",
      description: "실거래 화면과 운영 로그를 어떻게 읽어야 하는지, 공개 범위와 한계를 안내합니다.",
      canonicalPath: "/operator-note",
    },
    "/contact": {
      title: "문의 | Lee Trader Lab",
      description: "서비스 문의, 콘텐츠 수정 요청, 광고·정책 관련 문의를 위한 연락 안내 페이지입니다.",
      canonicalPath: "/contact",
      structuredData: {
        "@context": "https://schema.org",
        "@type": "ContactPage",
        name: "문의",
        url: buildAbsoluteUrl("/contact"),
      },
    },
    "/privacy": {
      title: "개인정보처리방침 | Lee Trader Lab",
      description: "쿠키, 로그, 광고, 문의 처리 과정에서의 개인정보 처리 기준을 안내합니다.",
      canonicalPath: "/privacy",
    },
    "/terms": {
      title: "이용약관 | Lee Trader Lab",
      description: "사이트 이용 조건, 콘텐츠 사용 범위, 책임 제한, 금지 행위를 정리한 이용약관입니다.",
      canonicalPath: "/terms",
    },
    "/disclaimer": {
      title: "면책조항 | Lee Trader Lab",
      description: "본 사이트의 금융 정보 제공 범위, 투자 책임, 데이터 한계를 설명하는 면책 문구입니다.",
      canonicalPath: "/disclaimer",
    },
    "/reports": {
      title: "시장 해설 | Lee Trader Lab",
      description: "국내 주식 시장 국면, 수급, 리스크, 운영 관찰 포인트를 설명형 리서치로 제공합니다.",
      canonicalPath: "/reports",
      structuredData: {
        "@context": "https://schema.org",
        "@type": "CollectionPage",
        name: "시장 해설",
        url: buildAbsoluteUrl("/reports"),
      },
    },
    "/blog": {
      title: "블로그 | Lee Trader Lab",
      description: "투자 기초, 리스크 관리, 데이터 읽기, 운영 노하우를 다루는 금융 정보 블로그입니다.",
      canonicalPath: "/blog",
      structuredData: {
        "@context": "https://schema.org",
        "@type": "Blog",
        name: "Lee Trader Lab 블로그",
        url: buildAbsoluteUrl("/blog"),
      },
    },
    "/app": {
      title: "운영 앱 | Lee Trader Lab",
      description: "Lee Trader Lab의 운영 화면과 데이터 대시보드입니다. 일반 독자는 해설 콘텐츠를 먼저 읽는 것을 권장합니다.",
      canonicalPath: "/app",
    },
  };
  return pages[normalized] || null;
}

function applyPublicPageMeta(html, pathname) {
  const meta = buildPublicPageMeta(pathname);
  if (!meta) return html;

  let next = html;
  const canonicalUrl = buildAbsoluteUrl(meta.canonicalPath || pathname || "/");
  const title = escapeHtml(meta.title || "Lee Trader Lab");
  const description = escapeHtml(meta.description || "");
  const headExtras = [
    '<link rel="preconnect" href="https://www.googletagmanager.com" crossorigin>',
    '<link rel="dns-prefetch" href="//www.googletagmanager.com">',
    '<link rel="dns-prefetch" href="//pagead2.googlesyndication.com">',
    `<link rel="canonical" href="${escapeHtml(canonicalUrl)}">`,
    `<meta property="og:type" content="${escapeHtml(meta.type || "website")}">`,
    `<meta property="og:title" content="${title}">`,
    `<meta property="og:description" content="${description}">`,
    `<meta property="og:url" content="${escapeHtml(canonicalUrl)}">`,
    '<meta name="twitter:card" content="summary_large_image">',
    `<meta name="twitter:title" content="${title}">`,
    `<meta name="twitter:description" content="${description}">`,
    renderJsonLd(meta.structuredData),
  ].filter(Boolean).join("\n");

  next = next.replace(/<title>[\s\S]*?<\/title>/i, `<title>${title}</title>`);
  if (/<meta\s+name=["']description["'][^>]*>/i.test(next)) {
    next = next.replace(/<meta\s+name=["']description["'][^>]*>/i, `<meta name="description" content="${description}">`);
  } else {
    next = next.replace("</head>", `  <meta name="description" content="${description}">\n</head>`);
  }
  return injectHeadSnippet(next, headExtras);
}

function renderGoogleAnalyticsSnippet() {
  if (!GA_MEASUREMENT_ID) return "";
  const id = escapeHtml(GA_MEASUREMENT_ID);
  return `
  <!-- Google tag (gtag.js) -->
  <script async src="https://www.googletagmanager.com/gtag/js?id=${id}"></script>
  <script>
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());
    gtag('config', '${id}');
  </script>`;
}

function renderNaverAnalyticsSnippet() {
  if (!NAVER_ANALYTICS_WA) return "";
  const wa = escapeHtml(NAVER_ANALYTICS_WA);
  return `
  <script type="text/javascript" src="https://wcs.pstatic.net/wcslog.js"></script>
  <script type="text/javascript">
    if (!window.wcs_add) window.wcs_add = {};
    window.wcs_add["wa"] = "${wa}";
    if (window.wcs) {
      window.wcs_do();
    }
  </script>`;
}

function renderAnalyticsHeadSnippet() {
  return `${renderGoogleAnalyticsSnippet()}${renderNaverAnalyticsSnippet()}`;
}

function injectHeadSnippet(html, snippet) {
  if (!snippet) return html;
  return html.replace("</head>", `${snippet}\n</head>`);
}

function injectBodySnippet(html, snippet) {
  if (!snippet) return html;
  return html.replace("</body>", `${snippet}\n</body>`);
}

function renderOpsUnifiedNavSnippet(fileName) {
  const targets = new Set([
    "index.html",
    "ranking.html",
    "meaningfulness.html",
    "ops-readiness.html",
    "score-check.html",
    "manual-trading.html",
    "holdings.html",
    "holdingsDetail.html",
    "paper-trading.html",
    "trade-history.html",
    "live-auto-trading.html",
    "rule-auto-trading.html",
    "detail.html",
  ]);
  if (!targets.has(fileName)) return "";
  return '<script src="/ops-unified-nav.js?v=20260429-rule-nav-v1"></script>';
}

function renderArticlePage(item, section) {
  const related = readSiteLibrary()
    .filter((entry) => entry.slug !== item.slug && entry.section === item.section)
    .slice(0, 4);
  const canonicalPath = `/${section}/${item.slug}`;
  const title = `${item.title} | Lee Trader Lab`;
  const description = stripHtml(item.excerpt || item.body).slice(0, 160);
  const canonicalUrl = buildAbsoluteUrl(canonicalPath);
  const articleSchema = {
    "@context": "https://schema.org",
    "@type": "Article",
    headline: item.title,
    description,
    datePublished: item.date || undefined,
    dateModified: item.date || undefined,
    mainEntityOfPage: canonicalUrl,
    author: {
      "@type": "Organization",
      name: "Lee Trader Lab",
    },
    publisher: {
      "@type": "Organization",
      name: "Lee Trader Lab",
      url: SITE_BASE_URL,
    },
  };

  return `<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${escapeHtml(title)}</title>
  <meta name="description" content="${escapeHtml(description)}">
  <link rel="canonical" href="${escapeHtml(canonicalUrl)}">
  <meta property="og:type" content="article">
  <meta property="og:title" content="${escapeHtml(item.title)}">
  <meta property="og:description" content="${escapeHtml(description)}">
  <meta property="og:url" content="${escapeHtml(canonicalUrl)}">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="${escapeHtml(item.title)}">
  <meta name="twitter:description" content="${escapeHtml(description)}">
  <link rel="preconnect" href="https://www.googletagmanager.com" crossorigin>
  <link rel="dns-prefetch" href="//www.googletagmanager.com">
  <link rel="stylesheet" href="/site.css">
  ${renderJsonLd(articleSchema)}
${renderAnalyticsHeadSnippet()}
</head>
<body class="site-body">
  <header class="site-header">
    <div class="site-container site-header__inner">
      <a class="site-brand" href="/">
        <div class="site-brand__mark">LT</div>
        <div>
          <p class="site-brand__title">Lee Trader Lab</p>
          <p class="site-brand__subtitle">국내 주식 운영 해석과 시장 브리프</p>
        </div>
      </a>
      <button class="site-nav-toggle" type="button" aria-expanded="false" aria-controls="siteNav">메뉴</button>
      <nav id="siteNav" class="site-nav">
        <span class="site-nav__section">둘러보기</span>
        <a href="/">홈</a>
        <a href="/about">소개</a>
        <a href="/methodology">방법론</a>
        <a href="/glossary">용어 해설</a>
        <a href="/operator-note">운영 안내</a>
        <span class="site-nav__section">콘텐츠</span>
        <a class="${section === "reports" ? "is-active" : ""}" href="/reports">시장 해설</a>
        <a class="${section === "blog" ? "is-active" : ""}" href="/blog">블로그</a>
        <span class="site-nav__section">바로가기</span>
        <a class="site-nav__app" href="/app">운영 앱</a>
        <a class="site-nav__minor" href="/contact">문의</a>
      </nav>
    </div>
  </header>
  <main class="site-main">
    <div class="site-container">
      <div class="content-layout">
        <div>
          <article class="article-shell">
            <div class="article-meta">
              <span>${escapeHtml(item.category || "-")}</span>
              <span>${escapeHtml(item.date || "-")}</span>
              <span>${escapeHtml(item.readingTime || "-")}</span>
            </div>
            <h1>${escapeHtml(item.title)}</h1>
            <p class="article-shell__excerpt">${escapeHtml(item.excerpt || "")}</p>
            <div class="article-body">${item.body || ""}</div>
          </article>
        </div>
        <aside>
          <section class="side-panel">
            <h3>글 정보</h3>
            <ul>
              <li>분류: ${escapeHtml(item.category || "-")}</li>
              <li>발행일: ${escapeHtml(item.date || "-")}</li>
              <li>읽는 시간: ${escapeHtml(item.readingTime || "-")}</li>
            </ul>
          </section>
          <section class="side-panel">
            <h3>같이 읽을 글</h3>
            <ul>
              ${related.length ? related.map((entry) => `<li><a class="text-link" href="/${entry.section === "report" ? "reports" : "blog"}/${entry.slug}">${escapeHtml(entry.title)}</a></li>`).join("") : "<li>같은 섹션의 글이 아직 많지 않습니다.</li>"}
            </ul>
          </section>
          <section class="side-panel">
            <h3>바로 가기</h3>
            <ul>
              <li><a class="text-link" href="/methodology">방법론 읽기</a></li>
              <li><a class="text-link" href="/glossary">용어 해설 보기</a></li>
              <li><a class="text-link" href="/about">회사 소개 보기</a></li>
              <li><a class="text-link" href="/disclaimer">면책조항 확인</a></li>
              <li><a class="text-link" href="/app">운영 앱 열기</a></li>
            </ul>
          </section>
        </aside>
      </div>
    </div>
  </main>
  <footer class="site-footer">
    <div class="site-container site-footer__inner">
      <div>Lee Trader Lab · 설명 중심 국내 주식 정보 플랫폼</div>
      <div>
        <a class="text-link" href="/privacy">개인정보처리방침</a>
        ·
        <a class="text-link" href="/terms">이용약관</a>
        ·
        <a class="text-link" href="/disclaimer">면책조항</a>
      </div>
    </div>
  </footer>
  <script src="/site-shell.js"></script>
</body>
</html>`;
}

function readCsv(filePath) {
  try {
    if (!fs.existsSync(filePath)) return null;
    const stat = fs.statSync(filePath);
    const cacheKey = `${stat.mtimeMs}:${stat.size}`;
    const cached = csvCache.get(filePath);
    if (cached && cached.cacheKey === cacheKey) {
      return cached.rows;
    }
    const content = fs.readFileSync(filePath, "utf-8");
    const normalized = content.replace(/^\uFEFF/, "");
    const records = [];
    let field = "";
    let record = [];
    let inQuotes = false;

    for (let i = 0; i < normalized.length; i += 1) {
      const ch = normalized[i];
      const next = normalized[i + 1];

      if (ch === '"') {
        if (inQuotes && next === '"') {
          field += '"';
          i += 1;
        } else {
          inQuotes = !inQuotes;
        }
        continue;
      }

      if (ch === "," && !inQuotes) {
        record.push(field);
        field = "";
        continue;
      }

      if ((ch === "\n" || ch === "\r") && !inQuotes) {
        if (ch === "\r" && next === "\n") i += 1;
        record.push(field);
        field = "";
        if (record.some((value) => String(value || "").length > 0)) {
          records.push(record);
        }
        record = [];
        continue;
      }

      field += ch;
    }

    if (field.length || record.length) {
      record.push(field);
      if (record.some((value) => String(value || "").length > 0)) {
        records.push(record);
      }
    }

    if (!records.length) return [];
    const headers = records.shift().map((value) => String(value || "").trim());
    const rows = records.map((values) => {
      const row = {};
      headers.forEach((header, idx) => {
        row[header] = values[idx] ?? "";
      });
      return row;
    });
    csvCache.set(filePath, { cacheKey, rows });
    return rows;
  } catch (e) {
    console.warn("readCsv error", e.message);
    return null;
  }
}

function readJson(filePath) {
  try {
    if (!fs.existsSync(filePath)) return null;
    const stat = fs.statSync(filePath);
    const cacheKey = `${stat.mtimeMs}:${stat.size}`;
    const cached = jsonCache.get(filePath);
    if (cached && cached.cacheKey === cacheKey) {
      return cached.value;
    }
    const raw = fs.readFileSync(filePath, "utf-8");
    const normalized = raw
      .replace(/^\uFEFF/, "")
      .replace(/\bNaN\b/g, "null")
      .replace(/\bInfinity\b/g, "null")
      .replace(/\b-Infinity\b/g, "null");
    const value = JSON.parse(normalized);
    jsonCache.set(filePath, { cacheKey, value });
    return value;
  } catch (e) {
    console.warn("readJson error", e.message);
    return null;
  }
}

function readText(filePath) {
  try {
    if (!fs.existsSync(filePath)) return null;
    return fs.readFileSync(filePath, "utf-8");
  } catch (e) {
    console.warn("readText error", e.message);
    return null;
  }
}

function writeJson(filePath, value) {
  try {
    fs.mkdirSync(path.dirname(filePath), { recursive: true });
    fs.writeFileSync(filePath, JSON.stringify(value, null, 2), "utf-8");
    jsonCache.delete(filePath);
    return true;
  } catch (e) {
    console.warn("writeJson error", e.message);
    return false;
  }
}

function parseCookieHeader(headerValue) {
  const out = {};
  const raw = String(headerValue || "").trim();
  if (!raw) return out;
  raw.split(";").forEach((part) => {
    const idx = part.indexOf("=");
    if (idx <= 0) return;
    const key = part.slice(0, idx).trim();
    const value = part.slice(idx + 1).trim();
    if (!key) return;
    out[key] = decodeURIComponent(value);
  });
  return out;
}

function shouldTrackPageView(req) {
  if (!req || String(req.method || "").toUpperCase() !== "GET") return false;
  const pathname = String(req.path || req.originalUrl || "").split("?")[0] || "";
  if (!pathname) return false;
  if (ANALYTICS_ROUTE_EXCLUDE_PREFIXES.some((prefix) => pathname.startsWith(prefix))) return false;
  const ext = path.extname(pathname).toLowerCase();
  if (ext && ANALYTICS_EXTENSIONS_EXCLUDE.has(ext)) return false;
  const ua = String(req.headers["user-agent"] || "");
  if (/bot|spider|crawl|slurp|facebookexternalhit|kakaotalk-scrap|discordbot|whatsapp|telegrambot|bingpreview/i.test(ua)) {
    return false;
  }
  return true;
}

function ensureVisitorId(req, res) {
  const cookies = parseCookieHeader(req.headers?.cookie || "");
  let visitorId = String(cookies[VISITOR_COOKIE_NAME] || "").trim();
  if (!visitorId) {
    visitorId = crypto.randomUUID();
    res.append(
      "Set-Cookie",
      [
        `${VISITOR_COOKIE_NAME}=${encodeURIComponent(visitorId)}`,
        "Path=/",
        `Max-Age=${VISITOR_COOKIE_MAX_AGE_SEC}`,
        "SameSite=Lax",
        "HttpOnly",
      ].join("; ")
    );
  }
  return visitorId;
}

function hashIp(ipValue) {
  const raw = String(ipValue || "").trim();
  if (!raw) return null;
  const salt = String(process.env.OPERATOR_AUTH_SECRET || process.env.DATABASE_URL || "lt-analytics").slice(0, 64);
  return crypto.createHash("sha256").update(`${salt}:${raw}`).digest("hex");
}

function toNum(v) {
  if (v === null || v === undefined) return null;
  if (typeof v === "number" && Number.isFinite(v)) return v;
  const s = String(v).replace(/,/g, "").trim();
  if (!s) return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
}

const RECENT_SURGE_THRESHOLDS = {
  soft: { ret5d: 0.12, ret10d: 0.20, rsi14: 70 },
  hard: { ret5d: 0.20, ret10d: 0.35, rsi14: 80 },
};

function buildRecentSurgeMeta(row = {}) {
  const ret5d = toNum(row.ret_5d);
  const ret10d = toNum(row.ret_10d);
  const mom20 = toNum(row.mom_20);
  const rsi14 = toNum(row.rsi_14);
  const providedSoftFlag = boolify(row.recent_surge_soft_flag);
  const hardFlag =
    (ret5d !== null && ret5d >= RECENT_SURGE_THRESHOLDS.hard.ret5d) ||
    (ret10d !== null && ret10d >= RECENT_SURGE_THRESHOLDS.hard.ret10d) ||
    (rsi14 !== null && rsi14 >= RECENT_SURGE_THRESHOLDS.hard.rsi14);
  const computedSoftFlag =
    (ret5d !== null && ret5d >= RECENT_SURGE_THRESHOLDS.soft.ret5d) ||
    (ret10d !== null && ret10d >= RECENT_SURGE_THRESHOLDS.soft.ret10d) ||
    (rsi14 !== null && rsi14 >= RECENT_SURGE_THRESHOLDS.soft.rsi14);
  const softFlag = hardFlag || providedSoftFlag === true || computedSoftFlag;
  const signals = [];

  if (ret5d !== null && ret5d >= RECENT_SURGE_THRESHOLDS.soft.ret5d) signals.push(`5일 ${formatPct(ret5d)}`);
  if (ret10d !== null && ret10d >= RECENT_SURGE_THRESHOLDS.soft.ret10d) signals.push(`10일 ${formatPct(ret10d)}`);
  if (rsi14 !== null && rsi14 >= RECENT_SURGE_THRESHOLDS.soft.rsi14) signals.push(`RSI ${rsi14.toFixed(1)}`);
  if (mom20 !== null && mom20 >= 0.30) signals.push(`20일 모멘텀 ${formatPct(mom20)}`);

  return {
    recent_surge_soft_flag: softFlag,
    recent_surge_hard_flag: hardFlag,
    recent_surge_label: hardFlag ? "과열 급등" : softFlag ? "급등 주의" : null,
    recent_surge_tone: hardFlag ? "drag" : softFlag ? "watch" : null,
    recent_surge_detail: signals.length ? signals.join(" / ") : null,
    ret_5d: ret5d,
    ret_10d: ret10d,
    mom_20: mom20,
    rsi_14: rsi14,
  };
}

function formatPct(value, digits = 1) {
  const n = toNum(value);
  if (!Number.isFinite(n)) return "-";
  return `${(n * 100).toFixed(digits)}%`;
}

function getLiveScore(row) {
  if (!row) return null;
  return toNum(row.live_score ?? row.final_score ?? row.score);
}

function getLiveRank(row) {
  if (!row) return null;
  return toNum(row.live_rank ?? row.rank_final ?? row.rank);
}

function getLiveScoreSource(row) {
  if (!row) return null;
  const source = String(row.live_score_source || "").trim();
  return source || "final_score";
}

function getConfidenceScore(row) {
  if (!row) return null;
  return (
    toNum(row.confidence_score) ??
    toNum(row.confidence_score_operational) ??
    toNum(row.confidence_score_research) ??
    toNum(row.raw_confidence_v2)
  );
}

function getConfidenceLabel(row) {
  if (!row) return null;
  return (
    row.confidence_grade ||
    row.confidence_label ||
    row.confidence_label_operational ||
    row.confidence_label_research ||
    null
  );
}

function getConfidenceExplainText(row) {
  if (!row) return null;
  return (
    row.confidence_explain_text ||
    row.score_explain_confidence ||
    row.confidence_reason ||
    null
  );
}

function getQualityRiskGuardShadowScore(row) {
  if (!row) return null;
  return toNum(row.shadow_final_score_quality_risk_guard);
}

function getQualityRiskGuardShadowRank(row) {
  if (!row) return null;
  return toNum(row.shadow_rank_quality_risk_guard);
}

function getQualityRiskGuardPenalty(row) {
  if (!row) return null;
  return toNum(row.shadow_quality_risk_guard_penalty);
}

function toIsoDate(v) {
  if (v === null || v === undefined) return "";
  const formatLocalDate = (date) => {
    if (!(date instanceof Date) || Number.isNaN(date.getTime())) return "";
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const day = String(date.getDate()).padStart(2, "0");
    return `${year}-${month}-${day}`;
  };
  if (v instanceof Date) {
    return formatLocalDate(v);
  }
  const s = String(v).trim();
  if (!s) return "";
  // yyyy-mm-dd string
  if (/^\d{4}-\d{2}-\d{2}$/.test(s)) return s;
  const d = new Date(s);
  if (!Number.isNaN(d.getTime())) return formatLocalDate(d);
  // fallback: first 10 chars
  return s.slice(0, 10);
}

function compareTradesChronologically(a, b) {
  const da = toIsoDate(a?.date || "");
  const db = toIsoDate(b?.date || "");
  if (da !== db) return da < db ? -1 : 1;

  const ca = String(a?.created_at || "");
  const cb = String(b?.created_at || "");
  if (ca !== cb) return ca < cb ? -1 : 1;

  const ia = Number(a?.trade_id) || 0;
  const ib = Number(b?.trade_id) || 0;
  return ia - ib;
}

function parseTimestampMs(v) {
  if (v === null || v === undefined) return NaN;
  const s = String(v).trim();
  if (!s) return NaN;
  let ts = Date.parse(s);
  if (Number.isFinite(ts)) return ts;
  ts = Date.parse(s.replace(" ", "T"));
  if (Number.isFinite(ts)) return ts;
  return NaN;
}

function isIsoDateString(v) {
  return /^\d{4}-\d{2}-\d{2}$/.test(String(v || "").trim());
}

function boolify(v) {
  const s = String(v).toLowerCase();
  if (["true", "1", "t", "yes"].includes(s)) return true;
  if (["false", "0", "f", "no"].includes(s)) return false;
  return null;
}

function getManualTradingTone(gateStatus, acceptanceStatus) {
  const gate = String(gateStatus || "").toUpperCase();
  const acceptance = String(acceptanceStatus || "").toUpperCase();
  if (gate === "BUY_ALLOWED" && acceptance === "ACCEPTED") {
    return {
      label: "공격 가능",
      note: "Gate와 walk-forward가 모두 허용 상태라면, 신규 진입 후보를 적극적으로 검토할 수 있습니다.",
    };
  }
  if (gate === "WATCH") {
    return {
      label: "선별 관찰",
      note: "상위 후보는 볼 수 있지만 시장과 gate 조건이 완전히 열리지 않아 선별적으로만 접근해야 합니다.",
    };
  }
  return {
    label: "관찰 중심",
    note: "운영 Gate가 보수 단계이므로 신규 매수보다 관찰과 검증이 우선입니다.",
  };
}

function translateBuyEligibilityReason(reason) {
  const key = String(reason || "").trim();
  const map = {
    "confidence_score below 55": "신뢰도 55 미만입니다.",
    "confidence_score below preferred 70": "신뢰도는 통과했지만 70 미만입니다.",
    "pred_return_60d below 4%": "60일 기대수익이 4% 미만입니다.",
    "pred_return_60d below preferred 8%": "60일 기대수익이 선호 구간 8%에 못 미칩니다.",
    "prob_top20_60d below 10%": "상위권 진입 확률이 10% 미만입니다.",
    "prob_top20_60d below preferred 18%": "상위권 진입 확률이 선호 구간 18%에 못 미칩니다.",
    "pred_mdd_60d worse than -30%": "예상 MDD가 -30%보다 나쁩니다.",
    "pred_mdd_60d worse than preferred -20%": "예상 MDD가 선호 구간 -20%보다 나쁩니다.",
    "market regime defensive": "시장 레짐이 defensive입니다.",
    "market regime neutral": "시장 레짐이 neutral입니다.",
    "portfolio gate hold": "포트폴리오 gate가 HOLD입니다.",
    "portfolio gate block": "포트폴리오 gate가 BLOCK입니다.",
  };
  return map[key] || key || null;
}

function translateBuyEligibilityReasons(reasons) {
  return (Array.isArray(reasons) ? reasons : [])
    .map(translateBuyEligibilityReason)
    .filter(Boolean);
}

async function getDailyRecommendationItem(code) {
  const target = String(code || "").trim();
  if (!target) return null;
  const daily = await readJsonPayloadDbFirst("daily_recommendations", [
    path.join(SERVING_DIR, "daily_recommendations.json"),
  ]);
  const items = Array.isArray(daily.items) ? daily.items : [];
  return items.find((item) => String(item?.security?.code || "").trim() === target) || null;
}

function summarizeIntradayChanges({ intraday, activeIntraday, dailyPriorityCandidates, dailyCautionCandidates }) {
  const dailyPriority = Array.isArray(dailyPriorityCandidates) ? dailyPriorityCandidates : [];
  const dailyCaution = Array.isArray(dailyCautionCandidates) ? dailyCautionCandidates : [];
  const intradayPriority = Array.isArray(intraday?.priority_candidates) ? intraday.priority_candidates : [];
  const intradayCaution = Array.isArray(intraday?.caution_candidates) ? intraday.caution_candidates : [];

  const dailyPriorityMap = new Map(dailyPriority.map((item) => [String(item?.code || "").trim(), item]));
  const dailyCautionMap = new Map(dailyCaution.map((item) => [String(item?.code || "").trim(), item]));
  const intradayPriorityMap = new Map(intradayPriority.map((item) => [String(item?.code || "").trim(), item]));
  const intradayCautionMap = new Map(intradayCaution.map((item) => [String(item?.code || "").trim(), item]));

  const promotedToPriority = intradayPriority
    .filter((item) => !dailyPriorityMap.has(String(item?.code || "").trim()))
    .slice(0, 6);
  const droppedFromPriority = dailyPriority
    .filter((item) => !intradayPriorityMap.has(String(item?.code || "").trim()))
    .map((item) => intradayCautionMap.get(String(item?.code || "").trim()) || item)
    .slice(0, 6);
  const cautionEscalations = intradayCaution
    .filter((item) => {
      const code = String(item?.code || "").trim();
      return !dailyCautionMap.has(code) || dailyPriorityMap.has(code);
    })
    .slice(0, 6);
  const holdingReview = intradayCaution
    .filter((item) => String(item?.intraday_verdict || "").toUpperCase() === "HOLDING_REVIEW")
    .slice(0, 6);
  const missingQuotes = intradayCaution
    .filter((item) => !Number.isFinite(toNum(item?.intraday_quote?.current_price)))
    .slice(0, 6);

  const priorityCount = intradayPriority.length;
  const cautionCount = intradayCaution.length;
  let headline = "장중 변화 정보가 아직 없습니다.";
  if (activeIntraday) {
    if (priorityCount > 0) {
      headline = `오후장 우선 검토 ${priorityCount}개, 보수 검토 ${cautionCount}개로 다시 정리했습니다.`;
    } else if (cautionCount > 0) {
      headline = `오후장 즉시 진입 후보 없이 보수 검토 ${cautionCount}개만 남았습니다.`;
    } else {
      headline = "오후장 기준 재선별을 돌렸지만 유효 후보가 남지 않았습니다.";
    }
  } else if (intraday?.entity === "intraday_recommendations") {
    headline = "장중 스케줄 산출물은 있지만 현재 추천 기준으로 채택되지는 않았습니다.";
  }

  return {
    is_active: Boolean(activeIntraday),
    status: intraday?.status || null,
    session_date: intraday?.session_date || null,
    source_daily_asof_date: intraday?.source_daily_asof_date || intraday?.asof_date || null,
    basis_label: intraday?.basis_label || null,
    quote_success_count: toNum(intraday?.counts?.quote_success) ?? 0,
    quote_failure_count: toNum(intraday?.counts?.quote_failure) ?? 0,
    priority_count: priorityCount,
    caution_count: cautionCount,
    promoted_to_priority_count: promotedToPriority.length,
    dropped_from_priority_count: droppedFromPriority.length,
    caution_escalation_count: cautionEscalations.length,
    holding_review_count: holdingReview.length,
    missing_quote_count: missingQuotes.length,
    headline,
    action_guide:
      intraday?.summary?.action_guide ||
      "장중 우선 검토는 오후장 실행 후보, 보수 검토는 관찰 또는 보유 관리 대상으로 읽으면 됩니다.",
    promoted_to_priority: promotedToPriority,
    dropped_from_priority: droppedFromPriority,
    caution_escalations: cautionEscalations,
    holding_review: holdingReview,
    missing_quotes: missingQuotes,
  };
}

function normalizeManualCandidate(item, rankingByCode) {
  const selection = item.selection || {};
  const scores = item.scores || {};
  const security = item.security || {};
  const buyEligibility = item.buy_eligibility || {};
  const marketSignals = item.market_signals || {};
  const rankRow = rankingByCode.get(String(security.code || "").trim()) || null;
  const liveRank = rankRow ? getLiveRank(rankRow) : null;
  const shadowRank = rankRow ? getQualityRiskGuardShadowRank(rankRow) : null;
  const surgeMeta = buildRecentSurgeMeta({
    ret_5d: marketSignals.ret_5d,
    ret_10d: marketSignals.ret_10d,
    mom_20: marketSignals.mom_20,
    rsi_14: marketSignals.rsi_14,
    recent_surge_soft_flag: selection.recent_surge_soft_flag,
  });
  return {
    code: String(security.code || "").trim(),
    name: security.name || null,
    sector: security.sector || null,
    dominant_theme: security.dominant_theme || null,
    buy_rank: toNum(item.buy_rank),
    buyability_status: selection.buyability_status || null,
    watchlist_tier: selection.buyability_watchlist_tier || null,
    promotion_readiness_score: toNum(selection.buyability_promotion_readiness_score),
    expected_action: selection.buyability_expected_action || null,
    supporting_reasons: Array.isArray(selection.buyability_supporting_reasons) ? selection.buyability_supporting_reasons : [],
    blocking_reasons: Array.isArray(selection.buyability_blocking_reasons) ? selection.buyability_blocking_reasons : [],
    confidence_state_v2: scores.confidence_state_v2 || null,
    raw_confidence_v2: toNum(scores.raw_confidence_v2),
    live_score: toNum(scores.live_score ?? scores.final_score),
    live_rank: liveRank,
    final_score: toNum(scores.final_score),
    shadow_quality_risk_guard_score: rankRow ? getQualityRiskGuardShadowScore(rankRow) : null,
    shadow_quality_risk_guard_rank: shadowRank,
    shadow_quality_risk_guard_penalty: rankRow ? getQualityRiskGuardPenalty(rankRow) : null,
    shadow_quality_risk_guard_rank_delta:
      Number.isFinite(liveRank) && Number.isFinite(shadowRank) ? liveRank - shadowRank : null,
    buy_eligibility_status: buyEligibility.status || null,
    buy_eligibility_score: toNum(buyEligibility.score),
    buy_eligibility_hard_block_reasons: translateBuyEligibilityReasons(buyEligibility.hard_block_reasons),
    buy_eligibility_caution_reasons: translateBuyEligibilityReasons(buyEligibility.caution_reasons),
    pred_return_60d: toNum(marketSignals.pred_return_60d),
    prob_top20_60d: toNum(marketSignals.prob_top20_60d),
    pred_mdd_60d: toNum(marketSignals.pred_mdd_60d),
    regime: marketSignals.regime || null,
    recent_surge_soft_flag: surgeMeta.recent_surge_soft_flag,
    recent_surge_hard_flag: surgeMeta.recent_surge_hard_flag,
    recent_surge_label: surgeMeta.recent_surge_label,
    recent_surge_detail: surgeMeta.recent_surge_detail,
    ret_5d: surgeMeta.ret_5d,
    ret_10d: surgeMeta.ret_10d,
    mom_20: surgeMeta.mom_20,
    rsi_14: surgeMeta.rsi_14,
    entry_rule_pass: Boolean(selection.entry_rule_pass),
  };
}

function isPriorityManualCandidate(item) {
  if (item.buy_eligibility_status === "BUY_ALLOWED") return true;
  if (item.buy_eligibility_status === "BLOCK") return false;
  if (item.buyability_status === "BUY_NOW") return true;
  if (item.watchlist_tier === "PROMOTION_READY") return true;
  if (item.buyability_status !== "WATCHLIST") return false;
  if (item.confidence_state_v2 === "BLOCKED") return false;
  return item.entry_rule_pass !== false;
}

function isCautionManualCandidate(item) {
  if (item.buy_eligibility_status === "BLOCK") return true;
  if (item.buyability_status === "BLOCK") return true;
  if (item.recent_surge_soft_flag) return true;
  if (item.confidence_state_v2 === "BLOCKED" || item.confidence_state_v2 === "WEAK") return true;
  if (item.entry_rule_pass === false) return true;

  const blockingReasons = Array.isArray(item.blocking_reasons) ? item.blocking_reasons : [];
  const cautionReasons = blockingReasons.filter(
    (reason) => !["gate_hold", "walkforward_conditional"].includes(String(reason || "").trim())
  );
  return cautionReasons.length > 0;
}

async function buildManualTradingSummary(options = {}) {
  const daily =
    options.daily ||
    await readJsonPayloadDbFirst("daily_recommendations", [path.join(SERVING_DIR, "daily_recommendations.json")]);
  const intraday =
    options.intraday ||
    await readJsonPayloadDbFirst("intraday_recommendations", [path.join(SERVING_DIR, "intraday_recommendations.json")]);
  const gate =
    options.gate ||
    await readJsonPayloadDbFirst("operational_buy_gate", [path.join(OUTPUTS_DIR, "operational_buy_gate.json")]);
  const walkforwardAcceptance =
    options.walkforwardAcceptance ||
    await readJsonPayloadDbFirst("walkforward_acceptance", [path.join(OUTPUTS_DIR, "walkforward_acceptance.json")]);
  const rankingContext = options.rankingRows
    ? {
        rankingRows: options.rankingRows || [],
        rankingLatestDate: options.rankingLatestDate || null,
      }
    : await getLatestRankingContext();
  const rankingRows = rankingContext.rankingRows || [];
  const rankingDates = rankingRows
    .map((row) => String(row.date || "").trim())
    .filter((value) => isIsoDateString(value));
  const rankingLatestDate =
    options.rankingLatestDate ||
    rankingContext.rankingLatestDate ||
    (rankingDates.length ? rankingDates.sort().pop() : null) ||
    null;
  const recommendationsDate = isIsoDateString(daily.asof_date) ? daily.asof_date : null;
  const recommendationsStale =
    Boolean(recommendationsDate) &&
    Boolean(rankingLatestDate) &&
    recommendationsDate < rankingLatestDate;
  const items = Array.isArray(daily.items) ? daily.items : [];
  const rankingLatestRows = rankingLatestDate
    ? rankingRows.filter((row) => String(row.date || "").trim() === rankingLatestDate)
    : rankingRows;
  const rankingByCode = new Map(
    rankingLatestRows.map((row) => [String(row.code || "").trim(), row])
  );
  const gateStatus = daily.gate_overall_status || gate.overall_status || null;
  const acceptanceStatus = daily.walkforward_acceptance_status || null;
  const tone = getManualTradingTone(gateStatus, acceptanceStatus);
  const staleChecklistLine = recommendationsStale && recommendationsDate && rankingLatestDate
    ? `daily_recommendations 기준일(${recommendationsDate})이 ranking_final 최신일(${rankingLatestDate})보다 오래되어 후보 목록을 숨깁니다. refresh 완료 후 다시 확인하세요.`
    : null;
  const intradaySessionDate = isIsoDateString(intraday.session_date) ? intraday.session_date : null;
  const intradaySourceDailyDate = isIsoDateString(intraday.source_daily_asof_date)
    ? intraday.source_daily_asof_date
    : (isIsoDateString(intraday.asof_date) ? intraday.asof_date : null);
  const baselineCandidates = items.map((item) => normalizeManualCandidate(item, rankingByCode));
  const baselinePriorityCandidates = baselineCandidates
    .filter(isPriorityManualCandidate)
    .sort((a, b) => {
      const aReady = toNum(a.promotion_readiness_score) ?? -1;
      const bReady = toNum(b.promotion_readiness_score) ?? -1;
      if (bReady !== aReady) return bReady - aReady;
      const aScore = toNum(a.live_score ?? a.final_score) ?? -1;
      const bScore = toNum(b.live_score ?? b.final_score) ?? -1;
      return bScore - aScore;
    })
    .slice(0, 8);
  const baselinePriorityCodes = new Set(baselinePriorityCandidates.map((item) => item.code));
  const baselineCautionCandidates = baselineCandidates
    .filter((item) => !baselinePriorityCodes.has(item.code))
    .filter(isCautionManualCandidate)
    .sort((a, b) => {
      const aPenalty =
        (a.buyability_status === "BLOCK" ? 3 : 0) +
        (a.confidence_state_v2 === "BLOCKED" ? 2 : a.confidence_state_v2 === "WEAK" ? 1 : 0) +
        (a.entry_rule_pass === false ? 1 : 0) +
        (a.recent_surge_soft_flag ? 1 : 0);
      const bPenalty =
        (b.buyability_status === "BLOCK" ? 3 : 0) +
        (b.confidence_state_v2 === "BLOCKED" ? 2 : b.confidence_state_v2 === "WEAK" ? 1 : 0) +
        (b.entry_rule_pass === false ? 1 : 0) +
        (b.recent_surge_soft_flag ? 1 : 0);
      if (bPenalty !== aPenalty) return bPenalty - aPenalty;
      const aScore = toNum(a.live_score ?? a.final_score) ?? -1;
      const bScore = toNum(b.live_score ?? b.final_score) ?? -1;
      return bScore - aScore;
    })
    .slice(0, 8);
  const activeIntraday =
    intraday.entity === "intraday_recommendations" &&
    ["current", "fallback"].includes(String(intraday.status || "").toLowerCase()) &&
    intradaySessionDate &&
    Array.isArray(intraday.priority_candidates) &&
    (!rankingLatestDate || intradaySourceDailyDate === rankingLatestDate);
  const intradaySummary = summarizeIntradayChanges({
    intraday,
    activeIntraday,
    dailyPriorityCandidates: baselinePriorityCandidates,
    dailyCautionCandidates: baselineCautionCandidates,
  });

  if (activeIntraday) {
    return {
      generated_at: intraday.generated_at || null,
      asof_date: intraday.asof_date || null,
      gate_status: intraday.gate_overall_status || gateStatus,
      walkforward_acceptance_status: intraday.walkforward_acceptance_status || acceptanceStatus,
      market_regime: gate.market_regime || null,
      manual_mode: {
        label: "오후장 대응",
        note: intraday.summary?.reason || "운영자가 수동 배포한 장중 기준 데이터로 오후장 대응 후보를 다시 정리합니다.",
      },
      source_status: intraday.status || "current",
      ranking_latest_date: rankingLatestDate,
      recommendations_stale: false,
      execution_basis: {
        label: intraday.basis_label || "장중 기준",
        note: "운영자가 수동 배포한 장중 기준 데이터입니다.",
      },
      intraday_summary: intradaySummary,
      checklist: Array.isArray(intraday.checklist) ? intraday.checklist : [],
      priority_candidates: Array.isArray(intraday.priority_candidates) ? intraday.priority_candidates : [],
      caution_candidates: Array.isArray(intraday.caution_candidates) ? intraday.caution_candidates : [],
      daily_priority_candidates: baselinePriorityCandidates,
      daily_caution_candidates: baselineCautionCandidates,
    };
  }

  const normalized = baselineCandidates;

  const priorityCandidates = normalized
    .filter(isPriorityManualCandidate)
    .sort((a, b) => {
      const aReady = toNum(a.promotion_readiness_score) ?? -1;
      const bReady = toNum(b.promotion_readiness_score) ?? -1;
      if (bReady !== aReady) return bReady - aReady;
      const aScore = toNum(a.live_score ?? a.final_score) ?? -1;
      const bScore = toNum(b.live_score ?? b.final_score) ?? -1;
      return bScore - aScore;
    })
    .slice(0, 8);

  const priorityCodes = new Set(priorityCandidates.map((item) => item.code));

  const cautionCandidates = normalized
    .filter((item) => !priorityCodes.has(item.code))
    .filter(isCautionManualCandidate)
    .sort((a, b) => {
      const aPenalty =
        (a.buyability_status === "BLOCK" ? 3 : 0) +
        (a.confidence_state_v2 === "BLOCKED" ? 2 : a.confidence_state_v2 === "WEAK" ? 1 : 0) +
        (a.entry_rule_pass === false ? 1 : 0) +
        (a.recent_surge_soft_flag ? 1 : 0);
      const bPenalty =
        (b.buyability_status === "BLOCK" ? 3 : 0) +
        (b.confidence_state_v2 === "BLOCKED" ? 2 : b.confidence_state_v2 === "WEAK" ? 1 : 0) +
        (b.entry_rule_pass === false ? 1 : 0) +
        (b.recent_surge_soft_flag ? 1 : 0);
      if (bPenalty !== aPenalty) return bPenalty - aPenalty;
      const aScore = toNum(a.live_score ?? a.final_score) ?? -1;
      const bScore = toNum(b.live_score ?? b.final_score) ?? -1;
      return bScore - aScore;
    })
    .slice(0, 8);

  return {
    generated_at: daily.generated_at || null,
    asof_date: daily.asof_date || null,
    gate_status: gateStatus,
    walkforward_acceptance_status: acceptanceStatus,
    market_regime: gate.market_regime || null,
    manual_mode: recommendationsStale
      ? {
          label: "추천 대기",
          note: `추천 데이터 기준일(${recommendationsDate})이 최신 랭킹(${rankingLatestDate})보다 오래되어 신규 후보 사용을 보류합니다.`,
        }
      : tone,
      source_status: recommendationsStale ? "stale" : (daily.source_status || null),
      execution_basis: {
        label: "마감 기준",
        note: "전일 마감 기준 후보입니다. 장중 payload가 없으면 이 기준으로 읽습니다.",
      },
      ranking_latest_date: rankingLatestDate,
    recommendations_stale: recommendationsStale,
    intraday_summary: intradaySummary,
    checklist: [
      ...(staleChecklistLine ? [staleChecklistLine] : []),
      "run_operational_refresh.py 실행 후 gate 상태와 기준일을 먼저 확인합니다.",
      "PROMOTION_READY 또는 WATCHLIST 후보만 보고 final_score, confidence_state_v2, buyability 사유를 확인합니다.",
      "실제 주문 전에는 HTS/MTS에서 실시간 가격, 기존 보유 중복, 주문 수량을 직접 확인합니다.",
      "장 초반 과열이 있거나 gate가 HOLD면 신규 진입보다 관찰을 우선합니다.",
      "실제 체결이 발생하면 가격, 시간, 수량, 메모를 거래 기록에 바로 남깁니다.",
    ],
    priority_candidates: recommendationsStale ? [] : priorityCandidates,
    caution_candidates: recommendationsStale ? [] : cautionCandidates,
  };
}

async function buildTradingPolicySummary() {
  const manual = await buildManualTradingSummary();
  const recommendationsStale = Boolean(manual.recommendations_stale);
  const gateStatus = String(manual.gate_status || "").toUpperCase();
  const walkforwardStatus = String(manual.walkforward_acceptance_status || "").toUpperCase();
  const modeTone =
    recommendationsStale ? "bad" :
    gateStatus === "BUY_ALLOWED" && walkforwardStatus === "ACCEPTED" ? "good" :
    gateStatus === "WATCH" ? "watch" :
    "info";

  return {
    asof_date: manual.asof_date || null,
    recommendations_stale: recommendationsStale,
    mode: {
      label: manual.manual_mode?.label || "관찰 중심",
      note: manual.manual_mode?.note || "신규 진입보다 관찰과 검증이 우선입니다.",
      tone: modeTone,
    },
    banner: [
      {
        title: "오늘 전략 모드",
        value: manual.manual_mode?.label || "관찰 중심",
        detail: manual.manual_mode?.note || "신규 진입보다 관찰과 검증이 우선입니다.",
        tone: modeTone,
      },
      {
        title: "신규 매수",
        value: recommendationsStale ? "보류" : (gateStatus === "BUY_ALLOWED" ? "허용" : gateStatus === "WATCH" ? "선별 허용" : "보수 접근"),
        detail: recommendationsStale
          ? "추천 기준일이 최신 랭킹보다 오래되면 신규 후보를 숨기고 refresh 이후에만 재개합니다."
          : "PROMOTION_READY 또는 상위 WATCHLIST만 보고 실시간 가격과 기존 보유 중복을 직접 확인합니다.",
        tone: recommendationsStale ? "bad" : (gateStatus === "BUY_ALLOWED" ? "good" : "watch"),
      },
      {
        title: "추가 매수",
        value: "기본 보수",
        detail: "신규 진입 후 3일은 유예 구간으로 보고, 초기 진입 근거가 안정되기 전까지 추가 매수는 보수적으로 봅니다.",
        tone: "watch",
      },
      {
        title: "보유 점검",
        value: "20일 / +15% / -8%",
        detail: "20일 도달, 수익률 +15% 이상, 손실 -8% 이하에서 보유 또는 축소/매도 근거를 다시 점검합니다.",
        tone: "info",
      },
      {
        title: "포트폴리오 한도",
        value: "5종목 / 24%",
        detail: "최대 5종목, 종목당 최대 비중 24%, 섹터 35%, 테마 35%, 재진입 쿨다운 10영업일 기준입니다.",
        tone: "info",
      },
    ],
    page_rules: {
      manual: [
        {
          title: "신규 진입 대상",
          value: "PROMOTION_READY / WATCHLIST",
          detail: "추천 후보 중에서도 buyability, confidence_state_v2, entry_rule_pass를 통과한 종목만 우선 검토합니다.",
          tone: "good",
        },
        {
          title: "Gate 우선 확인",
          value: gateStatus || "HOLD",
          detail: "gate가 HOLD면 관찰 중심, WATCH면 선별 진입, BUY_ALLOWED면 신규 진입 검토 강도를 높입니다.",
          tone: gateStatus === "BUY_ALLOWED" ? "good" : gateStatus === "WATCH" ? "watch" : "info",
        },
        {
          title: "추천 신선도",
          value: recommendationsStale ? "stale" : "current",
          detail: recommendationsStale
            ? "daily_recommendations와 ranking_final 기준일이 어긋나면 신규 후보 사용을 보류합니다."
            : "추천 기준일과 최신 랭킹이 맞는 상태에서만 후보를 사용합니다.",
          tone: recommendationsStale ? "bad" : "good",
        },
        {
          title: "실행 체크",
          value: "가격 / 중복 / 수량",
          detail: "HTS/MTS에서 실시간 가격, 기존 보유 섹터 중복, 주문 수량을 최종 확인한 뒤 체결합니다.",
          tone: "info",
        },
      ],
      holdings: [
        {
          title: "신규 3일 유예",
          value: "추가매수 보수",
          detail: "신규 포지션은 3일 동안 REVIEW를 완화하고, 초기 진입 근거 유지 여부를 우선 확인합니다.",
          tone: "watch",
        },
        {
          title: "수익 구간 점검",
          value: "+15% 이상",
          detail: "수익률 +15% 이상이면 이익 보호와 분할 축소 여부를 먼저 검토합니다.",
          tone: "good",
        },
        {
          title: "보유 기간 점검",
          value: "20일 도달",
          detail: "보유 20일 시점에는 ret/prob/confidence 지지와 점수 유지 여부를 다시 봅니다.",
          tone: "info",
        },
        {
          title: "손실 관리",
          value: "-8% 이하",
          detail: "손실 -8% 이하 또는 종합 점수 급약화 시에는 매도검토 우선순위를 높입니다.",
          tone: "bad",
        },
      ],
      ranking: [
        {
          title: "추천 해석",
          value: "아이디어 + 집행 분리",
          detail: "랭킹 상위라는 이유만으로 매수하지 않고, 왜 지금인지와 어떤 조건에서 틀렸는지를 함께 봅니다.",
          tone: "info",
        },
        {
          title: "우선 판단 축",
          value: "ret / prob / confidence",
          detail: "상방 기대, 확률, 신뢰도 축이 동시에 버텨주는지 먼저 확인합니다.",
          tone: "good",
        },
        {
          title: "리스크 확인",
          value: "risk_penalty / invalidation",
          detail: "risk drag가 높거나 무효화 조건이 약하면 즉시 매수보다 관찰이 우선입니다.",
          tone: "watch",
        },
        {
          title: "포지션 연결",
          value: "신규 vs 교체",
          detail: "기존 보유와 섹터 중복이 크면 신규 진입보다 교체 또는 관찰 우선으로 해석합니다.",
          tone: "info",
        },
      ],
      portfolio: [
        {
          title: "최대 보유 종목",
          value: "5개",
          detail: "포트폴리오 집중도를 유지하기 위한 기본 상한입니다.",
          tone: "info",
        },
        {
          title: "종목당 최대 비중",
          value: "24%",
          detail: "한 종목 비중을 과도하게 키우지 않도록 제한합니다.",
          tone: "info",
        },
        {
          title: "섹터 / 테마 캡",
          value: "각 35%",
          detail: "같은 축에 포지션이 몰리지 않도록 섹터와 테마 노출을 제한합니다.",
          tone: "info",
        },
        {
          title: "재진입 쿨다운",
          value: "10영업일",
          detail: "방금 나온 종목을 즉시 다시 추격하지 않도록 재진입 간격을 둡니다.",
          tone: "watch",
        },
      ],
    },
  };
}

function parseKeyValueMarkdown(filePath) {
  const raw = readText(filePath);
  return parseKeyValueMarkdownText(raw);
}

function parseKeyValueMarkdownText(raw) {
  if (!raw) return null;
  const out = {};
  String(raw).split(/\r?\n/).forEach((line) => {
    const trimmed = String(line || "").trim();
    const match = trimmed.match(/^- ([^:]+):\s*(.*)$/);
    if (!match) return;
    out[String(match[1] || "").trim()] = String(match[2] || "").trim();
  });
  return out;
}

function mapWalkforwardReasonCode(code, acceptance) {
  const top20 = acceptance?.selection_summary?.top20 || {};
  const top50 = acceptance?.selection_summary?.top50 || {};
  const universe = acceptance?.selection_summary?.universe || {};
  switch (String(code || "").trim()) {
    case "top20_excess_return_positive":
      return Number.isFinite(toNum(top20.excess_return))
        ? `top20 60일 초과수익은 플러스입니다 (${formatPct(top20.excess_return, 1)}).`
        : "top20 60일 초과수익은 플러스입니다.";
    case "ordering_not_stable":
      return Number.isFinite(toNum(top20.avg_return)) && Number.isFinite(toNum(top50.avg_return)) && Number.isFinite(toNum(universe.avg_return))
        ? `정렬력이 약합니다. 60일 평균수익이 top20 ${formatPct(top20.avg_return, 1)}, top50 ${formatPct(top50.avg_return, 1)}, universe ${formatPct(universe.avg_return, 1)}입니다.`
        : "정렬력이 약합니다. top20 > top50 > universe 구조가 유지되지 않았습니다.";
    case "drawdown_too_deep":
      return Number.isFinite(toNum(top20.avg_mdd))
        ? `top20 60일 평균 MDD가 ${formatPct(top20.avg_mdd, 1)}로 깊습니다.`
        : "top20 60일 평균 MDD가 기준보다 깊습니다.";
    case "confidence_monotonicity_missing":
      return "confidence 구간이 높을수록 성과가 좋아지는 단조성이 아직 확인되지 않았습니다.";
    case "execution_evidence_ok_or_unavailable":
      return "체결 증거는 차단 사유는 아니지만 아직 판정을 강화할 만큼 충분하지 않습니다.";
    default:
      return code ? `walkforward 사유: ${code}` : null;
  }
}

function formatOpsMetricReason(item) {
  if (!item || !item.metric) return null;
  const metric = String(item.metric || "");
  const valueText = item.value_display || (Number.isFinite(Number(item.value)) ? String(item.value) : null);
  const lookup = {
    overlap_final_ret_top20: `final score 상위 20개와 ret_score 상위 20개의 겹침이 낮습니다 (${valueText || "-"})`,
    overlap_final_prob_top20: `final score 상위 20개와 prob_score 상위 20개의 겹침이 낮습니다 (${valueText || "-"})`,
    overlap_final_tech_top20: `final score 상위 20개와 tech_score 상위 20개의 겹침이 낮습니다 (${valueText || "-"})`,
    corr_final_risk_penalty_abs: `final score가 risk penalty 영향에 과하게 끌리고 있습니다 (${valueText || "-"})`,
    corr_final_safety_abs: `final score가 safety 계열 보정에 과하게 끌리고 있습니다 (${valueText || "-"})`,
    corr_final_tech_score: `tech_score와 final score의 정렬 관계가 약합니다 (${valueText || "-"})`,
    top20_top_driver_share: `top20이 일부 driver에 몰려 있습니다 (${valueText || "-"})`,
  };
  return lookup[metric] || `${metric} 상태가 ${item.status || "-"} 입니다 (${valueText || "-"})`;
}

function buildMarketRegimeInterpretation(regimeInput) {
  const regime = regimeInput || {};
  const regimeName = String(regime.regime || "").toLowerCase();
  const trueCount = toNum(regime.true_count);
  const breadth = toNum(regime.breadth_20d);
  const recentReturn = toNum(regime.recent_20d_return);
  const vol5d = toNum(regime.volatility_5d);
  const summary =
    regimeName === "bull"
      ? "시장 레짐은 bull입니다. 신규 진입을 전면 확대할 단계는 아니어도 추세 확인 종목은 적극 검토할 수 있습니다."
      : regimeName === "defensive"
      ? "시장 레짐은 defensive입니다. 신규 진입보다 보유 방어와 현금 비중 관리가 우선입니다."
      : "시장 레짐은 neutral입니다. 추세는 일부 살아 있지만 확산과 변동성 구조가 아직 공격적으로 열리지는 않았습니다.";
  const stance =
    regimeName === "bull"
      ? "상위 후보 중 entry rule을 통과한 종목은 후보군으로 유지하되, 섹터 쏠림만 별도로 점검합니다."
      : regimeName === "defensive"
      ? "신규 진입은 최소화하고, watchlist 중심으로만 보면서 손실 확대 구간을 먼저 방어합니다."
      : "신규 진입은 보수적으로 보고, watchlist와 수동 검토 위주로 운영하는 편이 맞습니다.";
  const actionItems = [];
  if (Number.isFinite(trueCount)) {
    actionItems.push(`5개 레짐 조건 중 ${trueCount}개만 충족했습니다.`);
  }
  if (regime.close_gt_ma20 === false) {
    actionItems.push("코스피 종가가 MA20 아래라 추세 확인이 덜 됐습니다.");
  }
  if (regime.breadth_20d_gt_0_55 === false) {
    actionItems.push(`시장 breadth가 낮습니다 (${Number.isFinite(breadth) ? breadth.toFixed(3) : "-" }).`);
  }
  if (regime.volatility_risk_flag === true) {
    actionItems.push(`단기 변동성 부담이 있습니다 (${Number.isFinite(vol5d) ? vol5d.toFixed(4) : "-"}).`);
  }
  if (regime.recent_20d_return_gt_0_03 === true && regimeName !== "defensive") {
    actionItems.push(`최근 20일 수익률은 아직 플러스입니다 (${Number.isFinite(recentReturn) ? formatPct(recentReturn, 1) : "-"}).`);
  }
  const tone = regimeName === "bull" ? "GOOD" : regimeName === "defensive" ? "ALERT" : "WATCH";
  return {
    summary,
    stance,
    action_items: actionItems.slice(0, 4),
    tone,
  };
}

async function buildOpsReadinessSummary() {
  const opsNotesPath = path.join(OUTPUTS_DIR, "ops_operator_notes.json");
  const schedulerStatus = await readJsonPayloadDbFirst("auto_ops_scheduler_status", [path.join(OUTPUTS_DIR, "auto_ops_scheduler_status.json")]);
  const schedulerRecoveryStatus = await readJsonPayloadDbFirst("auto_ops_recovery_scheduler_status", [path.join(OUTPUTS_DIR, "auto_ops_recovery_scheduler_status.json")]);
  const gate = await readJsonPayloadDbFirst("operational_buy_gate", [path.join(OUTPUTS_DIR, "operational_buy_gate.json")]);
  const walkforwardAcceptance = await readJsonPayloadDbFirst("walkforward_acceptance", [path.join(OUTPUTS_DIR, "walkforward_acceptance.json")]);
  const shadowRepeatability =
    await readJsonPayloadDbFirst("shadow_quality_risk_guard_repeatability_report", [path.join(OUTPUTS_DIR, "shadow_quality_risk_guard_repeatability_report.json")]);
  const kpi = await readJsonPayloadDbFirst("score_kpi_monitor", [
    path.join(DATA_DIR, "score_kpi_monitor.json"),
    path.join(OUTPUTS_DIR, "score_kpi_monitor.json"),
  ]);
  const inventoryPayload = await readJsonPayloadDbFirst("ranking_snapshot_inventory");
  const inventory = inventoryPayload.summary || parseKeyValueMarkdown(path.join(DATA_DIR, "history", "ranking_snapshot_inventory.md")) || {};
  const inventoryRows = Array.isArray(inventoryPayload.rows) ? inventoryPayload.rows : (readCsv(path.join(DATA_DIR, "history", "ranking_snapshot_inventory.csv")) || []);
  const gateHistoryPayload = await readJsonPayloadDbFirst("operational_buy_gate_history");
  const gateHistoryRows = Array.isArray(gateHistoryPayload.rows) ? gateHistoryPayload.rows : (readCsv(path.join(DATA_DIR, "history", "operational_buy_gate_history.csv")) || []);
  const kpiHistoryPayload = await readJsonPayloadDbFirst("score_kpi_monitor_history");
  const kpiHistoryRows = Array.isArray(kpiHistoryPayload.rows) ? kpiHistoryPayload.rows : (readCsv(path.join(DATA_DIR, "history", "score_kpi_monitor_history.csv")) || []);
  const daily = await readJsonPayloadDbFirst("daily_recommendations", [path.join(SERVING_DIR, "daily_recommendations.json")]);
  const dailyCycle = await readJsonPayloadDbFirst("operational_daily_cycle_status", [path.join(OUTPUTS_DIR, "operational_daily_cycle_status.json")]);
  const opsNotes = await readJsonPayloadDbFirst("ops_operator_notes", [opsNotesPath]);
  const visitorAnalytics = await buildVisitorAnalyticsSummary();
  const rankingContext = await getLatestRankingContext();
  const rankingRows = rankingContext.rankingRows || [];
  const rankingDates = rankingRows
    .map((row) => String(row.date || "").trim())
    .filter((value) => isIsoDateString(value));
  const rankingLatestDate =
    rankingContext.rankingLatestDate ||
    (rankingDates.length ? rankingDates.sort().pop() : null) ||
    (isIsoDateString(daily.asof_date) ? daily.asof_date : null) ||
    (isIsoDateString(gate.asof_date) ? gate.asof_date : null) ||
    (isIsoDateString(inventory["latest snapshot date"]) ? inventory["latest snapshot date"] : null);
  const latestRanking = rankingLatestDate
    ? rankingRows.filter((row) => String(row.date || "").trim() === rankingLatestDate)
    : rankingRows;
  const manual = await buildManualTradingSummary({ rankingLatestDate, rankingRows, daily, gate, walkforwardAcceptance });
  const shadowCandidates = latestRanking
    .map((row) => {
      const liveRank = getLiveRank(row);
      const shadowRank = getQualityRiskGuardShadowRank(row);
      const rankDelta =
        Number.isFinite(liveRank) && Number.isFinite(shadowRank)
          ? liveRank - shadowRank
          : null;
      return {
        code: String(row.code || "").trim(),
        name: row.name || getName(row.code) || null,
        sector: row.sector || getSector(row.code) || null,
        market: row.market || getMarket(row.code) || null,
        live_score: getLiveScore(row),
        live_rank: liveRank,
        final_score: toNum(row.final_score),
        shadow_quality_risk_guard_score: getQualityRiskGuardShadowScore(row),
        shadow_quality_risk_guard_rank: shadowRank,
        shadow_quality_risk_guard_penalty: getQualityRiskGuardPenalty(row),
        shadow_quality_risk_guard_rank_delta: rankDelta,
        confidence_score: getConfidenceScore(row),
        risk_penalty: toNum(row.risk_penalty),
        qual_score: toNum(row.qual_score),
        dominant_theme: row.dominant_theme || null,
      };
    })
    .filter((item) => item.code)
    .filter((item) => Number.isFinite(item.shadow_quality_risk_guard_rank_delta) && item.shadow_quality_risk_guard_rank_delta > 0)
    .sort((a, b) => {
      const deltaDiff = (b.shadow_quality_risk_guard_rank_delta || 0) - (a.shadow_quality_risk_guard_rank_delta || 0);
      if (deltaDiff !== 0) return deltaDiff;
      const aPenalty = Number.isFinite(a.shadow_quality_risk_guard_penalty) ? a.shadow_quality_risk_guard_penalty : 999;
      const bPenalty = Number.isFinite(b.shadow_quality_risk_guard_penalty) ? b.shadow_quality_risk_guard_penalty : 999;
      if (aPenalty !== bPenalty) return aPenalty - bPenalty;
      return (b.live_score || 0) - (a.live_score || 0);
    })
    .slice(0, 5);

  const snapshotCount = toNum(inventory["total snapshot count"]);
  const matured20 = toNum(inventory["matured snapshot count 20d"]);
  const matured60 = toNum(inventory["matured snapshot count 60d"]);
  const matured90 = toNum(inventory["matured snapshot count 90d"]);
  const readiness60 = inventory["confidence calibration readiness 60d"] || null;

  const gateDecision = Array.isArray(gate.decisions) && gate.decisions.length ? gate.decisions[0] : {};
  const trustedRatio = toNum(gateDecision.confidence_v2?.trusted_ratio_top20);
  const maturedBenchmarkDates = toNum(gateDecision.benchmark?.matured_dates_max);
  const walkforwardStatus = gateDecision.walkforward_acceptance?.status || null;
  const buyability = gateDecision.buyability || {};

  const kpiSummary = kpi.summary || {};
  const kpiMetadata = kpi.metadata || {};
  const kpiRows = Array.isArray(kpi.kpi_snapshot) ? kpi.kpi_snapshot : (Array.isArray(kpi.kpis) ? kpi.kpis : []);
  const alertMetrics = kpiRows.filter((item) => String(item.status || "").toUpperCase() === "ALERT");
  const watchMetrics = kpiRows.filter((item) => String(item.status || "").toUpperCase() === "WATCH");
  const topConfidenceMetric = kpiRows.find((item) => item.metric === "top20_mean_confidence_score");
  const wfMetric = kpiRows.find((item) => item.metric === "walkforward_top20_avg_return_60d");
  const walkforwardReasons = Array.isArray(gateDecision.walkforward_acceptance?.reason_codes) && gateDecision.walkforward_acceptance.reason_codes.length
    ? gateDecision.walkforward_acceptance.reason_codes
    : (Array.isArray(walkforwardAcceptance.reason_codes) ? walkforwardAcceptance.reason_codes : []);
  const marketRegimeInterpretation = buildMarketRegimeInterpretation(gate.market_regime || gateDecision.market_regime || {});

  const trendRows = inventoryRows
    .map((row, idx) => ({
      as_of_date: row.as_of_date || null,
      row_count: toNum(row.row_count),
      has_top20: String(row.has_top20 || "").toLowerCase() === "true",
      matured_20d: String(row.matured_20d || "").toLowerCase() === "true",
      matured_60d: String(row.matured_60d || "").toLowerCase() === "true",
      matured_90d: String(row.matured_90d || "").toLowerCase() === "true",
      snapshot_index: idx + 1,
    }))
    .filter((row) => row.as_of_date);
  let matured60Running = 0;
  const readinessTrend = trendRows.map((row) => {
    matured60Running += row.matured_60d ? 1 : 0;
    return {
      ...row,
      matured_60d_cumulative: matured60Running,
    };
  });
  const recentReadinessTrend = readinessTrend.slice(-12);
  const recentStateBadges = readinessTrend.slice(-7).map((row) => ({
    date: row.as_of_date,
    label: row.as_of_date ? String(row.as_of_date).slice(5) : "-",
    kind: row.matured_60d ? "good" : (row.has_top20 ? "info" : "bad"),
    detail: row.matured_60d ? "60d 성숙 표본 포함" : (row.has_top20 ? "snapshot 누적 중" : "snapshot 이상"),
  }));

  const gateTrend = gateHistoryRows
    .map((row) => ({
      as_of_date: row.as_of_date || null,
      overall_status: row.overall_status || null,
      daily_cycle_status: row.daily_cycle_status || null,
      matured_benchmark_dates: toNum(row.matured_benchmark_dates),
      trusted_ratio_top20: toNum(row.trusted_ratio_top20),
      buy_now_count: toNum(row.buy_now_count),
      watchlist_count: toNum(row.watchlist_count),
      blocked_count: toNum(row.blocked_count),
    }))
    .filter((row) => row.as_of_date)
    .slice(-7);
  const kpiTrend = kpiHistoryRows
    .map((row) => ({
      as_of_date: row.as_of_date || null,
      overall_status: row.overall_status || null,
      alert_metric_count: toNum(row.alert_metric_count),
      watch_metric_count: toNum(row.watch_metric_count),
      top20_mean_confidence_score: toNum(row.top20_mean_confidence_score),
      walkforward_top20_avg_return_60d: toNum(row.walkforward_top20_avg_return_60d),
      confidence_high_bucket_hit_rate_60d: toNum(row.confidence_high_bucket_hit_rate_60d),
      confidence_calibration_usable_bucket_count: toNum(row.confidence_calibration_usable_bucket_count),
    }))
    .filter((row) => row.as_of_date)
    .slice(-7);

  const cycleSteps = Array.isArray(dailyCycle.steps) ? dailyCycle.steps : [];
  const cycleWaitCount = cycleSteps.filter((step) => String(step.status || "").toUpperCase() === "WAIT").length;
  const cycleSuccessCount = cycleSteps.filter((step) => String(step.status || "").toUpperCase() === "SUCCESS").length;
  const expectedPrimaryTime = process.env.SCHEDULER_DAILY_TIME || "16:00";
  const expectedIntradayTime = process.env.SCHEDULER_RECOVERY_DAILY_TIME || "12:00";

  const normalizeScheduler = (status, role, expectedTime) => {
    const configuredDailyTime = status.configured_daily_time || null;
    return {
      role,
      mode: status.scheduler_mode || null,
      status: status.status || null,
      timezone: status.timezone || null,
      configured_daily_time: configuredDailyTime,
      expected_daily_time: expectedTime,
      schedule_matches_expected:
        configuredDailyTime && expectedTime ? configuredDailyTime === expectedTime : null,
      run_policy: status.run_policy || null,
      skip_catchup_on_start: boolify(status.skip_catchup_on_start),
      bootstrap_skip_until_date: status.bootstrap_skip_until_date || null,
      last_attempt_at: status.last_attempt_at || null,
      last_success_at: status.last_success_at || null,
      last_success_date: status.last_success_date || null,
      last_failure_at: status.last_failure_at || null,
      last_error: status.last_error || null,
      status_note: status.status_note || null,
    };
  };

  const primaryScheduler = normalizeScheduler(schedulerStatus, "close", expectedPrimaryTime);
  const intradayScheduler = normalizeScheduler(schedulerRecoveryStatus, "intraday", expectedIntradayTime);
  const primarySuccessTs = primaryScheduler.last_success_at ? Date.parse(primaryScheduler.last_success_at) : NaN;
  const intradaySuccessTs = intradayScheduler.last_success_at ? Date.parse(intradayScheduler.last_success_at) : NaN;
  const hasPrimarySuccess = Number.isFinite(primarySuccessTs);
  const hasIntradaySuccess = Number.isFinite(intradaySuccessTs);
  const latestScheduler =
    hasPrimarySuccess && hasIntradaySuccess
      ? (primarySuccessTs >= intradaySuccessTs ? primaryScheduler : intradayScheduler)
      : hasPrimarySuccess
      ? primaryScheduler
      : hasIntradaySuccess
      ? intradayScheduler
      : primaryScheduler;
  const artifactRefreshCandidates = [
    daily && daily.generated_at,
    gate && gate.generated_at,
    kpiMetadata && kpiMetadata.generated_at,
    dailyCycle && dailyCycle.generated_at,
  ]
    .map((value) => {
      const ts = parseTimestampMs(value);
      return Number.isFinite(ts) ? { value, ts } : null;
    })
    .filter(Boolean);
  const latestArtifactRefresh = artifactRefreshCandidates.length
    ? artifactRefreshCandidates.sort((a, b) => b.ts - a.ts)[0].value
    : null;
  const executionBasis = {
    current_basis: latestScheduler.role || "close",
    label: latestScheduler.role === "intraday" ? "장중 기준" : "마감 기준",
    description:
      latestScheduler.role === "intraday"
        ? "운영자가 수동 배포한 장중 기준 결과입니다. 오후장 대응용 판단으로 읽어야 합니다."
        : "운영자가 수동 배포한 마감 기준 결과입니다. 다음 영업일 기준본으로 읽습니다.",
    source_scheduler: latestScheduler.role === "intraday" ? "scheduler-recovery" : "scheduler",
    expected_daily_time: latestScheduler.expected_daily_time || null,
    last_refresh_at: latestArtifactRefresh || latestScheduler.last_success_at || null,
    last_auto_success_at: latestScheduler.last_success_at || null,
    last_artifact_refresh_at: latestArtifactRefresh || null,
    status_file_time: latestScheduler.configured_daily_time || null,
    status_time_matches_expected: latestScheduler.schedule_matches_expected,
  };
  const schedulerConfigWarnings = [primaryScheduler, intradayScheduler]
    .filter((item) => item.configured_daily_time && item.expected_daily_time && item.configured_daily_time !== item.expected_daily_time)
    .map((item) => {
      const schedulerName = item.role === "intraday" ? "scheduler-recovery" : "scheduler";
      return `${schedulerName} 상태 파일 기준 시각이 ${item.configured_daily_time}로 남아 있습니다. 기대 기준은 ${item.expected_daily_time}이며 수동 배포 후 다시 확인해야 합니다.`;
    });

  const goReasons = [];
  let goDecision = "WAIT";
  if ((readiness60 || "").toUpperCase() !== "READY") goReasons.push("60d confidence calibration readiness가 아직 READY가 아닙니다.");
  if (!Number.isFinite(maturedBenchmarkDates) || maturedBenchmarkDates < 3) goReasons.push("matured benchmark dates가 3 미만입니다.");
  if ((String(gate.overall_status || "").toUpperCase() || "HOLD") === "HOLD") goReasons.push("buy gate overall_status가 아직 HOLD입니다.");
  if (!Number.isFinite(trustedRatio) || trustedRatio < 0.30) goReasons.push("confidence_v2 trusted ratio가 아직 낮습니다.");
  if ((String(walkforwardStatus || "").toUpperCase() || "CONDITIONAL") === "CONDITIONAL") goReasons.push("walkforward acceptance가 아직 CONDITIONAL 단계입니다.");
  if (!goReasons.length) goDecision = "GO_CHECK";

  const transitionChecklist = [
    {
      id: "live_score_unified",
      label: "운영 해석은 live_score 기준으로 통일",
      passed: true,
      detail: "API/대시보드 계산에서 live_score, live_rank, live_score_source를 우선 사용합니다.",
    },
    {
      id: "snapshot_ready_60d",
      label: "60d confidence calibration readiness가 READY인지",
      passed: String(readiness60 || "").toUpperCase() === "READY",
      detail: `현재 값: ${readiness60 || "WAIT"}`,
    },
    {
      id: "matured_benchmark_dates",
      label: "matured benchmark dates가 3 이상인지",
      passed: Number.isFinite(maturedBenchmarkDates) && maturedBenchmarkDates >= 3,
      detail: `현재 값: ${Number.isFinite(maturedBenchmarkDates) ? maturedBenchmarkDates : "-"}`,
    },
    {
      id: "buy_gate_not_hold",
      label: "buy gate overall_status가 HOLD를 벗어났는지",
      passed: !["", "HOLD"].includes(String(gate.overall_status || "").toUpperCase()),
      detail: `현재 값: ${gate.overall_status || "-"}`,
    },
    {
      id: "trusted_ratio",
      label: "confidence_v2 trusted ratio가 0.30 이상인지",
      passed: Number.isFinite(trustedRatio) && trustedRatio >= 0.30,
      detail: `현재 값: ${Number.isFinite(trustedRatio) ? trustedRatio.toFixed(2) : "-"}`,
    },
    {
      id: "walkforward_acceptance",
      label: "walkforward acceptance가 CONDITIONAL 이상인지",
      passed: ["ACCEPTED", "PASS", "READY"].includes(String(walkforwardStatus || "").toUpperCase()),
      detail: `현재 값: ${walkforwardStatus || "-"}`,
    },
  ];

  const summaryDecision =
    goDecision === "GO_CHECK"
      ? "운영 전환 재검토 가능"
      : gate.overall_status === "BLOCK"
      ? "오늘은 매수보다 관찰 중심"
      : "아직 운영 전환 대기";
  const summaryReason =
    !Number.isFinite(matured60) || matured60 <= 0
      ? "snapshot은 누적되고 있지만 60일 성숙 표본이 아직 없어 운영 근거가 부족합니다."
      : (readiness60 || "").toUpperCase() !== "READY"
      ? "성숙 표본은 생기기 시작했지만 confidence calibration이 아직 READY가 아닙니다."
      : !Number.isFinite(trustedRatio) || trustedRatio < 0.30
      ? "상위 후보 trusted ratio가 아직 낮아 적극 운영 신뢰도가 부족합니다."
      : "핵심 readiness 조건은 대체로 충족했지만 남은 경고 항목을 함께 확인해야 합니다.";
  const actionGuide =
    gate.overall_status === "BLOCK"
      ? "추천 종목은 watchlist 위주로만 보고, 실제 매수는 소액 테스트 또는 보류가 맞습니다."
      : gate.overall_status === "WATCH" || goDecision !== "GO_CHECK"
      ? "상위 후보를 수동 검토하되, 체결 전 갭 상승과 거래대금, 기존 보유 중복을 먼저 확인하십시오."
      : "운영 전환 체크를 다시 검토하고, 하루 이상 shadow 비교 결과와 함께 승격 여부를 결정하십시오.";
  const cardInterpretations = {
    readiness:
      !Number.isFinite(matured60) || matured60 <= 0
        ? "snapshot 적재는 정상입니다. 다만 60일이 지나 결과를 검증할 수 있는 표본은 아직 없습니다."
        : `60일 성숙 표본 ${matured60}개가 쌓였습니다. 이제 calibration readiness를 함께 해석하면 됩니다.`,
    transition: `운영 전환 체크 ${transitionChecklist.filter((item) => item.passed).length}/${transitionChecklist.length}개가 통과 상태입니다.`,
    gate:
      gate.overall_status === "BLOCK"
        ? "오늘 gate는 BLOCK입니다. 적극 매수보다 관찰과 선별이 우선입니다."
        : gate.overall_status === "WATCH"
        ? "오늘 gate는 WATCH입니다. 상위 후보는 보되 기계적 매수는 아직 이릅니다."
        : "오늘 gate는 상대적으로 양호하지만, 아래 조건을 함께 보고 판단해야 합니다.",
    kpi:
      alertMetrics.length > 0
        ? `KPI 경고가 ${alertMetrics.length}개 남아 있습니다. 일부 숫자가 좋아 보여도 아직 구조적 안정성은 미완입니다.`
        : "현재 KPI 경고는 크지 않습니다. 다만 walkforward와 confidence 추이를 함께 봐야 합니다.",
  };
  const criticalReasons = [];
  if ((String(gate.overall_status || "").toUpperCase() || "WAIT") === "BLOCK") {
    criticalReasons.push(
      gateDecision.reason_summary
        ? `BUY GATE is BLOCK. ${gateDecision.reason_summary}`
        : "BUY GATE is BLOCK, so observation is prioritized over new entries today."
    );
  }
  if ((String(walkforwardStatus || "").toUpperCase() || "-") === "REJECTED") {
    walkforwardReasons
      .map((code) => mapWalkforwardReasonCode(code, walkforwardAcceptance))
      .filter(Boolean)
      .forEach((reason) => criticalReasons.push(`Walkforward REJECTED: ${reason}`));
  }
  if ((String(readiness60 || "").toUpperCase() || "WAIT") !== "READY") {
    criticalReasons.push(
      Number.isFinite(matured60) && matured60 > 0
        ? `60일 성숙 표본 ${matured60}개가 있지만 calibration readiness는 아직 ${readiness60 || "WAIT"}입니다.`
        : "60일 성숙 표본이 아직 없어 calibration readiness를 READY로 볼 수 없습니다."
    );
  }
  alertMetrics
    .slice(0, 3)
    .map((item) => formatOpsMetricReason(item))
    .filter(Boolean)
    .forEach((reason) => criticalReasons.push(`KPI ALERT: ${reason}`));
  if (watchMetrics.length && criticalReasons.length < 6) {
    const watchReason = formatOpsMetricReason(watchMetrics[0]);
    if (watchReason) criticalReasons.push(`KPI WATCH: ${watchReason}`);
  }


  const gateAsOfDate = isIsoDateString(gate.asof_date) ? gate.asof_date : null;
  const manualAsOfDate = isIsoDateString(manual.asof_date) ? manual.asof_date : null;
  const effectiveAsOfDate = rankingLatestDate || gateAsOfDate || manualAsOfDate || null;

  return {
    generated_at: new Date().toISOString(),
    asof_date: effectiveAsOfDate,
    readiness: {
      snapshot_count: snapshotCount,
      matured_snapshot_count_20d: matured20,
      matured_snapshot_count_60d: matured60,
      matured_snapshot_count_90d: matured90,
      oldest_snapshot_date: inventory["oldest snapshot date"] || null,
      latest_snapshot_date: inventory["latest snapshot date"] || null,
      confidence_calibration_readiness_60d: readiness60,
      note: inventory.note || null,
    },
    gate: {
      asof_date: gateAsOfDate,
      overall_status: gate.overall_status || null,
      primary_bucket: gate.primary_bucket ?? null,
      daily_cycle_status: gate.daily_cycle_status || null,
      reason_summary: gateDecision.reason_summary || null,
      market_regime: gate.market_regime || gateDecision.market_regime || null,
      market_regime_interpretation: marketRegimeInterpretation,
      matured_benchmark_dates: maturedBenchmarkDates,
      trusted_ratio_top20: trustedRatio,
      walkforward_acceptance: walkforwardStatus,
      buy_now_count: toNum(buyability.buy_now_count),
      watchlist_count: toNum(buyability.watchlist_count),
      blocked_count: toNum(buyability.blocked_count),
      paper_only_count: toNum(buyability.paper_only_count),
    },
    kpi: {
      overall_status: kpiSummary.overall_status || kpi.overall_status || null,
      latest_date: kpiSummary.latest_date || kpi.latest_date || null,
      score_formula_version: kpiMetadata.score_formula_version || kpi.score_formula_version || null,
      alert_metric_count: alertMetrics.length,
      watch_metric_count: watchMetrics.length,
      top20_mean_confidence_score: toNum(topConfidenceMetric?.value),
      walkforward_top20_avg_return_60d: toNum(wfMetric?.value),
      alert_metrics: alertMetrics.slice(0, 6),
      watch_metrics: watchMetrics.slice(0, 6),
    },
    outputs: {
      ranking_latest_date: rankingLatestDate,
      ranking_row_count: latestRanking.length,
      daily_recommendations_date: manualAsOfDate,
      gate_asof_date: gateAsOfDate,
      priority_candidate_count: Array.isArray(manual.priority_candidates) ? manual.priority_candidates.length : 0,
      caution_candidate_count: Array.isArray(manual.caution_candidates) ? manual.caution_candidates.length : 0,
      daily_cycle_status: dailyCycle.overall_status || null,
      daily_cycle_wait_count: cycleWaitCount,
      daily_cycle_success_count: cycleSuccessCount,
    },
    execution_basis: executionBasis,
    scheduler: {
      mode: latestScheduler.mode || null,
      status: latestScheduler.status || null,
      timezone: latestScheduler.timezone || null,
      configured_daily_time: latestScheduler.configured_daily_time || null,
      expected_daily_time: latestScheduler.expected_daily_time || null,
      skip_catchup_on_start: latestScheduler.skip_catchup_on_start,
      bootstrap_skip_until_date: latestScheduler.bootstrap_skip_until_date || null,
      last_attempt_at: latestScheduler.last_attempt_at || null,
      last_success_at: latestScheduler.last_success_at || null,
      last_success_date: latestScheduler.last_success_date || null,
      last_failure_at: latestScheduler.last_failure_at || null,
      last_error: latestScheduler.last_error || null,
      status_note: latestScheduler.status_note || null,
      current_role: latestScheduler.role || null,
      current_label: executionBasis.label,
      config_warning_count: schedulerConfigWarnings.length,
      config_warnings: schedulerConfigWarnings,
    },
    schedulers: {
      primary: primaryScheduler,
      intraday: intradayScheduler,
    },
    go_no_go: {
      decision: goDecision,
      reasons: goReasons,
    },
    interpretation: {
      summary_decision: summaryDecision,
      summary_reason: summaryReason,
      action_guide: actionGuide,
      critical_reasons: criticalReasons.slice(0, 6),
      cards: cardInterpretations,
    },
    trends: {
      readiness: recentReadinessTrend,
      recent_state_badges: recentStateBadges,
      gate: gateTrend,
      kpi: kpiTrend,
    },
    transition_checklist: transitionChecklist,
    visitor_analytics: visitorAnalytics,
    notes: {
      operator_memo: typeof opsNotes.operator_memo === "string" ? opsNotes.operator_memo : "",
      last_updated_at: opsNotes.last_updated_at || null,
      last_updated_by: opsNotes.last_updated_by || null,
    },
    manual: {
      gate_status: manual.gate_status || null,
      walkforward_acceptance_status: manual.walkforward_acceptance_status || null,
      intraday_summary: manual.intraday_summary || null,
      priority_candidates: manual.priority_candidates || [],
      caution_candidates: manual.caution_candidates || [],
      checklist: manual.checklist || [],
    },
    shadow: {
      quality_risk_guard_candidates: shadowCandidates,
      repeatability: {
        summary: shadowRepeatability.summary || {},
        usable_dates: Array.isArray(shadowRepeatability.usable_dates) ? shadowRepeatability.usable_dates : [],
        top_repeaters: Array.isArray(shadowRepeatability.top_repeaters)
          ? shadowRepeatability.top_repeaters.slice(0, 5)
          : [],
      },
    },
  };
}

async function queryRows(sql, params = []) {
  const { rows } = await pool.query(sql, params);
  return rows;
}

async function ensurePageViewSchema() {
  if (pageViewSchemaReady) return pageViewSchemaReady;
  pageViewSchemaReady = (async () => {
    await queryRows(`
      CREATE TABLE IF NOT EXISTS public.page_view_events (
        id BIGSERIAL PRIMARY KEY,
        visitor_id TEXT NOT NULL,
        path TEXT NOT NULL,
        referrer TEXT NULL,
        user_agent TEXT NULL,
        ip_hash TEXT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
      )
    `);
    await queryRows("CREATE INDEX IF NOT EXISTS idx_page_view_events_created_at ON public.page_view_events(created_at DESC)");
    await queryRows("CREATE INDEX IF NOT EXISTS idx_page_view_events_visitor_created_at ON public.page_view_events(visitor_id, created_at DESC)");
    await queryRows("CREATE INDEX IF NOT EXISTS idx_page_view_events_path_created_at ON public.page_view_events(path, created_at DESC)");
  })().catch((error) => {
    pageViewSchemaReady = null;
    throw error;
  });
  return pageViewSchemaReady;
}

async function ensureLiveTradeReviewSchema() {
  if (liveTradeReviewSchemaReady) return liveTradeReviewSchemaReady;
  liveTradeReviewSchemaReady = (async () => {
    await queryRows("CREATE SCHEMA IF NOT EXISTS research");
    await queryRows(`
      CREATE TABLE IF NOT EXISTS research.live_trade_review (
        review_id BIGSERIAL PRIMARY KEY,
        intent_id TEXT NULL,
        request_id TEXT NULL,
        code VARCHAR(10) NOT NULL,
        review_date DATE NOT NULL DEFAULT CURRENT_DATE,
        pre_tags TEXT[] NULL,
        post_tags TEXT[] NULL,
        outcome_label TEXT NULL,
        review_note TEXT NULL,
        next_action_note TEXT NULL,
        reviewer TEXT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
      )
    `);
    await queryRows(`
      CREATE INDEX IF NOT EXISTS idx_live_trade_review_code_date
      ON research.live_trade_review(code, review_date DESC)
    `);
    await queryRows(`
      CREATE INDEX IF NOT EXISTS idx_live_trade_review_request
      ON research.live_trade_review(request_id)
    `);
    await queryRows(`
      CREATE INDEX IF NOT EXISTS idx_live_trade_review_intent
      ON research.live_trade_review(intent_id)
    `);
  })().catch((error) => {
    liveTradeReviewSchemaReady = null;
    throw error;
  });
  return liveTradeReviewSchemaReady;
}

async function ensureMeaningfulnessReviewSchema() {
  if (meaningfulnessReviewSchemaReady) return meaningfulnessReviewSchemaReady;
  meaningfulnessReviewSchemaReady = (async () => {
    await queryRows("CREATE SCHEMA IF NOT EXISTS research");
    await queryRows(`
      CREATE TABLE IF NOT EXISTS research.meaningfulness_review_note (
        analysis_date DATE NOT NULL,
        code TEXT NOT NULL,
        decision TEXT NULL,
        note TEXT NULL,
        updated_by TEXT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        PRIMARY KEY (analysis_date, code)
      )
    `);
    await queryRows(`
      CREATE INDEX IF NOT EXISTS idx_meaningfulness_review_note_updated
      ON research.meaningfulness_review_note(updated_at DESC)
    `);
  })().catch((error) => {
    meaningfulnessReviewSchemaReady = null;
    throw error;
  });
  return meaningfulnessReviewSchemaReady;
}

async function buildMeaningfulnessOutcomes({ analysisDate, codes }) {
  const normalizedCodes = Array.from(new Set((codes || []).map((code) => String(code || "").trim()).filter(Boolean)));
  if (!analysisDate || !normalizedCodes.length) return [];

  const rows = await queryRows(
    `
    WITH price_rows AS (
      SELECT
        code,
        date::date AS date,
        COALESCE(adj_close, close) AS px
      FROM fact_price_daily
      WHERE code = ANY($1::text[])
        AND date >= $2::date
        AND COALESCE(adj_close, close) IS NOT NULL
    ),
    first_rows AS (
      SELECT DISTINCT ON (code)
        code,
        date AS entry_date,
        px AS entry_close
      FROM price_rows
      ORDER BY code, date ASC
    ),
    latest_rows AS (
      SELECT DISTINCT ON (code)
        code,
        date AS latest_date,
        px AS latest_close
      FROM price_rows
      ORDER BY code, date DESC
    ),
    agg_rows AS (
      SELECT
        code,
        MAX(px) AS peak_close,
        MIN(px) AS trough_close,
        COUNT(*)::int AS observed_days
      FROM price_rows
      GROUP BY code
    )
    SELECT
      f.code,
      f.entry_date,
      f.entry_close,
      l.latest_date,
      l.latest_close,
      a.peak_close,
      a.trough_close,
      a.observed_days
    FROM first_rows f
    JOIN latest_rows l USING (code)
    JOIN agg_rows a USING (code)
    ORDER BY f.code ASC
    `,
    [normalizedCodes, analysisDate]
  );

  return rows.map((row) => {
    const entryClose = toNum(row.entry_close);
    const latestClose = toNum(row.latest_close);
    const peakClose = toNum(row.peak_close);
    const troughClose = toNum(row.trough_close);
    const latestReturn = entryClose && latestClose ? (latestClose / entryClose) - 1 : null;
    const peakReturn = entryClose && peakClose ? (peakClose / entryClose) - 1 : null;
    const troughReturn = entryClose && troughClose ? (troughClose / entryClose) - 1 : null;
    return {
      code: row.code,
      entry_date: row.entry_date,
      entry_close: entryClose,
      latest_date: row.latest_date,
      latest_close: latestClose,
      peak_close: peakClose,
      trough_close: troughClose,
      observed_days: toNum(row.observed_days),
      latest_return: latestReturn,
      peak_return: peakReturn,
      trough_return: troughReturn,
    };
  });
}

async function recordPageView(req, res) {
  if (!shouldTrackPageView(req)) return;
  try {
    const visitorId = ensureVisitorId(req, res);
    await ensurePageViewSchema();
    const pathname = String(req.path || req.originalUrl || "").split("?")[0] || "/";
    const referrer = String(req.get("referer") || "").slice(0, 500) || null;
    const userAgent = String(req.get("user-agent") || "").slice(0, 500) || null;
    const ipHash = hashIp(req.ip || req.headers["x-forwarded-for"] || "");
    await queryRows(
      `
      INSERT INTO public.page_view_events (visitor_id, path, referrer, user_agent, ip_hash)
      VALUES ($1, $2, $3, $4, $5)
      `,
      [visitorId, pathname, referrer, userAgent, ipHash]
    );
  } catch (error) {
    console.warn("recordPageView error", error.message);
  }
}

async function buildVisitorAnalyticsSummary() {
  try {
    await ensurePageViewSchema();
    const [todayRows, recentRows, trendRows, topPagesRows] = await Promise.all([
      queryRows(
        `
        SELECT COUNT(*)::int AS pageviews,
               COUNT(DISTINCT visitor_id)::int AS unique_visitors
        FROM public.page_view_events
        WHERE created_at >= date_trunc('day', now())
        `
      ),
      queryRows(
        `
        SELECT COUNT(*)::int AS pageviews,
               COUNT(DISTINCT visitor_id)::int AS unique_visitors
        FROM public.page_view_events
        WHERE created_at >= date_trunc('day', now()) - interval '6 day'
        `
      ),
      queryRows(
        `
        SELECT to_char(day_bucket, 'YYYY-MM-DD') AS as_of_date,
               pageviews,
               unique_visitors
        FROM (
          SELECT date_trunc('day', created_at) AS day_bucket,
                 COUNT(*)::int AS pageviews,
                 COUNT(DISTINCT visitor_id)::int AS unique_visitors
          FROM public.page_view_events
          WHERE created_at >= date_trunc('day', now()) - interval '6 day'
          GROUP BY 1
        ) ranked
        ORDER BY day_bucket ASC
        `
      ),
      queryRows(
        `
        SELECT path,
               COUNT(*)::int AS pageviews,
               COUNT(DISTINCT visitor_id)::int AS unique_visitors
        FROM public.page_view_events
        WHERE created_at >= date_trunc('day', now()) - interval '6 day'
        GROUP BY path
        ORDER BY pageviews DESC, unique_visitors DESC, path ASC
        LIMIT 5
        `
      ),
    ]);
    const today = todayRows[0] || {};
    const recent = recentRows[0] || {};
    return {
      available: true,
      today_pageviews: toNum(today.pageviews) || 0,
      today_unique_visitors: toNum(today.unique_visitors) || 0,
      last_7d_pageviews: toNum(recent.pageviews) || 0,
      last_7d_unique_visitors: toNum(recent.unique_visitors) || 0,
      trend_7d: trendRows.map((row) => ({
        as_of_date: toIsoDate(row.as_of_date),
        pageviews: toNum(row.pageviews) || 0,
        unique_visitors: toNum(row.unique_visitors) || 0,
      })),
      top_pages_7d: topPagesRows.map((row) => ({
        path: row.path || "/",
        pageviews: toNum(row.pageviews) || 0,
        unique_visitors: toNum(row.unique_visitors) || 0,
      })),
    };
  } catch (error) {
    console.warn("buildVisitorAnalyticsSummary error", error.message);
    return {
      available: false,
      today_pageviews: 0,
      today_unique_visitors: 0,
      last_7d_pageviews: 0,
      last_7d_unique_visitors: 0,
      trend_7d: [],
      top_pages_7d: [],
      error: error.message,
    };
  }
}

async function readPayloadFromDb(payloadKey) {
  if (!payloadKey) return null;
  try {
    const rows = await queryRows(
      `
      SELECT payload_json, asof_date, generated_at, source_path, updated_at
      FROM research.app_payload_store
      WHERE payload_key = $1
      LIMIT 1
      `,
      [payloadKey]
    );
    return rows[0]?.payload_json || null;
  } catch (e) {
    console.warn("readPayloadFromDb error", payloadKey, e.message);
    return null;
  }
}

async function readJsonPayloadDbFirst(payloadKey, fallbackPaths = []) {
  const dbPayload = await readPayloadFromDb(payloadKey);
  if (dbPayload && typeof dbPayload === "object") return dbPayload;
  for (const filePath of fallbackPaths) {
    const value = readJson(filePath);
    if (value && typeof value === "object") return value;
  }
  return {};
}

async function getLatestRankingContext() {
  try {
    const latestRows = await queryRows("SELECT MAX(date) AS latest_date FROM daily_ranking");
    const rankingLatestDate = toIsoDate(latestRows[0]?.latest_date);
    if (!rankingLatestDate) {
      throw new Error("daily_ranking latest date missing");
    }
    const rankingRows = await queryRows(
      `
      SELECT *
      FROM daily_ranking
      WHERE date = $1
      ORDER BY final_score DESC NULLS LAST, code ASC
      `,
      [rankingLatestDate]
    );
    return { rankingLatestDate, rankingRows };
  } catch (e) {
    const rankingRows = readCsv(path.join(DATA_DIR, "ranking_final.csv")) || [];
    const rankingDates = rankingRows
      .map((row) => String(row.date || "").trim())
      .filter((value) => isIsoDateString(value));
    return {
      rankingLatestDate: rankingDates.length ? rankingDates.sort().pop() : null,
      rankingRows,
    };
  }
}

function normalizePaperRun(row) {
  if (!row) return null;
  return {
    paper_run_id: toNum(row.paper_run_id),
    run_tag: row.run_tag || null,
    source_mode: row.source_mode || null,
    asof_date: toIsoDate(row.asof_date),
    hold_days: toNum(row.hold_days),
    initial_nav: toNum(row.initial_nav),
    entry_fee_bps: toNum(row.entry_fee_bps),
    exit_fee_bps: toNum(row.exit_fee_bps),
    entry_slippage_bps: toNum(row.entry_slippage_bps),
    exit_slippage_bps: toNum(row.exit_slippage_bps),
    positions_row_count: toNum(row.positions_row_count),
    nav_row_count: toNum(row.nav_row_count),
    source_positions_csv: row.source_positions_csv || null,
    source_nav_csv: row.source_nav_csv || null,
    source_report_md: row.source_report_md || null,
    comment: row.comment || null,
    created_at: row.created_at || null,
    updated_at: row.updated_at || null,
  };
}

function parseBooleanLike(value) {
  if (typeof value === "boolean") return value;
  const text = String(value ?? "").trim().toLowerCase();
  if (!text) return false;
  return ["true", "1", "yes", "y", "on"].includes(text);
}

function normalizeRuleSignalRow(row) {
  return {
    date: toIsoDate(row.date) || String(row.date || "").trim() || null,
    code: String(row.code || "").trim().padStart(6, "0"),
    name: row.name || null,
    sector: row.sector || null,
    market: row.market || null,
    close: toNum(row.close),
    open: toNum(row.open),
    expected_gap: toNum(row.expected_gap),
    expected_entry_price: toNum(row.expected_entry_price),
    actual_open_gap: toNum(row.actual_open_gap),
    trading_value: toNum(row.trading_value),
    trading_value_ma_20: toNum(row.trading_value_ma_20),
    trading_value_pass: parseBooleanLike(row.trading_value_pass),
    trading_value_block_reason: row.trading_value_block_reason || null,
    gap_risk_blocked: parseBooleanLike(row.gap_risk_blocked),
    gap_risk_reason: row.gap_risk_reason || null,
    market_defensive_mode: parseBooleanLike(row.market_defensive_mode),
    market_entry_allowed: parseBooleanLike(row.market_entry_allowed),
    rule_score: toNum(row.rule_score),
    rule_score_v2: toNum(row.rule_score_v2),
    liquidity_score: toNum(row.liquidity_score),
    vol_20: toNum(row.vol_20),
    entry_signal: parseBooleanLike(row.entry_signal),
    strong_entry_signal: parseBooleanLike(row.strong_entry_signal),
    signal_strength: row.signal_strength || "none",
    strategy_id: row.strategy_id || null,
    engine_type: row.engine_type || null,
    run_mode: row.run_mode || null,
  };
}

function loadLatestRuleSignals() {
  const portfolio = readJson(path.join(OUTPUTS_DIR, "rule_portfolio_plan.json")) || {};
  const preview = readJson(path.join(OUTPUTS_DIR, "rule_order_preview.json")) || {};
  const portfolioItems = Array.isArray(portfolio.items) ? portfolio.items : [];
  const previewItems = Array.isArray(preview.items) ? preview.items : [];
  if (portfolio.as_of_date && portfolioItems.length) {
    const previewByCode = new Map(
      previewItems.map((item) => [String(item.code || item.symbol || "").trim().padStart(6, "0"), item])
    );
    const items = portfolioItems.map((item) => {
      const code = String(item.code || "").trim().padStart(6, "0");
      const previewItem = previewByCode.get(code) || {};
      return {
        date: portfolio.as_of_date,
        code,
        name: item.name || previewItem.name || null,
        sector: item.sector || null,
        market: null,
        close: null,
        open: null,
        expected_gap: null,
        expected_entry_price: toNum(item.expected_entry_price) ?? toNum(previewItem.expected_execution_price),
        actual_open_gap: null,
        trading_value: null,
        trading_value_ma_20: null,
        trading_value_pass: String(item.trading_value_block_reason || "none") === "none",
        trading_value_block_reason: item.trading_value_block_reason || "none",
        gap_risk_blocked: String(item.gap_risk_reason || "none") !== "none",
        gap_risk_reason: item.gap_risk_reason || "none",
        market_defensive_mode: !!item.market_defensive_mode,
        market_entry_allowed: !item.market_defensive_mode,
        rule_score: toNum(item.rule_score),
        rule_score_v2: toNum(item.rule_score_v2),
        liquidity_score: toNum(item.liquidity_score),
        vol_20: toNum(item.vol_20),
        entry_signal: !!item.entry_signal,
        strong_entry_signal: !!item.strong_entry_signal,
        signal_strength: item.signal_strength || previewItem.signal_strength || "none",
        strategy_id: portfolio.strategy_id || preview.strategy_id || null,
        engine_type: portfolio.engine_type || preview.engine_type || null,
        run_mode: portfolio.run_mode || preview.run_mode || null,
      };
    }).sort((a, b) => {
      const strengthRank = { strong_entry: 2, entry: 1, none: 0 };
      const strengthDiff = (strengthRank[b.signal_strength] || 0) - (strengthRank[a.signal_strength] || 0);
      if (strengthDiff) return strengthDiff;
      const v2Diff = (b.rule_score_v2 || 0) - (a.rule_score_v2 || 0);
      if (v2Diff) return v2Diff;
      return (b.rule_score || 0) - (a.rule_score || 0);
    });
    return { latestDate: portfolio.as_of_date, items };
  }

  const rows = readCsv(path.join(DATA_DIR, "rule_signals.csv")) || [];
  if (!rows.length) {
    return { latestDate: null, items: [] };
  }
  const latestDate = rows
    .map((row) => toIsoDate(row.date) || String(row.date || "").trim())
    .filter(Boolean)
    .sort()
    .pop();
  const strengthRank = { strong_entry: 2, entry: 1, none: 0 };
  const items = rows
    .filter((row) => (toIsoDate(row.date) || String(row.date || "").trim()) === latestDate)
    .map(normalizeRuleSignalRow)
    .sort((a, b) => {
      const strengthDiff = (strengthRank[b.signal_strength] || 0) - (strengthRank[a.signal_strength] || 0);
      if (strengthDiff) return strengthDiff;
      const v2Diff = (b.rule_score_v2 || 0) - (a.rule_score_v2 || 0);
      if (v2Diff) return v2Diff;
      return (b.rule_score || 0) - (a.rule_score || 0);
    });
  return { latestDate, items };
}

function summarizeCountsBy(items, key, fallback = "none") {
  const counts = {};
  items.forEach((item) => {
    const raw = item?.[key];
    const value = String(raw == null || raw === "" ? fallback : raw).trim() || fallback;
    counts[value] = (counts[value] || 0) + 1;
  });
  return Object.entries(counts)
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name));
}

function buildRuleDashboardSummary() {
  const { latestDate, items: signalItems } = loadLatestRuleSignals();
  const portfolio = readJson(path.join(OUTPUTS_DIR, "rule_portfolio_plan.json")) || {};
  const preview = readJson(path.join(OUTPUTS_DIR, "rule_order_preview.json")) || {};
  const paperState = readJson(path.join(OUTPUTS_DIR, "rule_account_paper_state.json")) || {};
  const liveState = readJson(path.join(OUTPUTS_DIR, "rule_account_live_state.json")) || {};
  const backtest = readJson(path.join(OUTPUTS_DIR, "rule_strategy_backtest_report.json")) || {};
  const execution = readJson(path.join(OUTPUTS_DIR, "rule_execution_results.json")) || {};
  const accountState = (String(execution.run_mode || preview.run_mode || "").toLowerCase() === "paper" || (!liveState.generated_at && !Array.isArray(liveState.positions)))
    ? paperState
    : liveState;

  const portfolioItems = Array.isArray(portfolio.items) ? portfolio.items : [];
  const previewItems = Array.isArray(preview.items) ? preview.items : [];
  const positions = Array.isArray(accountState.positions) ? accountState.positions : [];

  return {
    generated_at: preview.generated_at || portfolio.generated_at || backtest.generated_at || null,
    as_of_date: preview.as_of_date || portfolio.as_of_date || latestDate || null,
    strategy_id: preview.strategy_id || portfolio.strategy_id || signalItems[0]?.strategy_id || null,
    engine_type: preview.engine_type || portfolio.engine_type || signalItems[0]?.engine_type || null,
    run_mode: preview.run_mode || portfolio.run_mode || signalItems[0]?.run_mode || null,
    counts: {
      total_candidates: signalItems.length,
      entry_signal_count: signalItems.filter((item) => item.entry_signal).length,
      strong_entry_count: signalItems.filter((item) => item.strong_entry_signal).length,
      gap_risk_blocked_count: signalItems.filter((item) => item.gap_risk_blocked).length,
      trading_value_failed_count: signalItems.filter((item) => !item.trading_value_pass).length,
      defensive_mode_count: signalItems.filter((item) => item.market_defensive_mode).length,
      hold_count: toNum(portfolio.summary?.hold_count) || 0,
      buy_count: toNum(portfolio.summary?.buy_count) || 0,
      reduce_count: toNum(portfolio.summary?.reduce_count) || 0,
      exit_count: toNum(portfolio.summary?.exit_count) || 0,
      skip_count: toNum(portfolio.summary?.skip_count) || 0,
      preview_request_count: toNum(preview.summary?.request_count) || 0,
      preview_buy_count: toNum(preview.summary?.buy_preview_count) || 0,
      preview_sell_count: toNum(preview.summary?.sell_preview_count) || 0,
      preview_allowed_count: toNum(preview.summary?.order_allowed_count) || 0,
      execution_submitted_count: toNum(execution.summary?.submitted_count) || 0,
      execution_failed_count: toNum(execution.summary?.failed_count) || 0,
      execution_skipped_count: toNum(execution.summary?.skipped_count) || 0,
      execution_filled_count: toNum(execution.summary?.filled_count) || 0,
      execution_partial_filled_count: toNum(execution.summary?.partial_filled_count) || 0,
      execution_unfilled_count: toNum(execution.summary?.unfilled_count) || 0,
      execution_canceled_count: toNum(execution.summary?.canceled_count) || 0,
      execution_simulated_filled_count: toNum(execution.summary?.simulated_filled_count) || 0,
      execution_simulated_unfilled_count: toNum(execution.summary?.simulated_unfilled_count) || 0,
      paper_position_count: positions.length,
    },
    distributions: {
      signal_strength: summarizeCountsBy(signalItems, "signal_strength", "none"),
      portfolio_action_reason: summarizeCountsBy(portfolioItems, "portfolio_action_reason", "none").slice(0, 10),
      order_block_reason: summarizeCountsBy(previewItems, "order_block_reason", "none").slice(0, 10),
      trading_value_block_reason: summarizeCountsBy(
        signalItems.filter((item) => !item.trading_value_pass),
        "trading_value_block_reason",
        "none"
      ),
      gap_risk_reason: summarizeCountsBy(
        signalItems.filter((item) => item.gap_risk_blocked),
        "gap_risk_reason",
        "none"
      ),
    },
    top_candidates: {
      strong: signalItems.filter((item) => item.strong_entry_signal).slice(0, 10),
      entry_only: signalItems.filter((item) => item.entry_signal && !item.strong_entry_signal).slice(0, 10),
    },
    paper_state: {
      total_equity: toNum(accountState.total_equity),
      cash: toNum(accountState.cash),
      recent_trade_count: Array.isArray(accountState.recent_trades) ? accountState.recent_trades.length : 0,
      cooldown_count: Array.isArray(accountState.cooldown_codes) ? accountState.cooldown_codes.length : 0,
      positions,
    },
    execution: execution || {},
    backtest: {
      summary: backtest.summary || {},
      portfolio_equity_curve: backtest.portfolio_equity_curve || {},
      score_distribution: backtest.score_distribution || {},
      trading_value_distribution: backtest.trading_value_distribution || {},
    },
  };
}

const RULE_DASHBOARD_SUMMARY_PATH = path.join(OUTPUTS_DIR, "rule_dashboard_summary.json");
const RULE_SIGNALS_LATEST_PATH = path.join(OUTPUTS_DIR, "rule_signals_latest.json");
const RULE_PORTFOLIO_PLAN_PATH = path.join(OUTPUTS_DIR, "rule_portfolio_plan.json");
const RULE_ORDER_PREVIEW_PATH = path.join(OUTPUTS_DIR, "rule_order_preview.json");
const RULE_PAPER_STATE_PATH = path.join(OUTPUTS_DIR, "rule_account_paper_state.json");
const RULE_LIVE_STATE_PATH = path.join(OUTPUTS_DIR, "rule_account_live_state.json");
const RULE_BACKTEST_PATH = path.join(OUTPUTS_DIR, "rule_strategy_backtest_report.json");
const RULE_EXECUTION_RESULTS_PATH = path.join(OUTPUTS_DIR, "rule_execution_results.json");

async function readRuleSummaryPayload() {
  return await readJsonPayloadDbFirst("rule_dashboard_summary", [RULE_DASHBOARD_SUMMARY_PATH]);
}

async function readRuleSignalsPayload(strength = "all", limit = 100) {
  const payload = await readJsonPayloadDbFirst("rule_signals_latest", [RULE_SIGNALS_LATEST_PATH]);
  const items = Array.isArray(payload.items) ? payload.items : [];
  const normalizedStrength = String(strength || "all").trim().toLowerCase() || "all";
  const filtered = items.filter((item) => {
    if (!normalizedStrength || normalizedStrength === "all") return true;
    if (normalizedStrength === "strong") return !!item.strong_entry_signal;
    if (normalizedStrength === "entry") return !!item.entry_signal;
    if (normalizedStrength === "none") return String(item.signal_strength || "none") === "none";
    return String(item.signal_strength || "none") === normalizedStrength;
  });
  return {
    as_of_date: payload.as_of_date || null,
    strength: normalizedStrength || "all",
    count: filtered.length,
    items: filtered.slice(0, limit),
  };
}

async function readRulePortfolioPlanPayload() {
  return await readJsonPayloadDbFirst("rule_portfolio_plan", [RULE_PORTFOLIO_PLAN_PATH]);
}

async function readRuleOrderPreviewPayload() {
  return await readJsonPayloadDbFirst("rule_order_preview", [RULE_ORDER_PREVIEW_PATH]);
}

async function readRuleExecutionResultsPayload() {
  return await readJsonPayloadDbFirst("rule_execution_results", [RULE_EXECUTION_RESULTS_PATH]);
}

async function readRulePaperStatePayload() {
  const execution = await readRuleExecutionResultsPayload();
  const paperPayload = await readJsonPayloadDbFirst("rule_account_paper_state", [RULE_PAPER_STATE_PATH]);
  const livePayload = await readJsonPayloadDbFirst("rule_account_live_state", [RULE_LIVE_STATE_PATH]);
  return String(execution.run_mode || "").toLowerCase() === "paper"
    ? paperPayload
    : ((livePayload && (livePayload.generated_at || Array.isArray(livePayload.positions))) ? livePayload : paperPayload);
}

async function readRuleBacktestPayload() {
  return await readJsonPayloadDbFirst("rule_strategy_backtest_report", [RULE_BACKTEST_PATH]);
}

function registerRuleApiRoutes(target) {
  target.get("/api/rule/summary", async (req, res) => {
    try {
      const payload = await readRuleSummaryPayload();
      if (!payload.as_of_date && !payload.counts?.total_candidates) {
        return res.status(404).json({ error: "rule artifacts not found" });
      }
      res.json(payload);
    } catch (e) {
      console.error("GET /api/rule/summary error", e);
      res.status(500).json({ error: "internal error" });
    }
  });

  target.get("/api/rule/signals/latest", async (req, res) => {
    try {
      const limit = Math.max(1, Math.min(300, Number(req.query.limit) || 100));
      const strength = String(req.query.strength || "").trim().toLowerCase();
      const payload = await readRuleSignalsPayload(strength, limit);
      if (!payload.as_of_date || !payload.items.length) {
        return res.status(404).json({ error: "rule signals not found" });
      }
      res.json(payload);
    } catch (e) {
      console.error("GET /api/rule/signals/latest error", e);
      res.status(500).json({ error: "internal error" });
    }
  });

  target.get("/api/rule/portfolio-plan", async (req, res) => {
    try {
      const payload = await readRulePortfolioPlanPayload();
      if (!payload.as_of_date && !Array.isArray(payload.items)) {
        return res.status(404).json({ error: "rule portfolio plan not found" });
      }
      res.json(payload);
    } catch (e) {
      console.error("GET /api/rule/portfolio-plan error", e);
      res.status(500).json({ error: "internal error" });
    }
  });

  target.get("/api/rule/order-preview", async (req, res) => {
    try {
      const payload = await readRuleOrderPreviewPayload();
      if (!payload.as_of_date && !Array.isArray(payload.items)) {
        return res.status(404).json({ error: "rule order preview not found" });
      }
      res.json(payload);
    } catch (e) {
      console.error("GET /api/rule/order-preview error", e);
      res.status(500).json({ error: "internal error" });
    }
  });

  target.get("/api/rule/paper-state", async (req, res) => {
    try {
      const payload = await readRulePaperStatePayload();
      if (!payload.as_of_date && !Array.isArray(payload.positions)) {
        return res.status(404).json({ error: "rule paper state not found" });
      }
      res.json(payload);
    } catch (e) {
      console.error("GET /api/rule/paper-state error", e);
      res.status(500).json({ error: "internal error" });
    }
  });

  target.get("/api/rule/backtest-summary", async (req, res) => {
    try {
      const payload = await readRuleBacktestPayload();
      if (!payload.latest_signal_date && !payload.generated_at) {
        return res.status(404).json({ error: "rule backtest summary not found" });
      }
      res.json(payload);
    } catch (e) {
      console.error("GET /api/rule/backtest-summary error", e);
      res.status(500).json({ error: "internal error" });
    }
  });

  target.get("/api/rule/execution-results", async (req, res) => {
    try {
      const payload = await readRuleExecutionResultsPayload();
      if (!payload.generated_at && !Array.isArray(payload.items)) {
        return res.status(404).json({ error: "rule execution results not found" });
      }
      res.json(payload);
    } catch (e) {
      console.error("GET /api/rule/execution-results error", e);
      res.status(500).json({ error: "internal error" });
    }
  });

  target.get("/api/rule/execution-history", async (req, res) => {
    try {
      const historyPath = path.join(OUTPUTS_DIR, "rule_execution_history.jsonl");
      if (!fs.existsSync(historyPath)) {
        return res.status(404).json({ error: "rule execution history not found" });
      }
      const limit = Math.max(1, Math.min(100, Number(req.query.limit) || 20));
      const lines = fs.readFileSync(historyPath, "utf-8")
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean);
      const items = lines.slice(-limit).map((line) => {
        try {
          return JSON.parse(line);
        } catch {
          return null;
        }
      }).filter(Boolean);
      res.json({ count: items.length, items });
    } catch (e) {
      console.error("GET /api/rule/execution-history error", e);
      res.status(500).json({ error: "internal error" });
    }
  });
}

async function getLatestDate(table) {
  const rows = await queryRows(`SELECT MAX(date) AS d FROM ${table}`);
  return rows[0]?.d || null;
}

async function getLatestPaperTradingRun() {
  const rows = await queryRows(
    `
    SELECT *
    FROM research.paper_trading_run
    ORDER BY asof_date DESC NULLS LAST, paper_run_id DESC
    LIMIT 1
    `
  );
  return rows[0] || null;
}

async function getPaperTradingRunById(paperRunId) {
  if (!Number.isFinite(paperRunId) || paperRunId <= 0) return null;
  const rows = await queryRows(
    `
    SELECT *
    FROM research.paper_trading_run
    WHERE paper_run_id = $1
    LIMIT 1
    `,
    [paperRunId]
  );
  return rows[0] || null;
}

// ---------------------
// Universe loader
// ---------------------
const UNIVERSE_CSV = path.join(DATA_DIR, "universe.csv");
let universeMap = new Map();
function getName(code) {
  const v = universeMap.get(code);
  return (v && v.name) || code;
}
function getMarket(code) {
  const v = universeMap.get(code);
  return (v && v.market) || null;
}
function getSector(code) {
  const v = universeMap.get(code);
  return (v && v.sector) || null;
}
function getShares(code) {
  const v = universeMap.get(code);
  return (v && v.shares) || null;
}
function getMktcap(code) {
  const v = universeMap.get(code);
  return (v && v.mktcap) || null;
}

function loadUniverse() {
  try {
    const rows = readCsv(UNIVERSE_CSV);
    const map = new Map();
    if (rows && rows.length) {
      rows.forEach((r) => {
        const code = (r.code || "").trim();
        if (!code) return;
        const shares = toNum(r.shares || r.shares_outstanding);
        const mktcap = toNum(r.mktcap || r.marketcap);
        map.set(code, {
          name: (r.name || code).trim(),
          market: (r.market || "").trim(),
          sector: (r.sector || "").trim(),
          shares: Number.isFinite(shares) ? shares : null,
          mktcap: Number.isFinite(mktcap) ? mktcap : null,
        });
      });
    }
    universeMap = map;
    console.log(`[universe] loaded ${universeMap.size} tickers`);
  } catch (e) {
    console.warn("Failed to load universe.csv", e.message);
  }
}

loadUniverse();
fs.watchFile(UNIVERSE_CSV, { interval: 5000 }, () => loadUniverse());

async function resolveTradeInstrumentInfo(code) {
  const normalizedCode = String(code || "").trim();
  if (!normalizedCode) {
    return { name: null, market: null, sector: null };
  }

  const universeInfo = universeMap.get(normalizedCode) || {};
  const hasUniverseInfo = Boolean(universeInfo.name || universeInfo.market || universeInfo.sector);
  if (hasUniverseInfo) {
    return {
      name: universeInfo.name || null,
      market: universeInfo.market || null,
      sector: universeInfo.sector || null,
    };
  }

  try {
    const latestByCode = await getRankingLatestByCode();
    const rank = latestByCode.get(normalizedCode) || {};
    return {
      name: (rank.name || "").trim() || null,
      market: (rank.market || "").trim() || null,
      sector: (rank.sector || "").trim() || null,
    };
  } catch (e) {
    console.warn("[resolveTradeInstrumentInfo] fallback failed:", e.message);
    return { name: null, market: null, sector: null };
  }
}

// ---------------------
// DB loaders
// ---------------------
async function loadMarketStatusLatest() {
  try {
    const rows = await queryRows(
      "SELECT * FROM market_status ORDER BY date DESC LIMIT 1"
    );
    if (rows.length) return rows[0];
  } catch (e) {
    console.warn("[market_status] DB load fail:", e.message);
  }

  try {
    const rows = readCsv(path.join(DATA_DIR, "market_status.csv")) || [];
    if (rows.length) {
      return rows
        .sort((a, b) => String(a.date || "").localeCompare(String(b.date || "")))
        .pop();
    }
  } catch (e) {
    console.warn("[market_status] CSV load fail:", e.message);
  }
  return null;
}

async function getPredictions() {
  try {
    return await queryRows("SELECT * FROM predictions");
  } catch (e) {
    console.warn("[predictions] DB load fail:", e.message);
    return readCsv(path.join(DATA_DIR, "predictions.csv")) || [];
  }
}

async function getFeatures(whereClause = "", params = []) {
  try {
    return await queryRows(
      `SELECT code, date, close, ma_5, ma_20, ma_60, rsi_14, vol_20, volume FROM features ${whereClause}`,
      params
    );
  } catch (e) {
    console.warn("[features] DB load fail:", e.message);
    return readCsv(path.join(DATA_DIR, "features.csv")) || [];
  }
}

async function getRanking(targetDate) {
  try {
    const date = targetDate || (await getLatestDate("daily_ranking"));
    if (!date) return null;
    const rows = await queryRows(
      "SELECT * FROM daily_ranking WHERE date = $1 ORDER BY COALESCE(live_score, final_score) DESC NULLS LAST",
      [date]
    );
    return { date, rows };
  } catch (e) {
    console.warn("[daily_ranking] DB load fail:", e.message);
    const csvRows = readCsv(path.join(DATA_DIR, "ranking_final.csv"));
    if (!csvRows || !csvRows.length) return null;
    const date = targetDate || csvRows.map((r) => r.date).filter(Boolean).sort().pop();
    const rows = csvRows.filter((r) => String(r.date || "") === String(date));
    return { date, rows };
  }
}

async function getRankingLatestByCode() {
  const res = await getRanking();
  const map = new Map();
  if (res && res.rows) {
    res.rows.forEach((r) => map.set(r.code, r));
  }
  return map;
}

function getScoreWeightProfile(regime) {
  const key = String(regime || "").trim().toLowerCase();
  if (key === "bull") {
    return { regime: "bull", ret: 0.35, prob: 0.26, tech: 0.29, qual: 0.06, valuation: 0.04, risk_penalty: 0.40 };
  }
  if (key === "neutral") {
    return { regime: "neutral", ret: 0.30, prob: 0.26, tech: 0.26, qual: 0.10, valuation: 0.08, risk_penalty: 0.65 };
  }
  return { regime: "defensive", ret: 0.27, prob: 0.24, tech: 0.15, qual: 0.19, valuation: 0.15, risk_penalty: 0.80 };
}

function clampScore(value) {
  const num = toNum(value);
  if (!Number.isFinite(num)) return null;
  return Math.max(0, Math.min(100, num));
}

function computeFinalScoreCheck(row) {
  const weights = getScoreWeightProfile(row?.regime);
  const retScore = toNum(row?.ret_score) ?? 0;
  const probScore = toNum(row?.prob_score) ?? 0;
  const techScore = toNum(row?.tech_score) ?? 0;
  const qualScore = toNum(row?.qual_score) ?? 0;
  const valuationScore = toNum(row?.valuation_score) ?? 0;
  const riskPenalty = toNum(row?.risk_penalty) ?? 0;
  const recomputed = clampScore(
    retScore * weights.ret +
    probScore * weights.prob +
    techScore * weights.tech +
    qualScore * weights.qual +
    valuationScore * weights.valuation -
    riskPenalty * weights.risk_penalty
  );
  const finalScore = toNum(row?.final_score);
  const liveScore = toNum(row?.live_score);
  const diff = Number.isFinite(finalScore) && Number.isFinite(recomputed) ? finalScore - recomputed : null;
  const fallbackFlags = [
    row?.ret_score_fallback_used ? "ret" : null,
    row?.prob_score_fallback_used ? "prob" : null,
    row?.qual_score_fallback_used ? "qual" : null,
    row?.tech_score_fallback_used ? "tech" : null,
    row?.safety_score_fallback_used ? "safety" : null,
    row?.liquidity_score_fallback_used ? "liquidity" : null,
  ].filter(Boolean);
  const missingFlags = [
    !Number.isFinite(toNum(row?.ret_score)) ? "ret" : null,
    !Number.isFinite(toNum(row?.prob_score)) ? "prob" : null,
    !Number.isFinite(toNum(row?.tech_score)) ? "tech" : null,
    !Number.isFinite(toNum(row?.qual_score)) ? "qual" : null,
  ].filter(Boolean);
  return {
    code: row?.code || null,
    name: row?.name || getName(row?.code || ""),
    date: row?.date || null,
    rank_final: toNum(row?.rank_final),
    live_rank: toNum(row?.live_rank),
    regime: row?.regime || weights.regime,
    weight_profile: row?.weight_profile || null,
    score_formula_version: row?.score_formula_version || null,
    final_score: finalScore,
    live_score: liveScore,
    live_score_source: getLiveScoreSource(row),
    recomputed_final_score: recomputed,
    final_diff: diff,
    abs_diff: Number.isFinite(diff) ? Math.abs(diff) : null,
    ret_score: toNum(row?.ret_score),
    prob_score: toNum(row?.prob_score),
    tech_score: toNum(row?.tech_score),
    qual_score: toNum(row?.qual_score),
    valuation_score: toNum(row?.valuation_score),
    safety_score: toNum(row?.safety_score),
    liquidity_score: toNum(row?.liquidity_score),
    risk_penalty: toNum(row?.risk_penalty),
    pred_return_60d: toNum(row?.pred_return_60d),
    prob_top20_60d: toNum(row?.prob_top20_60d),
    pred_mdd_60d: toNum(row?.pred_mdd_60d),
    fallback_flags: fallbackFlags,
    fallback_count: fallbackFlags.length,
    missing_flags: missingFlags,
    weights,
  };
}

function classifyPaperPositionReview({
  holdingAgeTradingDays,
  remainingHoldingDays,
  currentReturn,
  liveScore,
  confidenceScore,
  liveRank,
  currentPrice,
  latestPriceDate,
  currentActionCode,
}) {
  const reasons = [];
  const pushReason = (reason) => {
    if (reason && !reasons.includes(reason)) reasons.push(reason);
  };

  let status = "KEEP";
  let label = "계속보유";
  let priority = 1;

  const setReview = (reason) => {
    if (priority < 2) {
      status = "REVIEW";
      label = "점검필요";
      priority = 2;
    }
    pushReason(reason);
  };
  const setExitReview = (reason) => {
    status = "EXIT_REVIEW";
    label = "청산검토";
    priority = 3;
    pushReason(reason);
  };
  const setBlock = (reason) => {
    status = "BLOCK";
    label = "차단";
    priority = 4;
    pushReason(reason);
  };

  const inGracePeriod = Number.isFinite(holdingAgeTradingDays) && holdingAgeTradingDays <= 3;

  if (!Number.isFinite(currentPrice) || !latestPriceDate) {
    setReview("latest_price_missing");
  }

  if (currentActionCode === "EXIT_REVIEW_SOON") {
    setReview("planned_exit_near");
  }

  if (!inGracePeriod && Number.isFinite(currentReturn) && currentReturn <= -0.08) {
    setExitReview("loss_below_minus_8pct");
  }

  if (!inGracePeriod && Number.isFinite(liveScore) && liveScore < 45) {
    setExitReview("live_score_weak");
  }

  if (!inGracePeriod && Number.isFinite(confidenceScore) && confidenceScore < 55) {
    setBlock("confidence_blocked");
  } else if (!inGracePeriod && Number.isFinite(confidenceScore) && confidenceScore < 70) {
    setReview("confidence_low");
  }

  if (!inGracePeriod && Number.isFinite(liveRank) && liveRank > 10) {
    setReview("rank_outside_top10");
  }

  if (Number.isFinite(holdingAgeTradingDays) && holdingAgeTradingDays >= 20 && priority < 3) {
    setExitReview("hold_day_20_reached");
  }

  if (!reasons.length) {
    pushReason("holding_support_maintained");
  }

  let actionNote = "보유 근거가 유지되고 있습니다. 현재 포지션을 크게 흔들기보다 추적을 이어갑니다.";
  if (status === "BLOCK") {
    actionNote = "신뢰도 차단 구간입니다. 신규 대응보다 보수적 관리와 노출 축소를 우선 검토합니다.";
  } else if (status === "EXIT_REVIEW") {
    if (reasons.includes("hold_day_20_reached")) {
      actionNote = "20거래일 보유 기준에 도달했습니다. 고정 보유 정책상 청산 검토 우선순위가 높습니다.";
    } else if (reasons.includes("loss_below_minus_8pct")) {
      actionNote = "손실 관리 기준을 이탈했습니다. 계속 보유보다 청산 검토가 우선입니다.";
    } else {
      actionNote = "포지션 약화 신호가 강합니다. 청산 또는 교체 검토가 필요합니다.";
    }
  } else if (status === "REVIEW") {
    if (reasons.includes("planned_exit_near")) {
      actionNote = "예정 청산일이 임박했습니다. 신규 행동보다 종료 준비와 대체 후보 점검이 우선입니다.";
    } else if (reasons.includes("confidence_low")) {
      actionNote = "신뢰도 지지가 약해졌습니다. 추가 보유 근거를 다시 확인해야 합니다.";
    } else if (reasons.includes("rank_outside_top10")) {
      actionNote = "현재 우선순위가 약화되었습니다. 대기 후보와의 상대 우위를 다시 봐야 합니다.";
    } else {
      actionNote = "관찰 구간입니다. 즉시 행동보다 보유 근거 재확인이 우선입니다.";
    }
  }

  return {
    system_review_status: status,
    system_review_label: label,
    system_review_priority: priority,
    system_review_reasons: reasons,
    system_action_note: actionNote,
  };
}

async function buildScoreCheckPayload(targetDate) {
  const ranking = await getRanking(targetDate);
  if (!ranking || !Array.isArray(ranking.rows) || !ranking.rows.length) {
    return null;
  }

  const checkedRows = ranking.rows.map(computeFinalScoreCheck);
  const top20 = checkedRows
    .slice()
    .sort((a, b) => {
      const ar = Number.isFinite(a.rank_final) ? a.rank_final : Number.MAX_SAFE_INTEGER;
      const br = Number.isFinite(b.rank_final) ? b.rank_final : Number.MAX_SAFE_INTEGER;
      if (ar !== br) return ar - br;
      const af = Number.isFinite(a.final_score) ? a.final_score : -Infinity;
      const bf = Number.isFinite(b.final_score) ? b.final_score : -Infinity;
      return bf - af;
    })
    .slice(0, 20);

  const absDiffRows = checkedRows
    .filter((row) => Number.isFinite(row.abs_diff))
    .sort((a, b) => b.abs_diff - a.abs_diff);
  const flaggedRows = checkedRows
    .filter((row) => row.fallback_count > 0 || row.missing_flags.length > 0 || (Number.isFinite(row.abs_diff) && row.abs_diff > 0.05))
    .sort((a, b) => {
      const ad = Number.isFinite(a.abs_diff) ? a.abs_diff : -1;
      const bd = Number.isFinite(b.abs_diff) ? b.abs_diff : -1;
      if (bd !== ad) return bd - ad;
      return (b.fallback_count || 0) - (a.fallback_count || 0);
    });

  const regimeCounts = checkedRows.reduce((acc, row) => {
    const key = String(row.regime || "unknown");
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});
  const formulaCounts = checkedRows.reduce((acc, row) => {
    const key = String(row.score_formula_version || "unknown");
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});

  const diffValues = absDiffRows.map((row) => row.abs_diff);
  const meanAbsDiff = diffValues.length ? diffValues.reduce((sum, value) => sum + value, 0) / diffValues.length : null;
  const maxAbsDiff = diffValues.length ? diffValues[0] : null;

  return {
    date: ranking.date,
    summary: {
      row_count: checkedRows.length,
      top20_count: top20.length,
      max_abs_diff: maxAbsDiff,
      mean_abs_diff: meanAbsDiff,
      exact_match_count: checkedRows.filter((row) => Number.isFinite(row.abs_diff) && row.abs_diff <= 0.0001).length,
      diff_gt_001_count: checkedRows.filter((row) => Number.isFinite(row.abs_diff) && row.abs_diff > 0.01).length,
      diff_gt_005_count: checkedRows.filter((row) => Number.isFinite(row.abs_diff) && row.abs_diff > 0.05).length,
      rows_with_live_override: checkedRows.filter((row) => String(row.live_score_source || "final_score") !== "final_score").length,
      rows_with_any_fallback: checkedRows.filter((row) => row.fallback_count > 0).length,
      rows_with_missing_core_scores: checkedRows.filter((row) => row.missing_flags.length > 0).length,
      top20_fallback_rows: top20.filter((row) => row.fallback_count > 0).length,
    },
    regime_counts: regimeCounts,
    formula_counts: formulaCounts,
    top20,
    biggest_diffs: absDiffRows.slice(0, 30),
    flagged_rows: flaggedRows.slice(0, 50),
  };
}

async function getFeatureStatsForCodes(codes) {
  if (!codes || !codes.length) {
    return {
      latestClose: new Map(),
      ret3m: new Map(),
      ret5d: new Map(),
      ret10d: new Map(),
      mom20: new Map(),
      rsi14: new Map(),
    };
  }
  const latestClose = new Map();
  const ret3m = new Map();
  const ret5d = new Map();
  const ret10d = new Map();
  const mom20 = new Map();
  const rsi14 = new Map();

  try {
    const rows = await queryRows(
      `
      WITH ranked AS (
        SELECT
          code,
          ret_5d,
          ret_10d,
          mom_20,
          rsi_14,
          close,
          ROW_NUMBER() OVER (PARTITION BY code ORDER BY date DESC) AS rn
        FROM features
        WHERE code = ANY($1)
      )
      SELECT
        code,
        MAX(CASE WHEN rn = 1 THEN close END) AS latest_close,
        MAX(CASE WHEN rn = 60 THEN close END) AS close_60,
        MAX(CASE WHEN rn = 1 THEN ret_5d END) AS ret_5d,
        MAX(CASE WHEN rn = 1 THEN ret_10d END) AS ret_10d,
        MAX(CASE WHEN rn = 1 THEN mom_20 END) AS mom_20,
        MAX(CASE WHEN rn = 1 THEN rsi_14 END) AS rsi_14
      FROM ranked
      WHERE rn IN (1, 60)
      GROUP BY code
      `,
      [codes]
    );

    for (const row of rows) {
      const code = row.code;
      const lastClose = toNum(row.latest_close);
      const prevClose = toNum(row.close_60);
      if (code) latestClose.set(code, lastClose);
      if (code) ret5d.set(code, toNum(row.ret_5d));
      if (code) ret10d.set(code, toNum(row.ret_10d));
      if (code) mom20.set(code, toNum(row.mom_20));
      if (code) rsi14.set(code, toNum(row.rsi_14));
      if (Number.isFinite(prevClose) && Number.isFinite(lastClose) && prevClose !== 0) {
        ret3m.set(code, lastClose / prevClose - 1);
      } else if (code) {
        ret3m.set(code, null);
      }
    }
    return { latestClose, ret3m, ret5d, ret10d, mom20, rsi14 };
  } catch (e) {
    console.warn("[feature-stats] DB load fail:", e.message);
  }

  const feats = readCsv(path.join(DATA_DIR, "features.csv")) || [];
  const codeSet = new Set(codes);
  let current = null;
  let buffer = [];

  const flush = () => {
    if (!buffer.length || !current) return;
    const last = buffer[buffer.length - 1];
    latestClose.set(current, toNum(last.close));
    ret5d.set(current, toNum(last.ret_5d));
    ret10d.set(current, toNum(last.ret_10d));
    mom20.set(current, toNum(last.mom_20));
    rsi14.set(current, toNum(last.rsi_14));
    if (buffer.length >= 60) {
      const prev = buffer[buffer.length - 60];
      const prevClose = toNum(prev.close);
      const lastClose = toNum(last.close);
      if (Number.isFinite(prevClose) && Number.isFinite(lastClose) && prevClose !== 0) {
        ret3m.set(current, lastClose / prevClose - 1);
      } else {
        ret3m.set(current, null);
      }
    } else {
      ret3m.set(current, null);
    }
    buffer = [];
  };

  for (const row of feats) {
    const code = row.code;
    if (!codeSet.has(code)) continue;
    if (current !== code) {
      flush();
      current = code;
    }
    buffer.push(row);
  }
  flush();
  return { latestClose, ret3m, ret5d, ret10d, mom20, rsi14 };
}

// ---------------------
// Trades
// ---------------------
let ensureTradesTablePromise = null;
let backfillTradeMetadataPromise = null;

async function backfillTradeMetadata() {
  if (!backfillTradeMetadataPromise) {
    backfillTradeMetadataPromise = (async () => {
      const { rows } = await pool.query(
        `
        SELECT trade_id, code, name, market, sector
        FROM trades
        WHERE NULLIF(name, '') IS NULL
           OR NULLIF(market, '') IS NULL
           OR NULLIF(sector, '') IS NULL
        ORDER BY trade_id ASC
        `
      );

      for (const row of rows) {
        const info = await resolveTradeInstrumentInfo(row.code);
        if (!info.name && !info.market && !info.sector) continue;
        await pool.query(
          `
          UPDATE trades
          SET
            name = COALESCE(NULLIF(name, ''), $2),
            market = COALESCE(NULLIF(market, ''), $3),
            sector = COALESCE(NULLIF(sector, ''), $4)
          WHERE trade_id = $1
          `,
          [row.trade_id, info.name, info.market, info.sector]
        );
      }
    })().catch((error) => {
      backfillTradeMetadataPromise = null;
      throw error;
    });
  }

  await backfillTradeMetadataPromise;
}

async function ensureTradesTable() {
  if (!ensureTradesTablePromise) {
    ensureTradesTablePromise = (async () => {
      const [{ rows: tradeRows }, { rows: auditRows }] = await Promise.all([
        pool.query("SELECT to_regclass('public.trades') AS regclass"),
        pool.query("SELECT to_regclass('public.trade_audit_log') AS regclass"),
      ]);

      const tradesExists = Boolean(tradeRows?.[0]?.regclass);
      const auditExists = Boolean(auditRows?.[0]?.regclass);

      if (!tradesExists || !auditExists) {
        await pool.query(
          `
          CREATE TABLE IF NOT EXISTS trades (
            trade_id    BIGSERIAL PRIMARY KEY,
            date        DATE NOT NULL,
            side        TEXT NOT NULL,
            code        TEXT NOT NULL,
            name        TEXT,
            market      TEXT,
            sector      TEXT,
            qty         NUMERIC,
            price       NUMERIC,
            amount      NUMERIC,
            fee         NUMERIC,
            memo        TEXT,
            created_at  TIMESTAMPTZ DEFAULT now()
          );
          CREATE INDEX IF NOT EXISTS idx_trades_code_date ON trades(code, date);
          CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(date);

          CREATE TABLE IF NOT EXISTS trade_audit_log (
            audit_id      BIGSERIAL PRIMARY KEY,
            trade_id      BIGINT,
            action        TEXT NOT NULL,
            trade_snapshot JSONB NOT NULL,
            actor         TEXT,
            reason        TEXT,
            created_at    TIMESTAMPTZ DEFAULT now()
          );
          CREATE INDEX IF NOT EXISTS idx_trade_audit_trade_id_created ON trade_audit_log(trade_id, created_at DESC);
          `
        );
      }

      await pool.query(
        `
        ALTER TABLE trades ADD COLUMN IF NOT EXISTS name TEXT;
        ALTER TABLE trades ADD COLUMN IF NOT EXISTS market TEXT;
        ALTER TABLE trades ADD COLUMN IF NOT EXISTS sector TEXT;
        ALTER TABLE trades ADD COLUMN IF NOT EXISTS qty NUMERIC;
        ALTER TABLE trades ADD COLUMN IF NOT EXISTS price NUMERIC;
        ALTER TABLE trades ADD COLUMN IF NOT EXISTS amount NUMERIC;
        ALTER TABLE trades ADD COLUMN IF NOT EXISTS fee NUMERIC;
        ALTER TABLE trades ADD COLUMN IF NOT EXISTS memo TEXT;
        ALTER TABLE trades ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT now();

        ALTER TABLE trade_audit_log ADD COLUMN IF NOT EXISTS trade_id BIGINT;
        ALTER TABLE trade_audit_log ADD COLUMN IF NOT EXISTS action TEXT;
        ALTER TABLE trade_audit_log ADD COLUMN IF NOT EXISTS trade_snapshot JSONB;
        ALTER TABLE trade_audit_log ADD COLUMN IF NOT EXISTS actor TEXT;
        ALTER TABLE trade_audit_log ADD COLUMN IF NOT EXISTS reason TEXT;
        ALTER TABLE trade_audit_log ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT now();

        CREATE INDEX IF NOT EXISTS idx_trades_code_date ON trades(code, date);
        CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(date);
        CREATE INDEX IF NOT EXISTS idx_trade_audit_trade_id_created ON trade_audit_log(trade_id, created_at DESC);
        `
      );

      await pool.query(
        `
        UPDATE trade_audit_log
        SET trade_snapshot = '{}'::jsonb
        WHERE trade_snapshot IS NULL
        `
      );

      await pool.query(
        `
        UPDATE trade_audit_log
        SET action = 'UNKNOWN'
        WHERE action IS NULL
        `
      );

      await pool.query(
        `
        ALTER TABLE trade_audit_log
        ALTER COLUMN action SET NOT NULL,
        ALTER COLUMN trade_snapshot SET NOT NULL
        `
      );

      await backfillTradeMetadata();
    })().catch((error) => {
      ensureTradesTablePromise = null;
      throw error;
    });
  }

  await ensureTradesTablePromise;
}

async function listTrades() {
  await ensureTradesTable();
  const rows = await queryRows(
    "SELECT * FROM trades ORDER BY date ASC, trade_id ASC"
  );
  return rows.map((r) => ({
    trade_id: r.trade_id,
    date: r.date ? String(r.date) : null,
    side: (r.side || "").toUpperCase(),
    code: r.code,
    name: r.name || null,
    market: r.market || null,
    sector: r.sector || null,
    qty: toNum(r.qty) || 0,
    price: toNum(r.price) || 0,
    amount: toNum(r.amount) || 0,
    fee: toNum(r.fee) || 0,
    memo: r.memo || "",
    created_at: r.created_at || null,
  }));
}

async function insertTrade(payload) {
  await ensureTradesTable();
  const {
    date,
    side,
    code,
    name = null,
    market = null,
    sector = null,
    qty,
    price,
    amount = null,
    fee = null,
    memo = null,
  } = payload;

  const { rows } = await pool.query(
    `
    INSERT INTO trades (date, side, code, name, market, sector, qty, price, amount, fee, memo)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
    RETURNING trade_id;
    `,
    [date, side, code, name, market, sector, qty, price, amount, fee, memo]
  );
  return rows[0];
}

function buildTradeAuditSnapshot(trade) {
  return {
    trade_id: trade.trade_id ?? null,
    date: trade.date ?? null,
    side: trade.side ?? null,
    code: trade.code ?? null,
    name: trade.name ?? null,
    market: trade.market ?? null,
    sector: trade.sector ?? null,
    qty: trade.qty ?? null,
    price: trade.price ?? null,
    amount: trade.amount ?? null,
    fee: trade.fee ?? null,
    memo: trade.memo ?? null,
    created_at: trade.created_at ?? null,
  };
}

async function insertTradeAuditLog({ tradeId = null, action, tradeSnapshot, actor = null, reason = null }) {
  await ensureTradesTable();
  await pool.query(
    `
    INSERT INTO trade_audit_log (trade_id, action, trade_snapshot, actor, reason)
    VALUES ($1, $2, $3::jsonb, $4, $5)
    `,
    [
      tradeId,
      action,
      JSON.stringify(tradeSnapshot || {}),
      actor,
      reason,
    ]
  );
}

async function getTradeById(tradeId) {
  await ensureTradesTable();
  const rows = await queryRows(
    "SELECT * FROM trades WHERE trade_id = $1 LIMIT 1",
    [tradeId]
  );
  return rows.length ? rows[0] : null;
}

async function deleteTradeById(tradeId) {
  await ensureTradesTable();
  const { rowCount } = await pool.query(
    "DELETE FROM trades WHERE trade_id = $1",
    [tradeId]
  );
  return rowCount > 0;
}

// ---------------------
// Holdings helper
// ---------------------
function computePositions(trades) {
  const stateByCode = new Map();
  const sorted = trades.slice().sort(compareTradesChronologically);

  for (const t of sorted) {
    const code = (t.code || "").trim();
    if (!code) continue;
    const side = (t.side || "").toUpperCase().trim();
    const qRaw = toNum(t.qty);
    const p = toNum(t.price);
    if (!Number.isFinite(qRaw) || !Number.isFinite(p) || p <= 0) continue;
    const q = Math.abs(qRaw);
    if (q <= 0) continue;

    if (!stateByCode.has(code)) {
      stateByCode.set(code, { qty: 0, avgPrice: 0, realizedAcc: 0, totalBuy: 0 });
    }
    const st = stateByCode.get(code);

    if (side === "BUY") {
      const newQty = st.qty + q;
      st.avgPrice = (st.avgPrice * st.qty + p * q) / newQty;
      st.qty = newQty;
      st.totalBuy += p * q;
    } else if (side === "SELL") {
      const sellQty = Math.min(q, st.qty);
      st.realizedAcc += (p - st.avgPrice) * sellQty;
      st.qty -= sellQty;
      if (st.qty <= 0) {
        st.qty = 0;
        st.avgPrice = 0;
      }
    }
  }
  return stateByCode;
}

function buildHoldings(trades, latestRankByCode) {
  const stateByCode = computePositions(trades);
  const holdings = [];
  const tradeMetaByCode = new Map();
  const latestTradeByCode = new Map();
  const classifyHoldingReview = ({
    holdingDays,
    unrealizedPct,
    finalScore,
    scoreDelta,
    retScore,
    probScore,
    riskPenalty,
    confidenceScore,
    currentPrice,
    latestRankDate,
  }) => {
    const reasons = [];
    const pushUniqueReason = (code) => {
      if (!code || reasons.includes(code)) return;
      reasons.push(code);
    };
    const inGracePeriod = Number.isFinite(holdingDays) && holdingDays <= 3;
    const setReview = (reason) => {
      if (priority < 2) {
        status = "REVIEW";
        label = "점검필요";
        priority = 2;
      }
      pushUniqueReason(reason);
    };
    let status = "KEEP";
    let label = "계속보유";
    let priority = 1;

    if (!Number.isFinite(currentPrice) || !latestRankDate) {
      setReview("latest_price_missing");
    }

    if (Number.isFinite(unrealizedPct) && unrealizedPct <= -8) {
      status = "EXIT_REVIEW";
      label = "매도검토";
      priority = 3;
      pushUniqueReason("loss_below_minus_8pct");
    }

    if (!inGracePeriod && Number.isFinite(finalScore) && finalScore < 45) {
      status = "EXIT_REVIEW";
      label = "매도검토";
      priority = 3;
      pushUniqueReason("final_score_weak");
    }

    if (!inGracePeriod && Number.isFinite(scoreDelta) && scoreDelta <= -5) {
      setReview("score_delta_down");
    }

    if (!inGracePeriod && Number.isFinite(retScore) && retScore < 55) {
      setReview("ret_score_weak");
    }

    if (!inGracePeriod && Number.isFinite(probScore) && probScore < 55) {
      setReview("prob_score_weak");
    }

    if (!inGracePeriod && Number.isFinite(confidenceScore) && confidenceScore < 70) {
      setReview("confidence_low");
    }

    if (!inGracePeriod && Number.isFinite(riskPenalty) && riskPenalty >= 3) {
      setReview("risk_penalty_high");
    }

    if (Number.isFinite(holdingDays) && holdingDays >= 20 && priority < 3) {
      setReview("hold_day_20_reached");
    }

    if (Number.isFinite(unrealizedPct) && unrealizedPct >= 15 && priority < 3) {
      setReview("profit_above_15pct");
    }

    if (inGracePeriod && priority < 2) {
      pushUniqueReason("new_position");
    }

    if (!reasons.length) {
      pushUniqueReason("holding_support_maintained");
    }

    let sellPriorityScore = 22;
    if (status === "REVIEW") sellPriorityScore = 58;
    if (status === "EXIT_REVIEW") sellPriorityScore = 82;
    if (Number.isFinite(holdingDays) && holdingDays >= 20) sellPriorityScore += 8;
    if (Number.isFinite(unrealizedPct) && unrealizedPct >= 15) sellPriorityScore += 6;
    if (Number.isFinite(unrealizedPct) && unrealizedPct <= -8) sellPriorityScore += 10;
    if (!inGracePeriod && Number.isFinite(finalScore) && finalScore < 45) sellPriorityScore += 10;
    if (!Number.isFinite(currentPrice) || !latestRankDate) sellPriorityScore += 6;
    if (!inGracePeriod && Number.isFinite(confidenceScore) && confidenceScore < 70) sellPriorityScore += 5;
    if (!inGracePeriod && Number.isFinite(retScore) && retScore < 55) sellPriorityScore += 4;
    if (!inGracePeriod && Number.isFinite(probScore) && probScore < 55) sellPriorityScore += 4;
    if (!inGracePeriod && Number.isFinite(riskPenalty) && riskPenalty >= 3) sellPriorityScore += 4;
    if (!inGracePeriod && Number.isFinite(scoreDelta) && scoreDelta <= -5) sellPriorityScore += 4;
    sellPriorityScore = Math.max(0, Math.min(99, sellPriorityScore));

    const hasReason = (code) => reasons.includes(code);
    let actionNote = "보유 근거가 유지되고 있습니다. 비중 확대보다 현재 보유 논리가 살아있는지 확인하는 것이 우선입니다.";
    if (status === "EXIT_REVIEW") {
      if (hasReason("loss_below_minus_8pct")) {
        actionNote = "손실 관리 기준을 이탈했습니다. 추가 보유 전에 매수 근거를 다시 점검해야 합니다.";
      } else if (hasReason("final_score_weak")) {
        actionNote = "종합 점수가 크게 약화되었습니다. 계속 보유보다 매도 검토 우선순위가 높습니다.";
      } else {
        actionNote = "리스크 신호가 겹쳤습니다. 이 포지션에 자금을 계속 둘지 다시 판단해야 합니다.";
      }
    } else if (status === "REVIEW") {
      if (hasReason("profit_above_15pct")) {
        actionNote = "수익 구간 점검 대상입니다. 추가 매수보다 이익 보호 여부를 먼저 검토합니다.";
      } else if (hasReason("hold_day_20_reached")) {
        actionNote = "보유 20일 점검 시점입니다. ret/prob/confidence 지지가 유지되는지 다시 확인합니다.";
      } else if (hasReason("confidence_low") || hasReason("ret_score_weak") || hasReason("prob_score_weak")) {
        actionNote = "지지 점수가 약해졌습니다. 다른 행동보다 보유 근거 재확인이 우선입니다.";
      } else if (hasReason("risk_penalty_high")) {
        actionNote = "리스크 패널티가 높습니다. 비중 확대 전에 변동성 확대 여부를 먼저 봐야 합니다.";
      } else {
        actionNote = "관찰 구간입니다. 새 행동에 앞서 현재 보유 근거를 다시 확인합니다.";
      }
    } else if (hasReason("new_position")) {
      actionNote = "신규 포지션 유예 구간입니다. 초기 진입 근거가 안정될 때까지 추가 매수는 보수적으로 봅니다.";
    }

    return {
      system_review_status: status,
      system_review_label: label,
      system_review_priority: priority,
      sell_priority_score: sellPriorityScore,
      system_review_reasons: reasons,
      system_action_note: actionNote,
    };
  };
  const sortedTrades = [...trades].sort((a, b) => {
    return compareTradesChronologically(a, b);
  });

  for (const trade of sortedTrades) {
    const code = String(trade.code || "").trim();
    if (!code) continue;

    const side = String(trade.side || "").toUpperCase();
    const tradeDate = toIsoDate(trade.date || "");
    if (!tradeMetaByCode.has(code)) {
      tradeMetaByCode.set(code, {
        firstBuyDate: null,
        lastBuyDate: null,
        lastTradeDate: null,
        buyCount: 0,
      });
    }
    const meta = tradeMetaByCode.get(code);

    if (tradeDate) {
      if (!meta.lastTradeDate || tradeDate > meta.lastTradeDate) {
        meta.lastTradeDate = tradeDate;
        latestTradeByCode.set(code, trade);
      }
    }

    if (side === "BUY") {
      meta.buyCount += 1;
      if (tradeDate) {
        if (!meta.firstBuyDate || tradeDate < meta.firstBuyDate) {
          meta.firstBuyDate = tradeDate;
        }
        if (!meta.lastBuyDate || tradeDate > meta.lastBuyDate) {
          meta.lastBuyDate = tradeDate;
        }
      }
    }
  }

  for (const [code, st] of stateByCode.entries()) {
    const qty = st.qty;
    if (qty <= 0) continue;

    const rankRow = latestRankByCode.get(code) || {};
    const tradeMeta = tradeMetaByCode.get(code) || {};
    const latestTrade = latestTradeByCode.get(code) || {};
    const name =
      rankRow.name ||
      latestTrade.name ||
      getName(code) ||
      code;
    const market =
      rankRow.market ||
      latestTrade.market ||
      getMarket(code) ||
      null;
    const sector =
      rankRow.sector ||
      latestTrade.sector ||
      getSector(code) ||
      null;
    const currentPrice = toNum(rankRow.close);
    const currentValue = Number.isFinite(currentPrice) ? currentPrice * qty : null;
    const latestRankDate = toIsoDate(rankRow.date || "") || tradeMeta.lastTradeDate || null;

    const avgBuyPrice = st.avgPrice > 0 ? st.avgPrice : null;
    const costBasis = avgBuyPrice && qty ? avgBuyPrice * qty : null;
    const unrealized =
      currentValue != null && costBasis != null ? currentValue - costBasis : null;
    const unrealizedPct =
      currentValue != null && costBasis
        ? (currentValue / costBasis - 1) * 100
        : null;

    // 20% 목표가 및 진행률
    const targetPrice = Number.isFinite(avgBuyPrice) ? avgBuyPrice * 1.2 : null;
    let progressToTarget = null;
    let targetHit = false;
    if (
      Number.isFinite(currentPrice) &&
      Number.isFinite(avgBuyPrice) &&
      Number.isFinite(targetPrice) &&
      targetPrice > avgBuyPrice
    ) {
      // 진행률은 매입가를 0%로, 목표가를 100%로 보는 방식
      const span = targetPrice - avgBuyPrice;
      progressToTarget = ((currentPrice - avgBuyPrice) / span) * 100;
      progressToTarget = Math.min(100, Math.max(0, progressToTarget));
      targetHit = currentPrice >= targetPrice;
    }

    const realizedPct = st.totalBuy > 0 ? (st.realizedAcc / st.totalBuy) * 100 : null;
    let holdingDays = null;
    if (tradeMeta.firstBuyDate && latestRankDate) {
      const start = new Date(`${tradeMeta.firstBuyDate}T00:00:00Z`);
      const end = new Date(`${latestRankDate}T00:00:00Z`);
      if (Number.isFinite(start.getTime()) && Number.isFinite(end.getTime()) && end >= start) {
        holdingDays = Math.floor((end - start) / 86400000) + 1;
      }
    }
    const finalScore = getLiveScore(rankRow);
    const scoreDelta =
      toNum(rankRow.score_delta) ??
      toNum(rankRow.final_score_delta) ??
      null;
    const retScore = toNum(rankRow.ret_score);
    const probScore = toNum(rankRow.prob_score);
    const riskPenalty = toNum(rankRow.risk_penalty);
    const confidenceScore =
      toNum(rankRow.confidence_score) ??
      toNum(rankRow.raw_confidence_v2) ??
      toNum(rankRow.confidence_score_operational);
    const reviewMeta = classifyHoldingReview({
      holdingDays,
      unrealizedPct,
      finalScore,
      scoreDelta,
      retScore,
      probScore,
      riskPenalty,
      confidenceScore,
      currentPrice,
      latestRankDate,
    });

    holdings.push({
      code,
      name,
      market,
      sector,
      current_qty: qty,
      avg_buy_price: avgBuyPrice,
      current_price: currentPrice,
      current_value: currentValue,
      cost_basis: costBasis,
      realized_pnl: st.realizedAcc,
      realized_pnl_pct: realizedPct,
      unrealized_pnl: unrealized,
      unrealized_pnl_pct: unrealizedPct,
      final_score: finalScore,
      score_delta: scoreDelta,
      ret_score: retScore,
      prob_score: probScore,
      risk_penalty: riskPenalty,
      confidence_score: confidenceScore,
      target_price: targetPrice,
      progress_to_target: progressToTarget,
      target_hit: targetHit,
      first_buy_date: tradeMeta.firstBuyDate || null,
      last_buy_date: tradeMeta.lastBuyDate || null,
      last_trade_date: tradeMeta.lastTradeDate || null,
      holding_days: holdingDays,
      buy_count: tradeMeta.buyCount || 0,
      latest_rank_date: latestRankDate,
      live_score_source: getLiveScoreSource(rankRow),
      ...reviewMeta,
    });
  }

  holdings.sort((a, b) => {
    const spa = Number(a.sell_priority_score);
    const spb = Number(b.sell_priority_score);
    const hasSpa = Number.isFinite(spa);
    const hasSpb = Number.isFinite(spb);
    if (hasSpa || hasSpb) {
      if (!hasSpa) return 1;
      if (!hasSpb) return -1;
      if (spb !== spa) return spb - spa;
    }

    const pa = Number(a.system_review_priority) || 0;
    const pb = Number(b.system_review_priority) || 0;
    if (pb !== pa) return pb - pa;

    const la = toNum(a.unrealized_pnl_pct);
    const lb = toNum(b.unrealized_pnl_pct);
    const hasLa = Number.isFinite(la);
    const hasLb = Number.isFinite(lb);
    if (hasLa || hasLb) {
      if (!hasLa) return 1;
      if (!hasLb) return -1;
      if (la !== lb) return la - lb;
    }

    const ua = Math.abs(toNum(a.unrealized_pnl_pct) || 0);
    const ub = Math.abs(toNum(b.unrealized_pnl_pct) || 0);
    if (ub !== ua) return ub - ua;

    return a.code.localeCompare(b.code);
  });

  return holdings;
}

// ---------------------
// Express setup
// ---------------------
app.use(cors());
app.use(express.json());
app.use((req, res, next) => {
  void recordPageView(req, res);
  next();
});
const PUBLIC_DIR = path.join(__dirname, "public");

function sendPublicPage(res, fileName) {
  const filePath = path.join(PUBLIC_DIR, fileName);
  try {
    const html = applyPublicPageMeta(fs.readFileSync(filePath, "utf-8"), res.req?.path || "/");
    const withHead = injectHeadSnippet(html, renderAnalyticsHeadSnippet());
    res.set("Cache-Control", "no-cache");
    return res.type("html").send(injectBodySnippet(withHead, renderOpsUnifiedNavSnippet(fileName)));
  } catch (e) {
    console.error("sendPublicPage error", fileName, e);
    return res.sendFile(filePath);
  }
}

app.get("/operator-login", (req, res) => sendPublicPage(res, "operator-login.html"));
app.get("/ops-readiness.html", operatorAccess.pageGuard, (req, res) => sendPublicPage(res, "ops-readiness.html"));
app.get("/live-auto-trading.html", (req, res) => sendPublicPage(res, "live-auto-trading.html"));
app.get("/manual-trading.html", (req, res) => sendPublicPage(res, "manual-trading.html"));
app.get("/holdings.html", (req, res) => sendPublicPage(res, "holdings.html"));
app.get("/holdingsDetail.html", (req, res) => sendPublicPage(res, "holdingsDetail.html"));
app.get("/detail.html", (req, res) => sendPublicPage(res, "detail.html"));
app.get("/ranking.html", (req, res) => sendPublicPage(res, "ranking.html"));
app.get("/meaningfulness.html", (req, res) => sendPublicPage(res, "meaningfulness.html"));
app.get("/paper-trading.html", (req, res) => sendPublicPage(res, "paper-trading.html"));
app.get("/rule-auto-trading.html", (req, res) => sendPublicPage(res, "rule-auto-trading.html"));
app.get("/trade-history.html", (req, res) => sendPublicPage(res, "trade-history.html"));
app.get("/score-check", operatorAccess.pageGuard, (req, res) => sendPublicPage(res, "score-check.html"));

app.get("/", (req, res) => sendPublicPage(res, "landing.html"));
app.get("/app", (req, res) => sendPublicPage(res, "index.html"));
app.get("/about", (req, res) => sendPublicPage(res, "about.html"));
app.get("/methodology", (req, res) => sendPublicPage(res, "methodology.html"));
app.get("/glossary", (req, res) => sendPublicPage(res, "glossary.html"));
app.get("/operator-note", (req, res) => sendPublicPage(res, "operator-note.html"));
app.get("/contact", (req, res) => sendPublicPage(res, "contact.html"));
app.get("/privacy", (req, res) => sendPublicPage(res, "privacy.html"));
app.get("/terms", (req, res) => sendPublicPage(res, "terms.html"));
app.get("/disclaimer", (req, res) => sendPublicPage(res, "disclaimer.html"));
app.get("/reports", (req, res) => sendPublicPage(res, "content-list.html"));
app.get("/blog", (req, res) => sendPublicPage(res, "content-list.html"));
app.get("/robots.txt", (req, res) => {
  res.type("text/plain").send(`User-agent: *\nAllow: /\n\nSitemap: ${buildAbsoluteUrl("/sitemap.xml")}\n`);
});
app.get("/sitemap.xml", (req, res) => {
  const items = readSiteLibrary();
  const urls = [
    "/",
    "/about",
    "/methodology",
    "/glossary",
    "/operator-note",
    "/contact",
    "/privacy",
    "/terms",
    "/disclaimer",
    "/reports",
    "/blog",
    ...items.map((item) => `/${item.section === "report" ? "reports" : "blog"}/${item.slug}`),
  ];
  const xml = `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${urls.map((url) => `  <url>\n    <loc>${escapeHtml(buildAbsoluteUrl(url))}</loc>\n  </url>`).join("\n")}\n</urlset>\n`;
  res.type("application/xml").send(xml);
});
app.get("/api/site-library", (req, res) => {
  res.json({ items: readSiteLibrary() });
});
app.get("/api/homepage-content", (req, res) => {
  res.json(readHomepageContent());
});
app.get("/api/operator-auth/status", operatorAccess.status);
app.post("/api/operator-auth/login", operatorAccess.login);
app.post("/api/operator-auth/logout", operatorAccess.logout);
app.get("/reports/:slug", (req, res) => {
  const item = readSiteLibrary().find((entry) => entry.section === "report" && entry.slug === req.params.slug);
  if (!item) return res.status(404).sendFile(path.join(PUBLIC_DIR, "content-detail.html"));
  return res.type("html").send(renderArticlePage(item, "reports"));
});
app.get("/blog/:slug", (req, res) => {
  const item = readSiteLibrary().find((entry) => entry.section === "blog" && entry.slug === req.params.slug);
  if (!item) return res.status(404).sendFile(path.join(PUBLIC_DIR, "content-detail.html"));
  return res.type("html").send(renderArticlePage(item, "blog"));
});
registerRuleApiRoutes(app);
app.use(express.static(PUBLIC_DIR, {
  etag: true,
  lastModified: true,
  setHeaders: (res, filePath) => {
    if (/\.(html|xml|txt)$/i.test(filePath)) {
      res.setHeader("Cache-Control", "no-cache");
      return;
    }
    if (/\.(css|js|mjs|png|jpg|jpeg|gif|svg|webp|ico|woff|woff2)$/i.test(filePath)) {
      res.setHeader("Cache-Control", "public, max-age=604800, immutable");
    }
  },
}));

// Admin page/API (protected by adminAuth; allow if ADMIN_TOKEN unset)
try {
  const adminDbPage = require("./routes/adminDbPage");
  const adminDbApi = require("./routes/adminDbApi");
  app.use(adminDbPage);
  // Admin API는 /api/admin 아래로만 매핑해 다른 엔드포인트에 영향 주지 않도록 범위를 제한
  app.use("/api/admin", adminDbApi);
} catch (e) {
  console.warn("admin routes load failed", e.message);
}

// Health
app.get("/api/health", (req, res) => {
  const demo = fs.existsSync(path.join(DATA_DIR, ".demo"));
  res.json({ status: "ok", message: "API running", demo });
});

app.get("/api/confidence-calibration", (req, res) => {
  try {
    const filePath = path.join(DATA_DIR, "confidence_calibration_map.json");
    if (!fs.existsSync(filePath)) return res.status(404).json({ error: "confidence calibration not found" });
    const raw = fs.readFileSync(filePath, "utf-8");
    const normalized = raw.replace(/\bNaN\b/g, "null").replace(/\b-Infinity\b/g, "null").replace(/\bInfinity\b/g, "null");
    res.type("application/json").send(normalized);
  } catch (e) {
    console.error("GET /api/confidence-calibration error", e);
    res.status(500).json({ error: "internal error" });
  }
});

app.post("/api/ops-readiness/notes", operatorAccess.apiGuard, async (req, res) => {
  try {
    const memo = typeof req.body?.operator_memo === "string" ? req.body.operator_memo.trim() : "";
    const updatedBy = typeof req.body?.updated_by === "string" ? req.body.updated_by.trim() : "";
    const payload = {
      operator_memo: memo,
      last_updated_at: new Date().toISOString(),
      last_updated_by: updatedBy || "local_operator",
    };
    const outPath = path.join(OUTPUTS_DIR, "ops_operator_notes.json");
    const ok = writeJson(outPath, payload);
    if (!ok) return res.status(500).json({ error: "failed to save notes" });
    await queryRows(
      `
      INSERT INTO research.app_payload_store
        (payload_key, payload_json, asof_date, generated_at, source_path, updated_at)
      VALUES
        ($1, $2::jsonb, NULL, $3, $4, now())
      ON CONFLICT (payload_key) DO UPDATE
      SET payload_json = EXCLUDED.payload_json,
          generated_at = EXCLUDED.generated_at,
          source_path = EXCLUDED.source_path,
          updated_at = now()
      `,
      ["ops_operator_notes", JSON.stringify(payload), payload.last_updated_at, outPath]
    );
    res.json({ ok: true, notes: payload });
  } catch (e) {
    console.error("POST /api/ops-readiness/notes error", e);
    res.status(500).json({ error: "internal error" });
  }
});

app.get("/api/score-kpi-monitor", async (req, res) => {
  try {
    const payload = await readJsonPayloadDbFirst("score_kpi_monitor", [
      path.join(DATA_DIR, "score_kpi_monitor.json"),
      path.join(OUTPUTS_DIR, "score_kpi_monitor.json"),
    ]);
    if (!payload || !Object.keys(payload).length) {
      return res.status(404).json({ error: "score kpi monitor not found" });
    }
    res.json(payload);
  } catch (e) {
    console.error("GET /api/score-kpi-monitor error", e);
    res.status(500).json({ error: "internal error" });
  }
});

app.get("/api/score-check", operatorAccess.apiGuard, async (req, res) => {
  try {
    const date = typeof req.query.date === "string" ? req.query.date.trim() : "";
    const payload = await buildScoreCheckPayload(date || undefined);
    if (!payload) {
      return res.status(404).json({ error: "score_check_not_found" });
    }
    res.json(payload);
  } catch (e) {
    console.error("GET /api/score-check error", e);
    res.status(500).json({ error: "score_check_failed" });
  }
});

app.get("/api/top20-meaningfulness", async (req, res) => {
  try {
    const payload = await readJsonPayloadDbFirst("top20_meaningfulness_report", [
      path.join(DATA_DIR, "top20_meaningfulness_report.json"),
      path.join(OUTPUTS_DIR, "top20_meaningfulness_report.json"),
    ]);
    if (!payload || !Object.keys(payload).length) {
      return res.status(404).json({ error: "top20 meaningfulness report not found" });
    }
    res.json(payload);
  } catch (e) {
    console.error("GET /api/top20-meaningfulness error", e);
    res.status(500).json({ error: "internal error" });
  }
});

app.get("/api/meaningfulness-review-notes", operatorAccess.apiGuard, async (req, res) => {
  try {
    await ensureMeaningfulnessReviewSchema();
    const date = typeof req.query.date === "string" ? req.query.date.trim() : "";
    const params = [];
    let whereClause = "";
    if (date) {
      params.push(date);
      whereClause = "WHERE analysis_date = $1";
    }
    const rows = await queryRows(
      `
      SELECT analysis_date, code, decision, note, updated_by, created_at, updated_at
      FROM research.meaningfulness_review_note
      ${whereClause}
      ORDER BY analysis_date DESC, updated_at DESC, code ASC
      `,
      params
    );
    res.json({ rows });
  } catch (e) {
    console.error("GET /api/meaningfulness-review-notes error", e);
    res.status(500).json({ error: "meaningfulness_review_notes_failed" });
  }
});

app.post("/api/meaningfulness-review-notes", operatorAccess.apiGuard, async (req, res) => {
  try {
    await ensureMeaningfulnessReviewSchema();
    const analysisDate = typeof req.body?.analysis_date === "string" ? req.body.analysis_date.trim() : "";
    const code = typeof req.body?.code === "string" ? req.body.code.trim() : "";
    const decision = typeof req.body?.decision === "string" ? req.body.decision.trim() : "";
    const note = typeof req.body?.note === "string" ? req.body.note.trim() : "";
    const updatedBy = typeof req.body?.updated_by === "string" ? req.body.updated_by.trim() : "";

    if (!analysisDate || !code) {
      return res.status(400).json({ error: "analysis_date_and_code_required" });
    }

    const rows = await queryRows(
      `
      INSERT INTO research.meaningfulness_review_note
        (analysis_date, code, decision, note, updated_by, updated_at)
      VALUES
        ($1::date, $2, NULLIF($3, ''), NULLIF($4, ''), NULLIF($5, ''), now())
      ON CONFLICT (analysis_date, code) DO UPDATE
      SET decision = EXCLUDED.decision,
          note = EXCLUDED.note,
          updated_by = EXCLUDED.updated_by,
          updated_at = now()
      RETURNING analysis_date, code, decision, note, updated_by, created_at, updated_at
      `,
      [analysisDate, code, decision, note, updatedBy || "operator"]
    );

    res.json({ ok: true, row: rows[0] || null });
  } catch (e) {
    console.error("POST /api/meaningfulness-review-notes error", e);
    res.status(500).json({ error: "meaningfulness_review_notes_save_failed" });
  }
});

function normalizeReviewTags(value) {
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => String(item || "").trim().toUpperCase())
    .filter(Boolean)
    .filter((item, idx, arr) => arr.indexOf(item) === idx)
    .slice(0, 20);
}

function normalizeOptionalText(value, maxLength = 2000) {
  if (typeof value !== "string") return "";
  return value.trim().slice(0, maxLength);
}

app.get("/api/live-trade-reviews", operatorAccess.apiGuard, async (req, res) => {
  try {
    await ensureLiveTradeReviewSchema();
    const code = typeof req.query.code === "string" ? req.query.code.trim().padStart(6, "0") : "";
    const requestId = typeof req.query.request_id === "string" ? req.query.request_id.trim() : "";
    const intentId = typeof req.query.intent_id === "string" ? req.query.intent_id.trim() : "";
    const rawLimit = Number(req.query.limit);
    const limit = Number.isFinite(rawLimit) ? Math.min(Math.max(Math.floor(rawLimit), 1), 200) : 100;

    const clauses = [];
    const params = [];
    if (code) {
      params.push(code);
      clauses.push(`code = $${params.length}`);
    }
    if (requestId) {
      params.push(requestId);
      clauses.push(`request_id = $${params.length}`);
    }
    if (intentId) {
      params.push(intentId);
      clauses.push(`intent_id = $${params.length}`);
    }
    params.push(limit);
    const whereClause = clauses.length ? `WHERE ${clauses.join(" AND ")}` : "";
    const rows = await queryRows(
      `
      SELECT review_id, intent_id, request_id, code, review_date,
             pre_tags, post_tags, outcome_label, review_note,
             next_action_note, reviewer, created_at, updated_at
      FROM research.live_trade_review
      ${whereClause}
      ORDER BY review_date DESC, updated_at DESC, review_id DESC
      LIMIT $${params.length}
      `,
      params
    );
    res.json({ count: rows.length, rows });
  } catch (e) {
    console.error("GET /api/live-trade-reviews error", e);
    res.status(500).json({ error: "live_trade_reviews_failed" });
  }
});

app.post("/api/live-trade-reviews", operatorAccess.apiGuard, async (req, res) => {
  try {
    await ensureLiveTradeReviewSchema();
    const reviewId = Number(req.body?.review_id);
    const code = normalizeOptionalText(req.body?.code, 10).padStart(6, "0");
    const intentId = normalizeOptionalText(req.body?.intent_id, 120);
    const requestId = normalizeOptionalText(req.body?.request_id, 160);
    const reviewDate = normalizeOptionalText(req.body?.review_date, 10) || new Date().toISOString().slice(0, 10);
    const preTags = normalizeReviewTags(req.body?.pre_tags);
    const postTags = normalizeReviewTags(req.body?.post_tags);
    const outcomeLabel = normalizeOptionalText(req.body?.outcome_label, 80);
    const reviewNote = normalizeOptionalText(req.body?.review_note, 4000);
    const nextActionNote = normalizeOptionalText(req.body?.next_action_note, 2000);
    const reviewer = normalizeOptionalText(req.body?.reviewer || req.body?.updated_by, 80) || "operator";

    if (!code || code === "000000") {
      return res.status(400).json({ error: "code_required" });
    }

    const params = [
      intentId || null,
      requestId || null,
      code,
      reviewDate,
      preTags.length ? preTags : null,
      postTags.length ? postTags : null,
      outcomeLabel || null,
      reviewNote || null,
      nextActionNote || null,
      reviewer,
    ];

    let rows;
    if (Number.isFinite(reviewId) && reviewId > 0) {
      rows = await queryRows(
        `
        UPDATE research.live_trade_review
        SET intent_id = $1,
            request_id = $2,
            code = $3,
            review_date = $4::date,
            pre_tags = $5::text[],
            post_tags = $6::text[],
            outcome_label = $7,
            review_note = $8,
            next_action_note = $9,
            reviewer = $10,
            updated_at = now()
        WHERE review_id = $11
        RETURNING review_id, intent_id, request_id, code, review_date,
                  pre_tags, post_tags, outcome_label, review_note,
                  next_action_note, reviewer, created_at, updated_at
        `,
        [...params, reviewId]
      );
      if (!rows.length) {
        return res.status(404).json({ error: "review_not_found" });
      }
    } else {
      rows = await queryRows(
        `
        INSERT INTO research.live_trade_review (
          intent_id, request_id, code, review_date, pre_tags, post_tags,
          outcome_label, review_note, next_action_note, reviewer, updated_at
        )
        VALUES ($1, $2, $3, $4::date, $5::text[], $6::text[], $7, $8, $9, $10, now())
        RETURNING review_id, intent_id, request_id, code, review_date,
                  pre_tags, post_tags, outcome_label, review_note,
                  next_action_note, reviewer, created_at, updated_at
        `,
        params
      );
    }

    res.json({ ok: true, row: rows[0] || null });
  } catch (e) {
    console.error("POST /api/live-trade-reviews error", e);
    res.status(500).json({ error: "live_trade_review_save_failed", detail: String(e) });
  }
});

app.post("/api/meaningfulness-outcomes", async (req, res) => {
  try {
    const analysisDate = typeof req.body?.analysis_date === "string" ? req.body.analysis_date.trim() : "";
    const codes = Array.isArray(req.body?.codes) ? req.body.codes : [];
    if (!analysisDate || !codes.length) {
      return res.status(400).json({ error: "analysis_date_and_codes_required" });
    }
    const rows = await buildMeaningfulnessOutcomes({ analysisDate, codes });
    res.json({ rows });
  } catch (e) {
    console.error("POST /api/meaningfulness-outcomes error", e);
    res.status(500).json({ error: "meaningfulness_outcomes_failed" });
  }
});

// Market status
app.get("/api/market/status", async (req, res) => {
  try {
    const row = await loadMarketStatusLatest();
    if (!row) return res.status(404).json({ error: "market_status not found" });
    const status_date =
      row.date || row.status_date || row.market_status_date || null;
    res.json({
      status_date,
      market_up: (() => {
        const b = boolify(row.market_up);
        return b === null ? null : b;
      })(),
      kospi_close: toNum(row.kospi_close ?? row.close ?? row.kospi),
      kospi_ma20: toNum(row.kospi_ma20 ?? row.ma20),
      vol_5d: toNum(row.vol_5d ?? row.volatility_5d),
      foreign_5d: toNum(row.foreign_5d ?? row.foreign_trading_5d ?? row.foreign_net_5d),
    });
  } catch (e) {
    console.error("GET /api/market/status error", e);
    res.status(500).json({ error: "internal error" });
  }
});

// Sectors list
app.get("/api/sectors", (req, res) => {
  try {
    const sectors = Array.from(universeMap.values())
      .map((v) => v.sector || "")
      .filter((s) => s);
    res.json(Array.from(new Set(sectors)).sort((a, b) => a.localeCompare(b, "ko")));
  } catch {
    res.json([]);
  }
});

// Stocks list (predictions + latest feature snapshot)
app.get("/api/stocks", async (req, res) => {
  try {
    const preds = await getPredictions();
    if (!preds || !preds.length) {
      return res.status(404).json({ error: "predictions not found" });
    }
    const feats = await getFeatures();

    const byCode = new Map();
    feats.forEach((r) => {
      const code = r.code;
      if (!code) return;
      if (!byCode.has(code)) byCode.set(code, []);
      byCode.get(code).push(r);
    });

    const latestCloseMap = new Map();
    const ret3mMap = new Map();
    for (const [code, arr] of byCode.entries()) {
      arr.sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
      if (!arr.length) continue;
      const last = arr[arr.length - 1];
      latestCloseMap.set(code, toNum(last.close));
      if (arr.length >= 60) {
        const prev = arr[arr.length - 60];
        const prevClose = toNum(prev.close);
        const lastClose = toNum(last.close);
        if (Number.isFinite(prevClose) && Number.isFinite(lastClose) && prevClose !== 0) {
          ret3mMap.set(code, lastClose / prevClose - 1);
        }
      }
    }

    const marketFilter = (req.query.market || "").toUpperCase();
    const sectorFilter = (req.query.sector || "");

    let data = preds.map((r) => {
      const code = r.code;
      const info = universeMap.get(code) || {};
      const close = latestCloseMap.get(code);
      const shares = info.shares;
      const mktcap = Number.isFinite(info.mktcap)
        ? info.mktcap
        : Number.isFinite(shares) && Number.isFinite(close)
        ? shares * close
        : null;
      return {
        date: r.date,
        code,
        name: getName(code),
        market: (info.market || "").toUpperCase(),
        sector: info.sector || null,
        close,
        mktcap,
        ret_3m: ret3mMap.get(code) ?? null,
        pred_return_60d: toNum(r.pred_return_60d),
        pred_return_90d: r.pred_return_90d !== undefined && r.pred_return_90d !== "" ? toNum(r.pred_return_90d) : null,
      };
    });

    if (marketFilter && marketFilter !== "ALL") {
      data = data.filter((d) => d.market === marketFilter);
    }
    if (sectorFilter && sectorFilter !== "ALL") {
      data = data.filter((d) => (d.sector || "") === sectorFilter);
    }

    res.json(data);
  } catch (e) {
    console.error("api/stocks error", e);
    res.status(500).json({ error: "internal error" });
  }
});

// Stock detail
app.get("/api/stocks/:code", async (req, res) => {
  try {
    const code = req.params.code;
    const limit = parseInt(req.query.limit || "180", 10);

    const feats = await getFeatures("WHERE code = $1 ORDER BY date", [code]);
    if (!feats || !feats.length) {
      return res.status(404).json({ error: `no data for code ${code}` });
    }
    const sliced = limit > 0 ? feats.slice(-limit) : feats;
    const rows = sliced.map((r) => ({
      date: r.date,
      close: toNum(r.close),
      ma_5: toNum(r.ma_5),
      ma_20: toNum(r.ma_20),
      ma_60: toNum(r.ma_60),
      rsi_14: toNum(r.rsi_14),
      vol_20: toNum(r.vol_20),
      volume: toNum(r.volume),
    }));
    const latest = rows[rows.length - 1];

    const pred = (
      await queryRows(
        "SELECT * FROM predictions WHERE code = $1 ORDER BY date DESC LIMIT 1",
        [code]
      )
    )[0];
    const rank = (
      await queryRows(
        "SELECT * FROM daily_ranking WHERE code = $1 ORDER BY date DESC LIMIT 1",
        [code]
      )
    )[0];
    const stockRow = (
      await queryRows(
        "SELECT code, name, market, sector FROM stocks WHERE code = $1 LIMIT 1",
        [code]
      )
    )[0];
    const dailyRecommendation = await getDailyRecommendationItem(code);
    const dailySecurity = dailyRecommendation?.security || {};
    const buyEligibility = dailyRecommendation?.buy_eligibility || {};
    const selection = dailyRecommendation?.selection || {};
    const universeName = typeof getName(code) === "string" ? getName(code).trim() : "";
    const resolvedName =
      (typeof dailySecurity.name === "string" && dailySecurity.name.trim()) ||
      (typeof rank?.name === "string" && rank.name.trim()) ||
      (typeof stockRow?.name === "string" && stockRow.name.trim()) ||
      (universeName && universeName !== code ? universeName : "") ||
      null;
    const resolvedMarket =
      (typeof rank?.market === "string" && rank.market.trim()) ||
      (typeof stockRow?.market === "string" && stockRow.market.trim()) ||
      (typeof dailySecurity.market === "string" && dailySecurity.market.trim()) ||
      (typeof getMarket(code) === "string" && getMarket(code).trim()) ||
      null;
    const resolvedSector =
      (typeof rank?.sector === "string" && rank.sector.trim()) ||
      (typeof stockRow?.sector === "string" && stockRow.sector.trim()) ||
      (typeof dailySecurity.sector === "string" && dailySecurity.sector.trim()) ||
      (typeof getSector(code) === "string" && getSector(code).trim()) ||
      null;

    res.json({
      code,
      name: resolvedName,
      market: resolvedMarket,
      sector: resolvedSector,
      count: rows.length,
      latest,
      volatility_20: latest ? toNum(latest.vol_20) : null,
      pred_return_60d: pred ? toNum(pred.pred_return_60d) : null,
      pred_return_90d: pred ? toNum(pred.pred_return_90d) : null,
      ret_score: rank ? toNum(rank.ret_score) : null,
      prob_score: rank ? toNum(rank.prob_score) : null,
      qual_score: rank ? toNum(rank.qual_score) : null,
      tech_score: rank ? toNum(rank.tech_score) : null,
      pred_score: rank ? toNum(rank.pred_score) : null,
      risk_penalty: rank ? toNum(rank.risk_penalty) : null,
      pred_mdd_60d: rank ? toNum(rank.pred_mdd_60d) : null,
      pred_mdd_90d: rank ? toNum(rank.pred_mdd_90d) : null,
      pred_mdd_mix: rank ? toNum(rank.pred_mdd_mix) : null,
      prob_top20_60d: rank ? toNum(rank.prob_top20_60d) : null,
      prob_top20_90d: rank ? toNum(rank.prob_top20_90d) : null,
      regime: rank ? (rank.regime || null) : null,
      live_rank: rank ? getLiveRank(rank) : null,
      live_score: rank ? getLiveScore(rank) : null,
      live_score_source: rank ? getLiveScoreSource(rank) : null,
      final_score_rank: rank ? toNum(rank.final_score) : null,
      final_score_raw: rank ? toNum(rank.final_score_raw) : null,
      shadow_quality_risk_guard_score: rank ? getQualityRiskGuardShadowScore(rank) : null,
      shadow_quality_risk_guard_rank: rank ? getQualityRiskGuardShadowRank(rank) : null,
      shadow_quality_risk_guard_penalty: rank ? getQualityRiskGuardPenalty(rank) : null,
      shadow_quality_risk_guard_rank_delta:
        rank && Number.isFinite(getLiveRank(rank)) && Number.isFinite(getQualityRiskGuardShadowRank(rank))
          ? getLiveRank(rank) - getQualityRiskGuardShadowRank(rank)
          : null,
      confidence_score: rank ? getConfidenceScore(rank) : null,
      confidence_grade: rank ? getConfidenceLabel(rank) : null,
      explain_text: rank ? (rank.explain_text || null) : null,
      confidence_explain_text: rank ? getConfidenceExplainText(rank) : null,
      contrib_ret: rank ? toNum(rank.contrib_ret) : null,
      contrib_prob: rank ? toNum(rank.contrib_prob) : null,
      contrib_qual: rank ? toNum(rank.contrib_qual) : null,
      contrib_tech: rank ? toNum(rank.contrib_tech) : null,
      contrib_safety: rank ? toNum(rank.contrib_safety) : null,
      contrib_liquidity: rank ? toNum(rank.contrib_liquidity) : null,
      contrib_penalty: rank ? toNum(rank.contrib_penalty) : null,
      top_positive_factor: rank ? (rank.top_positive_factor || null) : null,
      top_positive_value: rank ? toNum(rank.top_positive_value) : null,
      top_negative_factor: rank ? (rank.top_negative_factor || null) : null,
      top_negative_value: rank ? toNum(rank.top_negative_value) : null,
      buy_eligibility_status: buyEligibility.status || null,
      buy_eligibility_score: toNum(buyEligibility.score),
      buy_eligibility_hard_block_reasons: translateBuyEligibilityReasons(buyEligibility.hard_block_reasons),
      buy_eligibility_caution_reasons: translateBuyEligibilityReasons(buyEligibility.caution_reasons),
      buyability_status: selection.buyability_status || null,
      buyability_watchlist_tier: selection.buyability_watchlist_tier || null,
      buyability_expected_action: selection.buyability_expected_action || null,
      buyability_blocking_reasons: Array.isArray(selection.buyability_blocking_reasons) ? selection.buyability_blocking_reasons : [],
      buyability_supporting_reasons: Array.isArray(selection.buyability_supporting_reasons) ? selection.buyability_supporting_reasons : [],
      rows,
    });
  } catch (e) {
    console.error("api/stocks/:code error", e);
    res.status(500).json({ error: "internal error" });
  }
});

// Ranking list
app.get("/api/ranking", async (req, res) => {
  try {
    const targetDate = (req.query.date || "").trim() || null;
    const rankingRes = await getRanking(targetDate);
    if (!rankingRes) return res.status(404).json({ error: "ranking data not found" });
    const { date, rows } = rankingRes;

    const codes = rows.map((r) => r.code);
    const { latestClose, ret3m, ret5d, ret10d, mom20, rsi14 } = await getFeatureStatsForCodes(codes);

    let marketUp = true;
    if (rows.length && rows[0].market_up !== undefined) {
      const b = boolify(rows[0].market_up);
      marketUp = b === null ? true : b;
    }

    const marketFilter = (req.query.market || "").toUpperCase();
    const sectorFilter = (req.query.sector || "");

    let data = rows.map((r) => {
      const code = r.code;
      const surgeMeta = buildRecentSurgeMeta({
        ...r,
        ret_5d: toNum(r.ret_5d) ?? ret5d.get(code) ?? null,
        ret_10d: toNum(r.ret_10d) ?? ret10d.get(code) ?? null,
        mom_20: toNum(r.mom_20) ?? mom20.get(code) ?? null,
        rsi_14: toNum(r.rsi_14) ?? rsi14.get(code) ?? null,
      });
      return {
        date: r.date,
        code,
        name: (r.name && r.name.trim()) || getName(code),
        market: ((r.market && r.market.trim()) || getMarket(code) || "").toUpperCase(),
        sector: (r.sector && r.sector.trim()) || getSector(code) || null,
        close: latestClose.get(code) ?? toNum(r.close),
        ret_3m: ret3m.get(code) ?? null,
        pred_return_60d: toNum(r.pred_return_60d),
        pred_return_90d: toNum(r.pred_return_90d),
        pred_mdd_60d: toNum(r.pred_mdd_60d),
        pred_mdd_90d: toNum(r.pred_mdd_90d),
        risk_penalty: toNum(r.risk_penalty),
        prob_top20_60d: toNum(r.prob_top20_60d),
        prob_top20_90d: toNum(r.prob_top20_90d),
        score: getLiveScore(r),
        live_score: getLiveScore(r),
        live_rank: getLiveRank(r),
        live_score_source: getLiveScoreSource(r),
        live_score: getLiveScore(r),
        final_score: toNum(r.final_score),
        shadow_quality_risk_guard_score: getQualityRiskGuardShadowScore(r),
        shadow_quality_risk_guard_rank: getQualityRiskGuardShadowRank(r),
        shadow_quality_risk_guard_penalty: getQualityRiskGuardPenalty(r),
        shadow_quality_risk_guard_rank_delta:
          Number.isFinite(getLiveRank(r)) && Number.isFinite(getQualityRiskGuardShadowRank(r))
            ? getLiveRank(r) - getQualityRiskGuardShadowRank(r)
            : null,
        final_score_raw: toNum(r.final_score_raw),
        tech_score: toNum(r.tech_score),
        qual_score: toNum(r.qual_score),
        ret_score: toNum(r.ret_score),
        prob_score: toNum(r.prob_score),
        pred_score: toNum(r.pred_score),
        safety_score: toNum(r.safety_score),
        liquidity_score: toNum(r.liquidity_score),
        confidence_score: getConfidenceScore(r),
        confidence_label: getConfidenceLabel(r),
        confidence_grade: getConfidenceLabel(r),
        confidence_reason: r.confidence_reason || null,
        explain_text: r.explain_text || null,
        confidence_explain_text: getConfidenceExplainText(r),
        score_explain_summary: r.score_explain_summary || null,
        score_explain_strengths: r.score_explain_strengths || null,
        score_explain_risks: r.score_explain_risks || null,
        score_explain_confidence: r.score_explain_confidence || null,
        score_explain_regime: r.score_explain_regime || null,
        score_driver_1: r.score_driver_1 || null,
        score_driver_2: r.score_driver_2 || null,
        score_driver_3: r.score_driver_3 || null,
        score_drag_1: r.score_drag_1 || null,
        score_drag_2: r.score_drag_2 || null,
        top_driver_1: r.top_driver_1 || null,
        top_driver_2: r.top_driver_2 || null,
        top_driver_3: r.top_driver_3 || null,
        risk_factor_1: r.risk_factor_1 || null,
        risk_factor_2: r.risk_factor_2 || null,
        action_note: r.action_note || null,
        regime: r.regime || null,
        regime_reason: r.regime_reason || null,
        weight_profile: r.weight_profile || null,
        ret_5d: surgeMeta.ret_5d,
        ret_10d: surgeMeta.ret_10d,
        mom_20: surgeMeta.mom_20,
        rsi_14: surgeMeta.rsi_14,
        recent_surge_soft_flag: surgeMeta.recent_surge_soft_flag,
        recent_surge_hard_flag: surgeMeta.recent_surge_hard_flag,
        recent_surge_label: surgeMeta.recent_surge_label,
        recent_surge_detail: surgeMeta.recent_surge_detail,
        contrib_ret: toNum(r.contrib_ret),
        contrib_prob: toNum(r.contrib_prob),
        contrib_qual: toNum(r.contrib_qual),
        contrib_tech: toNum(r.contrib_tech),
        contrib_safety: toNum(r.contrib_safety),
        contrib_liquidity: toNum(r.contrib_liquidity),
        contrib_penalty: toNum(r.contrib_penalty),
        score_contribution_ret: toNum(r.score_contribution_ret),
        score_contribution_prob: toNum(r.score_contribution_prob),
        score_contribution_tech: toNum(r.score_contribution_tech),
        score_contribution_qual: toNum(r.score_contribution_qual),
        score_contribution_safety: toNum(r.score_contribution_safety),
        score_contribution_liquidity: toNum(r.score_contribution_liquidity),
        score_contribution_risk: toNum(r.score_contribution_risk),
        top_positive_factor: r.top_positive_factor || null,
        top_positive_value: toNum(r.top_positive_value),
        top_negative_factor: r.top_negative_factor || null,
        top_negative_value: toNum(r.top_negative_value),
        market_up: marketUp,
        market_status_date: r.market_status_date || null,
        market_kospi_close: toNum(r.market_kospi_close),
        market_kospi_ma20: toNum(r.market_kospi_ma20),
        market_vol_5d: toNum(r.market_vol_5d),
        market_foreign_5d: toNum(r.market_foreign_5d),
      };
    });

    if (marketFilter && marketFilter !== "ALL") {
      data = data.filter((d) => d.market === marketFilter);
    }
    if (sectorFilter && sectorFilter !== "ALL") {
      data = data.filter((d) => (d.sector || "") === sectorFilter);
    }

    data.sort((a, b) => {
      const af = Number.isFinite(+a.live_score) ? +a.live_score : -Infinity;
      const bf = Number.isFinite(+b.live_score) ? +b.live_score : -Infinity;
      return bf - af;
    });

    // 기존 프런트 호환: 배열 형태를 그대로 반환
    res.setHeader("X-Ranking-Date", date);
    res.json(data);
  } catch (e) {
    console.error("api/ranking error", e);
    res.status(500).json({ error: "failed to read ranking", detail: String(e) });
  }
});

// Top20 summary
app.get("/api/top20", async (req, res) => {
  try {
    const rankingRes = await getRanking(req.query.date || null);
    if (!rankingRes) return res.status(404).json({ error: "ranking data is empty" });

    const { date, rows } = rankingRes;
    const sorted = rows
      .slice()
      .sort(
        (a, b) =>
          (Number.isFinite(+getLiveScore(b)) ? +getLiveScore(b) : -Infinity) -
          (Number.isFinite(+getLiveScore(a)) ? +getLiveScore(a) : -Infinity)
      )
      .slice(0, 20);

    if (!sorted.length) return res.status(404).json({ error: "no ranking rows" });

    const first = sorted[0];
    const marketMeta = {
      market_up: (() => {
        const b = boolify(first.market_up);
        return b === null ? true : b;
      })(),
      status_date: first.market_status_date || null,
      kospi_close: toNum(first.market_kospi_close),
      kospi_ma20: toNum(first.market_kospi_ma20),
      vol_5d: toNum(first.market_vol_5d),
      foreign_5d: toNum(first.market_foreign_5d),
    };

    const fmtPct = (v, d = 1) => {
      const n = toNum(v);
      if (!Number.isFinite(n)) return "-";
      return (n * 100).toFixed(d) + "%";
    };

    const items = sorted.map((r, idx) => {
      const code = r.code;
      const name = (r.name && r.name.trim()) || getName(code);
      const sector = r.sector || getSector(code) || null;
      const market = (r.market || getMarket(code) || "").toUpperCase();
      const close = toNum(r.close);
      const pred60 = toNum(r.pred_return_60d);
      const pred90 = toNum(r.pred_return_90d);
      const mdd60 = toNum(r.pred_mdd_60d);
      const mdd90 = toNum(r.pred_mdd_90d);
      const prob60 = toNum(r.prob_top20_60d);
      const prob90 = toNum(r.prob_top20_90d);
      const retScore = toNum(r.ret_score);
      const probScore = toNum(r.prob_score);
      const riskPenalty = toNum(r.risk_penalty);
      const finalScore = getLiveScore(r);

      const summary_ko = [
        `(${idx + 1}) ${name} (${code})${sector ? ` · 섹터: ${sector}` : ""}${market ? ` · 시장: ${market}` : ""}`,
        `- 예상 수익률 60d ${fmtPct(pred60)}, 90d ${fmtPct(pred90)}`,
        `- 상위20% 확률: 60d ${fmtPct(prob60)}, 90d ${fmtPct(prob90)}`,
        `- 예상 MDD: 60d ${fmtPct(Math.abs(mdd60))}, 90d ${fmtPct(Math.abs(mdd90))}`,
        `- 점수: 수익 ${retScore?.toFixed?.(1) ?? "-"}, 확률 ${probScore?.toFixed?.(1) ?? "-"}, 리스크 ${riskPenalty?.toFixed?.(1) ?? "-"}`,
        `- 실운영 점수(live_score): ${Number.isFinite(finalScore) ? finalScore.toFixed(2) : "-"}`,
      ].join("\n");

      return {
        rank: idx + 1,
        date: r.date,
        code,
        name,
        sector,
        market,
        close,
        pred_return_60d: pred60,
        pred_return_90d: pred90,
        pred_mdd_60d: mdd60,
        pred_mdd_90d: mdd90,
        prob_top20_60d: prob60,
        prob_top20_90d: prob90,
        ret_score: retScore,
        prob_score: probScore,
        risk_penalty: riskPenalty,
        final_score: finalScore,
        live_score: finalScore,
        live_rank: getLiveRank(r),
        live_score_source: getLiveScoreSource(r),
        shadow_quality_risk_guard_score: getQualityRiskGuardShadowScore(r),
        shadow_quality_risk_guard_rank: getQualityRiskGuardShadowRank(r),
        shadow_quality_risk_guard_penalty: getQualityRiskGuardPenalty(r),
        shadow_quality_risk_guard_rank_delta:
          Number.isFinite(getLiveRank(r)) && Number.isFinite(getQualityRiskGuardShadowRank(r))
            ? getLiveRank(r) - getQualityRiskGuardShadowRank(r)
            : null,
        score: finalScore,
        summary_ko,
      };
    });

    res.json({ date, market: marketMeta, count: items.length, items });
  } catch (e) {
    console.error("api/top20 error", e);
    res.status(500).json({ error: "failed to build top20 summary", detail: String(e) });
  }
});

// Signals top20
app.get("/api/signals/top20", async (req, res) => {
  try {
    const horizon = req.query.horizon === "90d" ? "90d" : "60d";
    const limit = Math.max(1, Math.min(100, Number(req.query.limit) || 20));
    const onlyNew = req.query.only_new === "1" || req.query.only_new === "true";
    const rankingRes = await getRanking(req.query.date || null);
    if (!rankingRes) return res.status(404).json({ error: "ranking data is empty" });
    const { date, rows } = rankingRes;

    const sortKey = horizon === "90d" ? "pred_return_90d" : "pred_return_60d";
    const mddKey = horizon === "90d" ? "pred_mdd_90d" : "pred_mdd_60d";
    const probKey = horizon === "90d" ? "prob_top20_90d" : "prob_top20_60d";

    const filtered = rows
      .slice()
      .sort((a, b) => (toNum(b[sortKey]) || 0) - (toNum(a[sortKey]) || 0));

    const latestByCode = new Map();
    rows.forEach((r) => {
      const code = (r.code || "").trim();
      if (!code) return;
      const prev = latestByCode.get(code);
      if (!prev || String(prev.date || "") < String(r.date || "")) {
        latestByCode.set(code, r);
      }
    });

    const trades = await listTrades();
    const holdings = buildHoldings(trades, latestByCode);
    const holdingCodes = new Set(holdings.map((h) => h.code));

    const items = [];
    for (const r of filtered) {
      const code = (r.code || "").trim();
      if (!code) continue;
      const isHolding = holdingCodes.has(code);
      if (onlyNew && isHolding) continue;
      items.push({
        rank: items.length + 1,
        code,
        name: r.name || getName(code) || code,
        market: r.market || getMarket(code) || "",
        sector: r.sector || getSector(code) || "",
        close: toNum(r.close),
        pred_return_60d: toNum(r.pred_return_60d),
        pred_return_90d: toNum(r.pred_return_90d),
        pred_mdd_60d: toNum(r.pred_mdd_60d),
        pred_mdd_90d: toNum(r.pred_mdd_90d),
        pred_return: toNum(r[sortKey]),
        pred_mdd: toNum(r[mddKey]),
        prob_top20: toNum(r[probKey]),
        live_score: getLiveScore(r),
        live_rank: getLiveRank(r),
        live_score_source: getLiveScoreSource(r),
        final_score: toNum(r.final_score),
        shadow_quality_risk_guard_score: getQualityRiskGuardShadowScore(r),
        shadow_quality_risk_guard_rank: getQualityRiskGuardShadowRank(r),
        shadow_quality_risk_guard_penalty: getQualityRiskGuardPenalty(r),
        shadow_quality_risk_guard_rank_delta:
          Number.isFinite(getLiveRank(r)) && Number.isFinite(getQualityRiskGuardShadowRank(r))
            ? getLiveRank(r) - getQualityRiskGuardShadowRank(r)
            : null,
        ret_score: toNum(r.ret_score),
        prob_score: toNum(r.prob_score),
        qual_score: toNum(r.qual_score),
        tech_score: toNum(r.tech_score),
        pred_score: toNum(r.pred_score),
        risk_penalty: toNum(r.risk_penalty),
        is_holding: isHolding,
      });
      if (items.length >= limit) break;
    }

    res.json({ date, horizon, items });
  } catch (e) {
    console.error("GET /api/signals/top20 error", e);
    res.status(500).json({ error: "internal error" });
  }
});

// Today actions
app.get("/api/dashboard/today-actions", async (req, res) => {
  try {
    const rankingRes = await getRanking(req.query.date || null);
    if (!rankingRes) return res.status(404).json({ error: "ranking_final is empty" });
    const { date: targetDate, rows } = rankingRes;

    const latestByCode = new Map();
    rows.forEach((r) => {
      const code = (r.code || "").trim();
      if (!code) return;
      const prev = latestByCode.get(code);
      if (!prev || String(prev.date || "") < String(r.date || "")) {
        latestByCode.set(code, r);
      }
    });

    const trades = await listTrades();
    const holdings = buildHoldings(trades, latestByCode);
    const holdingCodes = new Set(holdings.map((h) => h.code));

    const BUY_MIN_RET = 0.30;
    const BUY_MIN_PROB = 0.40;
    const BUY_MIN_MDD = -0.35;
    const ADD_MIN_RET = 0.25;
    const ADD_MIN_PROB = 0.30;
    const ADD_MIN_MDD = -0.40;
    const TRIM_MAX_RET = 0.05;
    const TRIM_MIN_MDD = -0.45;

    const sorted = rows
      .slice()
      .sort((a, b) => (toNum(b.pred_return_60d) || 0) - (toNum(a.pred_return_60d) || 0))
      .slice(0, 50);

    const buyCandidates = [];
    const addCandidates = [];
    const trimCandidates = [];

    for (let idx = 0; idx < sorted.length; idx++) {
      const r = sorted[idx];
      const code = (r.code || "").trim();
      if (!code) continue;

      const predRet = toNum(r.pred_return_60d);
      const predMdd = toNum(r.pred_mdd_60d);
      const prob = toNum(r.prob_top20_60d);
      const finalScore = getLiveScore(r);
      const isHolding = holdingCodes.has(code);

      if (
        !isHolding &&
        idx < 10 &&
        predRet != null &&
        predRet >= BUY_MIN_RET &&
        prob != null &&
        prob >= BUY_MIN_PROB &&
        predMdd != null &&
        predMdd >= BUY_MIN_MDD
      ) {
        buyCandidates.push({
          code,
          name: r.name || getName(code) || code,
          market: r.market || getMarket(code) || "",
          sector: r.sector || getSector(code) || "",
          close: toNum(r.close),
          pred_return_60d: predRet,
          pred_mdd_60d: predMdd,
          prob_top20_60d: prob,
          live_score: finalScore,
          final_score: finalScore,
          reason: "예상 수익/확률 양호 + 리스크 완화 범위",
        });
      }

      if (
        isHolding &&
        predRet != null &&
        predRet >= ADD_MIN_RET &&
        prob != null &&
        prob >= ADD_MIN_PROB &&
        predMdd != null &&
        predMdd >= ADD_MIN_MDD
      ) {
        addCandidates.push({
          code,
          name: r.name || getName(code) || code,
          market: r.market || getMarket(code) || "",
          sector: r.sector || getSector(code) || "",
          close: toNum(r.close),
          pred_return_60d: predRet,
          pred_mdd_60d: predMdd,
          prob_top20_60d: prob,
          live_score: finalScore,
          final_score: finalScore,
          reason: "보유 중이며 모멘텀/확률 충분",
        });
      }

      if (
        isHolding &&
        ((predRet != null && predRet < TRIM_MAX_RET) ||
          (predMdd != null && predMdd < TRIM_MIN_MDD))
      ) {
        trimCandidates.push({
          code,
          name: r.name || getName(code) || code,
          market: r.market || getMarket(code) || "",
          sector: r.sector || getSector(code) || "",
          close: toNum(r.close),
          pred_return_60d: predRet,
          pred_mdd_60d: predMdd,
          prob_top20_60d: prob,
          live_score: finalScore,
          final_score: finalScore,
          reason: "수익 기대치 낮거나 MDD 위험 확대",
        });
      }
    }

    res.json({
      date: targetDate,
      horizon: "60d",
      buy_candidates: buyCandidates.slice(0, 5),
      add_candidates: addCandidates.slice(0, 5),
      trim_candidates: trimCandidates.slice(0, 5),
    });
  } catch (e) {
    console.error("GET /api/dashboard/today-actions error", e);
    res.status(500).json({ error: "internal error" });
  }
});

// Trades list
app.get("/api/trades", operatorAccess.apiGuard, async (req, res) => {
  try {
    const trades = await listTrades();
    res.json({ count: trades.length, items: trades });
  } catch (e) {
    console.error("api/trades GET error", e);
    res.status(500).json({ error: "failed to load trades", detail: String(e) });
  }
});

// Create trade
app.post("/api/trades", operatorAccess.apiGuard, async (req, res) => {
  try {
    const { side, code, date, qty, price, fee, memo } = req.body || {};
    const s = (side || "").toUpperCase();
    if (!["BUY", "SELL"].includes(s)) {
      return res.status(400).json({ error: "side must be BUY or SELL" });
    }
    if (!code) return res.status(400).json({ error: "code required" });
    if (!date) return res.status(400).json({ error: "date required" });

    const q = Number(qty);
    const p = Number(price);
    if (!Number.isFinite(q) || q <= 0) return res.status(400).json({ error: "qty > 0" });
    if (!Number.isFinite(p) || p <= 0) return res.status(400).json({ error: "price > 0" });

    const amount = q * p;
    const info = await resolveTradeInstrumentInfo(code);
    const inserted = await insertTrade({
      date,
      side: s,
      code,
      name: info.name || null,
      market: info.market || null,
      sector: info.sector || null,
      qty: q,
      price: p,
      amount,
      fee: fee ?? null,
      memo: memo ?? null,
    });

    await insertTradeAuditLog({
      tradeId: inserted.trade_id,
      action: "INSERT",
      tradeSnapshot: buildTradeAuditSnapshot({
        trade_id: inserted.trade_id,
        date,
        side: s,
        code,
        name: info.name || null,
        market: info.market || null,
        sector: info.sector || null,
        qty: q,
        price: p,
        amount,
        fee: fee ?? null,
        memo: memo ?? null,
      }),
      actor: typeof req.body?.updated_by === "string" ? req.body.updated_by.trim() : null,
      reason: "api_insert_trade",
    });

    res.json({ success: true, trade_id: inserted.trade_id });
  } catch (e) {
    console.error("[POST /api/trades] error:", e);
    res.status(500).json({ error: "failed to save", detail: String(e) });
  }
});

app.delete("/api/trades/:tradeId", operatorAccess.apiGuard, async (req, res) => {
  try {
    const tradeId = Number(req.params.tradeId);
    if (!Number.isFinite(tradeId) || tradeId <= 0) {
      return res.status(400).json({ error: "valid trade_id required" });
    }

    const existing = await getTradeById(tradeId);
    if (!existing) {
      return res.status(404).json({ error: "trade not found" });
    }

    const deleted = await deleteTradeById(tradeId);
    if (!deleted) {
      return res.status(500).json({ error: "trade delete failed" });
    }

    await insertTradeAuditLog({
      tradeId,
      action: "DELETE",
      tradeSnapshot: buildTradeAuditSnapshot(existing),
      actor: typeof req.query.updated_by === "string" ? req.query.updated_by.trim() : null,
      reason: typeof req.query.reason === "string" ? req.query.reason.trim() : "api_delete_trade",
    });

    res.json({ success: true, trade_id: tradeId });
  } catch (e) {
    console.error("[DELETE /api/trades/:tradeId] error:", e);
    res.status(500).json({ error: "failed to delete trade", detail: String(e) });
  }
});

// Holdings summary
app.get("/api/holdings", async (req, res) => {
  try {
    const trades = await listTrades();
    if (!trades.length) return res.json({ count: 0, items: [] });

    const latestRankByCode = await getRankingLatestByCode();
    const holdings = buildHoldings(trades, latestRankByCode);

    let totalValue = 0;
    let totalCost = 0;
    let totalRealized = 0;
    holdings.forEach((h) => {
      if (Number.isFinite(h.current_value)) totalValue += h.current_value;
      if (Number.isFinite(h.current_value) && Number.isFinite(h.cost_basis)) totalCost += h.cost_basis;
      if (Number.isFinite(h.realized_pnl)) totalRealized += h.realized_pnl;
    });

    const totalUnrealized = totalValue - totalCost;
    const totalUnrealizedPct = totalCost > 0 ? (totalUnrealized / totalCost) * 100 : null;
    // 화면 요구에 맞춰 총 평가손익은 현재 평가손익(미실현 기준)으로 계산
    const totalPnl = totalUnrealized;
    const totalPnlPct = totalCost > 0 ? (totalPnl / totalCost) * 100 : null;

    res.json({
      count: holdings.length,
      total_cost: totalCost,
      total_value: totalValue,
      total_realized_pnl: totalRealized,
      total_unrealized_pnl: totalUnrealized,
      total_unrealized_pnl_pct: totalUnrealizedPct,
      total_pnl: totalPnl,
      total_pnl_pct: totalPnlPct,
      items: holdings,
    });
  } catch (e) {
    console.error("api/holdings error", e);
    res.status(500).json({ error: "failed to build holdings", detail: String(e) });
  }
});

// Trades history with running stats
app.get("/api/trades/history", async (req, res) => {
  try {
    const rows = await listTrades();
    if (!rows || !rows.length) return res.json({ items: [] });

    const codeFilter = (req.query.code || "").trim();
    const q = (req.query.q || "").trim().toLowerCase();
    const from = (req.query.from || "").trim();
    const to = (req.query.to || "").trim();

    const stateByCode = new Map();
    const items = [];
    let totalRealizedAcc = 0;

    // 1) 계산용: 오름차순(날짜 asc, trade_id asc)
    rows.sort((a, b) => {
      return compareTradesChronologically(a, b);
    });

    for (const r of rows) {
      const date = toIsoDate(r.date || "");
      const code = String(r.code || "").trim();
      if (!code) continue;

      const name = r.name || getName(code) || code;
      const inDate = (!from || date >= from) && (!to || date <= to);
      const inCode = !codeFilter || code === codeFilter;
      let include = inDate && inCode;
      if (q) {
        include = include && (code.toLowerCase().includes(q) || name.toLowerCase().includes(q));
      }

      const side = (r.side || "").toUpperCase().trim();
      const qty = toNum(r.qty) || 0;
      const price = toNum(r.price) || 0;
      const amount = qty * price * (side === "SELL" ? -1 : 1);

      if (!stateByCode.has(code)) {
        stateByCode.set(code, { qty: 0, avgPrice: 0, realizedAcc: 0 });
      }
      const st = stateByCode.get(code);

      let realized = 0;
      if (side === "BUY") {
        const newQty = st.qty + qty;
        const newCost = st.qty * st.avgPrice + qty * price;
        st.qty = newQty;
        st.avgPrice = newQty > 0 ? newCost / newQty : 0;
      } else if (side === "SELL") {
        const sellQty = Math.min(qty, st.qty > 0 ? st.qty : qty);
        realized = (price - st.avgPrice) * sellQty;
        st.qty -= sellQty;
        if (st.qty < 0) st.qty = 0;
        if (st.qty === 0) st.avgPrice = 0;
        st.realizedAcc += realized;
        totalRealizedAcc += realized;
      }

      if (include) {
        items.push({
          trade_id: r.trade_id,
          date: String(date).slice(0, 10), // YYYY-MM-DD
          code,
          name,
          side,
          qty: side === "SELL" ? -qty : qty,
          price,
          amount,
          realized,
          realized_acc_code: st.realizedAcc,
          realized_acc_total: totalRealizedAcc,
          remain_qty: st.qty,
          avg_price: st.avgPrice,
        });
      }
    }

    // 2) 응답용: 시간순(날짜 asc, trade_id asc) + 날짜 포맷 고정
    items.sort((a, b) => {
      return compareTradesChronologically(a, b);
    });

    res.json({ items });
  } catch (e) {
    console.error("GET /api/trades/history error", e);
    res.status(500).json({ error: "internal error" });
  }
});

// Holding detail
app.get("/api/holding/:code", async (req, res) => {
  try {
    const rawCode = (req.params.code || "").trim();
    if (!rawCode) return res.status(400).json({ error: "code required" });
    const code = rawCode;

    const tradesAll = await listTrades();
    const trades = tradesAll
      .filter((t) => (t.code || "").trim() === code)
      .sort(compareTradesChronologically);

    const rank = (
      await queryRows(
        "SELECT * FROM daily_ranking WHERE code = $1 ORDER BY date DESC LIMIT 1",
        [code]
      )
    )[0];

    const name = (rank && rank.name) || (trades[0] && trades[0].name) || getName(code) || code;
    const market = (rank && rank.market) || (trades[0] && trades[0].market) || getMarket(code) || null;
    const sector = (rank && rank.sector) || (trades[0] && trades[0].sector) || getSector(code) || null;

    if (!trades.length) {
      return res.json({ code, name, market, sector, holding: null, latest: rank || null, trades: [] });
    }

    let positionQty = 0;
    let avgCost = 0;
    let realizedPnl = 0;
    let totalBuyAmount = 0;
    let firstBuyDate = null;
    let lastTradeDate = null;

    const tradesWithRun = trades.map((t) => {
      const side = (t.side || "").toUpperCase();
      const q = toNum(t.qty);
      const p = toNum(t.price);
      const dateStr = String(t.date || "");

      if (!firstBuyDate && side === "BUY") firstBuyDate = dateStr;
      if (side === "BUY" && dateStr && firstBuyDate && dateStr < firstBuyDate) firstBuyDate = dateStr;
      if (!lastTradeDate || (dateStr && dateStr > lastTradeDate)) lastTradeDate = dateStr;

      if (Number.isFinite(q) && Number.isFinite(p) && q > 0 && p > 0) {
        if (side === "BUY") {
          const newQty = positionQty + q;
          avgCost = (avgCost * positionQty + p * q) / newQty;
          positionQty = newQty;
          totalBuyAmount += p * q;
        } else if (side === "SELL" && positionQty > 0) {
          const sellQty = Math.min(q, positionQty);
          realizedPnl += (p - avgCost) * sellQty;
          positionQty -= sellQty;
        }
      }

      return {
        trade_id: t.trade_id,
        date: t.date,
        side,
        qty: t.qty,
        price: t.price,
        amount: t.amount,
        fee: t.fee,
        memo: t.memo,
        running_qty: positionQty,
        running_avg_price: avgCost,
        running_realized_pnl: realizedPnl,
      };
    });

    let holding = null;
    if (positionQty > 0) {
      const currentPrice =
        rank && rank.close !== undefined && rank.close !== "" ? Number(rank.close) : null;
      const avgBuyPrice = avgCost > 0 ? avgCost : null;
      const costBasis =
        Number.isFinite(avgBuyPrice) && Number.isFinite(positionQty)
          ? avgBuyPrice * positionQty
          : null;
      const currentValue =
        Number.isFinite(currentPrice) && Number.isFinite(positionQty)
          ? currentPrice * positionQty
          : null;

      let unrealizedPnl = null;
      let unrealizedPnlPct = null;
      if (Number.isFinite(currentValue) && Number.isFinite(costBasis) && costBasis !== 0) {
        unrealizedPnl = currentValue - costBasis;
        unrealizedPnlPct = (currentValue / costBasis - 1) * 100;
      }

      let realizedPnlPct = null;
      if (Number.isFinite(realizedPnl) && totalBuyAmount > 0) {
        realizedPnlPct = (realizedPnl / totalBuyAmount) * 100;
      }

      holding = {
        code,
        name,
        market,
        sector,
        current_qty: positionQty,
        avg_buy_price: avgBuyPrice,
        current_price: currentPrice,
        current_value: currentValue,
        cost_basis: costBasis,
        unrealized_pnl: unrealizedPnl,
        unrealized_pnl_pct: unrealizedPnlPct,
        realized_pnl: realizedPnl,
        realized_pnl_pct: realizedPnlPct,
        live_score: rank ? getLiveScore(rank) : null,
        live_rank: rank ? getLiveRank(rank) : null,
        live_score_source: rank ? getLiveScoreSource(rank) : null,
        final_score: rank ? Number(rank.final_score) : null,
        first_buy_date: firstBuyDate,
        last_trade_date: lastTradeDate,
      };
    }

    res.json({ code, name, market, sector, latest: rank || null, holding, trades: tradesWithRun });
  } catch (e) {
    console.error("api/holding error", e);
    res.status(500).json({ error: "failed to build holding detail", detail: String(e) });
  }
});

app.get("/api/paper-trading/summary", async (req, res) => {
  try {
    const requestedRunId = toNum(req.query.paper_run_id);
    const runRow = Number.isFinite(requestedRunId)
      ? await getPaperTradingRunById(requestedRunId)
      : await getLatestPaperTradingRun();
    if (!runRow) {
      return res.status(404).json({ error: "paper trading run not found" });
    }

    const strategyRows = await queryRows(
      `
      WITH latest_nav AS (
        SELECT DISTINCT ON (strategy)
          strategy,
          date,
          nav,
          cumulative_return,
          drawdown,
          active_position_count,
          closed_trade_count,
          closed_trade_count_cum,
          closed_win_rate
        FROM research.paper_trading_nav
        WHERE paper_run_id = $1
        ORDER BY strategy, date DESC
      )
      SELECT *
      FROM latest_nav
      ORDER BY strategy
      `,
      [runRow.paper_run_id]
    );

    const openPositionRows = await queryRows(
      `
      SELECT strategy, COUNT(*) AS open_position_count
      FROM research.paper_trading_position
      WHERE paper_run_id = $1
        AND COALESCE(status, 'OPEN') = 'OPEN'
      GROUP BY strategy
      ORDER BY strategy
      `,
      [runRow.paper_run_id]
    );

    const openCountMap = new Map(
      openPositionRows.map((row) => [String(row.strategy || ""), toNum(row.open_position_count) || 0])
    );

    const strategies = strategyRows.map((row) => ({
      strategy: row.strategy,
      latest_date: toIsoDate(row.date),
      nav: toNum(row.nav),
      cumulative_return: toNum(row.cumulative_return),
      drawdown: toNum(row.drawdown),
      active_position_count: toNum(row.active_position_count),
      open_position_count: openCountMap.get(String(row.strategy || "")) || 0,
      closed_trade_count: toNum(row.closed_trade_count_cum ?? row.closed_trade_count),
      closed_win_rate: toNum(row.closed_win_rate),
    }));

    res.json({
      run: normalizePaperRun(runRow),
      strategies,
    });
  } catch (e) {
    console.error("GET /api/paper-trading/summary error", e);
    res.status(500).json({ error: "internal error" });
  }
});

app.get("/api/paper-trading/nav", async (req, res) => {
  try {
    const requestedRunId = toNum(req.query.paper_run_id);
    const strategy = String(req.query.strategy || "").trim();
    const limit = Math.max(1, Math.min(500, Number(req.query.limit) || 120));
    const runRow = Number.isFinite(requestedRunId)
      ? await getPaperTradingRunById(requestedRunId)
      : await getLatestPaperTradingRun();
    if (!runRow) {
      return res.status(404).json({ error: "paper trading run not found" });
    }

    const params = [runRow.paper_run_id];
    let strategyFilter = "";
    if (strategy) {
      params.push(strategy);
      strategyFilter = `AND strategy = $${params.length}`;
    }
    params.push(limit);

    const rows = await queryRows(
      `
      SELECT
        strategy,
        date,
        cash,
        market_value,
        nav,
        daily_return,
        active_position_count,
        opened_today,
        duplicate_skip_count,
        deployed_cash,
        cumulative_return,
        running_nav_max,
        drawdown,
        closed_trade_count,
        closed_win_rate,
        closed_win_count,
        closed_trade_count_cum,
        closed_win_count_cum
      FROM research.paper_trading_nav
      WHERE paper_run_id = $1
      ${strategyFilter}
      ORDER BY date DESC, strategy ASC
      LIMIT $${params.length}
      `,
      params
    );

    const items = rows
      .slice()
      .reverse()
      .map((row) => ({
        strategy: row.strategy,
        date: toIsoDate(row.date),
        cash: toNum(row.cash),
        market_value: toNum(row.market_value),
        nav: toNum(row.nav),
        daily_return: toNum(row.daily_return),
        cumulative_return: toNum(row.cumulative_return),
        drawdown: toNum(row.drawdown),
        active_position_count: toNum(row.active_position_count),
        opened_today: toNum(row.opened_today),
        duplicate_skip_count: toNum(row.duplicate_skip_count),
        deployed_cash: toNum(row.deployed_cash),
        closed_trade_count: toNum(row.closed_trade_count),
        closed_win_rate: toNum(row.closed_win_rate),
        closed_win_count: toNum(row.closed_win_count),
        closed_trade_count_cum: toNum(row.closed_trade_count_cum),
        closed_win_count_cum: toNum(row.closed_win_count_cum),
      }));

    res.json({
      run: normalizePaperRun(runRow),
      strategy: strategy || null,
      count: items.length,
      items,
    });
  } catch (e) {
    console.error("GET /api/paper-trading/nav error", e);
    res.status(500).json({ error: "internal error" });
  }
});

app.get("/api/paper-trading/positions", async (req, res) => {
  try {
    const requestedRunId = toNum(req.query.paper_run_id);
    const strategy = String(req.query.strategy || "").trim();
    const status = String(req.query.status || "").trim().toUpperCase();
    const limit = Math.max(1, Math.min(500, Number(req.query.limit) || 200));
    const runRow = Number.isFinite(requestedRunId)
      ? await getPaperTradingRunById(requestedRunId)
      : await getLatestPaperTradingRun();
    if (!runRow) {
      return res.status(404).json({ error: "paper trading run not found" });
    }

    const params = [runRow.paper_run_id];
    const filters = [];
    if (strategy) {
      params.push(strategy);
      filters.push(`strategy = $${params.length}`);
    }
    if (status) {
      params.push(status);
      filters.push(`COALESCE(status, 'OPEN') = $${params.length}`);
    }
    params.push(limit);
    const whereExtra = filters.length ? `AND ${filters.join(" AND ")}` : "";

    const rows = await queryRows(
      `
      SELECT
        strategy,
        code,
        name,
        entry_date,
        planned_exit_date,
        exit_date,
        entry_price_close,
        entry_exec_price,
        exit_price_close,
        exit_exec_price,
        shares,
        entry_notional_gross,
        exit_notional_net,
        entry_cost_amount,
        exit_cost_amount,
        gross_return,
        net_return,
        source_rank,
        selection_stage,
        dominant_theme,
        confidence_score,
        final_score,
        holding_age_trading_days,
        remaining_holding_days,
        holding_policy_code,
        entry_action_code,
        entry_action_reason,
        current_action_code,
        current_action_reason,
        exit_action_code,
        exit_action_reason,
        status
      FROM research.paper_trading_position
      WHERE paper_run_id = $1
      ${whereExtra}
      ORDER BY entry_date DESC, strategy ASC, code ASC
      LIMIT $${params.length}
      `,
      params
    );

    const latestRankByCode = await getRankingLatestByCode();

    const items = rows.map((row) => {
      const code = String(row.code || "").trim();
      const rankRow = latestRankByCode.get(code) || {};
      const currentPrice = toNum(rankRow.close);
      const shares = toNum(row.shares);
      const entryNotionalGross = toNum(row.entry_notional_gross);
      const entryCostAmount = toNum(row.entry_cost_amount);
      const currentValue =
        Number.isFinite(currentPrice) && Number.isFinite(shares) ? currentPrice * shares : null;
      const currentPnlAmount =
        Number.isFinite(currentValue) && Number.isFinite(entryNotionalGross)
          ? currentValue - entryNotionalGross - (Number.isFinite(entryCostAmount) ? entryCostAmount : 0)
          : null;
      const currentReturn =
        Number.isFinite(currentPnlAmount) &&
        Number.isFinite(entryNotionalGross) &&
        entryNotionalGross > 0
          ? currentPnlAmount / entryNotionalGross
          : null;
      const confidenceScore = getConfidenceScore(row);
      const liveScore = getLiveScore(rankRow) ?? toNum(row.final_score);
      const liveRank = getLiveRank(rankRow);
      const currentActionCode = row.current_action_code || null;
      const review = classifyPaperPositionReview({
        holdingAgeTradingDays: toNum(row.holding_age_trading_days),
        remainingHoldingDays: toNum(row.remaining_holding_days),
        currentReturn,
        liveScore,
        confidenceScore,
        liveRank,
        currentPrice,
        latestPriceDate: toIsoDate(rankRow.date),
        currentActionCode,
      });

      return {
        strategy: row.strategy,
        code,
        name: row.name || getName(code) || code,
        entry_date: toIsoDate(row.entry_date),
        planned_exit_date: toIsoDate(row.planned_exit_date),
        exit_date: toIsoDate(row.exit_date),
      entry_price_close: toNum(row.entry_price_close),
      entry_exec_price: toNum(row.entry_exec_price),
      exit_price_close: toNum(row.exit_price_close),
      exit_exec_price: toNum(row.exit_exec_price),
      shares: toNum(row.shares),
      entry_notional_gross: toNum(row.entry_notional_gross),
      exit_notional_net: toNum(row.exit_notional_net),
      entry_cost_amount: toNum(row.entry_cost_amount),
      exit_cost_amount: toNum(row.exit_cost_amount),
      gross_return: toNum(row.gross_return),
      net_return: toNum(row.net_return),
      source_rank: toNum(row.source_rank),
      selection_stage: row.selection_stage || null,
        dominant_theme: row.dominant_theme || null,
        confidence_score: confidenceScore,
        live_score: liveScore,
        live_rank: liveRank,
        live_score_source: getLiveScoreSource(rankRow),
        final_score: toNum(row.final_score),
        holding_age_trading_days: toNum(row.holding_age_trading_days),
        remaining_holding_days: toNum(row.remaining_holding_days),
        holding_policy_code: row.holding_policy_code || null,
        entry_action_code: row.entry_action_code || null,
        entry_action_reason: row.entry_action_reason || null,
        current_action_code: currentActionCode,
        current_action_reason: row.current_action_reason || null,
        exit_action_code: row.exit_action_code || null,
        exit_action_reason: row.exit_action_reason || null,
        status: row.status || "OPEN",
        current_price: currentPrice,
        current_value: currentValue,
        current_pnl_amount: currentPnlAmount,
        current_return: currentReturn,
        latest_price_date: toIsoDate(rankRow.date),
        system_review_status: review.system_review_status,
        system_review_label: review.system_review_label,
        system_review_priority: review.system_review_priority,
        system_review_reasons: review.system_review_reasons,
        system_action_note: review.system_action_note,
      };
    });

    res.json({
      run: normalizePaperRun(runRow),
      strategy: strategy || null,
      status: status || null,
      count: items.length,
      items,
    });
  } catch (e) {
    console.error("GET /api/paper-trading/positions error", e);
    res.status(500).json({ error: "internal error" });
  }
});

app.get("/api/rule/summary", async (req, res) => {
  try {
    const payload = await readRuleSummaryPayload();
    if (!payload.as_of_date && !payload.counts?.total_candidates) {
      return res.status(404).json({ error: "rule artifacts not found" });
    }
    res.json(payload);
  } catch (e) {
    console.error("GET /api/rule/summary error", e);
    res.status(500).json({ error: "internal error" });
  }
});

app.get("/api/rule/signals/latest", async (req, res) => {
  try {
    const limit = Math.max(1, Math.min(300, Number(req.query.limit) || 100));
    const strength = String(req.query.strength || "").trim().toLowerCase();
    const payload = await readRuleSignalsPayload(strength, limit);
    if (!payload.as_of_date || !payload.items.length) {
      return res.status(404).json({ error: "rule signals not found" });
    }
    res.json(payload);
  } catch (e) {
    console.error("GET /api/rule/signals/latest error", e);
    res.status(500).json({ error: "internal error" });
  }
});

app.get("/api/rule/portfolio-plan", async (req, res) => {
  try {
    const payload = await readRulePortfolioPlanPayload();
    if (!payload.as_of_date && !Array.isArray(payload.items)) {
      return res.status(404).json({ error: "rule portfolio plan not found" });
    }
    res.json(payload);
  } catch (e) {
    console.error("GET /api/rule/portfolio-plan error", e);
    res.status(500).json({ error: "internal error" });
  }
});

app.get("/api/rule/order-preview", async (req, res) => {
  try {
    const payload = await readRuleOrderPreviewPayload();
    if (!payload.as_of_date && !Array.isArray(payload.items)) {
      return res.status(404).json({ error: "rule order preview not found" });
    }
    res.json(payload);
  } catch (e) {
    console.error("GET /api/rule/order-preview error", e);
    res.status(500).json({ error: "internal error" });
  }
});

app.get("/api/rule/paper-state", async (req, res) => {
  try {
    const payload = await readRulePaperStatePayload();
    if (!payload.as_of_date && !Array.isArray(payload.positions)) {
      return res.status(404).json({ error: "rule paper state not found" });
    }
    res.json(payload);
  } catch (e) {
    console.error("GET /api/rule/paper-state error", e);
    res.status(500).json({ error: "internal error" });
  }
});

app.get("/api/rule/backtest-summary", async (req, res) => {
  try {
    const payload = await readRuleBacktestPayload();
    if (!payload.latest_signal_date && !payload.generated_at) {
      return res.status(404).json({ error: "rule backtest summary not found" });
    }
    res.json(payload);
  } catch (e) {
    console.error("GET /api/rule/backtest-summary error", e);
    res.status(500).json({ error: "internal error" });
  }
});

app.get("/api/rule/execution-results", async (req, res) => {
  try {
    const payload = await readRuleExecutionResultsPayload();
    if (!payload.generated_at && !Array.isArray(payload.items)) {
      return res.status(404).json({ error: "rule execution results not found" });
    }
    res.json(payload);
  } catch (e) {
    console.error("GET /api/rule/execution-results error", e);
    res.status(500).json({ error: "internal error" });
  }
});

app.get("/api/rule/execution-history", async (req, res) => {
  try {
    const historyPath = path.join(OUTPUTS_DIR, "rule_execution_history.jsonl");
    if (!fs.existsSync(historyPath)) {
      return res.status(404).json({ error: "rule execution history not found" });
    }
    const limit = Math.max(1, Math.min(100, Number(req.query.limit) || 20));
    const lines = fs.readFileSync(historyPath, "utf-8")
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);
    const items = lines.slice(-limit).map((line) => {
      try {
        return JSON.parse(line);
      } catch {
        return null;
      }
    }).filter(Boolean);
    res.json({ count: items.length, items });
  } catch (e) {
    console.error("GET /api/rule/execution-history error", e);
    res.status(500).json({ error: "internal error" });
  }
});

app.get("/api/live-account/summary", async (req, res) => {
  try {
    const summary = await readJsonPayloadDbFirst("live_account_balance_summary", [path.join(OUTPUTS_DIR, "live_account_balance_summary.json")]);
    const preview = await readJsonPayloadDbFirst("live_order_preview", [path.join(OUTPUTS_DIR, "live_order_preview.json")]);
    const execution = await readJsonPayloadDbFirst("order_requests_execution", [path.join(OUTPUTS_DIR, "order_requests_execution.json")]);
    const holdingsPayload = await readJsonPayloadDbFirst("live_account_holdings");
    const holdings = Array.isArray(holdingsPayload?.items) ? holdingsPayload.items : (readCsv(path.join(DATA_DIR, "live_account_holdings.csv")) || []);
    const visibleHoldingCount = holdings.filter((row) => Number(toNum(row.qty)) > 0).length;
    if (!summary && !preview && !execution && !visibleHoldingCount) {
      return res.status(404).json({ error: "live account artifacts not found" });
    }
    res.json({
      summary: summary || null,
      holding_count: visibleHoldingCount,
      order_preview_count: Array.isArray(preview?.items) ? preview.items.length : 0,
      preview_gate_status: preview?.gate_status || null,
      preview_gate_display_status: preview?.gate_display_status || preview?.gate_status || null,
      order_execution_count: Array.isArray(execution?.items) ? execution.items.length : 0,
      last_execution_at: execution?.executed_at || null,
      last_execution_submitted_count: execution?.summary?.submitted_count ?? null,
    });
  } catch (e) {
    console.error("GET /api/live-account/summary error", e);
    res.status(500).json({ error: "internal error" });
  }
});

app.get("/api/live-account/holdings", async (req, res) => {
  try {
    const payload = await readJsonPayloadDbFirst("live_account_holdings");
    const rows = Array.isArray(payload?.items) ? payload.items : (readCsv(path.join(DATA_DIR, "live_account_holdings.csv")) || []);
    const items = rows.map((row) => ({
      code: String(row.code || "").trim(),
      name: row.name || getName(String(row.code || "").trim()) || null,
      qty: toNum(row.qty),
      avg_price: toNum(row.avg_price),
      current_price: toNum(row.current_price),
      eval_amount: toNum(row.eval_amount),
      pnl_amount: toNum(row.pnl_amount),
      pnl_pct: toNum(row.pnl_pct),
      weight: toNum(row.weight),
      status: row.status || "OPEN",
    })).filter((row) => Number(row.qty) > 0);
    if (!items.length) {
      return res.status(404).json({ error: "live account holdings not found" });
    }
    res.json({ count: items.length, items });
  } catch (e) {
    console.error("GET /api/live-account/holdings error", e);
    res.status(500).json({ error: "internal error" });
  }
});

app.get("/api/live-account/order-preview", async (req, res) => {
  try {
    const preview = await readJsonPayloadDbFirst("live_order_preview", [path.join(OUTPUTS_DIR, "live_order_preview.json")]);
    if (!preview) {
      return res.status(404).json({ error: "live order preview not found" });
    }
    res.json(preview);
  } catch (e) {
    console.error("GET /api/live-account/order-preview error", e);
    res.status(500).json({ error: "internal error" });
  }
});

app.get("/api/trade-intents", async (req, res) => {
  try {
    const payload = await readJsonPayloadDbFirst("trade_intents", [path.join(OUTPUTS_DIR, "trade_intents.json")]);
    if (!payload) {
      return res.status(404).json({ error: "trade intents not found" });
    }
    res.json(payload);
  } catch (e) {
    console.error("GET /api/trade-intents error", e);
    res.status(500).json({ error: "internal error" });
  }
});

app.get("/api/order-requests-preview", async (req, res) => {
  try {
    const payload = await readJsonPayloadDbFirst("order_requests_preview", [path.join(OUTPUTS_DIR, "order_requests_preview.json")]);
    if (!payload) {
      return res.status(404).json({ error: "order requests preview not found" });
    }
    res.json(payload);
  } catch (e) {
    console.error("GET /api/order-requests-preview error", e);
    res.status(500).json({ error: "internal error" });
  }
});

app.get("/api/order-requests-execution", async (req, res) => {
  try {
    const payload = await readJsonPayloadDbFirst("order_requests_execution", [path.join(OUTPUTS_DIR, "order_requests_execution.json")]);
    if (!payload) {
      return res.status(404).json({ error: "order requests execution not found" });
    }
    res.json(payload);
  } catch (e) {
    console.error("GET /api/order-requests-execution error", e);
    res.status(500).json({ error: "internal error" });
  }
});

app.get("/api/live-trade-consistency", async (req, res) => {
  try {
    const payload = await readJsonPayloadDbFirst("live_trade_consistency_report", [
      path.join(OUTPUTS_DIR, "live_trade_consistency_report.json"),
    ]);
    if (!payload || !Object.keys(payload).length) {
      return res.status(404).json({ error: "live trade consistency report not found" });
    }
    res.json(payload);
  } catch (e) {
    console.error("GET /api/live-trade-consistency error", e);
    res.status(500).json({ error: "internal error" });
  }
});

app.get("/api/live-trade-review-report", async (req, res) => {
  try {
    const payload = await readJsonPayloadDbFirst("live_trade_review_report", [
      path.join(OUTPUTS_DIR, "live_trade_review_report.json"),
    ]);
    if (!payload || !Object.keys(payload).length) {
      return res.status(404).json({ error: "live trade review report not found" });
    }
    res.json(payload);
  } catch (e) {
    console.error("GET /api/live-trade-review-report error", e);
    res.status(500).json({ error: "internal error" });
  }
});

app.get("/api/live-trade-review-summary", async (req, res) => {
  try {
    const payload = await readJsonPayloadDbFirst("live_trade_review_summary", [
      path.join(OUTPUTS_DIR, "live_trade_review_summary.json"),
    ]);
    if (!payload || !Object.keys(payload).length) {
      return res.status(404).json({ error: "live trade review summary not found" });
    }
    res.json(payload);
  } catch (e) {
    console.error("GET /api/live-trade-review-summary error", e);
    res.status(500).json({ error: "internal error" });
  }
});

app.get("/api/live-kpi-daily-report", async (req, res) => {
  try {
    const payload = await readJsonPayloadDbFirst("live_kpi_daily_report", [
      path.join(OUTPUTS_DIR, "live_kpi_daily_report.json"),
    ]);
    if (!payload || !Object.keys(payload).length) {
      return res.status(404).json({ error: "live KPI daily report not found" });
    }
    res.json(payload);
  } catch (e) {
    console.error("GET /api/live-kpi-daily-report error", e);
    res.status(500).json({ error: "internal error" });
  }
});

app.get("/api/quality-risk-guard-live-review", async (req, res) => {
  try {
    const payload = await readJsonPayloadDbFirst("quality_risk_guard_live_review", [
      path.join(OUTPUTS_DIR, "quality_risk_guard_live_review.json"),
    ]);
    if (!payload || !Object.keys(payload).length) {
      return res.status(404).json({ error: "quality risk guard live review not found" });
    }
    res.json(payload);
  } catch (e) {
    console.error("GET /api/quality-risk-guard-live-review error", e);
    res.status(500).json({ error: "internal error" });
  }
});

app.get("/api/live-closed-trade-report", async (req, res) => {
  try {
    const payload = await readJsonPayloadDbFirst("live_closed_trade_report", [
      path.join(OUTPUTS_DIR, "live_closed_trade_report.json"),
    ]);
    if (!payload || !Object.keys(payload).length) {
      return res.status(404).json({ error: "live closed trade report not found" });
    }
    res.json(payload);
  } catch (e) {
    console.error("GET /api/live-closed-trade-report error", e);
    res.status(500).json({ error: "internal error" });
  }
});

app.get("/api/live-quality-guard-output-check", async (req, res) => {
  try {
    const payload = await readJsonPayloadDbFirst("live_quality_guard_output_check", [
      path.join(OUTPUTS_DIR, "live_quality_guard_output_check.json"),
    ]);
    if (!payload || !Object.keys(payload).length) {
      return res.status(404).json({ error: "live quality guard output check not found" });
    }
    res.json(payload);
  } catch (e) {
    console.error("GET /api/live-quality-guard-output-check error", e);
    res.status(500).json({ error: "internal error" });
  }
});

app.get("/api/watch-auto-buy-simulation", async (req, res) => {
  try {
    const payload = await readJsonPayloadDbFirst("watch_auto_buy_simulation", [path.join(OUTPUTS_DIR, "watch_auto_buy_simulation.json")]);
    if (!payload) {
      return res.status(404).json({ error: "watch auto buy simulation not found" });
    }
    res.json(payload);
  } catch (e) {
    console.error("GET /api/watch-auto-buy-simulation error", e);
    res.status(500).json({ error: "internal error" });
  }
});

app.get("/api/auto-trading/runtime-status", async (req, res) => {
  try {
    const policyPayload = await readJsonPayloadDbFirst("auto_trading_policy", [path.join(OUTPUTS_DIR, "auto_trading_policy.json")]);
    const envPolicy = {
      auto_trade_execute: ["1", "true", "yes", "on"].includes(String(process.env.AUTO_TRADE_EXECUTE || "").toLowerCase()),
      auto_trade_allow_buy: ["1", "true", "yes", "on"].includes(String(process.env.AUTO_TRADE_ALLOW_BUY || "").toLowerCase()),
      buy_approval_required: !["0", "false", "no", "off"].includes(String(process.env.AUTO_TRADE_BUY_APPROVAL_REQUIRED || "1").toLowerCase()),
      confirm_configured: String(process.env.AUTO_TRADE_CONFIRM_TEXT || "").trim() === "LIVE_ORDER",
      source: "process_env",
    };
    const payload = {
      close_scheduler: await readJsonPayloadDbFirst("auto_ops_scheduler_status", [path.join(OUTPUTS_DIR, "auto_ops_scheduler_status.json")]),
      intraday_scheduler: await readJsonPayloadDbFirst("auto_ops_recovery_scheduler_status", [path.join(OUTPUTS_DIR, "auto_ops_recovery_scheduler_status.json")]),
      auto_buy_scheduler: await readJsonPayloadDbFirst("auto_ops_auto_buy_scheduler_status", [path.join(OUTPUTS_DIR, "auto_ops_auto_buy_scheduler_status.json")]),
      live_account_sync_scheduler: await readJsonPayloadDbFirst("auto_ops_live_account_sync_scheduler_status", [path.join(OUTPUTS_DIR, "auto_ops_live_account_sync_scheduler_status.json")]),
      policy: (policyPayload && Object.keys(policyPayload).length ? {
        auto_trade_execute: !!policyPayload.auto_trade_execute,
        auto_trade_allow_buy: !!policyPayload.auto_trade_allow_buy,
        buy_approval_required: typeof policyPayload.buy_approval_required === "boolean" ? policyPayload.buy_approval_required : true,
        confirm_configured: !!policyPayload.confirm_configured,
        source: policyPayload.source || "payload_store",
        generated_at: policyPayload.generated_at || null,
      } : envPolicy),
    };
    if (!payload.close_scheduler && !payload.intraday_scheduler && !payload.auto_buy_scheduler && !payload.live_account_sync_scheduler) {
      return res.status(404).json({ error: "auto trading runtime status not found" });
    }
    res.json(payload);
  } catch (e) {
    console.error("GET /api/auto-trading/runtime-status error", e);
    res.status(500).json({ error: "internal error" });
  }
});

app.get("/api/manual-trading/summary", async (req, res) => {
  try {
    const payload = await buildManualTradingSummary();
    res.json(payload);
  } catch (e) {
    console.error("GET /api/manual-trading/summary error", e);
    res.status(500).json({ error: "internal error" });
  }
});

app.get("/api/trading-policy", async (req, res) => {
  try {
    const payload = await buildTradingPolicySummary();
    res.json(payload);
  } catch (e) {
    console.error("GET /api/trading-policy error", e);
    res.status(500).json({ error: "internal error" });
  }
});

app.get("/api/ops-readiness", operatorAccess.apiGuard, async (req, res) => {
  try {
    const payload = await buildOpsReadinessSummary();
    res.json(payload);
  } catch (e) {
    console.error("GET /api/ops-readiness error", e);
    res.status(500).json({ error: "internal error" });
  }
});

app.get("/api/analytics/summary", operatorAccess.apiGuard, async (req, res) => {
  try {
    const payload = await buildVisitorAnalyticsSummary();
    res.json(payload);
  } catch (e) {
    console.error("GET /api/analytics/summary error", e);
    res.status(500).json({ error: "internal error" });
  }
});

// Debug data dir
app.get("/api/debug/data-dir", (req, res) => {
  res.json({ DATA_DIR });
});

app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
