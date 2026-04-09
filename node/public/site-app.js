(function () {
  const items = Array.isArray(window.SiteLibrary) ? window.SiteLibrary : [];

  function formatDate(value) {
    const text = String(value || "").trim();
    return text || "-";
  }

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (m) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[m]));
  }

  function itemHref(item) {
    return item.section === "report" ? `/reports/${item.slug}` : `/blog/${item.slug}`;
  }

  function readingMinutesValue(item) {
    const match = String(item.readingTime || "").match(/\d+/);
    return match ? Number(match[0]) : 0;
  }

  function summarizeExcerpt(text) {
    const excerpt = String(text || "").trim();
    return excerpt.length > 84 ? `${excerpt.slice(0, 84).trim()}...` : excerpt;
  }

  function sectionLabel(item) {
    return item.section === "report" ? "시장 해설" : "블로그";
  }

  function buildCard(item, options = {}) {
    const featured = options.featured || item.featured;
    const toneClass = item.section === "report" ? "article-card--report" : "article-card--blog";
    return `
      <article class="article-card ${toneClass}">
        ${featured ? '<span class="article-card__badge">추천 글</span>' : ""}
        <div class="article-card__meta">
          <span>${escapeHtml(sectionLabel(item))}</span>
          <span>${escapeHtml(item.category)}</span>
          <span>${escapeHtml(formatDate(item.date))}</span>
        </div>
        <h3>${escapeHtml(item.title)}</h3>
        <p class="article-card__excerpt">${escapeHtml(item.excerpt)}</p>
        <div class="article-card__footer">
          <span class="article-card__time">${escapeHtml(item.readingTime || "읽기 시간 미정")}</span>
          <a class="article-card__link" href="${itemHref(item)}">자세히 읽기</a>
        </div>
      </article>
    `;
  }

  function buildFeaturedMini(item) {
    return `
      <a class="featured-mini" href="${itemHref(item)}">
        <span class="featured-mini__section">${escapeHtml(sectionLabel(item))}</span>
        <strong>${escapeHtml(item.title)}</strong>
        <span>${escapeHtml(item.category)} · ${escapeHtml(item.readingTime || "-")}</span>
      </a>
    `;
  }

  function recommendedStarterSlugs(pageType) {
    if (pageType === "report") {
      return [
        "weekly-market-regime-neutral-example",
        "what-walkforward-acceptance-means",
        "case-study-good-score-but-no-entry",
      ];
    }
    if (pageType === "blog") {
      return [
        "how-this-site-analyzes-korean-stocks",
        "why-score-is-not-buy-signal",
        "difference-between-watchlist-and-buy-allowed",
      ];
    }
    return [];
  }

  function updateListHeroStats(filtered) {
    const totalEl = document.getElementById("heroTotalCount");
    const topCategoryEl = document.getElementById("heroTopCategory");
    if (totalEl) totalEl.textContent = `${filtered.length}개`;
    if (topCategoryEl) {
      const counts = new Map();
      filtered.forEach((item) => counts.set(item.category, (counts.get(item.category) || 0) + 1));
      const top = [...counts.entries()].sort((a, b) => b[1] - a[1])[0];
      topCategoryEl.textContent = top ? `${top[0]} ${top[1]}건` : "-";
    }
  }

  function renderHome() {
    const featuredEl = document.getElementById("featuredGrid");
    if (!featuredEl) return;
    featuredEl.innerHTML = items.filter((item) => item.featured).slice(0, 6).map((item) => buildCard(item, { featured: true })).join("");
    document.getElementById("latestReports").innerHTML = items.filter((item) => item.section === "report").slice(0, 3).map(buildCard).join("");
    document.getElementById("latestPosts").innerHTML = items.filter((item) => item.section === "blog").slice(0, 3).map(buildCard).join("");
  }

  function renderList() {
    const listEl = document.getElementById("contentList");
    if (!listEl) return;
    const pageType = document.body.dataset.pageType || "all";
    const labelEl = document.getElementById("listTitleNote");
    const filterButtons = Array.from(document.querySelectorAll("[data-filter]"));
    const featuredStripEl = document.getElementById("featuredStrip");
    const starterStripEl = document.getElementById("starterStrip");
    const starterTitleEl = document.getElementById("starterTitle");
    const starterDescEl = document.getElementById("starterDesc");
    let activeFilter = "all";

    if (starterTitleEl && starterDescEl) {
      if (pageType === "report") {
        starterTitleEl.textContent = "처음 읽을 시장 해설";
        starterDescEl.textContent = "오늘 시장 흐름과 운영 판단을 이해하는 데 먼저 도움이 되는 글 3개입니다.";
      } else if (pageType === "blog") {
        starterTitleEl.textContent = "처음 읽을 블로그";
        starterDescEl.textContent = "용어와 판단 기준을 가장 빠르게 이해하는 데 도움이 되는 블로그 3개입니다.";
      }
    }

    if (starterStripEl) {
      const starterItems = recommendedStarterSlugs(pageType)
        .map((slug) => items.find((item) => item.slug === slug))
        .filter(Boolean);
      starterStripEl.innerHTML = starterItems.length
        ? starterItems.map(buildFeaturedMini).join("")
        : '<article class="article-card"><h3>추천 글을 준비 중입니다</h3><p class="article-card__excerpt">곧 처음 읽기 좋은 글을 따로 정리해 드리겠습니다.</p></article>';
    }

    function filterItems(filter) {
      return items.filter((item) => {
        if (pageType === "report") return filter === "all" ? item.section === "report" : item.section === "report" && item.category === filter;
        if (pageType === "blog") return filter === "all" ? item.section === "blog" : item.section === "blog" && item.category === filter;
        return filter === "all" ? true : item.category === filter;
      });
    }

    function applyFilter(filter) {
      activeFilter = filter;
      filterButtons.forEach((btn) => btn.classList.toggle("is-active", btn.dataset.filter === filter));
      const filtered = filterItems(filter);
      if (labelEl) {
        const avgRead = filtered.length ? Math.round(filtered.reduce((acc, item) => acc + readingMinutesValue(item), 0) / filtered.length) : 0;
        labelEl.textContent = `${filtered.length}개 글 · 평균 읽기 ${avgRead || "-"}분`;
      }
      updateListHeroStats(filtered);
      if (featuredStripEl) {
        const featured = filtered.filter((item) => item.featured).slice(0, 3);
        featuredStripEl.style.display = featured.length ? "grid" : "none";
        featuredStripEl.innerHTML = featured.length ? featured.map(buildFeaturedMini).join("") : "";
      }
      listEl.innerHTML = filtered.length
        ? filtered.map((item) => buildCard(item, { featured: item.featured && filter === "all" })).join("")
        : `<article class="article-card"><h3>아직 준비 중입니다</h3><p class="article-card__excerpt">해당 조건에 맞는 공개 글이 아직 없습니다. 다른 분류를 먼저 확인해 주세요.</p></article>`;
    }

    filterButtons.forEach((btn) => btn.addEventListener("click", () => applyFilter(btn.dataset.filter)));
    applyFilter(activeFilter);
  }

  function renderDetail() {
    const detailRoot = document.getElementById("articleDetail");
    if (!detailRoot) return;
    const slug = document.body.dataset.slug || "";
    const item = items.find((entry) => entry.slug === slug);
    if (!item) {
      detailRoot.innerHTML = `<article class="article-shell"><h1>글을 찾을 수 없습니다</h1><p class="article-shell__excerpt">요청한 콘텐츠가 아직 준비되지 않았거나 주소가 잘못되었습니다.</p></article>`;
      return;
    }

    detailRoot.innerHTML = `
      <article class="article-shell">
        <div class="article-shell__hero">
          <div class="article-meta">
            <span>${escapeHtml(sectionLabel(item))}</span>
            <span>${escapeHtml(item.category)}</span>
            <span>${escapeHtml(formatDate(item.date))}</span>
            <span>${escapeHtml(item.readingTime || "-")}</span>
          </div>
          <h1>${escapeHtml(item.title)}</h1>
          <p class="article-shell__excerpt">${escapeHtml(item.excerpt)}</p>
          <div class="article-shell__chips">
            <span class="chip chip--accent">핵심 주제 ${escapeHtml(item.category)}</span>
            <span class="chip">${escapeHtml(sectionLabel(item))}</span>
            <span class="chip chip--good">읽기 ${escapeHtml(item.readingTime || "-")}</span>
          </div>
        </div>
        <div class="article-body">${item.body}</div>
      </article>
    `;

    const breadcrumbEl = document.getElementById("detailBreadcrumb");
    if (breadcrumbEl) {
      breadcrumbEl.innerHTML = `
        <a href="/">홈</a>
        <span>/</span>
        <a href="${item.section === "report" ? "/reports" : "/blog"}">${escapeHtml(sectionLabel(item))}</a>
        <span>/</span>
        <strong>${escapeHtml(item.title)}</strong>
      `;
    }

    const summaryEl = document.getElementById("articleSummaryText");
    if (summaryEl) {
      summaryEl.textContent = summarizeExcerpt(item.excerpt);
    }

    const metaEl = document.getElementById("articleMeta");
    if (metaEl) {
      metaEl.innerHTML = `
        <li>분류: ${escapeHtml(sectionLabel(item))}</li>
        <li>주제: ${escapeHtml(item.category)}</li>
        <li>발행일: ${escapeHtml(formatDate(item.date))}</li>
        <li>읽기 시간: ${escapeHtml(item.readingTime || "-")}</li>
      `;
    }

    const relatedEl = document.getElementById("relatedList");
    if (relatedEl) {
      const related = items
        .filter((entry) => entry.slug !== item.slug && entry.section === item.section)
        .slice(0, 4);
      relatedEl.innerHTML = related.length
        ? related.map((entry) => `<li><a class="text-link" href="${itemHref(entry)}">${escapeHtml(entry.title)}</a></li>`).join("")
        : "<li>같은 분류의 글이 아직 많지 않습니다.</li>";
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    renderHome();
    renderList();
    renderDetail();
  });
})();
