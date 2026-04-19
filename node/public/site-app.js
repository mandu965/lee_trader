(function () {
  let items = Array.isArray(window.SiteLibrary) ? window.SiteLibrary : [];
  let homepageContent = null;

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (match) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[match]));
  }

  function itemHref(item) {
    return item.section === "report" ? `/reports/${item.slug}` : `/blog/${item.slug}`;
  }

  function sectionLabel(item) {
    return item.section === "report" ? "시장 해설" : "블로그";
  }

  function readingMinutesValue(item) {
    const match = String(item.readingTime || "").match(/\d+/);
    return match ? Number(match[0]) : 0;
  }

  function summarizeExcerpt(text) {
    const excerpt = String(text || "").trim();
    return excerpt.length > 96 ? `${excerpt.slice(0, 96).trim()}...` : excerpt;
  }

  function buildCard(item, options = {}) {
    const toneClass = item.section === "report" ? "article-card--report" : "article-card--blog";
    const featured = options.featured || item.featured;
    return `
      <article class="article-card ${toneClass}">
        ${featured ? '<span class="article-card__badge">추천 글</span>' : ""}
        <div class="article-card__meta">
          <span>${escapeHtml(sectionLabel(item))}</span>
          <span>${escapeHtml(item.category || "")}</span>
          <span>${escapeHtml(item.date || "-")}</span>
        </div>
        <h3>${escapeHtml(item.title)}</h3>
        <p class="article-card__excerpt">${escapeHtml(item.excerpt || "")}</p>
        <div class="article-card__footer">
          <span class="article-card__time">${escapeHtml(item.readingTime || "읽는 시간 미정")}</span>
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
        <span>${escapeHtml(item.category || "-")} · ${escapeHtml(item.readingTime || "-")}</span>
      </a>
    `;
  }

  function buildPickCard(pick) {
    const toneClass = pick.tag_tone === "accent" ? "chip chip--accent" : pick.tag_tone === "good" ? "chip chip--good" : "chip";
    return `
      <article class="pick-card">
        <div class="pick-card__head">
          <div>
            <p class="pick-card__ticker">${escapeHtml(pick.ticker || "-")}</p>
            <h3>${escapeHtml(pick.name || "-")}</h3>
          </div>
          <span class="${toneClass}">${escapeHtml(pick.tag || "관찰 후보")}</span>
        </div>
        <p class="pick-card__lead">${escapeHtml(pick.lead || "")}</p>
        <ul class="pick-card__reasons">
          ${(Array.isArray(pick.reasons) ? pick.reasons : []).map((reason) => `<li>${escapeHtml(reason)}</li>`).join("")}
        </ul>
        <p class="pick-card__risk">유의사항: ${escapeHtml(pick.risk || "-")}</p>
      </article>
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

  function renderHomeSummary() {
    const summaryRoot = document.getElementById("homeSummary");
    const picksRoot = document.getElementById("homePicks");
    const dateEl = document.getElementById("marketSummaryDate");
    if (!summaryRoot || !picksRoot || !homepageContent) return;

    const summary = homepageContent.marketSummary || {};
    const picks = Array.isArray(homepageContent.picks) ? homepageContent.picks : [];

    if (dateEl && summary.as_of_date) {
      dateEl.textContent = `기준일 ${summary.as_of_date}`;
    }

    summaryRoot.innerHTML = `
      <article class="summary-card summary-card--focus">
        <span class="home-summary__badge">핵심 결론</span>
        <h3>${escapeHtml(summary.headline || "오늘 시장 요약을 준비 중입니다")}</h3>
        <p>${escapeHtml(summary.summary || "최신 요약 문구를 곧 반영하겠습니다.")}</p>
      </article>
      <article class="summary-card">
        <h3>${escapeHtml(summary.market_mood_title || "시장 분위기")}</h3>
        <p>${escapeHtml(summary.market_mood || "-")}</p>
      </article>
      <article class="summary-card">
        <h3>${escapeHtml(summary.watchpoints_title || "오늘의 관찰 포인트")}</h3>
        <p>${escapeHtml(summary.watchpoints || "-")}</p>
      </article>
    `;

    picksRoot.innerHTML = picks.length
      ? picks.map(buildPickCard).join("")
      : '<article class="pick-card"><h3>추천 종목 준비 중</h3><p class="pick-card__lead">운영 데이터가 준비되면 이 영역에 자동 반영됩니다.</p></article>';
  }

  function renderHome() {
    const featuredEl = document.getElementById("featuredGrid");
    const reportEl = document.getElementById("latestReports");
    const blogEl = document.getElementById("latestPosts");
    const studyEl = document.getElementById("studyPosts");
    const reports = items.filter((item) => item.section === "report");
    const posts = items.filter((item) => item.section === "blog");

    renderHomeSummary();

    if (featuredEl) {
      featuredEl.innerHTML = items.filter((item) => item.featured).slice(0, 6).map((item) => buildCard(item, { featured: true })).join("");
    }
    if (reportEl) reportEl.innerHTML = reports.slice(0, 3).map(buildCard).join("");
    if (blogEl) blogEl.innerHTML = posts.slice(0, 3).map(buildCard).join("");
    if (studyEl) studyEl.innerHTML = posts.slice(0, 3).map(buildCard).join("");
  }

  function renderList() {
    const listEl = document.getElementById("contentList");
    if (!listEl) return;

    const pageType = document.body.dataset.pageType || "all";
    const labelEl = document.getElementById("listTitleNote");
    const filtersRoot = document.getElementById("listFilters");
    const featuredStripEl = document.getElementById("featuredStrip");
    const starterStripEl = document.getElementById("starterStrip");
    const starterTitleEl = document.getElementById("starterTitle");
    const starterDescEl = document.getElementById("starterDesc");
    const scopedItems = items.filter((item) => (pageType === "all" ? true : item.section === pageType));

    if (filtersRoot && !filtersRoot.children.length) {
      const categories = [...new Set(scopedItems.map((item) => item.category).filter(Boolean))];
      filtersRoot.innerHTML = [`<button class="filter-pill is-active" data-filter="all">전체</button>`]
        .concat(categories.map((category) => `<button class="filter-pill" data-filter="${category}">${category}</button>`))
        .join("");
    }

    const filterButtons = Array.from(document.querySelectorAll("[data-filter]"));
    let activeFilter = "all";

    if (starterTitleEl && starterDescEl) {
      if (pageType === "report") {
        starterTitleEl.textContent = "처음 읽을 시장 해설";
        starterDescEl.textContent = "시장 국면, 검증 지표, 실제 운영 사례를 함께 읽으면 오늘의 맥락이 훨씬 선명해집니다.";
      } else if (pageType === "blog") {
        starterTitleEl.textContent = "처음 읽을 블로그";
        starterDescEl.textContent = "점수 읽는 법, 리스크 관리, 서비스 목적을 먼저 이해하면 나머지 콘텐츠도 훨씬 쉽게 읽힙니다.";
      }
    }

    if (starterStripEl) {
      const starterItems = recommendedStarterSlugs(pageType)
        .map((slug) => items.find((item) => item.slug === slug))
        .filter(Boolean);
      starterStripEl.innerHTML = starterItems.length
        ? starterItems.map(buildFeaturedMini).join("")
        : '<article class="article-card"><h3>추천 글 준비 중</h3><p class="article-card__excerpt">선별해서 먼저 읽기 좋은 글을 계속 보강하고 있습니다.</p></article>';
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
      filterButtons.forEach((button) => button.classList.toggle("is-active", button.dataset.filter === filter));
      const filtered = filterItems(filter);
      const avgRead = filtered.length
        ? Math.round(filtered.reduce((sum, item) => sum + readingMinutesValue(item), 0) / filtered.length)
        : 0;
      if (labelEl) labelEl.textContent = `${filtered.length}개 글 · 평균 읽는 시간 ${avgRead || "-"}분`;
      updateListHeroStats(filtered);
      if (featuredStripEl) {
        const featured = filtered.filter((item) => item.featured).slice(0, 3);
        featuredStripEl.style.display = featured.length ? "grid" : "none";
        featuredStripEl.innerHTML = featured.length ? featured.map(buildFeaturedMini).join("") : "";
      }
      listEl.innerHTML = filtered.length
        ? filtered.map((item) => buildCard(item, { featured: item.featured && filter === "all" })).join("")
        : '<article class="article-card"><h3>해당 조건의 글이 없습니다</h3><p class="article-card__excerpt">다른 카테고리를 선택하거나 전체 목록으로 다시 보세요.</p></article>';
    }

    filterButtons.forEach((button) => button.addEventListener("click", () => applyFilter(button.dataset.filter)));
    applyFilter(activeFilter);
  }

  function renderDetail() {
    const detailRoot = document.getElementById("articleDetail");
    if (!detailRoot) return;

    const slug = document.body.dataset.slug || "";
    const item = items.find((entry) => entry.slug === slug);
    if (!item) {
      detailRoot.innerHTML = '<article class="article-shell"><h1>글을 찾을 수 없습니다</h1><p class="article-shell__excerpt">주소가 잘못되었거나 게시물이 이동되었습니다.</p></article>';
      return;
    }

    detailRoot.innerHTML = `
      <article class="article-shell">
        <div class="article-shell__hero">
          <div class="article-meta">
            <span>${escapeHtml(sectionLabel(item))}</span>
            <span>${escapeHtml(item.category || "-")}</span>
            <span>${escapeHtml(item.date || "-")}</span>
            <span>${escapeHtml(item.readingTime || "-")}</span>
          </div>
          <h1>${escapeHtml(item.title)}</h1>
          <p class="article-shell__excerpt">${escapeHtml(item.excerpt || "")}</p>
          <div class="article-shell__chips">
            <span class="chip chip--accent">주제 ${escapeHtml(item.category || "-")}</span>
            <span class="chip">${escapeHtml(sectionLabel(item))}</span>
            <span class="chip chip--good">${escapeHtml(item.readingTime || "-")}</span>
          </div>
        </div>
        <div class="article-body">${item.body || ""}</div>
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
    if (summaryEl) summaryEl.textContent = summarizeExcerpt(item.excerpt);

    const metaEl = document.getElementById("articleMeta");
    if (metaEl) {
      metaEl.innerHTML = `
        <li>분류: ${escapeHtml(sectionLabel(item))}</li>
        <li>주제: ${escapeHtml(item.category || "-")}</li>
        <li>발행일: ${escapeHtml(item.date || "-")}</li>
        <li>읽는 시간: ${escapeHtml(item.readingTime || "-")}</li>
      `;
    }

    const relatedEl = document.getElementById("relatedList");
    if (relatedEl) {
      const related = items
        .filter((entry) => entry.slug !== item.slug && entry.section === item.section)
        .slice(0, 4);
      relatedEl.innerHTML = related.length
        ? related.map((entry) => `<li><a class="text-link" href="${itemHref(entry)}">${escapeHtml(entry.title)}</a></li>`).join("")
        : "<li>같은 섹션의 다른 글을 준비 중입니다.</li>";
    }
  }

  function loadSiteLibrary() {
    if (Array.isArray(window.SiteLibrary) && window.SiteLibrary.length) {
      items = window.SiteLibrary;
      return Promise.resolve(items);
    }
    if (window.SiteLibraryReady && typeof window.SiteLibraryReady.then === "function") {
      return window.SiteLibraryReady.then((loaded) => {
        items = Array.isArray(loaded) ? loaded : [];
        return items;
      });
    }
    return fetch("/api/site-library")
      .then((response) => response.json())
      .then((payload) => {
        items = Array.isArray(payload.items) ? payload.items : [];
        window.SiteLibrary = items;
        return items;
      })
      .catch(() => {
        items = [];
        return items;
      });
  }

  function loadHomepageContent() {
    return fetch("/api/homepage-content")
      .then((response) => response.json())
      .then((payload) => {
        homepageContent = payload || { marketSummary: {}, picks: [] };
        return homepageContent;
      })
      .catch(() => {
        homepageContent = { marketSummary: {}, picks: [] };
        return homepageContent;
      });
  }

  document.addEventListener("DOMContentLoaded", async () => {
    await Promise.all([loadSiteLibrary(), loadHomepageContent()]);
    renderHome();
    renderList();
    renderDetail();
  });
})();
