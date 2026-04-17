(() => {
  const ITEMS = [
    { key: "main", label: "메인", href: "/app" },
    { key: "ranking", label: "리서치랭킹", href: "/ranking.html" },
    { key: "meaningfulness", label: "리서치리뷰", href: "/meaningfulness.html" },
    { key: "operator", label: "운영자", href: "/ops-readiness.html" },
    { key: "score-check", label: "점수검증", href: "/score-check" },
    { key: "manual-trading", label: "수동매매", href: "/manual-trading.html" },
    { key: "holdings", label: "보유종목", href: "/holdings.html" },
    { key: "paper-trading", label: "모의투자", href: "/paper-trading.html" },
    { key: "trade-history", label: "매매기록", href: "/trade-history.html" },
    { key: "live-auto-trading", label: "실자동매매", href: "/live-auto-trading.html" },
  ];

  function splitKeys(rawValue, fallback = "") {
    return new Set(String(rawValue || fallback)
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean));
  }

  function renderItem(container, item) {
    const element = String(container.dataset.navElement || "button").trim().toLowerCase() === "a" ? "a" : "button";
    const baseClass = String(container.dataset.navClass || "").trim();
    const activeClass = String(container.dataset.navActiveClass || "is-active").trim();
    const secondaryClass = String(container.dataset.navSecondaryClass || "").trim();
    const activeKeys = splitKeys(container.dataset.navActive);
    const secondaryKeys = splitKeys(container.dataset.navSecondaryKeys, "ranking,meaningfulness");
    const classNames = [baseClass];
    if (activeKeys.has(item.key) && activeClass) classNames.push(activeClass);
    else if (secondaryKeys.has(item.key) && secondaryClass) classNames.push(secondaryClass);
    const classAttr = classNames.filter(Boolean).join(" ");
    if (element === "a") {
      return `<a class="${classAttr}" href="${item.href}">${item.label}</a>`;
    }
    return `<button type="button" class="${classAttr}" onclick="location.href='${item.href}'">${item.label}</button>`;
  }

  function renderInto(container, options = {}) {
    if (!container) return;
    if (options.element) container.dataset.navElement = options.element;
    if (options.baseClass !== undefined) container.dataset.navClass = options.baseClass;
    if (options.activeClass) container.dataset.navActiveClass = options.activeClass;
    if (options.secondaryClass) container.dataset.navSecondaryClass = options.secondaryClass;
    if (options.activeKey) container.dataset.navActive = options.activeKey;
    container.innerHTML = ITEMS.map((item) => renderItem(container, item)).join("");
  }

  const path = window.location.pathname || "/";
  document.querySelectorAll("[data-unified-ops-nav]").forEach((container) => renderInto(container));

  if (path === "/holdings.html") {
    renderInto(document.querySelector("header .toolbar"), { baseClass: "btn", activeClass: "primary", secondaryClass: "secondary", activeKey: "holdings" });
  } else if (path === "/trade-history.html") {
    renderInto(document.querySelector(".page-header .toolbar"), { baseClass: "btn-outline", activeClass: "is-active", secondaryClass: "is-secondary", activeKey: "trade-history" });
  } else if (path === "/ranking.html") {
    renderInto(document.querySelector(".top-nav-strip"), { baseClass: "btn", activeKey: "ranking" });
  } else if (path === "/meaningfulness.html") {
    renderInto(document.querySelector(".top-nav-strip"), { baseClass: "nav-tab", activeClass: "active", activeKey: "meaningfulness" });
  } else if (path === "/detail.html") {
    renderInto(document.querySelector(".top-nav-links"), { element: "a", baseClass: "action-btn", activeClass: "nav-active" });
  } else if (path === "/holdingsDetail.html") {
    renderInto(document.querySelector("header .toolbar"), { baseClass: "", activeKey: "holdings" });
  } else if (path === "/score-check") {
    renderInto(document.querySelector(".page-header .page-actions"), { activeKey: "score-check" });
  } else if (path === "/paper-trading.html") {
    renderInto(document.querySelector(".paper-page .page-actions"), { secondaryClass: "is-secondary", activeKey: "paper-trading" });
  }

  if (path === "/app") {
    const primaryNav = document.querySelector(".toolbar-nav .nav-tabs--primary");
    const secondaryNav = document.querySelector(".toolbar-nav .nav-tabs--secondary");
    if (primaryNav && !document.getElementById("liveAutoTradingBtn")) {
      primaryNav.insertAdjacentHTML(
        "beforeend",
        `<button id="scoreCheckBtn" class="nav-tab" aria-label="점수 검증 화면으로 이동">점수검증</button>
         <button id="liveAutoTradingBtn" class="nav-tab" aria-label="실자동매매 화면으로 이동">실자동매매</button>`
      );
      document.getElementById("scoreCheckBtn")?.addEventListener("click", (e) => { e.preventDefault(); window.location.href = "/score-check"; });
      document.getElementById("liveAutoTradingBtn")?.addEventListener("click", (e) => { e.preventDefault(); window.location.href = "./live-auto-trading.html"; });
    }
    if (secondaryNav && !secondaryNav.querySelector('[data-nav-key="ranking"]')) {
      secondaryNav.querySelectorAll("button").forEach((button) => {
        if (button.id === "recoBtn") button.dataset.navKey = "ranking";
        if (button.id === "meaningfulnessBtn") button.dataset.navKey = "meaningfulness";
      });
    }
  }
})();
