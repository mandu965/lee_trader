(() => {
  const ITEMS = [
    { key: "main", label: "메인", href: "/app" },
    { key: "ranking", label: "리서치 랭킹", href: "/ranking.html" },
    { key: "meaningfulness", label: "리서치 리뷰", href: "/meaningfulness.html" },
    { key: "operator", label: "운영자", href: "/ops-readiness.html" },
    { key: "score-check", label: "점수검증", href: "/score-check" },
    { key: "manual-trading", label: "수동매매", href: "/manual-trading.html" },
    { key: "holdings", label: "보유종목", href: "/holdings.html" },
    { key: "paper-trading", label: "모의투자", href: "/paper-trading.html" },
    { key: "trade-history", label: "매매기록", href: "/trade-history.html" },
    { key: "rule-auto-trading", label: "RULE 자동매매", href: "/rule-auto-trading.html" },
    { key: "live-auto-trading", label: "실자동매매", href: "/live-auto-trading.html" },
  ];

  const OPERATOR_ONLY_KEYS = new Set(["score-check", "holdings", "trade-history"]);

  function splitKeys(rawValue, fallback = "") {
    return new Set(
      String(rawValue || fallback)
        .split(",")
        .map((value) => value.trim())
        .filter(Boolean)
    );
  }

  function canShowOperatorOnly(auth) {
    if (!auth || typeof auth !== "object") return false;
    return !auth.enabled || auth.authorized;
  }

  async function fetchOperatorAuthStatus() {
    try {
      const res = await fetch("/api/operator-auth/status", { credentials: "same-origin" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (error) {
      console.warn("ops unified nav auth status unavailable", error);
      return { enabled: false, authorized: true };
    }
  }

  function getItemsForContainer(container, auth) {
    const hiddenKeys = splitKeys(container?.dataset?.navHiddenKeys);
    const showOperatorOnly = canShowOperatorOnly(auth);
    return ITEMS.filter((item) => {
      if (hiddenKeys.has(item.key)) return false;
      if (OPERATOR_ONLY_KEYS.has(item.key) && !showOperatorOnly) return false;
      return true;
    });
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
    return `<button type="button" class="${classAttr}" data-nav-key="${item.key}" onclick="location.href='${item.href}'">${item.label}</button>`;
  }

  function renderInto(container, auth, options = {}) {
    if (!container) return;
    if (options.element) container.dataset.navElement = options.element;
    if (options.baseClass !== undefined) container.dataset.navClass = options.baseClass;
    if (options.activeClass) container.dataset.navActiveClass = options.activeClass;
    if (options.secondaryClass) container.dataset.navSecondaryClass = options.secondaryClass;
    if (options.activeKey) container.dataset.navActive = options.activeKey;
    if (options.hiddenKeys !== undefined) container.dataset.navHiddenKeys = options.hiddenKeys;

    const items = getItemsForContainer(container, auth);
    container.innerHTML = items.map((item) => renderItem(container, item)).join("");
  }

  function appendAppButtons(auth) {
    const primaryNav = document.querySelector(".toolbar-nav .nav-tabs--primary");
    const secondaryNav = document.querySelector(".toolbar-nav .nav-tabs--secondary");
    const showOperatorOnly = canShowOperatorOnly(auth);
    if (primaryNav && !document.getElementById("liveAutoTradingBtn")) {
      const extraButtons = [];
      if (showOperatorOnly) {
        extraButtons.push(
          `<button id="scoreCheckBtn" class="nav-tab" data-operator-only="1" aria-label="점수 검증 화면으로 이동">점수검증</button>`
        );
      }
      extraButtons.push(
        `<button id="ruleAutoTradingBtn" class="nav-tab" aria-label="RULE 자동매매 화면으로 이동">RULE 자동매매</button>`,
        `<button id="liveAutoTradingBtn" class="nav-tab" aria-label="실자동매매 화면으로 이동">실자동매매</button>`
      );
      primaryNav.insertAdjacentHTML("beforeend", extraButtons.join(""));
      document.getElementById("scoreCheckBtn")?.addEventListener("click", (e) => {
        e.preventDefault();
        window.location.href = "/score-check";
      });
      document.getElementById("ruleAutoTradingBtn")?.addEventListener("click", (e) => {
        e.preventDefault();
        window.location.href = "./rule-auto-trading.html";
      });
      document.getElementById("liveAutoTradingBtn")?.addEventListener("click", (e) => {
        e.preventDefault();
        window.location.href = "./live-auto-trading.html";
      });
    }
    if (secondaryNav && !secondaryNav.querySelector('[data-nav-key="ranking"]')) {
      secondaryNav.querySelectorAll("button").forEach((button) => {
        if (button.id === "recoBtn") button.dataset.navKey = "ranking";
        if (button.id === "meaningfulnessBtn") button.dataset.navKey = "meaningfulness";
      });
    }
  }

  async function init() {
    const auth = await fetchOperatorAuthStatus();
    const path = window.location.pathname || "/";

    document.querySelectorAll("[data-unified-ops-nav]").forEach((container) => renderInto(container, auth));

    if (path === "/holdings.html") {
      renderInto(document.querySelector("header .toolbar"), auth, { baseClass: "btn", activeClass: "primary", secondaryClass: "secondary", activeKey: "holdings" });
    } else if (path === "/trade-history.html") {
      renderInto(document.querySelector(".page-header .toolbar"), auth, { baseClass: "btn-outline", activeClass: "is-active", secondaryClass: "is-secondary", activeKey: "trade-history" });
    } else if (path === "/ranking.html") {
      renderInto(document.querySelector(".top-nav-strip"), auth, { baseClass: "btn", activeKey: "ranking" });
    } else if (path === "/meaningfulness.html") {
      renderInto(document.querySelector(".top-nav-strip"), auth, { baseClass: "nav-tab", activeClass: "active", activeKey: "meaningfulness" });
    } else if (path === "/detail.html") {
      renderInto(document.querySelector(".top-nav-links"), auth, { element: "a", baseClass: "action-btn", activeClass: "nav-active" });
    } else if (path === "/holdingsDetail.html") {
      renderInto(document.querySelector("header .toolbar"), auth, { baseClass: "", activeKey: "holdings" });
    } else if (path === "/score-check") {
      renderInto(document.querySelector(".page-header .page-actions"), auth, { activeKey: "score-check" });
    } else if (path === "/paper-trading.html") {
      renderInto(document.querySelector(".paper-page .page-actions"), auth, { secondaryClass: "is-secondary", activeKey: "paper-trading" });
    }

    if (path === "/app") {
      appendAppButtons(auth);
    }
  }

  init().catch((error) => {
    console.error("ops unified nav init failed", error);
  });
})();
