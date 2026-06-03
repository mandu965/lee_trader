(() => {
  const ITEMS = [
    { key: "main", label: "\uBA54\uC778", href: "/app" },
    { key: "live-auto-trading", label: "AI \uC2E4\uC790\uB3D9\uB9E4\uB9E4", href: "/live-auto-trading.html" },
    { key: "rule-auto-trading", label: "\uC218\uB3D9\uB9E4\uB9E4", href: "/rule-auto-trading.html" },
    { key: "ranking", label: "\uB7AD\uD0B9 \uBD84\uC11D", href: "/ranking.html" },
    { key: "meaningfulness", label: "AI \uC131\uACFC", href: "/meaningfulness.html" },
    { key: "paper-trading", label: "\uBAA8\uC758\uD22C\uC790", href: "/paper-trading.html" },
    { key: "ops-readiness", label: "\uC6B4\uC601\uC790", href: "/ops-readiness.html" },
  ];

  const BOTTOM_NAV_ITEMS = [
    {
      key: "main",
      label: "\uBA54\uC778",
      href: "/app",
      icon: '<svg width="22" height="22" viewBox="0 0 24 24"><path d="M3 9.5L12 3l9 6.5V20a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V9.5z"/><polyline points="9 21 9 12 15 12 15 21"/></svg>',
    },
    {
      key: "live-auto-trading",
      label: "AI\uB9E4\uB9E4",
      href: "/live-auto-trading.html",
      icon: '<svg width="22" height="22" viewBox="0 0 24 24"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>',
    },
    {
      key: "ranking",
      label: "\uB7AD\uD0B9",
      href: "/ranking.html",
      icon: '<svg width="22" height="22" viewBox="0 0 24 24"><rect x="18" y="3" width="4" height="18" rx="1"/><rect x="10" y="8" width="4" height="13" rx="1"/><rect x="2" y="13" width="4" height="8" rx="1"/></svg>',
    },
    {
      key: "meaningfulness",
      label: "AI\uC131\uACFC",
      href: "/meaningfulness.html",
      icon: '<svg width="22" height="22" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>',
    },
    {
      key: "paper-trading",
      label: "\uBAA8\uC758\uD22C\uC790",
      href: "/paper-trading.html",
      icon: '<svg width="22" height="22" viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M7 8h10M7 12h6M7 16h8"/></svg>',
    },
  ];

  function detectActive() {
    const pathname = location.pathname;
    if (pathname === "/app") return "main";
    if (pathname.includes("live-auto-trading")) return "live-auto-trading";
    if (pathname.includes("rule-auto-trading")) return "rule-auto-trading";
    if (pathname.includes("ranking")) return "ranking";
    if (pathname.includes("meaningfulness")) return "meaningfulness";
    if (pathname.includes("paper-trading")) return "paper-trading";
    if (pathname.includes("ops-readiness")) return "ops-readiness";
    if (pathname.includes("score-check")) return "score-check";
    if (pathname.includes("alerts")) return "alerts";
    return null;
  }

  function renderInto(container) {
    if (!container) return;
    const active = container.dataset.navActive || detectActive() || "";
    const tagName = (container.dataset.navElement || "a").toLowerCase();
    const baseClass = container.dataset.navClass || "snav__item";
    const activeClass = container.dataset.navActiveClass || "is-active";
    const secondaryClass = container.dataset.navSecondaryClass || "";
    const secondaryKeys = new Set(
      String(container.dataset.navSecondaryKeys || "")
        .split(",")
        .map((value) => value.trim())
        .filter(Boolean)
    );

    if (!container.dataset.navClass) {
      container.className = "snav";
    }

    container.innerHTML = ITEMS.map((item) => {
      const classes = [baseClass];
      if (item.key === active && activeClass) classes.push(activeClass);
      if (secondaryClass && secondaryKeys.has(item.key)) classes.push(secondaryClass);
      const attrs = [
        `class="${classes.join(" ")}"`,
        `href="${item.href}"`,
        `aria-current="${item.key === active ? "page" : "false"}"`,
      ];
      return `<${tagName} ${attrs.join(" ")}>${item.label}</${tagName}>`;
    }).join("");
  }

  function injectBottomNav() {
    if (document.querySelector(".mobile-bottom-nav")) return;
    const active = detectActive() || "";
    const nav = document.createElement("nav");
    nav.className = "mobile-bottom-nav";
    nav.setAttribute("aria-label", "모바일 하단 내비게이션");
    nav.innerHTML = BOTTOM_NAV_ITEMS.map((item) => {
      const isActive = item.key === active;
      return `<a class="mobile-bottom-nav__item${isActive ? " is-active" : ""}" href="${item.href}" aria-current="${isActive ? "page" : "false"}">${item.icon}<span>${item.label}</span></a>`;
    }).join("");
    document.body.appendChild(nav);
  }

  function init() {
    document.querySelectorAll("[data-unified-ops-nav]").forEach(renderInto);
    injectBottomNav();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
