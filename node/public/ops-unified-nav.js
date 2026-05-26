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

  function init() {
    document.querySelectorAll("[data-unified-ops-nav]").forEach(renderInto);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
