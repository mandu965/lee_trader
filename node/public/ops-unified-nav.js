(() => {
  const ITEMS = [
    { key: "main", label: "\uBA54\uC778", href: "/app" },
    { key: "live-auto-trading", label: "AI \uC2E4\uC790\uB3D9\uB9E4\uB9E4", href: "/live-auto-trading.html" },
    { key: "paper-trading", label: "\uBAA8\uC758\uD22C\uC790", href: "/paper-trading.html" },
    { key: "ranking", label: "\uB7AD\uD0B9 \uBD84\uC11D", href: "/ranking.html" },
    { key: "us-ranking", label: "US \uB7AD\uD0B9", href: "/us-ranking" },
    { key: "us-trading", label: "US \uC790\uB3D9\uB9E4\uB9E4", href: "/us-trading" },
    { key: "meaningfulness", label: "AI \uC131\uACFC", href: "/meaningfulness.html" },
  ];

  function detectActive() {
    const pathname = location.pathname;
    if (pathname === "/app") return "main";
    if (pathname.includes("manual-trading")) return "manual-trading";
    if (pathname.includes("live-auto-trading")) return "live-auto-trading";
    if (pathname.includes("paper-trading")) return "paper-trading";
    if (pathname.includes("us-trading")) return "us-trading";
    if (pathname.includes("us-ranking")) return "us-ranking";
    if (pathname.includes("ranking")) return "ranking";
    if (pathname.includes("meaningfulness")) return "meaningfulness";
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
