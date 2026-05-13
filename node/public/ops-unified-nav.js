(() => {
  const ITEMS = [
    { key: "main",              label: "메인",          href: "/app" },
    { key: "manual-trading",    label: "수동매매",        href: "/manual-trading.html" },
    { key: "live-auto-trading", label: "AI 실자동매매",    href: "/live-auto-trading.html" },
    { key: "rule-auto-trading", label: "RULE 자동매매",   href: "/rule-auto-trading.html" },
    { key: "paper-trading",     label: "모의투자",        href: "/paper-trading.html" },
    { key: "ranking",           label: "리서치 랭킹",     href: "/ranking.html" },
    { key: "meaningfulness",    label: "리서치 리뷰",     href: "/meaningfulness.html" },
  ];

  function detectActive() {
    const p = location.pathname;
    if (p === "/app") return "main";
    if (p.includes("manual-trading"))    return "manual-trading";
    if (p.includes("live-auto-trading")) return "live-auto-trading";
    if (p.includes("rule-auto-trading")) return "rule-auto-trading";
    if (p.includes("paper-trading"))     return "paper-trading";
    if (p.includes("ranking"))           return "ranking";
    if (p.includes("meaningfulness"))    return "meaningfulness";
    return null;
  }

  function renderInto(container) {
    if (!container) return;
    const active = container.dataset.navActive || detectActive() || "";
    container.className = "snav";
    container.innerHTML = ITEMS.map((item) =>
      `<a class="snav__item${item.key === active ? " is-active" : ""}" href="${item.href}" aria-current="${item.key === active ? "page" : "false"}">${item.label}</a>`
    ).join("");
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
