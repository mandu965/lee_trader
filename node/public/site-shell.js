document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".site-footer__inner > div:last-child").forEach((links) => {
    if (links.querySelector('a[href="/editorial-policy"]')) return;
    const about = links.querySelector('a[href="/about"]');
    if (!about) return;
    about.insertAdjacentHTML(
      "afterend",
      ' · <a class="text-link" href="/editorial-policy">편집·정정 정책</a>'
    );
  });

  const toggle = document.querySelector(".site-nav-toggle");
  const nav = document.querySelector(".site-nav");
  if (!toggle || !nav) return;

  const closeNav = () => {
    nav.classList.remove("is-open");
    toggle.setAttribute("aria-expanded", "false");
  };

  toggle.addEventListener("click", () => {
    const open = nav.classList.toggle("is-open");
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
  });

  document.addEventListener("click", (event) => {
    if (!window.matchMedia("(max-width: 760px)").matches) return;
    if (!nav.classList.contains("is-open")) return;
    if (nav.contains(event.target) || toggle.contains(event.target)) return;
    closeNav();
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeNav();
  });

  nav.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => {
      if (window.matchMedia("(max-width: 760px)").matches) closeNav();
    });
  });

  window.addEventListener("resize", () => {
    if (!window.matchMedia("(max-width: 760px)").matches) closeNav();
  });
});
