document.addEventListener("DOMContentLoaded", () => {
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
