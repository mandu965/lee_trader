window.SiteLibrary = Array.isArray(window.SiteLibrary) ? window.SiteLibrary : [];
window.SiteLibraryReady = fetch("/api/site-library")
  .then((response) => response.json())
  .then((payload) => {
    window.SiteLibrary = Array.isArray(payload.items) ? payload.items : [];
    document.dispatchEvent(new CustomEvent("site-library:ready", { detail: window.SiteLibrary }));
    return window.SiteLibrary;
  })
  .catch(() => {
    window.SiteLibrary = [];
    document.dispatchEvent(new CustomEvent("site-library:ready", { detail: window.SiteLibrary }));
    return window.SiteLibrary;
  });
