(() => {
  if (window.TradingPolicyUI) return;

  const STYLE_ID = "trading-policy-ui-style";

  function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
      .trading-policy-strip {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
      }
      .trading-policy-card,
      .trading-policy-rule {
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.03);
        padding: 14px 16px;
      }
      .trading-policy-card.tone-good,
      .trading-policy-rule.tone-good {
        border-color: rgba(34, 197, 94, 0.28);
        background: rgba(34, 197, 94, 0.08);
      }
      .trading-policy-card.tone-watch,
      .trading-policy-rule.tone-watch {
        border-color: rgba(250, 204, 21, 0.28);
        background: rgba(250, 204, 21, 0.08);
      }
      .trading-policy-card.tone-bad,
      .trading-policy-rule.tone-bad {
        border-color: rgba(248, 113, 113, 0.28);
        background: rgba(248, 113, 113, 0.08);
      }
      .trading-policy-kicker {
        font-size: 11px;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--color-text-secondary, #94a3b8);
      }
      .trading-policy-value {
        margin-top: 8px;
        font-size: 20px;
        font-weight: 800;
        line-height: 1.2;
        color: var(--color-text-primary, #e5e7eb);
      }
      .trading-policy-detail {
        margin-top: 8px;
        font-size: 12px;
        line-height: 1.55;
        color: var(--color-text-secondary, #94a3b8);
      }
      .trading-policy-section {
        display: grid;
        gap: 12px;
      }
      .trading-policy-section-head {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        align-items: flex-end;
        flex-wrap: wrap;
      }
      .trading-policy-section-title {
        margin: 0;
        font-size: 17px;
      }
      .trading-policy-section-note {
        font-size: 12px;
        color: var(--color-text-secondary, #94a3b8);
      }
      .trading-policy-rule-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 12px;
      }
      .trading-policy-rule-value {
        margin-top: 8px;
        font-size: 16px;
        font-weight: 700;
        color: var(--color-text-primary, #e5e7eb);
      }
      @media (max-width: 820px) {
        .trading-policy-strip,
        .trading-policy-rule-grid {
          grid-template-columns: 1fr;
        }
      }
    `;
    document.head.appendChild(style);
  }

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (m) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }[m]));
  }

  function renderStrip(targetId, items) {
    ensureStyle();
    const el = document.getElementById(targetId);
    if (!el) return;
    const rows = Array.isArray(items) ? items : [];
    el.innerHTML = rows.length
      ? `<div class="trading-policy-strip">${rows.map((item) => `
          <article class="trading-policy-card tone-${escapeHtml(item.tone || "info")}">
            <div class="trading-policy-kicker">${escapeHtml(item.title || "-")}</div>
            <div class="trading-policy-value">${escapeHtml(item.value || "-")}</div>
            <div class="trading-policy-detail">${escapeHtml(item.detail || "-")}</div>
          </article>
        `).join("")}</div>`
      : `<div class="trading-policy-card"><div class="trading-policy-detail">전략 정책을 불러오지 못했습니다.</div></div>`;
  }

  function renderRuleSection(targetId, options = {}) {
    ensureStyle();
    const el = document.getElementById(targetId);
    if (!el) return;
    const items = Array.isArray(options.items) ? options.items : [];
    el.innerHTML = `
      <div class="trading-policy-section">
        <div class="trading-policy-section-head">
          <div>
            <h3 class="trading-policy-section-title">${escapeHtml(options.title || "전략 규칙")}</h3>
            <div class="trading-policy-section-note">${escapeHtml(options.note || "")}</div>
          </div>
        </div>
        <div class="trading-policy-rule-grid">
          ${items.map((item) => `
            <article class="trading-policy-rule tone-${escapeHtml(item.tone || "info")}">
              <div class="trading-policy-kicker">${escapeHtml(item.title || "-")}</div>
              <div class="trading-policy-rule-value">${escapeHtml(item.value || "-")}</div>
              <div class="trading-policy-detail">${escapeHtml(item.detail || "-")}</div>
            </article>
          `).join("")}
        </div>
      </div>
    `;
  }

  window.TradingPolicyUI = {
    renderStrip,
    renderRuleSection,
  };
})();
