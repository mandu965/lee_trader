const fmtNum = (value, digits = 0) => {
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  return n.toLocaleString("ko-KR", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
};

const fmtPct = (value, digits = 1) => {
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  return `${(n * 100).toFixed(digits)}%`;
};

const escapeHtml = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (m) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[m]));

async function fetchJsonMaybe(url) {
  const res = await fetch(url, { credentials: "same-origin" });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`${url} HTTP ${res.status}`);
  return res.json();
}

function toneClass(kind) {
  const value = String(kind || "").toUpperCase();
  if (["BUY", "GOOD", "OPEN", "READY", "EXECUTABLE", "SELL"].includes(value)) return "good";
  if (["REVIEW", "WATCH", "TRIM", "HOLD"].includes(value)) return "watch";
  if (["BLOCK", "BAD", "ERROR", "EXIT"].includes(value)) return "bad";
  return "warn";
}

function orderStateChip(row) {
  if (row.executable_now) return `<span class="chip good">executable</span>`;
  if (row.blocked_reason) return `<span class="chip bad">blocked</span>`;
  return `<span class="chip warn">preview</span>`;
}

function intentStateChip(row) {
  if (row.executable) return `<span class="chip good">실행 후보</span>`;
  if (String(row.intent_type || "").toUpperCase() === "REVIEW") return `<span class="chip watch">review</span>`;
  return `<span class="chip warn">설명용</span>`;
}

function holdingStateChip(row) {
  const pnlPct = Number(row.pnl_pct);
  if (String(row.status || "").toUpperCase() !== "OPEN") return `<span class="chip warn">${escapeHtml(row.status || "closed")}</span>`;
  if (Number.isFinite(pnlPct) && pnlPct >= 0.1) return `<span class="chip good">수익</span>`;
  if (Number.isFinite(pnlPct) && pnlPct <= -0.05) return `<span class="chip bad">손실</span>`;
  return `<span class="chip watch">보유중</span>`;
}

function renderHero(summary, intents, preview, holdings) {
  const summaryInfo = summary?.summary || {};
  const heroGrid = document.getElementById("heroGrid");
  heroGrid.innerHTML = `
    <article class="hero-card">
      <div class="card-label">기준일</div>
      <div class="card-value">${escapeHtml(intents?.asof_date || preview?.asof_date || "-")}</div>
      <div class="card-detail">intent 생성 ${escapeHtml(intents?.generated_at || "-")}</div>
    </article>
    <article class="hero-card">
      <div class="card-label">Gate</div>
      <div class="card-value">${escapeHtml(intents?.gate_status || preview?.gate_status || summary?.preview_gate_status || "-")}</div>
      <div class="card-detail">${escapeHtml(intents?.gate_guidance || "gate 설명 정보 없음")}</div>
    </article>
    <article class="hero-card">
      <div class="card-label">Intent / Preview</div>
      <div class="card-value">${fmtNum(intents?.intent_count)} / ${fmtNum(preview?.summary?.request_count)}</div>
      <div class="card-detail">실행 후보 ${fmtNum((intents?.intents || []).filter((item) => item.executable).length)}건 | dry-run 주문 ${fmtNum(summary?.order_preview_count ?? preview?.summary?.request_count)}</div>
    </article>
    <article class="hero-card">
      <div class="card-label">실계좌 보유</div>
      <div class="card-value">${fmtNum(summary?.holding_count ?? holdings?.count)}</div>
      <div class="card-detail">계좌 요약 파일 ${summaryInfo ? "연결" : "없음"} | holdings csv ${holdings?.count ? "존재" : "없음"}</div>
    </article>
  `;
}

function renderStatus(summary, intents, preview) {
  const summaryInfo = summary?.summary || {};
  const cash = summaryInfo?.output2 || summaryInfo?.cash || {};
  const cards = [
    {
      label: "계좌 요약",
      value: summaryInfo?.tot_evlu_amt ?? summaryInfo?.total_evaluation_amount,
      detail: `예수금 ${fmtNum(cash?.dnca_tot_amt ?? cash?.ord_psbl_cash)} | 평가손익 ${fmtNum(summaryInfo?.evlu_pfls_smtl_amt ?? summaryInfo?.pnl_amount)}`,
    },
    {
      label: "Intent 분포",
      value: (intents?.intents || []).length,
      detail: `BUY ${fmtNum((intents?.intents || []).filter((item) => item.intent_type === "BUY").length)} | TRIM ${fmtNum((intents?.intents || []).filter((item) => item.intent_type === "TRIM").length)} | REVIEW ${fmtNum((intents?.intents || []).filter((item) => item.intent_type === "REVIEW").length)}`,
    },
    {
      label: "주문 초안 상태",
      value: preview?.summary?.request_count,
      detail: `실행 가능 ${fmtNum((preview?.items || []).filter((item) => item.executable_now).length)} | 차단 ${fmtNum((preview?.items || []).filter((item) => item.blocked_reason).length)}`,
    },
  ];

  document.getElementById("statusGrid").innerHTML = cards.map((item) => `
    <article class="hero-card">
      <div class="card-label">${escapeHtml(item.label)}</div>
      <div class="card-value">${fmtNum(item.value)}</div>
      <div class="card-detail">${escapeHtml(item.detail)}</div>
    </article>
  `).join("");
}

function renderRunSummary(intents, preview, holdings) {
  const kv = document.getElementById("runSummaryKv");
  const chips = document.getElementById("runSummaryChips");
  const help = document.getElementById("runSummaryHelp");
  const blockedOrders = (preview?.items || []).filter((item) => item.blocked_reason);
  const missingHoldingQty = blockedOrders.filter((item) => item.blocked_reason === "holding_qty_missing").length;
  kv.innerHTML = [
    ["policy version", intents?.policy_version || "-"],
    ["holdings source", intents?.holdings_source || "-"],
    ["preview gate", preview?.gate_status || "-"],
    ["실행 가능 intent", fmtNum((intents?.intents || []).filter((item) => item.executable).length)],
    ["차단된 주문 초안", fmtNum(blockedOrders.length)],
  ].map(([label, value]) => `
    <div class="kv-row">
      <span class="muted">${escapeHtml(label)}</span>
      <strong>${escapeHtml(String(value))}</strong>
    </div>
  `).join("");
  chips.innerHTML = `
    <span class="chip ${toneClass(intents?.gate_status)}">${escapeHtml(intents?.gate_status || "gate unknown")}</span>
    <span class="chip ${blockedOrders.length ? "bad" : "good"}">preview blocked ${fmtNum(blockedOrders.length)}</span>
    <span class="chip ${holdings?.count ? "good" : "bad"}">holdings ${fmtNum(holdings?.count)}</span>
  `;
  help.textContent = missingHoldingQty
    ? `현재 주문 초안 중 ${missingHoldingQty}건이 holding_qty_missing 상태입니다. 실계좌 보유 CSV와 intent 대상 코드 정합성을 먼저 맞춰야 합니다.`
    : "intent, preview, holdings 간의 기본 연결은 확인된 상태입니다.";
}

function renderFocus(intents, preview, holdings) {
  const focusKv = document.getElementById("focusKv");
  const focusChips = document.getElementById("focusChips");
  const focusHelp = document.getElementById("focusHelp");
  const topIntent = (intents?.intents || []).slice().sort((a, b) => (Number(b.priority) || 0) - (Number(a.priority) || 0))[0];
  const topHolding = (holdings?.items || []).slice().sort((a, b) => (Number(b.weight) || 0) - (Number(a.weight) || 0))[0];
  const topPreview = (preview?.items || []).slice().sort((a, b) => (Number(b.priority) || 0) - (Number(a.priority) || 0))[0];
  focusKv.innerHTML = [
    ["최우선 intent", topIntent ? `${topIntent.code} ${topIntent.name || ""}`.trim() : "-"],
    ["최우선 주문초안", topPreview ? `${topPreview.side} ${topPreview.code}` : "-"],
    ["최대 보유비중", topHolding ? `${topHolding.code} ${fmtPct(topHolding.weight, 1)}` : "-"],
  ].map(([label, value]) => `
    <div class="kv-row">
      <span class="muted">${escapeHtml(label)}</span>
      <strong>${escapeHtml(String(value))}</strong>
    </div>
  `).join("");
  focusChips.innerHTML = `
    ${topIntent ? `<span class="chip ${toneClass(topIntent.intent_type)}">${escapeHtml(topIntent.intent_type)}</span>` : ""}
    ${topPreview ? `<span class="chip ${toneClass(topPreview.side)}">${escapeHtml(topPreview.side)}</span>` : ""}
    ${topHolding ? `<span class="chip watch">largest holding ${escapeHtml(topHolding.code)}</span>` : ""}
  `;
  focusHelp.textContent = topIntent
    ? `${topIntent.code} ${topIntent.name || ""} intent의 우선순위가 가장 높습니다. 다만 preview 단계에서 blocked면 실제 주문 엔진으로 넘기면 안 됩니다.`
    : "아직 생성된 intent가 없습니다.";
}

function renderIntents(intents) {
  const tbody = document.getElementById("intentsTbody");
  const rows = intents?.intents || [];
  if (!rows.length) {
    document.getElementById("intentsWrap").innerHTML = `<div class="empty-state">trade intents 산출물이 아직 없습니다.</div>`;
    return;
  }
  tbody.innerHTML = rows.map((row) => `
    <tr>
      <td>${intentStateChip(row)}</td>
      <td class="mono">${escapeHtml(row.code || "-")}</td>
      <td>${escapeHtml(row.name || "-")}</td>
      <td>${escapeHtml(row.intent_type || "-")}</td>
      <td class="right">${fmtNum(row.priority)}</td>
      <td class="right">${fmtPct(row.target_weight, 1)}</td>
      <td>${escapeHtml(row.reason || "-")}</td>
    </tr>
  `).join("");
}

function renderPreview(preview) {
  const tbody = document.getElementById("previewTbody");
  const rows = preview?.items || [];
  if (!rows.length) {
    document.getElementById("previewWrap").innerHTML = `<div class="empty-state">order requests preview 산출물이 아직 없습니다.</div>`;
    return;
  }
  tbody.innerHTML = rows.map((row) => `
    <tr>
      <td>${orderStateChip(row)}</td>
      <td class="mono">${escapeHtml(row.request_id || "-")}</td>
      <td class="mono">${escapeHtml(row.code || "-")}</td>
      <td>${escapeHtml(row.name || "-")}</td>
      <td>${escapeHtml(row.side || "-")}</td>
      <td>${escapeHtml(row.intent_type || "-")}</td>
      <td class="right">${fmtNum(row.final_request_qty)}</td>
      <td class="right">${fmtNum(row.allowed_qty)}</td>
      <td>${escapeHtml(row.blocked_reason || "-")}</td>
      <td>${escapeHtml(row.reason || "-")}</td>
    </tr>
  `).join("");
}

function renderHoldings(holdings) {
  const tbody = document.getElementById("holdingsTbody");
  const rows = holdings?.items || [];
  if (!rows.length) {
    document.getElementById("holdingsWrap").innerHTML = `<div class="empty-state">실계좌 보유 CSV가 아직 없습니다.</div>`;
    return;
  }
  tbody.innerHTML = rows.map((row) => `
    <tr>
      <td>${holdingStateChip(row)}</td>
      <td class="mono">${escapeHtml(row.code || "-")}</td>
      <td>${escapeHtml(row.name || "-")}</td>
      <td class="right">${fmtNum(row.qty)}</td>
      <td class="right">${fmtNum(row.avg_price, 2)}</td>
      <td class="right">${fmtNum(row.current_price, 2)}</td>
      <td class="right">${fmtNum(row.eval_amount)}</td>
      <td class="right ${Number(row.pnl_amount) >= 0 ? "pos" : "neg"}">${fmtNum(row.pnl_amount)}</td>
      <td class="right ${Number(row.pnl_pct) >= 0 ? "pos" : "neg"}">${fmtPct(row.pnl_pct, 2)}</td>
      <td class="right">${fmtPct(row.weight, 1)}</td>
    </tr>
  `).join("");
}

async function main() {
  const state = document.getElementById("pageState");
  state.textContent = "실자동매매 데이터를 불러오는 중입니다.";
  try {
    const [summary, intents, preview, holdings] = await Promise.all([
      fetchJsonMaybe("/api/live-account/summary"),
      fetchJsonMaybe("/api/trade-intents"),
      fetchJsonMaybe("/api/order-requests-preview"),
      fetchJsonMaybe("/api/live-account/holdings"),
    ]);

    renderHero(summary, intents, preview, holdings);
    renderStatus(summary, intents, preview);
    renderRunSummary(intents, preview, holdings);
    renderFocus(intents, preview, holdings);
    renderIntents(intents);
    renderPreview(preview);
    renderHoldings(holdings);

    const loaded = [
      summary ? "summary" : null,
      intents ? "intents" : null,
      preview ? "preview" : null,
      holdings ? "holdings" : null,
    ].filter(Boolean);
    state.textContent = loaded.length
      ? `불러온 데이터: ${loaded.join(", ")}`
      : "실자동매매 산출물이 아직 없습니다.";
  } catch (error) {
    console.error(error);
    state.textContent = `실자동매매 데이터를 불러오지 못했습니다: ${error.message}`;
  }
}

void main();
