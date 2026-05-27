/* ─────────────────────────────────────────────
   AI 실자동매매 운영 보드
   계좌: AI 계좌 (KIS_CANO=43510321)
   RULE 계좌 데이터는 절대 사용하지 않음
───────────────────────────────────────────── */

// ── Formatters ──────────────────────────────
const fmtNum = (value, digits = 0) => {
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  return n.toLocaleString("ko-KR", { minimumFractionDigits: digits, maximumFractionDigits: digits });
};

const fmtPct = (value, digits = 1) => {
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  return `${(n * 100).toFixed(digits)}%`;
};

const fmtWon = (v) => {
  const n = Number(v);
  if (!Number.isFinite(n)) return "-";
  const abs = Math.abs(n);
  const sign = n < 0 ? "-" : n > 0 ? "+" : "";
  return `${sign}${Math.round(abs).toLocaleString("ko-KR")}원`;
};

const signedClass = (value) => {
  const n = Number(value);
  if (!Number.isFinite(n) || n === 0) return "";
  return n > 0 ? "pos" : "neg";
};

const metricHtml = (value, formatter, digits = 0) => {
  const cls = signedClass(value);
  const rendered = escapeHtml(formatter(value, digits));
  return cls ? `<span class="${cls}">${rendered}</span>` : rendered;
};

const escapeHtml = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (m) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m])
  );

const fmtRuntimeDate = (value) => {
  if (!value) return "-";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return String(value);
  return parsed.toLocaleDateString("ko-KR");
};

const fmtRuntimeDateTime = (value) => {
  if (!value) return "-";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return String(value);
  return parsed.toLocaleString("ko-KR", {
    year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false,
  });
};

// ── Chip helpers ─────────────────────────────
const toneClass = (kind) => {
  const v = String(kind || "").toUpperCase();
  if (["BUY", "GOOD", "OPEN", "READY", "EXECUTABLE", "BUY_ALLOWED"].includes(v)) return "good";
  if (["REVIEW", "WATCH", "PILOT", "TRIM", "HOLD"].includes(v)) return "watch";
  if (["BLOCK", "BAD", "ERROR", "EXIT"].includes(v)) return "bad";
  return "warn";
};

const opsToneClass = (kind) => {
  const v = String(kind || "").toLowerCase();
  if (v === "normal") return "good";
  if (v === "risk" || v === "stopped") return "bad";
  if (v === "warning") return "warn";
  return "watch";
};

const orderStateChip = (row) => {
  if (row.executable_now) return `<span class="chip good">제출 가능</span>`;
  if (row.blocked_reason) return `<span class="chip bad">차단됨</span>`;
  return `<span class="chip warn">초안</span>`;
};

const intentStateChip = (row) => {
  if (row.executable) return `<span class="chip good">실행 후보</span>`;
  if (String(row.intent_type || "").toUpperCase() === "REVIEW") return `<span class="chip watch">검토용</span>`;
  return `<span class="chip warn">설명용</span>`;
};

const holdingStateChip = (row) => {
  const pnlPct = Number(row.pnl_pct);
  if (String(row.status || "").toUpperCase() !== "OPEN") return `<span class="chip warn">${escapeHtml(row.status || "closed")}</span>`;
  if (Number.isFinite(pnlPct) && pnlPct > 0) return `<span class="chip good">수익</span>`;
  if (Number.isFinite(pnlPct) && pnlPct < 0) return `<span class="chip bad">손실</span>`;
  return `<span class="chip watch">보유중</span>`;
};

const executionStateChip = (row) => {
  const s = String(row.submission_status || "").toLowerCase();
  if (s === "submitted") return `<span class="chip good">제출됨</span>`;
  if (s === "failed") return `<span class="chip bad">실패</span>`;
  if (s === "skipped") return `<span class="chip watch">보류</span>`;
  return `<span class="chip warn">미상</span>`;
};

const statusText = (value) => {
  const key = String(value || "").toUpperCase();
  return ({
    ACTIONABLE: "운영 참고 가능", MONITOR_ONLY: "관찰 전용", REVIEW_READY: "검토 가능",
    PROMOTE_CANDIDATE: "운영 반영 후보", KEEP_SHADOW: "데이터 축적 중", REJECT: "반영 보류",
    PASS: "검증 통과", FAIL: "검증 실패",
  })[key] || key || "-";
};

const sampleTone = (status) => {
  const v = String(status || "").toUpperCase();
  if (["ACTIONABLE", "PROMOTE_CANDIDATE"].includes(v)) return "good";
  if (["MONITOR_ONLY", "REVIEW_READY"].includes(v)) return "watch";
  if (v === "REJECT") return "bad";
  return "warn";
};

const statusChip = (value) => {
  const raw = String(value || "-").toUpperCase();
  const label = statusText(raw);
  return `<span class="chip ${sampleTone(raw)}">${escapeHtml(label)}</span>`;
};

const validationChip = (value) => {
  const raw = String(value || "MISSING").toUpperCase();
  const ok = raw === "PASS";
  return `<span class="chip ${ok ? "good" : "bad"}">${escapeHtml(statusText(raw))}</span>`;
};

const gateLabel = (value) => {
  const key = String(value || "").toUpperCase();
  return ({ BUY_ALLOWED: "매수 허용", PILOT: "제한 실운용", WATCH: "관찰 진입", BLOCK: "신규 진입 차단" })[key] || key || "-";
};

const gateChipText = (value) => {
  const raw = String(value || "").toUpperCase();
  const label = gateLabel(raw);
  return raw && raw !== label && raw !== "-" ? `${label} · ${raw}` : label;
};

const gateStatusChip = (ok, label, value, failTone = "warn") =>
  `<span class="chip ${ok ? "good" : failTone}">${escapeHtml(label)} ${escapeHtml(value)}</span>`;

const helpTip = (text) =>
  `<span class="help-tip" title="${escapeHtml(text)}">?</span>`;

const metricLabel = (label, help) =>
  `${escapeHtml(label)}${help ? helpTip(help) : ""}`;

const helpDetails = (items) => `
  <details class="details-help">
    <summary>용어와 판단 기준</summary>
    <dl class="help-list">
      ${items.map((item) => `
        <div>
          <dt>${escapeHtml(item.term)}</dt>
          <dd>${escapeHtml(item.desc)}</dd>
        </div>
      `).join("")}
    </dl>
  </details>
`;

// ── Network ──────────────────────────────────
async function fetchJsonMaybe(url) {
  const res = await fetch(url, { credentials: "same-origin" });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`${url} HTTP ${res.status}`);
  return res.json();
}

// ── Tab navigation ───────────────────────────
function initTabs() {
  const tabNav = document.getElementById("tabNav");
  if (!tabNav) return;
  tabNav.addEventListener("click", (e) => {
    const btn = e.target.closest(".tab-btn");
    if (!btn) return;
    const tabId = btn.dataset.tab;
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("is-active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("is-active"));
    btn.classList.add("is-active");
    const panel = document.getElementById(`tab${tabId.charAt(0).toUpperCase() + tabId.slice(1)}`);
    if (panel) panel.classList.add("is-active");
  });
}

// ── Safety banner ────────────────────────────
function renderSafety(summary, runtime, intents, preview) {
  const banner = document.getElementById("safetyBanner");
  const textEl = document.getElementById("safetyText");
  if (!banner || !textEl) return;

  const executeOn = !!runtime?.policy?.auto_trade_execute;
  const buyOn = !!runtime?.policy?.auto_trade_allow_buy;
  const gate = String(intents?.gate_status || preview?.gate_status || "").toUpperCase();
  const info = summary?.summary || {};
  const derived = info?.derived_metrics || {};
  const weeklyPnl = derived.weekly_total_pnl;
  const weeklyPct = derived.weekly_loss_pct;

  let cls = "ok";
  let text = "";

  if (!executeOn) {
    cls = "bad";
    text = "실주문 비활성 — AUTO_TRADE_EXECUTE=0";
  } else if (!buyOn) {
    cls = "warn";
    text = "매수 차단 — AUTO_TRADE_ALLOW_BUY=0";
  } else if (gate === "BLOCK") {
    cls = "bad";
    const blockReasons = Array.isArray(intents?.gate_block_reasons_ko) ? intents.gate_block_reasons_ko : [];
    text = "신규 진입 차단 — Gate BLOCK" + (blockReasons.length ? "  ·  " + blockReasons.slice(0, 2).join("  ·  ") : "");
  } else if (gate === "WATCH") {
    cls = "warn";
    text = "제한 진입 구간 — Gate WATCH";
  } else if (gate === "PILOT") {
    cls = "warn";
    text = "PILOT 제한 실운용";
  } else if (gate === "BUY_ALLOWED") {
    cls = "ok";
    text = "실매매 정상 — BUY_ALLOWED";
  } else {
    cls = "warn";
    text = `실매매 활성 — Gate ${gate || "미확인"}`;
  }

  // 주간 손익 정보 추가
  if (Number.isFinite(weeklyPnl) && weeklyPnl !== 0) {
    const pctStr = Number.isFinite(weeklyPct) ? ` (${fmtPct(weeklyPct, 2)})` : "";
    text += `  ·  주간 손익 ${fmtWon(weeklyPnl)}${pctStr}`;
  }

  banner.className = `safety-banner ${cls}`;
  textEl.textContent = text;
}

// ── Tab badges ───────────────────────────────
function updateBadges(intents, preview, execution, holdings, liveKpiDaily) {
  const previewRows = Array.isArray(preview?.items) ? preview.items : [];
  const executableCount = previewRows.filter((r) => r.executable_now).length;
  const submittedCount = Number(execution?.summary?.submitted_count || 0);
  const holdingCount = Number(holdings?.count || 0);
  const warningCount = Number(liveKpiDaily?.consistency?.warning_count || 0);

  const orderBadge = document.getElementById("badgeOrders");
  const analysisBadge = document.getElementById("badgeAnalysis");
  const accountBadge = document.getElementById("badgeAccount");

  if (orderBadge) {
    const orderNum = executableCount || submittedCount;
    orderBadge.textContent = orderNum > 0 ? String(orderNum) : "";
    orderBadge.className = `tab-badge${orderNum > 0 ? " active" : ""}`;
  }
  if (analysisBadge) {
    analysisBadge.textContent = warningCount > 0 ? String(warningCount) : "";
    analysisBadge.className = `tab-badge${warningCount > 0 ? " active" : ""}`;
  }
  if (accountBadge) {
    accountBadge.textContent = holdingCount > 0 ? String(holdingCount) : "";
    accountBadge.className = `tab-badge${holdingCount > 0 ? " active" : ""}`;
  }
}

// ── Decision banner ──────────────────────────
function renderDecisionBanner(summary, intents, preview, execution, runtime, consistency) {
  const root = document.getElementById("decisionBanner");
  if (!root) return;

  const intentRows = Array.isArray(intents?.intents) ? intents.intents : [];
  const previewRows = Array.isArray(preview?.items) ? preview.items : [];
  const executionRows = Array.isArray(execution?.items) ? execution.items : [];
  const gate = String(intents?.gate_status || preview?.gate_status || summary?.preview_gate_status || "").toUpperCase();
  const executeOn = !!runtime?.policy?.auto_trade_execute;
  const buyOn = !!runtime?.policy?.auto_trade_allow_buy;

  const executablePreviewCount = previewRows.filter((r) => r.executable_now).length;
  const blockedPreviewCount = previewRows.filter((r) => r.blocked_reason).length;
  const submittedCount = Number(execution?.summary?.submitted_count || 0);
  const skippedCount = Number(execution?.summary?.skipped_count || 0);
  const buyIntentCount = intentRows.filter((r) => String(r.intent_type || "").toUpperCase() === "BUY").length;
  const filledCount = Number(consistency?.counts?.filled_count || 0);
  const accountSyncedAt = summary?.summary?.generated_at || runtime?.live_account_sync_scheduler?.last_success_at || "-";

  let headline = "현재 상태 판정 불가";
  let headlineTone = "warn";
  let headlineDetail = "필수 산출물이 부족합니다.";

  if (!executeOn) {
    headline = "실주문 비활성";
    headlineTone = "warn";
    headlineDetail = `주문 초안 ${executablePreviewCount}건이 있어도 execute 스위치가 OFF라 실주문은 나가지 않습니다.`;
  } else if (gate === "BLOCK") {
    headline = "신규 진입 차단";
    headlineTone = "bad";
    const blockReasons = Array.isArray(intents?.gate_block_reasons_ko) ? intents.gate_block_reasons_ko : [];
    headlineDetail = "Gate BLOCK — 신규 BUY 차단" + (blockReasons.length ? ": " + blockReasons.join(" / ") : " (기존 보유 정리 중심으로 운영)");
  } else if (gate === "WATCH") {
    headline = "WATCH 제한 진입 구간";
    headlineTone = "warn";
    headlineDetail = "소액 제한 진입만 허용되는 구간입니다. BUY_ALLOWED 전면 운용과는 다릅니다.";
  } else if (gate === "PILOT") {
    headline = "PILOT 제한 실운용";
    headlineTone = "primary";
    headlineDetail = "WATCH보다 넓은 진입이 허용되지만 풀 비중 자동매수는 아직 보류됩니다.";
  } else if (submittedCount > 0) {
    headline = `실주문 제출 ${submittedCount}건`;
    headlineTone = "primary";
    headlineDetail = "가장 최근 사이클에서 실제 주문 제출이 있었습니다. 체결 상태를 같이 확인하세요.";
  } else if (skippedCount > 0) {
    headline = `주문 보류 ${skippedCount}건`;
    headlineTone = "warn";
    headlineDetail = "주문 초안이 있었지만 제출 단계에서 보류되었습니다. 주문 탭에서 차단 사유를 확인하세요.";
  } else if (executablePreviewCount > 0) {
    headline = `제출 가능 초안 ${executablePreviewCount}건`;
    headlineTone = "primary";
    headlineDetail = "차단되지 않은 주문 초안이 존재합니다. 실행 스위치와 승인 정책을 확인하세요.";
  }

  root.innerHTML = `
    <article class="decision-card ${headlineTone}">
      <h2 class="decision-title">오늘 결론</h2>
      <div class="decision-value">${escapeHtml(headline)}</div>
      <div class="decision-detail">${escapeHtml(headlineDetail)}</div>
    </article>
    <article class="decision-card">
      <h2 class="decision-title">진입 모드</h2>
      <div class="decision-value">${escapeHtml(gateChipText(gate))}</div>
      <div class="decision-detail">BUY 판단 ${fmtNum(buyIntentCount)}건 · 제출가능 ${fmtNum(executablePreviewCount)}건 · 차단 ${fmtNum(blockedPreviewCount)}건 · 매수 ${buyOn ? "ON" : "OFF"}</div>
    </article>
    <article class="decision-card">
      <h2 class="decision-title">계좌 동기화</h2>
      <div class="decision-value">${escapeHtml(String(accountSyncedAt).slice(0, 16) || "-")}</div>
      <div class="decision-detail">제출 ${fmtNum(submittedCount)}건 · 체결 ${fmtNum(filledCount)}건${consistency?.as_of_date ? ` · 기준 ${escapeHtml(consistency.as_of_date)}` : ""}</div>
    </article>
  `;
}

// ── Hero grid ─────────────────────────────────
function renderHero(summary, intents, preview, holdings, execution) {
  const root = document.getElementById("heroGrid");
  if (!root) return;

  const info = summary?.summary || {};
  const derived = info?.derived_metrics || {};
  const raw = info?.summary_row || {};
  const gate = String(intents?.gate_status || preview?.gate_status || "").toUpperCase();
  const previewRows = Array.isArray(preview?.items) ? preview.items : [];
  const executableCount = previewRows.filter((r) => r.executable_now).length;
  const submittedBuyCount = (Array.isArray(execution?.items) ? execution.items : []).filter((r) =>
    String(r.submission_status || "").toLowerCase() === "submitted" && String(r.side || "").toUpperCase() === "BUY"
  ).length;

  const totalAssets = derived.total_assets ?? raw.tot_evlu_amt;
  const cashRatio = derived.cash_ratio;
  const evalPnl = raw.evlu_pfls_smtl_amt ?? derived.holding_pnl_amount;
  const weeklyPnl = derived.weekly_total_pnl;
  const weeklyPct = derived.weekly_loss_pct;

  root.innerHTML = `
    <article class="hero-card">
      <div class="card-label">기준일</div>
      <div class="card-value">${escapeHtml(intents?.asof_date || preview?.asof_date || "-")}</div>
      <div class="card-detail">판단 생성: ${escapeHtml(intents?.generated_at || "-")}</div>
    </article>
    <article class="hero-card">
      <div class="card-label">총자산 (AI 계좌)</div>
      <div class="card-value">${escapeHtml(Number.isFinite(Number(totalAssets)) ? fmtNum(totalAssets) + "원" : "-")}</div>
      <div class="card-detail">현금비중 ${escapeHtml(fmtPct(cashRatio, 1))} · 보유 ${fmtNum(holdings?.count ?? summary?.holding_count)}종목</div>
    </article>
    <article class="hero-card">
      <div class="card-label">평가손익</div>
      <div class="card-value ${signedClass(evalPnl)}">${escapeHtml(fmtWon(evalPnl))}</div>
      <div class="card-detail">제출가능 ${fmtNum(executableCount)}건 · BUY 제출 ${fmtNum(submittedBuyCount)}건</div>
    </article>
    <article class="hero-card">
      <div class="card-label">주간 손익 (AI 계좌)</div>
      <div class="card-value ${signedClass(weeklyPnl)}">${escapeHtml(fmtWon(weeklyPnl))}</div>
      <div class="card-detail">주간 손익률 ${escapeHtml(fmtPct(weeklyPct, 2))}</div>
    </article>
  `;
}

// ── Account detail ────────────────────────────
function renderAccountDetails(summary, runtime) {
  const root = document.getElementById("accountDetailGrid");
  if (!root) return;

  const info = summary?.summary || {};
  const raw = info?.summary_row || {};
  const derived = info?.derived_metrics || {};
  const policy = runtime?.policy || {};

  const items = [
    {
      label: "예수금",
      valueHtml: escapeHtml(fmtNum(derived.cash_amount ?? raw.dnca_tot_amt)),
      valueClass: "",
      detail: `D+1 ${fmtNum(raw.nxdy_excc_amt)} · 전일정산 ${fmtNum(raw.prvs_rcdl_excc_amt)}`,
    },
    {
      label: "증권평가",
      valueHtml: escapeHtml(fmtNum(raw.scts_evlu_amt)),
      valueClass: "",
      detail: `매입원가 ${fmtNum(raw.pchs_amt_smtl_amt)} · 평가금액 ${fmtNum(raw.evlu_amt_smtl_amt)}`,
    },
    {
      label: "총자산",
      valueHtml: escapeHtml(Number.isFinite(Number(derived.total_assets ?? raw.tot_evlu_amt)) ? fmtNum(derived.total_assets ?? raw.tot_evlu_amt) + "원" : "-"),
      valueClass: "",
      detail: `전일총자산 ${fmtNum(raw.bfdy_tot_asst_evlu_amt)} · 자산증감 ${fmtNum(raw.asst_icdc_amt)}`,
    },
    {
      label: "평가손익",
      valueHtml: metricHtml(raw.evlu_pfls_smtl_amt, fmtNum),
      valueClass: signedClass(raw.evlu_pfls_smtl_amt),
      detail: `보유합산 ${fmtNum(derived.holding_pnl_amount)} · 자산증감률 ${fmtPct(raw.asst_icdc_erng_rt, 2)}`,
    },
    {
      label: "현금 비중",
      valueHtml: escapeHtml(fmtPct(derived.cash_ratio, 1)),
      valueClass: "",
      detail: `투자비중 ${fmtPct(derived.invested_ratio, 1)} · 평균보유비중 ${fmtPct(derived.avg_position_weight, 1)}`,
    },
    {
      label: "주간 손익",
      valueHtml: metricHtml(derived.weekly_total_pnl, fmtNum),
      valueClass: signedClass(derived.weekly_total_pnl),
      detail: `주간 손익률 ${fmtPct(derived.weekly_loss_pct, 2)} · execute ${policy.auto_trade_execute ? "ON" : "OFF"} · buy ${policy.auto_trade_allow_buy ? "ALLOW" : "BLOCK"}`,
    },
  ];

  root.innerHTML = items.map((item) => `
    <article class="hero-card">
      <div class="card-label">${escapeHtml(item.label)}</div>
      <div class="card-value ${escapeHtml(item.valueClass)}">${item.valueHtml}</div>
      <div class="card-detail">${escapeHtml(item.detail)}</div>
    </article>
  `).join("");
}

// ── Run summary ───────────────────────────────
function renderRunSummary(intents, preview, holdings, runtime) {
  const kv = document.getElementById("runSummaryKv");
  const chips = document.getElementById("runSummaryChips");
  const help = document.getElementById("runSummaryHelp");
  if (!kv) return;

  const blockedOrders = (preview?.items || []).filter((r) => r.blocked_reason);
  const missingHoldingQty = blockedOrders.filter((r) => r.blocked_reason === "holding_qty_missing").length;
  const policy = runtime?.policy || {};

  kv.innerHTML = [
    ["정책 버전", intents?.policy_version || "-"],
    ["보유 데이터 기준", intents?.holdings_source || "-"],
    ["주문 초안 gate", preview?.gate_status || "-"],
    ["실행 가능 intent", fmtNum((intents?.intents || []).filter((r) => r.executable).length)],
    ["차단된 주문 초안", fmtNum(blockedOrders.length)],
    ["주문 초안 생성시각", preview?.generated_at || "-"],
  ].map(([label, value]) => `
    <div class="kv-row">
      <span class="muted">${escapeHtml(label)}</span>
      <strong>${escapeHtml(String(value))}</strong>
    </div>
  `).join("");

  chips.innerHTML = `
    <span class="chip ${toneClass(intents?.gate_status)}">${escapeHtml(intents?.gate_status || "gate unknown")}</span>
    <span class="chip ${blockedOrders.length ? "bad" : "good"}">차단된 초안 ${fmtNum(blockedOrders.length)}</span>
    <span class="chip ${holdings?.count ? "good" : "bad"}">보유 ${fmtNum(holdings?.count)}종목</span>
    <span class="chip ${policy.auto_trade_execute ? "watch" : "warn"}">실주문 ${policy.auto_trade_execute ? "ON" : "OFF"}</span>
    <span class="chip ${policy.auto_trade_allow_buy ? "watch" : "good"}">매수 ${policy.auto_trade_allow_buy ? "허용" : "차단"}</span>
  `;

  if (help) {
    help.textContent = missingHoldingQty
      ? `주문 초안 중 ${missingHoldingQty}건이 holding_qty_missing 상태입니다. 실계좌 보유 CSV와 intent 코드 정합성을 먼저 확인하세요.`
      : "intent, preview, holdings 간의 기본 연결은 확인된 상태입니다.";
  }
}

// ── Focus memo ────────────────────────────────
function renderFocus(intents, preview, holdings) {
  const focusKv = document.getElementById("focusKv");
  const focusChips = document.getElementById("focusChips");
  const focusHelp = document.getElementById("focusHelp");
  if (!focusKv) return;

  const topIntent = (intents?.intents || []).slice().sort((a, b) => (Number(b.priority) || 0) - (Number(a.priority) || 0))[0];
  const topHolding = (holdings?.items || []).slice().sort((a, b) => (Number(b.weight) || 0) - (Number(a.weight) || 0))[0];
  const topPreview = (preview?.items || []).filter((r) => r.executable_now).slice().sort((a, b) => (Number(b.priority) || 0) - (Number(a.priority) || 0))[0];

  focusKv.innerHTML = [
    ["최우선 intent", topIntent ? `${topIntent.code} ${topIntent.name || ""}`.trim() : "-"],
    ["제출가능 초안 1순위", topPreview ? `${topPreview.side} ${topPreview.code}` : "-"],
    ["최대 보유비중", topHolding ? `${topHolding.code} ${fmtPct(topHolding.weight, 1)}` : "-"],
  ].map(([label, value]) => `
    <div class="kv-row">
      <span class="muted">${escapeHtml(label)}</span>
      <strong>${escapeHtml(String(value))}</strong>
    </div>
  `).join("");

  focusChips.innerHTML = `
    ${topIntent ? `<span class="chip ${toneClass(topIntent.intent_type)}">${escapeHtml(topIntent.intent_type)}</span>` : ""}
    ${topPreview ? `<span class="chip good">${escapeHtml(topPreview.side)} ${escapeHtml(topPreview.code)}</span>` : ""}
    ${topHolding ? `<span class="chip watch">최대 보유 ${escapeHtml(topHolding.code)}</span>` : ""}
  `;

  if (focusHelp) {
    focusHelp.textContent = topIntent
      ? `${topIntent.code} ${topIntent.name || ""}의 우선순위가 가장 높습니다.`
      : "아직 생성된 intent가 없습니다.";
  }
}

// ── Operational explain ───────────────────────
function renderOperationalExplain(intents, preview, runtime, holdings) {
  const root = document.getElementById("operationalExplain");
  if (!root) return;
  const gate = String(intents?.gate_status || preview?.gate_status || "").toUpperCase();
  const intentRows = intents?.intents || [];
  const buyRows = intentRows.filter((r) => String(r.intent_type || "").toUpperCase() === "BUY");
  const sellRows = intentRows.filter((r) => ["EXIT", "TRIM"].includes(String(r.intent_type || "").toUpperCase()));
  const hasHoldings = Number(holdings?.count || 0) > 0;
  const executeOn = !!runtime?.policy?.auto_trade_execute;

  let title = "현재는 실운영 정리 화면입니다";
  let body = "아래 표는 실제 운영 산출물 기준입니다. 연구용 후보와 섞어 읽지 않습니다.";

  if (gate === "BLOCK") {
    title = "현재 Gate가 BLOCK이라 신규 매수가 보이지 않습니다";
    body = hasHoldings
      ? `실계좌 보유 ${fmtNum(holdings?.count)}종목이 있고 Gate가 BLOCK이라 신규 진입보다 보유 축소가 우선됩니다.`
      : "빈 계좌여도 BLOCK에서는 신규 매수 후보를 실운영 intent로 올리지 않습니다.";
  } else if (gate === "WATCH") {
    title = "현재는 WATCH 제한 진입 구간입니다";
    body = "소액 제한 진입만 허용됩니다. 연구용 가정 후보와 섞어 보지 않습니다.";
  } else if (gate === "PILOT") {
    title = "현재는 PILOT 제한 실운용 구간입니다";
    body = "WATCH보다 넓게 진입할 수 있지만 BUY_ALLOWED 전면 자동매수와는 다릅니다.";
  } else if (buyRows.length) {
    title = `실운영 신규 매수 후보 ${fmtNum(buyRows.length)}건이 보입니다`;
    body = "현재 Gate 기준에서 실제 BUY intent가 올라온 상태입니다. 주문 탭에서 상세 내용을 확인하세요.";
  } else if (sellRows.length) {
    title = `현재는 매수보다 정리 후보 ${fmtNum(sellRows.length)}건이 우선입니다`;
    body = "정책상 신규 진입보다 리스크 축소가 먼저 선택된 상태입니다.";
  }

  if (!executeOn) {
    body += " 현재 실주문 스위치가 꺼져 있어 이 화면은 주문 모니터 역할입니다.";
  }

  root.innerHTML = `
    <h3 class="explain-title">${escapeHtml(title)}</h3>
    <div class="explain-body">${escapeHtml(body)}</div>
  `;
}

// ── Diagnostic summary ────────────────────────
function summarizeDiagnostics(diagnostics) {
  const items = Array.isArray(diagnostics?.diagnostics) ? diagnostics.diagnostics : [];
  const groups = new Map();
  for (const item of items) {
    const key = [
      item?.type || "",
      item?.broker_error_code || "",
      item?.message_ko || item?.raw_reason || "",
    ].join("|");
    const current = groups.get(key) || { item, count: 0 };
    current.count += 1;
    groups.set(key, current);
  }
  const top = Array.from(groups.values()).sort((a, b) => b.count - a.count)[0] || null;
  return { items, top, uniqueCount: groups.size };
}

function splitDiagnosticAction(value) {
  return String(value || "")
    .split(/\s*\/\s*/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function renderDiagnosticSummary(diagnostics) {
  const root = document.getElementById("diagnosticSummaryPanel");
  if (!root) return;
  const summary = diagnostics?.summary || {};
  const runId = diagnostics?.run_id || "-";
  const diagnosticSummary = summarizeDiagnostics(diagnostics);
  const topDiag = diagnosticSummary.top;
  const topItem = topDiag?.item || {};
  const topMessage = topItem.message_ko || topItem.raw_reason || "";
  const topCode = topItem.broker_error_code || "";
  root.innerHTML = `
    <div class="card-label" style="margin-bottom:10px;">진단 요약</div>
    <div class="kv">
      <div class="kv-row"><span>run_id</span><strong class="mono">${escapeHtml(runId)}</strong></div>
      <div class="kv-row"><span>AI 추천 수</span><strong>${fmtNum(summary.recommendation_count)}</strong></div>
      <div class="kv-row"><span>주문 후보</span><strong>${fmtNum(summary.order_candidate_count)}</strong></div>
      <div class="kv-row"><span>제출 가능</span><strong>${fmtNum(summary.submit_allowed_count)}</strong></div>
      <div class="kv-row"><span>정책 차단</span><strong>${fmtNum(summary.policy_blocked_count)}</strong></div>
      <div class="kv-row"><span>브로커 거부</span><strong>${fmtNum(summary.broker_rejected_count)}</strong></div>
      <div class="kv-row"><span>매도 후보</span><strong>${fmtNum(summary.sell_candidate_count)}</strong></div>
      <div class="kv-row"><span>매도 제출 가능</span><strong>${fmtNum(summary.sell_submit_allowed_count)}</strong></div>
      <div class="kv-row"><span>신규 BUY 허용</span><strong>${summary.new_buy_allowed ? "YES" : "NO"}</strong></div>
      <div class="kv-row"><span>live grade</span><strong>${escapeHtml(summary.live_grade || "-")}</strong></div>
      <div class="kv-row"><span>시장 상태</span><strong>${escapeHtml(summary.market_status_ko || summary.market_status || "-")}</strong></div>
      <div class="kv-row"><span>최근 실행</span><strong>${escapeHtml(summary.last_run_at || "-")}</strong></div>
      <div class="kv-row"><span>최근 주문 시도</span><strong>${escapeHtml(summary.last_order_attempt_at || summary.last_execution_at || "-")}</strong></div>
      <div class="kv-row"><span>Scheduler 상태</span><strong>${escapeHtml(summary.scheduler_status || "-")}</strong></div>
      <div class="kv-row"><span>Refresh 상태</span><strong>${escapeHtml(summary.refresh_status || "-")}</strong></div>
      <div class="kv-row"><span>Refresh 실패 step</span><strong>${escapeHtml(summary.refresh_failing_step || "-")}</strong></div>
    </div>
    <div class="chip-row">
      <span class="chip ${Number(summary.broker_rejected_count || 0) ? "bad" : "good"}">브로커 거부 ${fmtNum(summary.broker_rejected_count)}</span>
      <span class="chip ${String(summary.scheduler_status || "").toLowerCase() === "error" ? "bad" : "good"}">scheduler ${escapeHtml(summary.scheduler_status || "-")}</span>
      <span class="chip ${diagnosticSummary.uniqueCount > 1 ? "warn" : "watch"}">진단 유형 ${fmtNum(diagnosticSummary.uniqueCount)}</span>
      ${topCode ? `<span class="chip bad">${escapeHtml(topCode)}</span>` : ""}
    </div>
    ${topDiag ? `<div class="state-line">반복 진단 ${fmtNum(topDiag.count)}건${topMessage ? `: ${escapeHtml(topMessage)}` : ""}</div>` : ""}
  `;
}

// ── Why no trade ──────────────────────────────
function renderWhyNoTrade(diagnostics) {
  const root = document.getElementById("whyNoTradeBox");
  if (!root) return;
  const summary = diagnostics?.summary || {};
  const items = Array.isArray(diagnostics?.diagnostics) ? diagnostics.diagnostics : [];
  const diagnosticSummary = summarizeDiagnostics(diagnostics);
  const topDiag = diagnosticSummary.top;
  const topItem = topDiag?.item || items[0] || {};
  const primary = items[0] || null;
  const actionItems = splitDiagnosticAction(topItem.recommended_action || primary?.recommended_action).slice(0, 4);
  const causes = Array.isArray(topItem.inferred_causes) ? topItem.inferred_causes.slice(0, 4) : [];
  const brokerCode = topItem.broker_error_code || "";
  const repeatedLine = topDiag?.count > 1
    ? `<br>동일 진단 ${fmtNum(topDiag.count)}회 반복${brokerCode ? ` · 코드 ${escapeHtml(brokerCode)}` : ""}`
    : brokerCode
      ? `<br>브로커 오류 코드: ${escapeHtml(brokerCode)}`
      : "";
  root.innerHTML = `
    <h3 class="explain-title">주문이 없었던 이유</h3>
    <div class="explain-body">
      ${escapeHtml(summary.main_user_message_ko || (
        (Number(summary.submit_allowed_count || 0) + Number(summary.sell_submit_allowed_count || 0)) > 0
          ? `주문 후보 ${Number(summary.submit_allowed_count || 0) + Number(summary.sell_submit_allowed_count || 0)}건이 제출 가능 상태였습니다. 아래 실행 결과를 확인하세요.`
          : "현재 화면에서 판단 근거를 확인하세요."
      ))}
      ${summary.main_block_reason ? `<br>이유: ${escapeHtml(summary.main_block_reason)}` : ""}
      ${repeatedLine}
      ${summary.scheduler_last_error ? `<br>Scheduler 오류: ${escapeHtml(summary.scheduler_last_error)}` : ""}
      ${summary.refresh_failing_step ? `<br>Refresh step: ${escapeHtml(summary.refresh_failing_step)}` : ""}
      ${summary.refresh_failure_reason ? `<br>Refresh reason: ${escapeHtml(summary.refresh_failure_reason)}` : ""}
    </div>
    ${actionItems.length ? `
      <div class="chip-row">
        ${actionItems.map((item) => `<span class="chip warn">${escapeHtml(item)}</span>`).join("")}
      </div>
    ` : ""}
    ${causes.length ? `<div class="state-line">추정 원인: ${causes.map((item) => escapeHtml(item)).join(" · ")}</div>` : ""}
  `;
}

// ── Intents table ─────────────────────────────
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

// ── Order preview table ───────────────────────
function describePreviewExecutionRisk(row, runtime) {
  const blockedReason = String(row?.blocked_reason || "").trim();
  if (blockedReason) return blockedReason;
  const side = String(row?.side || "").toUpperCase();
  if (side === "BUY" && !runtime?.policy?.auto_trade_execute) return "execute 스위치 OFF — 실주문 미시도";
  if (side === "BUY" && !runtime?.policy?.auto_trade_allow_buy) return "BUY 스위치 OFF — 매수 보류";
  if (side === "BUY" && runtime?.policy?.buy_approval_required) return "매수 승인 필수 — 승인 파일 확인";
  return "현재 preview 기준 제출 가능 상태입니다.";
}

function translateBlockedReason(row) {
  const key = String(row?.blocked_reason || "").trim();
  const gapPct = Number(row?.entry_price_gap_pct);
  const gapStr = Number.isFinite(gapPct) ? ` (${gapPct >= 0 ? "+" : ""}${(gapPct * 100).toFixed(1)}%)` : "";
  if (key === "entry_gap_up_hard_blocked")
    return `진입 급등 차단${gapStr} — 기준가 대비 급등으로 매수 보류`;
  if (key === "entry_gap_up_blocked")
    return `진입 갭업 차단${gapStr} — 가격 상승으로 매수 보류`;
  if (key === "buy_qty_zero_budget_below_one_share")
    return "예산 부족 — 목표 예산이 1주 가격 미달";
  if (key === "holding_qty_missing")
    return "보유수량 미확인 — 계좌 CSV 확인 필요";
  if (key === "invalid_final_request_qty")
    return "주문수량 0 이하 — 제출 불가";
  if (key === "trim_weight_unavailable")
    return "TRIM 비중 정보 없음 — 매도 수량 산출 불가";
  if (key === "trim_ratio_zero")
    return "TRIM 비율 0 — 목표비중이 현재비중 이상, 매도 불필요";
  if (key === "market_guard_kill_active")
    return "Market Guard 발동 — 시장 급락 감지로 매수 차단";
  if (key === "kill_switch_active")
    return "Kill Switch 활성 — 전체 매수 차단 중";
  if (key === "sync_stale")
    return "계좌 동기화 만료 — 최신 잔고 데이터 없음";
  if (key === "daily_loss_pct_unavailable")
    return "일간 손실률 미수신 — 안전을 위해 매수 차단";
  if (key === "daily_loss_exceeded")
    return "일간 손실 한도 초과 — 당일 매수 차단";
  if (key === "weekly_loss_exceeded")
    return "주간 손실 한도 초과 — 주간 매수 차단";
  if (key === "daily_buy_limit_exceeded")
    return "일간 매수 한도 초과";
  if (key === "weekly_buy_limit_exceeded")
    return "주간 매수 한도 초과";
  if (key.includes("gap_up"))
    return `가격 갭업 차단${gapStr}`;
  return key || "";
}

function buildPreviewBlockDetail(row) {
  const parts = [];
  if (row.blocked_reason) parts.push(`차단: ${row.blocked_reason}`);
  if (row.entry_price_gate_reason && row.entry_price_gate_reason !== row.blocked_reason)
    parts.push(`진입게이트: ${row.entry_price_gate_reason}`);
  const gapPct = Number(row.entry_price_gap_pct);
  if (Number.isFinite(gapPct))
    parts.push(`갭: ${gapPct >= 0 ? "+" : ""}${(gapPct * 100).toFixed(1)}%`);
  if (row.reference_price_source)
    parts.push(`갭 기준가 출처: ${translateReferenceSource(row.reference_price_source)}`);
  if (row.ranking_close != null)
    parts.push(`DB 종가: ${fmtNum(row.ranking_close)}`);
  if (row.live_previous_close != null)
    parts.push(`KIS 전일종가: ${fmtNum(row.live_previous_close)}`);
  if (row.live_price != null)
    parts.push(`실시간가: ${fmtNum(row.live_price)}`);
  if (row.quote_checked_at)
    parts.push(`시세 확인: ${fmtRuntimeDateTime(row.quote_checked_at)}`);
  const referenceNote = translateEntryReferenceNote(row.entry_reference_note);
  if (referenceNote)
    parts.push(referenceNote);
  const hold = String(row.expected_hold_reason || "");
  if (hold && !hold.startsWith("No expected"))
    parts.push(hold);
  if (!parts.length) return row.raw_reason || row.blocked_reason || "-";
  return parts.join(" | ");
}

function translateReferenceSource(source) {
  const key = String(source || "").trim().toLowerCase();
  if (key === "ranking_close") return "DB 종가";
  if (key === "kis_previous_close") return "KIS 전일종가";
  return key || "-";
}

function translateEntryReferenceNote(note) {
  const key = String(note || "").trim().toLowerCase();
  if (!key) return "";
  if (key.startsWith("ignored_kis_previous_close_during_")) {
    const marketStatus = key.replace("ignored_kis_previous_close_during_", "").toUpperCase();
    return `${marketStatus} 시간대라 KIS 전일종가 대신 DB 종가를 갭 기준가로 사용`;
  }
  return key;
}

function formatPreviewBlockedMessage(row, runtime) {
  if (row?.blocked_reason === "buy_qty_zero_budget_below_one_share") {
    return "남은 목표 비중 없음 - 배정 예산이 1주 가격 미만";
  }
  return translateBlockedReason(row) || describePreviewExecutionRisk(row, runtime);
}

function renderPreview(preview, runtime) {
  const tbody = document.getElementById("previewTbody");
  const rows = preview?.items || [];
  if (!rows.length) {
    document.getElementById("previewWrap").innerHTML = `<div class="empty-state">order requests preview 산출물이 아직 없습니다.</div>`;
    return;
  }
  tbody.innerHTML = rows.map((row) => {
    const isPolicyAllow = String(row.policy_status || "").toUpperCase() === "ALLOW";
    const blockMsg = row.blocked_reason && isPolicyAllow
      ? formatPreviewBlockedMessage(row, runtime)
      : row.user_message_ko;
    return `
    <tr>
      <td>${orderStateChip(row)}</td>
      <td class="mono">${escapeHtml(row.request_id || "-")}</td>
      <td class="mono">${escapeHtml(row.code || "-")}</td>
      <td>${escapeHtml(row.name || "-")}</td>
      <td>${escapeHtml(row.side || "-")}</td>
      <td>${escapeHtml(row.intent_type || "-")}</td>
      <td>${escapeHtml(row.policy_status || "-")}</td>
      <td>${escapeHtml(row.block_type || "-")}</td>
      <td>${escapeHtml(row.severity || "-")}</td>
      <td class="right">${fmtNum(row.final_request_qty)}</td>
      <td class="right">${fmtNum(row.allowed_qty)}</td>
      <td>${escapeHtml(blockMsg || describePreviewExecutionRisk(row, runtime))}</td>
      <td><details><summary>보기</summary>${escapeHtml(buildPreviewBlockDetail(row))}</details></td>
    </tr>
  `;
  }).join("");
}

// ── Execution table ───────────────────────────
function describeExecutionReason(reason, row, runtime) {
  const key = String(reason || "").trim();
  if (!key) return "-";
  const buyApprovalRequired = !!runtime?.policy?.buy_approval_required;
  if (key.startsWith("policy_blocked:")) return row?.user_message_ko || "정책 기준으로 주문이 차단되었습니다.";
  switch (key) {
    case "buy_approval_required": return buyApprovalRequired ? "매수 승인 목록에 없어 보류되었습니다." : "매수 승인 조건으로 보류되었습니다.";
    case "buy_requires_allow_buy": return "BUY 실주문 스위치가 꺼져 있어 매수가 보류되었습니다.";
    case "duplicate_request_id": return "이미 성공 처리된 요청 ID라 중복 제출을 건너뛰었습니다.";
    case "invalid_final_request_qty": return "최종 주문 수량이 0 이하라 제출하지 않았습니다.";
    case "holding_qty_missing": return "실계좌 보유수량을 찾지 못해 매도 주문을 만들지 못했습니다.";
    case "buy_qty_zero_budget_below_one_share": return "예산 부족 — 목표 예산이 1주 가격 미달";
    default:
      if (key.includes("market_closed")) return "장 운영 시간이 아니어서 주문이 제출되지 않았습니다.";
      if (key.includes("LIVE_ORDER")) return "실주문 확인 문구가 맞지 않아 제출이 차단되었습니다.";
      return key;
  }
}

function renderExecution(execution, preview, runtime) {
  const wrap = document.getElementById("executionWrap");
  const tbody = document.getElementById("executionTbody");
  const rows = execution?.items || [];
  if (!rows.length) {
    wrap.innerHTML = `<div class="empty-state">order requests execution 산출물이 아직 없습니다.</div>`;
    return;
  }

  const executedAt = execution?.executed_at || execution?.generated_at;
  const asofDate = execution?.asof_date;
  const existingMeta = wrap.previousElementSibling;
  if (existingMeta && existingMeta.classList.contains("execution-meta")) existingMeta.remove();
  if (executedAt) {
    const meta = document.createElement("div");
    meta.className = "execution-meta section-note";
    meta.style.marginBottom = "6px";
    meta.textContent = `실행: ${String(executedAt).slice(0, 16)}${asofDate ? `  ·  기준일: ${asofDate}` : ""}  ·  총 ${rows.length}건`;
    wrap.insertAdjacentElement("beforebegin", meta);
  }

  tbody.innerHTML = rows.map((row) => `
    <tr>
      <td>${executionStateChip(row)}</td>
      <td class="mono">${escapeHtml(row.request_id || "-")}</td>
      <td class="mono">${escapeHtml(row.code || "-")}</td>
      <td>${escapeHtml(row.name || "-")}</td>
      <td>${escapeHtml(row.side || "-")}</td>
      <td>${escapeHtml(row.broker_result || (String(row.submission_status || "").toLowerCase() === "failed" ? "BROKER_REJECT" : row.block_type || "-"))}</td>
      <td class="mono">${escapeHtml(row.broker_error_code || ((String(row.skip_reason || "").match(/msg_cd=([A-Z0-9_-]+)/i) || [])[1]) || "-")}</td>
      <td class="right">${fmtNum(row.final_request_qty)}</td>
      <td class="mono">${escapeHtml(row.broker_order_id || "-")}</td>
      <td class="mono">${escapeHtml(row.submitted_at ? String(row.submitted_at).slice(0, 16) : "-")}</td>
      <td>${escapeHtml(row.broker_error_message || describeExecutionReason(row.skip_reason, row, runtime))}</td>
    </tr>
  `).join("");
}

// ── Consistency ───────────────────────────────
function renderConsistency(consistency) {
  const root = document.getElementById("consistencyPanel");
  if (!root) return;
  if (!consistency || !Object.keys(consistency).length) {
    root.innerHTML = `<div class="empty-state">live_trade_consistency_report 산출물이 아직 없습니다.</div>`;
    return;
  }
  const counts = consistency.counts || {};
  const warnings = Array.isArray(consistency.warnings) ? consistency.warnings : [];
  const missingFills = Array.isArray(consistency.submitted_without_fill) ? consistency.submitted_without_fill : [];
  const warningCount = Number(consistency.warning_count || warnings.length || 0);
  const fillCount = Number(counts.filled_count || 0);
  const submittedCount = Number(counts.submitted_count || 0);

  const missingHtml = missingFills.length
    ? `
      <div class="table-wrap" style="margin-top:12px;">
        <table class="status-table">
          <thead><tr><th>request_id</th><th>주문번호</th><th>종목</th><th>구분</th><th>제출시각</th></tr></thead>
          <tbody>
            ${missingFills.slice(0, 8).map((row) => `
              <tr>
                <td class="mono">${escapeHtml(row.request_id || "-")}</td>
                <td class="mono">${escapeHtml(row.broker_order_id || "-")}</td>
                <td class="mono">${escapeHtml(row.code || "-")}</td>
                <td>${escapeHtml(row.side || "-")}</td>
                <td>${escapeHtml(row.submitted_at || "-")}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    `
    : "";

  root.innerHTML = `
    <div class="kv">
      <div class="kv-row"><span>기준일</span><strong>${escapeHtml(consistency.as_of_date || "-")}</strong></div>
      <div class="kv-row"><span>체결 / 제출</span><strong>${fmtNum(fillCount)} / ${fmtNum(submittedCount)}</strong></div>
      <div class="kv-row"><span>정합성 경고</span><strong class="${warningCount ? "neg" : "pos"}">${fmtNum(warningCount)}건</strong></div>
    </div>
    <div class="chip-row">
      <span class="chip ${warningCount ? "bad" : "good"}">경고 ${fmtNum(warningCount)}</span>
      <span class="chip ${missingFills.length ? "warn" : "good"}">미체결 ${fmtNum(missingFills.length)}</span>
    </div>
    ${warnings.slice(0, 4).map((item) => `<div class="state-line">${escapeHtml(item)}</div>`).join("")}
    ${missingHtml}
  `;
}

// ── Live KPI ──────────────────────────────────
function latestHorizonRow(rows, horizon) {
  return (Array.isArray(rows) ? rows : []).find((r) => Number(r.horizon) === horizon) || {};
}

function guardAppliedRow(rows, applied, horizon = 5) {
  return (Array.isArray(rows) ? rows : []).find((r) =>
    Number(r.horizon) === horizon && Boolean(r.shadow_quality_risk_guard_applied) === applied
  ) || {};
}

function translateLiveKpiWarning(text) {
  const v = String(text || "");
  if (v.includes("Some fills have no ranking context")) {
    return "일부 체결에 랭킹 문맥이 없습니다. 승격 분석 전 ledger/ranking 동기화를 확인하세요.";
  }
  if (v.includes("Live trade consistency report has warnings")) {
    return "자동매매 정합성 리포트에 경고가 있습니다. 주문 탭의 정합성 상태를 확인하세요.";
  }
  return v;
}

function translateGuardBucket(value) {
  const v = String(value || "");
  if (!v || v === "-") return "-";
  if (v === "penalty_unknown") return "페널티 미상";
  if (v.startsWith("penalty_")) return `페널티 ${v.replace("penalty_", "")}`;
  return v;
}

function translateShadowDelta(value) {
  const v = String(value || "");
  if (v === "shadow_rank_up") return "순위 개선";
  if (v === "shadow_rank_down") return "순위 하락";
  if (v === "shadow_rank_same") return "순위 유지";
  if (v === "shadow_rank_unknown") return "변화 미상";
  return v || "-";
}

function renderAnalysisSummary(liveKpi, guardReview, validation, closedTrade) {
  const root = document.getElementById("analysisSummaryGrid");
  if (!root) return;

  const d0 = latestHorizonRow(liveKpi?.horizon_summary, 0);
  const d5 = latestHorizonRow(liveKpi?.horizon_summary, 5);
  const overview = liveKpi?.overview || {};
  const today = liveKpi?.today_counts || {};
  const warningCount = Number(liveKpi?.consistency?.warning_count || 0);
  const missingRanking = Number(overview.missing_ranking_context_count || 0);

  const guardD5 = latestHorizonRow(guardReview?.horizon_summary, 5);
  const notAppliedD5 = guardAppliedRow(guardReview?.by_guard_applied, false, 5);
  const promotionStatus = String(guardReview?.promotion_status || "").toUpperCase();
  const canPromote = promotionStatus === "PROMOTE_CANDIDATE";
  const validationStatus = String(validation?.validation_status || "").toUpperCase();

  const closed = guardReview?.closed_trade_summary || closedTrade?.overview || {};
  const closedPnl = Number(closed.realized_net_pnl);
  const closedObserved = Number(closed.observed_count || 0);
  const closedWinRate = Number(closed.win_rate);

  const sampleReady = Number(d5.observed_count || guardD5.observed_count || 0) >= 30;
  const consistencyOk = warningCount === 0 && missingRanking === 0;
  const validationText = validationStatus === "PASS" ? "검증 통과" : (validationStatus || "검증 미확인");

  root.innerHTML = `
    <div class="analysis-summary-card ${canPromote ? "good" : "warn"}">
      <div class="analysis-summary-label">운영 반영 판단</div>
      <div class="analysis-summary-value">${canPromote ? "반영 후보" : "반영 보류"}</div>
      <div class="analysis-summary-detail">${escapeHtml(validationText)} · 미적용군 ${fmtNum(notAppliedD5.observed_count)}/30 · ${escapeHtml(statusText(guardReview?.promotion_status || "KEEP_SHADOW"))}</div>
    </div>
    <div class="analysis-summary-card ${sampleReady ? "good" : "warn"}">
      <div class="analysis-summary-label">성과 표본</div>
      <div class="analysis-summary-value">${sampleReady ? "관찰 적용" : "데이터 축적 중"}</div>
      <div class="analysis-summary-detail">D+5 ${escapeHtml(fmtPct(d5.avg_return ?? guardD5.avg_return, 2))} / ${fmtNum(d5.observed_count ?? guardD5.observed_count)}건 · D0 ${escapeHtml(fmtPct(d0.avg_return, 2))}</div>
    </div>
    <div class="analysis-summary-card ${consistencyOk ? "good" : "warn"}">
      <div class="analysis-summary-label">정합성</div>
      <div class="analysis-summary-value">${consistencyOk ? "정상" : `경고 ${fmtNum(warningCount + missingRanking)}건`}</div>
      <div class="analysis-summary-detail">랭킹 누락 ${fmtNum(missingRanking)}건 · 체결 ${fmtNum(today.fill_count)}/${fmtNum(today.execution_count)}</div>
    </div>
    <div class="analysis-summary-card ${Number.isFinite(closedPnl) && closedPnl < 0 ? "bad" : "good"}">
      <div class="analysis-summary-label">청산 손익</div>
      <div class="analysis-summary-value ${signedClass(closedPnl)}">${escapeHtml(fmtNum(closedPnl))}</div>
      <div class="analysis-summary-detail">관찰 ${fmtNum(closedObserved)}건${Number.isFinite(closedWinRate) ? ` · 승률 ${escapeHtml(fmtPct(closedWinRate, 1))}` : ""}</div>
    </div>
  `;
}

function renderLiveKpiDaily(report) {
  const root = document.getElementById("liveKpiPanel");
  if (!root) return;
  if (!report || !Object.keys(report).length) {
    root.innerHTML = `<div class="empty-state">live_kpi_daily_report 산출물이 아직 없습니다.</div>`;
    return;
  }
  const overview = report.overview || {};
  const account = report.account || {};
  const today = report.today_counts || {};
  const d0 = latestHorizonRow(report.horizon_summary, 0);
  const d5 = latestHorizonRow(report.horizon_summary, 5);
  const warnings = Array.isArray(report.warnings) ? report.warnings.slice(0, 3) : [];

  root.innerHTML = `
    <p class="card-guide">오늘 자동매매 산출물이 정상적으로 쌓였는지와 체결 후 수익률 표본이 충분한지 확인하는 카드입니다.</p>
    <div class="kv">
      <div class="kv-row"><span>${metricLabel("기준일", "이 리포트가 계산된 거래 기준일입니다.")}</span><strong>${escapeHtml(report.as_of_date || "-")}</strong></div>
      <div class="kv-row"><span>${metricLabel("표본 상태", "성과 판단에 필요한 관찰 건수 충분 여부입니다.")}</span><strong>${statusChip(report.sample_status)}</strong></div>
      <div class="kv-row"><span>${metricLabel("총자산 / 현금비중", "실계좌 요약 기준 총자산과 현금 비중입니다.")}</span><strong>${escapeHtml(fmtNum(account.total_assets))} / ${escapeHtml(fmtPct(account.cash_ratio, 1))}</strong></div>
      <div class="kv-row"><span>${metricLabel("판단 / 요청 / 제출 / 체결", "전략 판단부터 체결까지 오늘 생성된 건수 흐름입니다.")}</span><strong>${fmtNum(today.decision_count)} / ${fmtNum(today.request_count)} / ${fmtNum(today.execution_count)} / ${fmtNum(today.fill_count)}</strong></div>
      <div class="kv-row"><span>${metricLabel("D0 평균 / 관찰", "체결 당일 수익률 평균과 표본 수입니다.")}</span><strong class="${signedClass(d0.avg_return)}">${escapeHtml(fmtPct(d0.avg_return, 2))} / ${fmtNum(d0.observed_count)}</strong></div>
      <div class="kv-row"><span>${metricLabel("D+5 평균 / 관찰", "체결 후 5거래일 수익률 — 승격 판단의 핵심 표본입니다.")}</span><strong class="${signedClass(d5.avg_return)}">${escapeHtml(fmtPct(d5.avg_return, 2))} / ${fmtNum(d5.observed_count)}</strong></div>
    </div>
    <div class="chip-row">
      <span class="chip ${Number(overview.missing_ranking_context_count || 0) ? "warn" : "good"}">랭킹 누락 ${fmtNum(overview.missing_ranking_context_count)}</span>
      <span class="chip ${Number(report?.consistency?.warning_count || 0) ? "warn" : "good"}">정합성 경고 ${fmtNum(report?.consistency?.warning_count)}</span>
    </div>
    ${warnings.map((item) => `<div class="state-line">${escapeHtml(translateLiveKpiWarning(item))}</div>`).join("")}
    ${helpDetails([
      { term: "표본 상태", desc: "ACTIONABLE이면 참고 가능, MONITOR_ONLY면 데이터 축적 단계입니다." },
      { term: "D0 / D+5", desc: "D0는 체결 당일, D+5는 체결 후 5거래일 성과입니다." },
      { term: "랭킹 누락", desc: "체결 또는 판단 데이터와 리서치 랭킹 문맥이 연결되지 않은 건수입니다." },
    ])}
  `;
}

// ── Quality Guard ─────────────────────────────
function translatePromotionBlocker(text) {
  const v = String(text || "");
  if (v.includes("D+5 observed_count is below 30")) return "D+5 관찰 표본이 30건 미만입니다.";
  if (v.includes("Guard-applied observed_count is below 30")) return "Guard 적용군 관찰 표본이 30건 미만입니다.";
  if (v.includes("Guard-not-applied observed_count is below 30")) return "Guard 미적용군 관찰 표본이 30건 미만입니다.";
  if (v.includes("Production top20 vs shadow top20 comparison is not available")) return "Production Top20과 Shadow Top20 비교가 아직 불가능합니다.";
  if (v.includes("Closed-trade report is not available")) return "Closed Trade 리포트가 아직 없습니다.";
  if (v.includes("Closed-trade observed_count is below 30")) return "Closed Trade 관찰 표본이 30건 미만입니다.";
  if (v.includes("Closed-trade PnL uses position snapshot avg_price fallback")) return "청산 손익 일부는 계좌 스냅샷 평균단가로 보조 계산했습니다. lot 단위 근거는 근사값입니다.";
  return v;
}

function renderClosedQualityGuardTable(rows) {
  const values = Array.isArray(rows) ? rows.slice(0, 6) : [];
  if (!values.length) return "";
  return `
    <div class="table-wrap" style="margin-top:12px;">
      <table class="status-table">
        <thead>
          <tr><th>페널티</th><th>Shadow 변화</th><th class="right">건수</th><th class="right">관찰</th><th class="right">실현손익</th><th class="right">평균</th><th class="right">승률</th></tr>
        </thead>
        <tbody>
          ${values.map((row) => `
            <tr>
              <td>${escapeHtml(translateGuardBucket(row.guard_penalty_bucket))}</td>
              <td>${escapeHtml(translateShadowDelta(row.shadow_rank_delta_bucket))}</td>
              <td class="right">${fmtNum(row.count)}</td>
              <td class="right">${fmtNum(row.observed_count)}</td>
              <td class="right ${signedClass(row.realized_net_pnl)}">${escapeHtml(fmtNum(row.realized_net_pnl))}</td>
              <td class="right ${signedClass(row.avg_realized_return)}">${escapeHtml(fmtPct(row.avg_realized_return, 2))}</td>
              <td class="right">${escapeHtml(fmtPct(row.win_rate, 1))}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderQualityGuardReview(report, validation) {
  const root = document.getElementById("qualityGuardPanel");
  if (!root) return;
  if (!report || !Object.keys(report).length) {
    root.innerHTML = `<div class="empty-state">quality_risk_guard_live_review 산출물이 아직 없습니다.</div>`;
    return;
  }
  const overview = report.overview || {};
  const closed = report.closed_trade_summary || {};
  const d5 = latestHorizonRow(report.horizon_summary, 5);
  const appliedD5 = guardAppliedRow(report.by_guard_applied, true, 5);
  const notAppliedD5 = guardAppliedRow(report.by_guard_applied, false, 5);
  const blockers = Array.isArray(report.promotion_blockers) ? report.promotion_blockers.slice(0, 4) : [];
  const productionBlocked = String(report.promotion_status || "").toUpperCase() !== "PROMOTE_CANDIDATE";
  const validationStatus = String(validation?.validation_status || "").toUpperCase();
  const validationOk = validationStatus === "PASS";

  root.innerHTML = `
    <p class="card-guide">quality_risk_guard를 실제 운영 점수에 반영해도 되는지 보는 카드입니다.</p>
    <div class="kv">
      <div class="kv-row"><span>${metricLabel("승격 상태", "Quality Guard를 production에 반영할 수 있는지의 판단입니다.")}</span><strong>${statusChip(report.promotion_status)}</strong></div>
      <div class="kv-row"><span>${metricLabel("산출물 검증", "필수 값과 차단 조건 검사 결과입니다.")}</span><strong>${validationChip(validationStatus)}</strong></div>
      <div class="kv-row"><span>${metricLabel("표본 상태", "성과 비교에 필요한 표본 충분 여부입니다.")}</span><strong>${statusChip(report.sample_status)}</strong></div>
      <div class="kv-row"><span>${metricLabel("Guard 적용 / 미적용", "Shadow 계산에서 품질 가드 페널티가 걸린 후보 수입니다.")}</span><strong>${fmtNum(overview.guard_applied_count)} / ${fmtNum(overview.guard_not_applied_count)}</strong></div>
      <div class="kv-row"><span>${metricLabel("D+5 평균 / 관찰", "체결 후 5거래일 성과 — 승격 판단의 핵심 표본입니다.")}</span><strong class="${signedClass(d5.avg_return)}">${escapeHtml(fmtPct(d5.avg_return, 2))} / ${fmtNum(d5.observed_count)}</strong></div>
      <div class="kv-row"><span>${metricLabel("Closed PnL / 관찰", "실제 매도까지 끝난 거래의 실현손익과 표본 수입니다.")}</span><strong class="${signedClass(closed.realized_net_pnl)}">${escapeHtml(fmtNum(closed.realized_net_pnl))} / ${fmtNum(closed.observed_count)}</strong></div>
    </div>
    <div class="chip-row">
      <span class="chip ${productionBlocked ? "bad" : "good"}">${productionBlocked ? "운영 반영 보류" : "운영 반영 후보"}</span>
      <span class="chip ${Number(d5.observed_count || 0) >= 30 ? "good" : "warn"}">D+5 관찰 ${fmtNum(d5.observed_count)}</span>
      ${gateStatusChip(Number(appliedD5.observed_count || 0) >= 30, "적용군", fmtNum(appliedD5.observed_count))}
      ${gateStatusChip(Number(notAppliedD5.observed_count || 0) >= 30, "미적용군", fmtNum(notAppliedD5.observed_count))}
      <span class="chip ${validationOk ? "good" : "bad"}">${validationOk ? "검증 통과" : `검증 ${escapeHtml(validationStatus || "누락")}`}</span>
    </div>
    ${blockers.map((item) => `<div class="state-line">${escapeHtml(translatePromotionBlocker(item))}</div>`).join("")}
    ${renderClosedQualityGuardTable(closed.by_quality_guard || [])}
    ${helpDetails([
      { term: "데이터 축적 중", desc: "KEEP_SHADOW 상태입니다. 실제 매매 로직은 바꾸지 않고 결과만 비교합니다." },
      { term: "Guard 적용군", desc: "품질 가드가 위험하다고 판단해 Shadow 점수에 페널티를 준 후보입니다." },
      { term: "snapshot fallback", desc: "매수 lot 원가가 부족해 계좌 스냅샷 평균단가로 보조 계산한 청산 거래입니다." },
    ])}
  `;
}

// ── Trade Review ──────────────────────────────
function extractSignedReturn(item) {
  const keys = ["return_d5", "return_d3", "return_d1", "return_d0"];
  const labels = ["D+5", "D+3", "D+1", "D0"];
  for (let i = 0; i < keys.length; i++) {
    const v = Number(item[keys[i]]);
    if (Number.isFinite(v)) return { horizon: labels[i], value: v };
  }
  return { horizon: "-", value: null };
}

function reviewOutcomeChip(outcome) {
  const v = String(outcome || "").toLowerCase();
  if (v.includes("positive") || v.includes("good")) return "good";
  if (v.includes("pending")) return "warn";
  if (v.includes("bad") || v.includes("negative")) return "bad";
  return "watch";
}

function reviewOutcomeLabel(outcome) {
  const v = String(outcome || "").toLowerCase();
  if (v === "positive") return "긍정";
  if (v === "negative") return "부정";
  if (v === "neutral") return "중립";
  if (v === "pending_price_data") return "가격 대기";
  if (v.includes("pending")) return "관찰 대기";
  return outcome || "-";
}

function intentTypeLabel(value) {
  const v = String(value || "").toUpperCase();
  return ({ BUY: "매수", SELL: "매도", EXIT: "청산", REVIEW: "검토", HOLD: "보유" })[v] || v || "-";
}

function rankBucketLabel(value) {
  const v = String(value || "");
  if (v === "rank_top5") return "랭킹 1-5";
  if (v === "rank_top10") return "랭킹 6-10";
  if (v === "rank_top20") return "랭킹 11-20";
  if (v === "rank_21_plus") return "랭킹 21+";
  return v || "-";
}

function translateClosedWarning(text) {
  const v = String(text || "");
  if (v.includes("position snapshot avg_price as fallback")) {
    return "일부 청산 거래는 계좌 스냅샷 평균단가를 원가 보조값으로 사용했습니다. lot 단위 손익은 근사값입니다.";
  }
  return v;
}

function matchStatusLabel(value) {
  const v = String(value || "").toUpperCase();
  if (v === "MATCHED") return "매칭 완료";
  if (v === "PARTIAL") return "부분 매칭";
  if (v === "UNMATCHED") return "미매칭";
  return v || "-";
}

function renderReviewSummaryRows(title, rows, keyName) {
  const values = Array.isArray(rows) ? rows.slice(0, 4) : [];
  if (!values.length) return "";
  const titleLabel = title === "Intent" ? "판단 유형" : title === "Rank" ? "랭킹 구간" : title;
  const labelValue = (value) => {
    if (keyName === "intent_type") return intentTypeLabel(value);
    if (keyName === "rank_bucket") return rankBucketLabel(value);
    return value || "-";
  };
  return `
    <div class="table-wrap" style="margin-top:12px;">
      <table class="status-table">
        <thead><tr><th>${escapeHtml(titleLabel)}</th><th class="right">건수</th><th class="right">관찰</th><th class="right">평균</th><th class="right">승률</th></tr></thead>
        <tbody>
          ${values.map((row) => `
            <tr>
              <td>${escapeHtml(labelValue(row[keyName]))}</td>
              <td class="right">${fmtNum(row.count)}</td>
              <td class="right">${fmtNum(row.observed_count)}</td>
              <td class="right ${signedClass(row.avg_signed_return)}">${escapeHtml(fmtPct(row.avg_signed_return, 2))}</td>
              <td class="right">${escapeHtml(fmtPct(row.win_rate, 1))}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function reviewItemSortKey(item) {
  const requestId = String(item?.request_id || "");
  const basisDate = requestId.split(":")[0] || "";
  const fillDate = String(item?.fill_date || "");
  const filledAt = String(item?.filled_at || "");
  return [basisDate, fillDate, filledAt, requestId].join("|");
}

function renderTradeReview(review, summary) {
  const root = document.getElementById("reviewPanel");
  if (!root) return;
  if (!review || !Object.keys(review).length) {
    root.innerHTML = `<div class="empty-state">live_trade_review_report 산출물이 아직 없습니다.</div>`;
    return;
  }
  const items = Array.isArray(review.items) ? review.items : [];
  const outcomeCounts = Array.isArray(review.outcome_counts) ? review.outcome_counts : [];
  const overview = summary?.overview || {};
  const rows = items
    .slice()
    .sort((a, b) => reviewItemSortKey(b).localeCompare(reviewItemSortKey(a)))
    .slice(0, 10);
  const countsHtml = outcomeCounts.length
    ? outcomeCounts.map((r) => `<span class="chip ${reviewOutcomeChip(r.outcome_label)}">${escapeHtml(reviewOutcomeLabel(r.outcome_label))} ${fmtNum(r.count)}</span>`).join("")
    : `<span class="chip warn">성과 요약 없음</span>`;

  const tableHtml = rows.length
    ? `
      <div class="table-wrap" style="margin-top:12px;">
        <table class="status-table">
          <thead><tr><th>요청ID</th><th>코드</th><th>종목명</th><th>구분</th><th>판단</th><th class="right">체결가</th><th class="right">성과</th><th>판정</th></tr></thead>
          <tbody>
            ${rows.map((item) => {
              const ret = extractSignedReturn(item);
              return `
                <tr>
                  <td class="mono">${escapeHtml(item.request_id || "")}</td>
                  <td class="mono">${escapeHtml(item.code || "")}</td>
                  <td>${escapeHtml(item.name || "")}</td>
                  <td>${escapeHtml(intentTypeLabel(item.side))}</td>
                  <td>${escapeHtml(intentTypeLabel(item.intent_type))}</td>
                  <td class="right">${escapeHtml(fmtNum(item.filled_price, 0))}</td>
                  <td class="right ${signedClass(ret.value)}">${escapeHtml(ret.value === null ? "-" : `${ret.horizon} ${fmtPct(ret.value, 2)}`)}</td>
                  <td><span class="chip ${reviewOutcomeChip(item.outcome_label)}">${escapeHtml(reviewOutcomeLabel(item.outcome_label))}</span></td>
                </tr>
              `;
            }).join("")}
          </tbody>
        </table>
      </div>
    `
    : `<div class="empty-state" style="margin-top:12px;">리뷰 대상 체결이 없습니다.</div>`;

  root.innerHTML = `
    <p class="card-guide">체결 이후 D0/D+5/D+10 성과를 자동으로 되짚어 매수 판단이 실제로 작동했는지 보는 영역입니다.</p>
    <div class="mini-stat-grid">
      <div class="mini-stat-card">
        <div class="mini-stat-label">누적 리뷰</div>
        <div class="mini-stat-value">${fmtNum(overview.review_count || review.reviewed_count || items.length)}</div>
        <div class="mini-stat-detail">관찰 ${fmtNum(overview.observed_count)} · 대기 ${fmtNum(overview.pending_count)}</div>
      </div>
      <div class="mini-stat-card">
        <div class="mini-stat-label">평균 성과</div>
        <div class="mini-stat-value ${signedClass(overview.avg_signed_return)}">${escapeHtml(fmtPct(overview.avg_signed_return, 2))}</div>
        <div class="mini-stat-detail">체결 후 관찰 가능 표본 기준</div>
      </div>
      <div class="mini-stat-card">
        <div class="mini-stat-label">승률</div>
        <div class="mini-stat-value">${escapeHtml(fmtPct(overview.win_rate, 1))}</div>
        <div class="mini-stat-detail">양수 성과 비율</div>
      </div>
      <div class="mini-stat-card">
        <div class="mini-stat-label">최신 리뷰</div>
        <div class="mini-stat-value">${escapeHtml(review.review_date || overview.latest_review_date || "-")}</div>
        <div class="mini-stat-detail">가격 최신일 ${escapeHtml(review.price_latest_date || "-")}</div>
      </div>
    </div>
    <div class="kv">
      <div class="kv-row"><span>기준일 / 리뷰일</span><strong>${escapeHtml(review.as_of_date || "-")} / ${escapeHtml(review.review_date || "-")}</strong></div>
      <div class="kv-row"><span>가격 최신일</span><strong>${escapeHtml(review.price_latest_date || "-")}</strong></div>
      <div class="kv-row"><span>리뷰 건수</span><strong>${fmtNum(review.reviewed_count || items.length)}</strong></div>
      <div class="kv-row"><span>누적 평균 / 승률</span><strong>${escapeHtml(fmtPct(overview.avg_signed_return, 2))} / ${escapeHtml(fmtPct(overview.win_rate, 1))}</strong></div>
    </div>
    <div class="chip-row">${countsHtml}</div>
    ${renderReviewSummaryRows("Intent", summary?.by_intent, "intent_type")}
    ${renderReviewSummaryRows("Rank", summary?.by_rank_bucket, "rank_bucket")}
    <details class="analysis-details">
      <summary>최근 리뷰 상세 보기</summary>
      ${tableHtml}
    </details>
  `;
}

// ── Closed Trade ──────────────────────────────
function renderClosedTradeReport(report) {
  const root = document.getElementById("closedTradePanel");
  if (!root) return;
  if (!report || !Object.keys(report).length) {
    root.innerHTML = `<div class="empty-state">live_closed_trade_report 산출물이 아직 없습니다.</div>`;
    return;
  }
  const overview = report.overview || {};
  const warnings = Array.isArray(report.warnings) ? report.warnings.slice(0, 3) : [];
  const recentRows = Array.isArray(report.recent_closed_trades) ? report.recent_closed_trades.slice(0, 6) : [];

  const recentHtml = recentRows.length
    ? `
      <div class="table-wrap">
        <table class="status-table">
          <thead><tr><th>종목</th><th>판단</th><th class="right">수량</th><th class="right">매칭</th><th class="right">실현손익</th><th class="right">수익률</th><th>매칭상태</th></tr></thead>
          <tbody>
            ${recentRows.map((row) => `
              <tr>
                <td><span class="mono">${escapeHtml(row.code || "")}</span> ${escapeHtml(row.name || "")}</td>
                <td>${escapeHtml(intentTypeLabel(row.intent_type))}</td>
                <td class="right">${fmtNum(row.sell_qty)}</td>
                <td class="right">${fmtNum(row.matched_qty)}</td>
                <td class="right ${signedClass(row.realized_net_pnl)}">${escapeHtml(fmtNum(row.realized_net_pnl))}</td>
                <td class="right ${signedClass(row.realized_return)}">${escapeHtml(fmtPct(row.realized_return, 2))}</td>
                <td><span class="chip ${row.match_status === "MATCHED" ? "good" : "warn"}">${escapeHtml(matchStatusLabel(row.match_status))}</span></td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    `
    : `<div class="empty-state" style="margin-top:12px;">닫힌 거래가 아직 없습니다.</div>`;

  root.innerHTML = `
    <p class="card-guide">매도까지 끝난 거래만 모아 실제로 돈을 벌었는지 확인하는 카드입니다. 표본이 적으면 방향성만 참고합니다.</p>
    <div class="mini-stat-grid">
      <div class="mini-stat-card">
        <div class="mini-stat-label">실현손익</div>
        <div class="mini-stat-value ${signedClass(overview.realized_net_pnl)}">${escapeHtml(fmtNum(overview.realized_net_pnl))}</div>
        <div class="mini-stat-detail">닫힌 거래 ${fmtNum(overview.closed_trade_count)}건</div>
      </div>
      <div class="mini-stat-card">
        <div class="mini-stat-label">평균 수익률</div>
        <div class="mini-stat-value ${signedClass(overview.avg_realized_return)}">${escapeHtml(fmtPct(overview.avg_realized_return, 2))}</div>
        <div class="mini-stat-detail">관찰 ${fmtNum(overview.observed_count)}건</div>
      </div>
      <div class="mini-stat-card">
        <div class="mini-stat-label">승률</div>
        <div class="mini-stat-value">${escapeHtml(fmtPct(overview.win_rate, 1))}</div>
        <div class="mini-stat-detail">최대손실 ${escapeHtml(fmtPct(overview.max_loss, 2))}</div>
      </div>
      <div class="mini-stat-card">
        <div class="mini-stat-label">원가 매칭</div>
        <div class="mini-stat-value">${fmtNum(overview.unmatched_count)} 미매칭</div>
        <div class="mini-stat-detail">스냅샷 보조 ${fmtNum(overview.snapshot_fallback_count)}건</div>
      </div>
    </div>
    <div class="kv">
      <div class="kv-row"><span>${metricLabel("최근 청산일")}</span><strong>${escapeHtml(report.latest_closed_date || "-")}</strong></div>
      <div class="kv-row"><span>${metricLabel("표본 상태")}</span><strong>${statusChip(report.sample_status)}</strong></div>
      <div class="kv-row"><span>${metricLabel("닫힌 거래 / 관찰")}</span><strong>${fmtNum(overview.closed_trade_count)} / ${fmtNum(overview.observed_count)}</strong></div>
      <div class="kv-row"><span>${metricLabel("실현손익")}</span><strong class="${signedClass(overview.realized_net_pnl)}">${escapeHtml(fmtNum(overview.realized_net_pnl))}</strong></div>
      <div class="kv-row"><span>${metricLabel("평균 수익률 / 승률")}</span><strong class="${signedClass(overview.avg_realized_return)}">${escapeHtml(fmtPct(overview.avg_realized_return, 2))} / ${escapeHtml(fmtPct(overview.win_rate, 1))}</strong></div>
      <div class="kv-row"><span>${metricLabel("최대손실 / 미매칭")}</span><strong class="${signedClass(overview.max_loss)}">${escapeHtml(fmtPct(overview.max_loss, 2))} / ${fmtNum(overview.unmatched_count)}</strong></div>
    </div>
    ${warnings.map((item) => `<div class="state-line">${escapeHtml(translateClosedWarning(item))}</div>`).join("")}
    <details class="analysis-details">
      <summary>최근 청산 상세 보기</summary>
      ${recentHtml}
    </details>
    ${helpDetails([
      { term: "닫힌 거래", desc: "매수 후 매도까지 발생해 실현손익을 계산할 수 있는 거래입니다." },
      { term: "FIFO", desc: "먼저 산 수량을 먼저 판 것으로 보고 매도 원가를 연결하는 방식입니다." },
      { term: "표본 기준", desc: "청산 관찰 표본 30건 이상을 최소 기준으로 봅니다." },
    ])}
  `;
}

// ── Holdings table ────────────────────────────
function renderHoldings(holdings) {
  const tbody = document.getElementById("holdingsTbody");
  const rows = holdings?.items || [];
  if (!rows.length) {
    document.getElementById("holdingsWrap").innerHTML = `<div class="empty-state">실계좌 보유 CSV가 아직 없습니다.</div>`;
    return;
  }
  tbody.innerHTML = rows.map((row) => {
    const peakPrice = Number(row.peak_price);
    const currentPrice = Number(row.current_price);
    let peakDrawdownHtml = "-";
    if (peakPrice > 0 && currentPrice > 0) {
      const drawdown = (currentPrice - peakPrice) / peakPrice;
      const cls = drawdown <= -0.05 ? "neg" : drawdown < 0 ? "" : "pos";
      peakDrawdownHtml = `<span class="${cls}">${drawdown >= 0 ? "+" : ""}${(drawdown * 100).toFixed(1)}%</span>`;
    }
    return `
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
      <td class="right">${row.holding_days != null ? fmtNum(row.holding_days) + "일" : "-"}</td>
      <td>${row.entry_date ? escapeHtml(row.entry_date) : "-"}</td>
      <td class="right">${peakDrawdownHtml}</td>
    </tr>`;
  }).join("");
}

// ── Analysis: feature importance ──────────────
const FEATURE_LABELS = {
  quality_score: "퀄리티 종합점수", ma_60: "이동평균 60일", vol_60: "거래량 60일",
  vol_ma_20: "거래량MA 20일", ret_120d: "수익률 120일", ma_20: "이동평균 20일",
  high_52w_ratio: "52주 고점 근접도", close: "현재가", ma_5: "이동평균 5일",
  vol_20: "거래량 20일", ret_60d: "수익률 60일", flow_inst_net_20d: "기관 순매수 20일",
  flow_foreign_net_20d: "외국인 순매수 20일", quality_factor_count: "퀄리티 충족 팩터 수",
  mom_20: "모멘텀 20일", ret_5d: "수익률 5일", rsi_14: "RSI 14일",
  flow_foreign_net_5d: "외국인 순매수 5일", flow_inst_net_5d: "기관 순매수 5일",
  atr_14: "변동성 ATR 14일", macd_hist: "MACD 히스토그램",
};

function renderFeatureImportance(data) {
  const root = document.getElementById("featureImportancePanel");
  if (!root) return;
  if (!data || !Array.isArray(data.features) || !data.features.length) {
    root.innerHTML = `<div class="empty-state">피처 중요도 데이터가 없습니다.</div>`;
    return;
  }
  const rows = data.features.slice(0, 15);
  const isFlow = (name) => name.startsWith("flow_");
  const top = rows[0];
  const flowCount = rows.filter((f) => isFlow(f.name)).length;
  root.innerHTML = `
    <div class="mini-stat-grid">
      <div class="mini-stat-card">
        <div class="mini-stat-label">최상위 근거</div>
        <div class="mini-stat-value">${escapeHtml(FEATURE_LABELS[top?.name] || top?.name || "-")}</div>
        <div class="mini-stat-detail">중요도 ${fmtNum(top?.importance)} · 상대 ${fmtNum(top?.pct)}%</div>
      </div>
      <div class="mini-stat-card">
        <div class="mini-stat-label">수급 피처</div>
        <div class="mini-stat-value">${fmtNum(flowCount)}개</div>
        <div class="mini-stat-detail">상위 15개 기준</div>
      </div>
      <div class="mini-stat-card">
        <div class="mini-stat-label">모델 버전</div>
        <div class="mini-stat-value">${escapeHtml(data.model_version || "-")}</div>
        <div class="mini-stat-detail">학습 ${escapeHtml(String(data.trained_at || "-").slice(0, 16))}</div>
      </div>
      <div class="mini-stat-card">
        <div class="mini-stat-label">표시 피처</div>
        <div class="mini-stat-value">${fmtNum(rows.length)}</div>
        <div class="mini-stat-detail">전체 ${fmtNum(data.features.length)}개 중 상위</div>
      </div>
    </div>
    <p style="font-size:12px;color:var(--color-text-secondary);margin:0 0 12px;">
      모델 버전: ${escapeHtml(data.model_version || "-")} · 학습일시: ${escapeHtml(String(data.trained_at || "-").slice(0, 16))}
    </p>
    <div style="display:flex;flex-direction:column;gap:6px;">
      ${rows.map((f) => {
        const label = FEATURE_LABELS[f.name] || f.name;
        const pct = Math.max(1, f.pct || 0);
        const isFlowFeature = isFlow(f.name);
        const barColor = isFlowFeature ? "rgba(96,165,250,0.6)" : "rgba(134,239,172,0.5)";
        return `
          <div style="display:grid;grid-template-columns:160px 1fr 48px;align-items:center;gap:8px;font-size:12px;">
            <span style="color:var(--color-text-secondary);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" title="${escapeHtml(f.name)}">
              ${isFlowFeature ? "수급 · " : ""}${escapeHtml(label)}
            </span>
            <div style="background:var(--color-bg-tertiary);border-radius:3px;height:10px;overflow:hidden;">
              <div style="width:${pct}%;height:100%;background:${barColor};border-radius:3px;"></div>
            </div>
            <span style="color:var(--color-text-secondary);text-align:right;">${fmtNum(f.importance)}</span>
          </div>`;
      }).join("")}
    </div>
    <div style="font-size:11px;color:var(--color-text-secondary);margin-top:10px;">수급 표시는 외국인·기관 순매수 관련 피처입니다.</div>
  `;
}

// ── Analysis: flow history chart ───────────────
function renderFlowHistory(data, holdings) {
  const root = document.getElementById("flowHistoryPanel");
  if (!root) return;
  const items = data?.items || {};
  const holdingCodes = (holdings?.items || []).map((h) => h.code);
  if (!holdingCodes.length) {
    root.innerHTML = `<div class="empty-state">보유 종목이 없습니다.</div>`;
    return;
  }
  const holdingMap = Object.fromEntries((holdings?.items || []).map((h) => [h.code, h.name || h.code]));

  const sections = holdingCodes.map((code) => {
    const rows = items[code] || [];
    if (!rows.length) return `<div style="margin-bottom:16px;"><strong>${escapeHtml(holdingMap[code])}</strong> — 수급 데이터 없음</div>`;
    const maxAbs = Math.max(...rows.flatMap((r) => [Math.abs(r.foreign_net), Math.abs(r.inst_net)]), 1);
    const barHtml = rows.slice(0, 15).map((r) => {
      const fBar = Math.round(Math.abs(r.foreign_net) / maxAbs * 60);
      const iBar = Math.round(Math.abs(r.inst_net) / maxAbs * 60);
      const fDir = r.foreign_net >= 0 ? "pos" : "neg";
      const iDir = r.inst_net >= 0 ? "pos" : "neg";
      const fColor = r.foreign_net >= 0 ? "rgba(134,239,172,0.6)" : "rgba(248,113,113,0.5)";
      const iColor = r.inst_net >= 0 ? "rgba(96,165,250,0.6)" : "rgba(251,191,36,0.5)";
      return `
        <div style="display:grid;grid-template-columns:72px 1fr 1fr;gap:4px;align-items:center;font-size:11px;padding:2px 0;">
          <span style="color:var(--color-text-secondary);">${escapeHtml(r.date.slice(5))}</span>
          <div style="display:flex;align-items:center;gap:2px;">
            <span style="width:24px;color:var(--color-text-secondary);text-align:right;font-size:10px;">외국인</span>
            <div style="width:${fBar}px;height:8px;background:${fColor};border-radius:2px;" title="${r.foreign_net > 0 ? "+" : ""}${Math.round(r.foreign_net).toLocaleString("ko-KR")}주"></div>
          </div>
          <div style="display:flex;align-items:center;gap:2px;">
            <span style="width:24px;color:var(--color-text-secondary);text-align:right;font-size:10px;">기관</span>
            <div style="width:${iBar}px;height:8px;background:${iColor};border-radius:2px;" title="${r.inst_net > 0 ? "+" : ""}${Math.round(r.inst_net).toLocaleString("ko-KR")}주"></div>
          </div>
        </div>`;
    }).join("");
    return `
      <div style="margin-bottom:20px;">
        <div style="font-size:13px;font-weight:600;margin-bottom:6px;">${escapeHtml(holdingMap[code])} <span class="mono" style="font-weight:400;font-size:11px;color:var(--color-text-secondary);">${escapeHtml(code)}</span></div>
        ${barHtml}
      </div>`;
  });
  root.innerHTML = `
    <div style="font-size:11px;color:var(--color-text-secondary);margin-bottom:10px;">초록=순매수 / 빨강·노랑=순매도 · 단위: 주(株) · 막대 위에 마우스 올리면 수치 확인</div>
    ${sections.join("")}
  `;
}

// ── Analysis: regime history ───────────────────
function renderRegimeHistory(data) {
  const root = document.getElementById("regimeHistoryPanel");
  if (!root) return;
  const items = Array.isArray(data?.items) ? data.items : [];
  if (!items.length) {
    root.innerHTML = `<div class="empty-state">레짐 히스토리 데이터가 없습니다.</div>`;
    return;
  }
  const statusColor = { RISK_ON: "good", NEUTRAL: "watch", RISK_OFF: "bad" };
  const statusLabel = { RISK_ON: "Risk-On ↑", NEUTRAL: "Neutral", RISK_OFF: "Risk-Off ↓" };
  root.innerHTML = `
    <div class="table-wrap">
      <table class="status-table">
        <thead><tr><th>KR 적용일</th><th>US Macro 상태</th></tr></thead>
        <tbody>
          ${items.slice(0, 20).map((r) => `
            <tr>
              <td>${escapeHtml(r.date)}</td>
              <td><span class="chip ${statusColor[r.macro_status] || "warn"}">${escapeHtml(statusLabel[r.macro_status] || r.macro_status)}</span></td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

// ── Main ──────────────────────────────────────
async function main() {
  initTabs();

  const state = document.getElementById("pageState");
  if (state) state.textContent = "AI 실자동매매 데이터를 불러오는 중...";

  try {
    const [
      summary, intents, preview, execution, runtime, holdings,
      consistency, tradeReview, tradeReviewSummary,
      liveKpiDaily, qualityGuardReview, closedTradeReport,
      qualityGuardOutputCheck, diagnostics,
      featureImportance, regimeHistory,
    ] = await Promise.all([
      fetchJsonMaybe("/api/live-account/summary").catch(() => null),
      fetchJsonMaybe("/api/trade-intents").catch(() => null),
      fetchJsonMaybe("/api/order-requests-preview").catch(() => null),
      fetchJsonMaybe("/api/order-requests-execution").catch(() => null),
      fetchJsonMaybe("/api/auto-trading/runtime-status").catch(() => null),
      fetchJsonMaybe("/api/live-account/holdings").catch(() => null),
      fetchJsonMaybe("/api/live-trade-consistency").catch(() => null),
      fetchJsonMaybe("/api/live-trade-review-report").catch(() => null),
      fetchJsonMaybe("/api/live-trade-review-summary").catch(() => null),
      fetchJsonMaybe("/api/live-kpi-daily-report").catch(() => null),
      fetchJsonMaybe("/api/quality-risk-guard-live-review").catch(() => null),
      fetchJsonMaybe("/api/live-closed-trade-report").catch(() => null),
      fetchJsonMaybe("/api/live-quality-guard-output-check").catch(() => null),
      fetchJsonMaybe("/api/live-auto-trading-diagnostics").catch(() => null),
      fetchJsonMaybe("/api/model-feature-importance").catch(() => null),
      fetchJsonMaybe("/api/regime-history?days=30").catch(() => null),
    ]);

    // ── 개요 탭 ──
    renderSafety(summary, runtime, intents, preview);
    renderDecisionBanner(summary, intents, preview, execution, runtime, consistency);
    renderHero(summary, intents, preview, holdings, execution);
    renderAccountDetails(summary, runtime);
    renderRunSummary(intents, preview, holdings, runtime);
    renderFocus(intents, preview, holdings);
    renderOperationalExplain(intents, preview, runtime, holdings);

    // ── 주문 탭 ──
    renderDiagnosticSummary(diagnostics);
    renderWhyNoTrade(diagnostics);
    renderIntents(intents);
    renderPreview(preview, runtime);
    renderExecution(execution, preview, runtime);
    renderConsistency(consistency);

    // ── 분석 탭 ──
    renderFeatureImportance(featureImportance);
    renderRegimeHistory(regimeHistory);
    renderLiveKpiDaily(liveKpiDaily);
    renderQualityGuardReview(qualityGuardReview, qualityGuardOutputCheck);
    renderTradeReview(tradeReview, tradeReviewSummary);
    renderClosedTradeReport(closedTradeReport);
    renderAnalysisSummary(liveKpiDaily, qualityGuardReview, qualityGuardOutputCheck, closedTradeReport);

    // flow history는 holdings 로드 후 별도 호출 (보유 종목 코드 필요)
    const holdingCodes = (holdings?.items || []).map((h) => h.code).join(",");
    if (holdingCodes) {
      fetchJsonMaybe(`/api/flow-history?codes=${holdingCodes}&days=20`)
        .then((flowData) => renderFlowHistory(flowData, holdings))
        .catch(() => renderFlowHistory(null, holdings));
    } else {
      renderFlowHistory(null, holdings);
    }

    // ── 계좌 탭 ──
    renderHoldings(holdings);

    // ── 탭 배지 ──
    updateBadges(intents, preview, execution, holdings, liveKpiDaily);

    const loaded = [
      summary && "summary", intents && "intents", preview && "preview", execution && "execution",
      runtime && "runtime", holdings && "holdings", consistency && "consistency",
      tradeReview && "review", liveKpiDaily && "kpi", qualityGuardReview && "qualityGuard",
      closedTradeReport && "closedTrade", diagnostics && "diagnostics",
    ].filter(Boolean);

    if (state) state.textContent = loaded.length
      ? `불러온 데이터: ${loaded.join(", ")}`
      : "AI 실자동매매 산출물이 아직 없습니다.";

  } catch (error) {
    console.error(error);
    if (state) state.textContent = `데이터를 불러오지 못했습니다: ${error.message}`;
  }
}

void main();
