/* US 자동매매 (Paper Trading) 운영 보드 */

const state = {
  summary: null,
  buyDecisions: [],
  sellDecisions: [],
  positions: [],
  guards: [],
  loading: false,
};

// ── 포맷 헬퍼 ──────────────────────────────────────────
function esc(v) {
  return String(v ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function fmt(v, digits = 1) {
  if (v == null || !Number.isFinite(Number(v))) return "—";
  return Number(v).toFixed(digits);
}
function fmtUsd(v, digits = 2) {
  if (v == null || !Number.isFinite(Number(v))) return "—";
  const n = Number(v);
  const sign = n >= 0 ? "+" : "";
  return sign + "$" + Math.abs(n).toFixed(digits);
}
function fmtPct(v, digits = 2) {
  if (v == null || !Number.isFinite(Number(v))) return "—";
  const n = Number(v) * 100;
  return (n >= 0 ? "+" : "") + n.toFixed(digits) + "%";
}
function posneg(v) {
  if (v == null || !Number.isFinite(Number(v))) return "";
  return Number(v) >= 0 ? "pos" : "neg";
}
function decisionChip(d) {
  if (!d) return '<span class="chip muted">—</span>';
  const map = {
    ALLOWED: "allowed", BLOCKED: "blocked",
    FULL_SELL: "sell", PARTIAL_SELL: "sell",
    HOLD: "hold", REVIEW_REQUIRED: "review",
  };
  return `<span class="chip ${map[d] || ""}">${esc(d)}</span>`;
}
function guardChip(status) {
  const map = { PASS: "pass", FAIL: "fail", WARN: "warn" };
  return `<span class="chip ${map[status] || ""}">${esc(status || "—")}</span>`;
}
function gradeChip(g) {
  const map = { STRONG_BUY: "allowed", BUY: "allowed", WATCH: "warn", HOLD: "hold", EXCLUDE: "blocked" };
  return `<span class="chip ${map[g] || ""}">${esc(g || "—")}</span>`;
}

// ── 탭 전환 ────────────────────────────────────────────
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("is-active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("is-active"));
    btn.classList.add("is-active");
    document.getElementById(`tab-${tab}`)?.classList.add("is-active");
  });
});

// ── Safety banner ──────────────────────────────────────
function updateBanner(summary) {
  const banner = document.getElementById("safetyBanner");
  const text = document.getElementById("safetyText");
  if (!summary || !summary.trade_date) {
    banner.className = "safety-banner warn";
    text.textContent = "Paper Trading — 데이터 없음 (파이프라인 실행 전)";
    return;
  }
  const mode = summary.mode || "PAPER";
  const weekPnl = Number(summary.week_pnl_usd || 0);
  const pct = summary.open_positions > 0 ? "" : " · 보유 없음";
  const weekStr = weekPnl === 0 ? "" : ` · 7일 손익 ${fmtUsd(weekPnl)}`;
  banner.className = "safety-banner " + (weekPnl >= 0 ? "ok" : "warn");
  text.textContent = `Paper Trading 활성 — 모드 ${mode}${weekStr}${pct}`;
}

// ── 개요 탭 ────────────────────────────────────────────
function renderOverview(summary) {
  if (!summary) return;
  const { trade_date, mode, allowed_count, blocked_count, total_decisions,
    sell_count, hold_count, open_positions, total_unrealized_pnl_usd,
    week_pnl_usd, generated_at } = summary;

  // 결론 카드
  const hasBuy = allowed_count > 0;
  const decCard = document.getElementById("decCardConclusion");
  const decVal = document.getElementById("decConclusion");
  const decDet = document.getElementById("decConclusionDetail");
  if (allowed_count > 0) {
    decCard.className = "decision-card primary";
    decVal.textContent = `매수 ${allowed_count}건`;
  } else if (sell_count > 0) {
    decCard.className = "decision-card warn";
    decVal.textContent = `매도 ${sell_count}건`;
  } else {
    decCard.className = "decision-card";
    decVal.textContent = "HOLD";
  }
  decDet.textContent = `매수 후보 ${total_decisions}건 검토 · 차단 ${blocked_count}건 · 매도 ${sell_count}건`;

  // 모드 카드
  document.getElementById("decMode").textContent = mode;
  document.getElementById("decModeDetail").textContent =
    mode === "PAPER" ? "Paper Trading 활성 — 가상 포트폴리오 주문" :
    mode === "SHADOW" ? "Shadow 모드 — 결정 로그만 기록" :
    mode === "LIVE" ? "LIVE 실주문 모드" : "—";

  // 갱신 카드
  const genAt = generated_at ? new Date(generated_at).toLocaleString("ko-KR", { timeZone: "Asia/Seoul" }) : "—";
  document.getElementById("decUpdated").textContent = genAt.slice(0, 16);
  document.getElementById("decUpdatedDetail").textContent = `기준일 ${trade_date || "—"}`;

  // 히어로 4개
  document.getElementById("heroDate").textContent = trade_date || "—";
  document.getElementById("heroDateDetail").textContent = "마지막 의사결정 기준일";

  const posEl = document.getElementById("heroPositions");
  posEl.textContent = open_positions ?? "—";
  posEl.className = "card-value";
  document.getElementById("heroPositionsDetail").textContent = "현재 오픈 포지션 수";

  const pnlEl = document.getElementById("heroPnl");
  pnlEl.textContent = fmtUsd(total_unrealized_pnl_usd);
  pnlEl.className = "card-value " + posneg(total_unrealized_pnl_usd);
  document.getElementById("heroPnlDetail").textContent = "미실현 평가손익 합계";

  const weekEl = document.getElementById("heroWeek");
  weekEl.textContent = fmtUsd(week_pnl_usd);
  weekEl.className = "card-value " + posneg(week_pnl_usd);
  const weekPct = week_pnl_usd !== 0 ? ` (${fmtPct(week_pnl_usd / 10000)})` : "";
  document.getElementById("heroWeekDetail").textContent = "최근 7일 실현 손익";

  // 배지
  document.getElementById("badgeOrders").textContent = total_decisions;
  document.getElementById("badgeOrders").className = "tab-badge" + (total_decisions > 0 ? " active" : "");
  document.getElementById("badgePositions").textContent = open_positions ?? 0;
  document.getElementById("badgePositions").className = "tab-badge" + (open_positions > 0 ? " active" : "");
}

// ── 포지션 미리보기 (개요 탭) ──────────────────────────
function renderPositionPreview(rows) {
  const body = document.getElementById("positionPreviewBody");
  const snap = rows[0]?.snapshot_date;
  if (snap) document.getElementById("posSnapshotDate").textContent = "스냅샷 기준: " + String(snap).slice(0, 10);

  if (!rows.length) {
    body.innerHTML = '<div class="empty-state">보유 포지션이 없습니다.</div>';
    return;
  }
  const html = `
    <div class="table-wrap"><table>
      <thead><tr>
        <th>심볼</th><th>종목명</th><th>섹터</th>
        <th class="right">현재가(USD)</th><th class="right">수량</th>
        <th class="right">미실현 손익(USD)</th><th class="right">수익률</th>
        <th class="right">보유일</th>
      </tr></thead>
      <tbody>
        ${rows.map((r) => `
          <tr>
            <td><strong>${esc(r.symbol)}</strong></td>
            <td class="muted" style="font-size:12px;">${esc(r.company_name || "—")}</td>
            <td class="muted" style="font-size:12px;">${esc(r.sector || "—")}</td>
            <td class="right">$${fmt(r.latest_price, 2)}</td>
            <td class="right">${fmt(r.remaining_quantity, 4)}</td>
            <td class="right ${posneg(r.unrealized_pnl)}">${fmtUsd(r.unrealized_pnl)}</td>
            <td class="right ${posneg(r.unrealized_pnl_pct)}">${fmtPct(r.unrealized_pnl_pct)}</td>
            <td class="right">${r.holding_days ?? "—"}일</td>
          </tr>`).join("")}
      </tbody>
    </table></div>`;
  body.innerHTML = html;
}

// ── 매수 결정 테이블 ────────────────────────────────────
function renderBuyDecisions(rows, tradeDate) {
  document.getElementById("buyDecisionDate").textContent = tradeDate ? "기준일: " + tradeDate : "";
  const tbody = document.getElementById("buyDecisionBody");
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="10" class="muted" style="text-align:center;padding:16px;">해당 날짜 데이터 없음</td></tr>';
    return;
  }
  tbody.innerHTML = rows.map((r) => {
    const blockReasons = (() => {
      try { const arr = JSON.parse(r.block_reasons || "[]"); return arr.join(", ") || "—"; } catch { return r.decision_reason_code || "—"; }
    })();
    return `<tr>
      <td><strong>${esc(r.symbol)}</strong></td>
      <td style="font-size:12px;">${esc(r.company_name || "—")}</td>
      <td style="font-size:11px;color:var(--color-text-secondary);">${esc(r.sector || "—")}</td>
      <td class="right">${r.rank_no ?? "—"}</td>
      <td>${gradeChip(r.recommend_grade)}</td>
      <td class="right">${fmt(r.total_score)}</td>
      <td>${decisionChip(r.decision)}</td>
      <td style="font-size:11px;color:var(--color-text-secondary);">${esc(blockReasons)}</td>
      <td class="right">${r.planned_order_amount_usd ? "$" + fmt(r.planned_order_amount_usd, 0) : "—"}</td>
      <td style="font-size:11px;">${esc(r.assumed_fill_status || "—")}</td>
    </tr>`;
  }).join("");
}

// ── 매도 결정 ───────────────────────────────────────────
function renderSellDecisions(rows, tradeDate) {
  document.getElementById("sellDecisionDate").textContent = tradeDate ? "기준일: " + tradeDate : "";
  const body = document.getElementById("sellDecisionBody");
  if (!rows.length) {
    body.innerHTML = '<div class="empty-state">매도 대상 포지션 없음 또는 데이터 없음</div>';
    return;
  }
  body.innerHTML = `
    <div class="table-wrap"><table>
      <thead><tr>
        <th>심볼</th><th>결정</th><th>매도 유형</th><th>종료 이유</th>
        <th class="right">현재가(USD)</th><th class="right">평균단가(USD)</th>
        <th class="right">수익률</th><th class="right">실현 손익(USD)</th>
      </tr></thead>
      <tbody>
        ${rows.map((r) => `<tr>
          <td><strong>${esc(r.symbol)}</strong></td>
          <td>${decisionChip(r.decision)}</td>
          <td style="font-size:12px;">${esc(r.sell_action || "—")}</td>
          <td style="font-size:11px;color:var(--color-text-secondary);">${esc(r.exit_reason || "—")}</td>
          <td class="right">$${fmt(r.latest_price, 2)}</td>
          <td class="right">$${fmt(r.avg_entry_price, 2)}</td>
          <td class="right ${posneg(r.unrealized_pnl_pct)}">${fmtPct(r.unrealized_pnl_pct)}</td>
          <td class="right ${posneg(r.realized_paper_pnl)}">${fmtUsd(r.realized_paper_pnl)}</td>
        </tr>`).join("")}
      </tbody>
    </table></div>`;
}

// ── 포지션 탭 ───────────────────────────────────────────
function renderPositions(rows, snapshotDate) {
  document.getElementById("positionsSnapshotDate").textContent = snapshotDate ? "스냅샷 기준: " + snapshotDate : "";
  const body = document.getElementById("positionsBody");
  if (!rows.length) {
    body.innerHTML = '<div class="empty-state">현재 오픈 포지션이 없습니다. 파이프라인 실행 후 매수 결정이 있어야 포지션이 생성됩니다.</div>';
    return;
  }
  body.innerHTML = `
    <div class="table-wrap"><table>
      <thead><tr>
        <th>심볼</th><th>종목명</th><th>섹터</th>
        <th class="right">현재가(USD)</th><th class="right">수량</th>
        <th class="right">52주 최고가(USD)</th>
        <th class="right">미실현 손익(USD)</th><th class="right">수익률</th>
        <th class="right">보유일</th>
      </tr></thead>
      <tbody>
        ${rows.map((r) => `<tr>
          <td><strong>${esc(r.symbol)}</strong></td>
          <td style="font-size:12px;">${esc(r.company_name || "—")}</td>
          <td style="font-size:11px;color:var(--color-text-secondary);">${esc(r.sector || "—")}</td>
          <td class="right">$${fmt(r.latest_price, 2)}</td>
          <td class="right">${fmt(r.remaining_quantity, 4)}</td>
          <td class="right">$${fmt(r.highest_price_since_entry, 2)}</td>
          <td class="right ${posneg(r.unrealized_pnl)}">${fmtUsd(r.unrealized_pnl)}</td>
          <td class="right ${posneg(r.unrealized_pnl_pct)}">${fmtPct(r.unrealized_pnl_pct)}</td>
          <td class="right">${r.holding_days ?? "—"}일</td>
        </tr>`).join("")}
      </tbody>
    </table></div>`;
}

// ── 리스크 가드 ─────────────────────────────────────────
function renderGuards(rows) {
  const grid = document.getElementById("guardGrid");
  const dateEl = document.getElementById("guardsDate");
  if (rows.length > 0 && rows[0].trade_date) {
    dateEl.textContent = "기준일: " + String(rows[0].trade_date).slice(0, 10);
  }
  if (!rows.length) {
    grid.innerHTML = '<div class="empty-state" style="grid-column:1/-1;">리스크 가드 데이터 없음 (파이프라인 실행 후 생성됩니다)</div>';
    return;
  }
  const classMap = { PASS: "pass", FAIL: "fail", WARN: "warn" };
  grid.innerHTML = rows.map((r) => `
    <div class="guard-card ${classMap[r.guard_status] || ""}">
      <div class="guard-name">${esc(r.guard_name)}</div>
      <div class="guard-status">
        ${guardChip(r.guard_status)}
        <span style="font-size:11px;margin-left:6px;color:var(--color-text-secondary);">${esc(r.guard_scope || "")}</span>
      </div>
      ${r.reason_detail ? `<div style="font-size:11px;margin-top:6px;color:var(--color-text-secondary);">${esc(r.reason_detail)}</div>` : ""}
      ${r.metric_value != null ? `<div style="font-size:11px;margin-top:4px;">값: ${fmt(r.metric_value)} / 기준: ${fmt(r.threshold_value)}</div>` : ""}
    </div>`).join("");
}

// ── 날짜 드롭다운 ───────────────────────────────────────
async function loadOrderDates() {
  try {
    const res = await fetch("/api/us/trading/dates");
    if (!res.ok) return;
    const data = await res.json();
    const select = document.getElementById("filterOrderDate");
    select.innerHTML = '<option value="">최신</option>' +
      (data.dates || []).map((d) => `<option value="${esc(d)}">${esc(d)}</option>`).join("");
  } catch {}
}

// ── 데이터 로드 함수들 ──────────────────────────────────
async function loadSummary() {
  try {
    const res = await fetch("/api/us/trading/summary");
    if (!res.ok) return null;
    return await res.json();
  } catch { return null; }
}

async function loadBuyDecisions(date) {
  const params = date ? `?date=${encodeURIComponent(date)}` : "";
  try {
    const res = await fetch("/api/us/trading/decisions/buy" + params);
    if (!res.ok) return { rows: [], trade_date: null };
    return await res.json();
  } catch { return { rows: [], trade_date: null }; }
}

async function loadSellDecisions(date) {
  const params = date ? `?date=${encodeURIComponent(date)}` : "";
  try {
    const res = await fetch("/api/us/trading/decisions/sell" + params);
    if (!res.ok) return { rows: [], trade_date: null };
    return await res.json();
  } catch { return { rows: [], trade_date: null }; }
}

async function loadPositions() {
  try {
    const res = await fetch("/api/us/trading/positions");
    if (!res.ok) return { rows: [], snapshot_date: null };
    return await res.json();
  } catch { return { rows: [], snapshot_date: null }; }
}

async function loadGuards() {
  try {
    const res = await fetch("/api/us/trading/guards");
    if (!res.ok) return { rows: [] };
    return await res.json();
  } catch { return { rows: [] }; }
}

// ── 주문 탭 새로고침 ────────────────────────────────────
async function refreshOrders() {
  const date = document.getElementById("filterOrderDate")?.value || "";
  const [buyData, sellData] = await Promise.all([loadBuyDecisions(date), loadSellDecisions(date)]);
  renderBuyDecisions(buyData.rows || [], buyData.trade_date);
  renderSellDecisions(sellData.rows || [], sellData.trade_date);
}

// ── 전체 초기화 ─────────────────────────────────────────
async function init() {
  const [summary, posData, guardData] = await Promise.all([
    loadSummary(),
    loadPositions(),
    loadGuards(),
  ]);

  state.summary = summary;
  state.positions = posData?.rows || [];
  state.guards = guardData?.rows || [];

  updateBanner(summary);
  renderOverview(summary);
  renderPositionPreview(state.positions.slice(0, 5));
  renderPositions(state.positions, posData?.snapshot_date);
  renderGuards(state.guards);

  // 주문 탭은 날짜 드롭다운 로드 후 최신 데이터 로드
  await loadOrderDates();
  await refreshOrders();
}

// ── 이벤트 바인딩 ───────────────────────────────────────
document.getElementById("btnRefreshOrders")?.addEventListener("click", refreshOrders);
document.getElementById("filterOrderDate")?.addEventListener("change", refreshOrders);

document.addEventListener("DOMContentLoaded", init);
