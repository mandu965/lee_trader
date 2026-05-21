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
function fmtMoney(v, digits = 2) {
  if (v == null || !Number.isFinite(Number(v))) return "—";
  const n = Number(v);
  return (n < 0 ? "-$" : "$") + Math.abs(n).toFixed(digits);
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
function severityLabel(severity) {
  const map = { CRITICAL: "치명", HIGH: "높음", MEDIUM: "중간", LOW: "낮음", INFO: "정보" };
  return map[String(severity || "").toUpperCase()] || (severity || "—");
}
function guardTone(status) {
  if (status === "FAIL") return "bad";
  if (status === "WARN") return "warn";
  return "good";
}
function guardMargin(metricValue, thresholdValue) {
  const metric = Number(metricValue);
  const threshold = Number(thresholdValue);
  if (!Number.isFinite(metric) || !Number.isFinite(threshold)) return null;
  return metric - threshold;
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
    week_pnl_usd, generated_at, cash_balance_usd, market_value_usd,
    equity_value_usd, total_pnl_usd, total_pnl_pct, account_id,
    initial_cash_usd, valuation_source } = summary;

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

  // 히어로 6개
  document.getElementById("heroDate").textContent = trade_date || "—";
  document.getElementById("heroDateDetail").textContent = "마지막 의사결정 기준일";

  const posEl = document.getElementById("heroPositions");
  posEl.textContent = open_positions ?? "—";
  posEl.className = "card-value";
  document.getElementById("heroPositionsDetail").textContent = "현재 오픈 포지션 수";

  const cashEl = document.getElementById("heroCash");
  cashEl.textContent = fmtMoney(cash_balance_usd);
  cashEl.className = "card-value " + posneg(cash_balance_usd);
  document.getElementById("heroCashDetail").textContent =
    `초기예산 ${fmtMoney(initial_cash_usd)} · 계정 ${account_id || "—"}`;

  const mvEl = document.getElementById("heroMarketValue");
  mvEl.textContent = fmtMoney(market_value_usd);
  mvEl.className = "card-value";
  document.getElementById("heroMarketValueDetail").textContent =
    valuation_source === "paper_account_snapshot" ? "계좌 스냅샷 기준 보유 평가금액" : "주문/포지션 기준 보유 평가금액";

  const equityEl = document.getElementById("heroEquity");
  equityEl.textContent = fmtMoney(equity_value_usd);
  equityEl.className = "card-value " + posneg(total_pnl_usd);
  document.getElementById("heroEquityDetail").textContent =
    total_pnl_pct != null ? `총수익률 ${fmtPct(total_pnl_pct)}` : "현금 + 보유 평가금액";

  const pnlEl = document.getElementById("heroPnl");
  pnlEl.textContent = fmtUsd(total_pnl_usd);
  pnlEl.className = "card-value " + posneg(total_pnl_usd);
  document.getElementById("heroPnlDetail").textContent =
    `미실현 ${fmtUsd(total_unrealized_pnl_usd)} · 최근7일 실현 ${fmtUsd(week_pnl_usd)}`;

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
        <th class="right">현재가(USD)</th><th class="right">수량</th><th class="right">보유금액(USD)</th>
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
            <td class="right">${fmt(r.remaining_quantity, 0)}</td>
            <td class="right">${fmtMoney(r.market_value_usd)}</td>
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

function renderPaperOrders(rows, tradeDate) {
  document.getElementById("paperOrderDate").textContent = tradeDate ? "기준일 " + tradeDate : "";
  const tbody = document.getElementById("paperOrderBody");
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="8" class="muted" style="text-align:center;padding:16px;">해당 날짜 paper 주문 데이터가 없습니다.</td></tr>';
    return;
  }
  tbody.innerHTML = rows.map((r) => {
    const note = r.sell_action || r.source_decision_id || "-";
    const statusText = r.assumed_fill_status || "-";
    const statusClass =
      statusText.includes("FILLED") ? "allowed" :
      statusText.includes("PENDING") ? "warn" :
      statusText.includes("PRICE") ? "blocked" : "hold";
    return `<tr>
      <td style="font-size:11px;">${esc(r.order_id || "-")}</td>
      <td><strong>${esc(r.symbol)}</strong></td>
      <td>${esc(r.side || "-")}</td>
      <td><span class="chip ${statusClass}">${esc(statusText)}</span></td>
      <td class="right">${fmt(r.order_qty, 0)}</td>
      <td class="right">${r.order_price != null ? "$" + fmt(r.order_price, 2) : "-"}</td>
      <td class="right">${r.order_amount != null ? "$" + fmt(r.order_amount, 2) : "-"}</td>
      <td style="font-size:11px;color:var(--color-text-secondary);">${esc(note)}</td>
    </tr>`;
  }).join("");
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
        <th class="right">평균단가(USD)</th><th class="right">보유금액(USD)</th><th class="right">52주 최고가(USD)</th>
        <th class="right">미실현 손익(USD)</th><th class="right">수익률</th>
        <th class="right">보유일</th>
      </tr></thead>
      <tbody>
        ${rows.map((r) => `<tr>
          <td><strong>${esc(r.symbol)}</strong></td>
          <td style="font-size:12px;">${esc(r.company_name || "—")}</td>
          <td style="font-size:11px;color:var(--color-text-secondary);">${esc(r.sector || "—")}</td>
          <td class="right">$${fmt(r.latest_price, 2)}</td>
          <td class="right">${fmt(r.remaining_quantity, 0)}</td>
          <td class="right">${r.avg_entry_price != null ? "$" + fmt(r.avg_entry_price, 2) : "—"}</td>
          <td class="right">${fmtMoney(r.market_value_usd)}</td>
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
  const summary = document.getElementById("guardSummary");
  const callout = document.getElementById("guardCallout");
  const symbolWrap = document.getElementById("guardSymbolTableWrap");
  const symbolNote = document.getElementById("guardSymbolNote");
  const dateEl = document.getElementById("guardsDate");
  if (rows.length > 0 && rows[0].trade_date) {
    dateEl.textContent = "기준일: " + String(rows[0].trade_date).slice(0, 10);
  }
  if (!rows.length) {
    summary.innerHTML = `
      <div class="guard-summary-card info">
        <div class="guard-summary-label">요약</div>
        <div class="guard-summary-value">0건</div>
        <div class="guard-summary-detail">리스크 가드 데이터가 아직 없습니다.</div>
      </div>`;
    callout.className = "guard-callout";
    callout.textContent = "리스크 가드 데이터 없음 (파이프라인 실행 후 생성됩니다).";
    symbolWrap.innerHTML = '<div class="empty-state">심볼별 리스크 가드 데이터가 없습니다.</div>';
    symbolNote.textContent = "";
    grid.innerHTML = '<div class="empty-state" style="grid-column:1/-1;">리스크 가드 데이터 없음 (파이프라인 실행 후 생성됩니다)</div>';
    return;
  }

  const symbolRows = rows.filter((r) => String(r.guard_scope || "").toUpperCase() === "SYMBOL");
  const otherRows = rows.filter((r) => String(r.guard_scope || "").toUpperCase() !== "SYMBOL");
  const passCount = rows.filter((r) => r.guard_status === "PASS").length;
  const warnCount = rows.filter((r) => r.guard_status === "WARN").length;
  const failCount = rows.filter((r) => r.guard_status === "FAIL").length;
  const highestSeverity = rows.find((r) => r.guard_status === "FAIL") ? "FAIL" : rows.find((r) => r.guard_status === "WARN") ? "WARN" : "PASS";
  const topRisk = rows.find((r) => r.guard_status !== "PASS") || rows[0];

  summary.innerHTML = `
    <div class="guard-summary-card ${highestSeverity === "FAIL" ? "bad" : highestSeverity === "WARN" ? "warn" : "good"}">
      <div class="guard-summary-label">전체 판정</div>
      <div class="guard-summary-value">${highestSeverity === "FAIL" ? "위험" : highestSeverity === "WARN" ? "주의" : "안정"}</div>
      <div class="guard-summary-detail">${rows.length}개 가드 중 PASS ${passCount}건 · WARN ${warnCount}건 · FAIL ${failCount}건</div>
    </div>
    <div class="guard-summary-card good">
      <div class="guard-summary-label">통과</div>
      <div class="guard-summary-value">${passCount}</div>
      <div class="guard-summary-detail">기준을 만족한 가드 수</div>
    </div>
    <div class="guard-summary-card ${warnCount > 0 ? "warn" : "info"}">
      <div class="guard-summary-label">주의</div>
      <div class="guard-summary-value">${warnCount}</div>
      <div class="guard-summary-detail">추가 확인이 필요한 가드 수</div>
    </div>
    <div class="guard-summary-card ${failCount > 0 ? "bad" : "info"}">
      <div class="guard-summary-label">차단</div>
      <div class="guard-summary-value">${failCount}</div>
      <div class="guard-summary-detail">진입을 막는 가드 수</div>
    </div>`;

  if (highestSeverity === "FAIL") {
    callout.className = "guard-callout bad";
    callout.textContent = `${esc(topRisk.guard_name)} 가드에서 FAIL이 발생했습니다. 실주문 전 차단 사유와 threshold 초과 원인을 먼저 확인해야 합니다.`;
  } else if (highestSeverity === "WARN") {
    callout.className = "guard-callout warn";
    callout.textContent = `${esc(topRisk.guard_name)} 가드에 WARN이 있습니다. 즉시 차단은 아니지만 여유 폭과 reason detail을 확인하는 편이 좋습니다.`;
  } else {
    callout.className = "guard-callout good";
    callout.textContent = "현재 리스크 가드는 모두 PASS입니다. 다만 PASS라도 threshold 대비 여유 폭이 작은 종목은 재점검 후보로 볼 수 있습니다.";
  }

  if (!symbolRows.length) {
    symbolNote.textContent = "";
    symbolWrap.innerHTML = '<div class="empty-state">심볼 범위 리스크 가드가 없습니다.</div>';
  } else {
    const sortedSymbolRows = [...symbolRows].sort((a, b) => {
      const statusRank = { FAIL: 0, WARN: 1, PASS: 2 };
      const aRank = statusRank[a.guard_status] ?? 9;
      const bRank = statusRank[b.guard_status] ?? 9;
      if (aRank !== bRank) return aRank - bRank;
      const aMargin = guardMargin(a.metric_value, a.threshold_value);
      const bMargin = guardMargin(b.metric_value, b.threshold_value);
      if (Number.isFinite(aMargin) && Number.isFinite(bMargin)) return aMargin - bMargin;
      return String(a.guard_name || "").localeCompare(String(b.guard_name || ""));
    });
    const closest = sortedSymbolRows
      .map((r) => ({ ...r, margin: guardMargin(r.metric_value, r.threshold_value) }))
      .filter((r) => Number.isFinite(r.margin))
      .sort((a, b) => a.margin - b.margin)[0];
    symbolNote.textContent = closest
      ? `threshold 여유가 가장 작은 종목: ${closest.guard_name} (${fmt(closest.margin)}pt)`
      : `심볼별 가드 ${symbolRows.length}건`;
    symbolWrap.innerHTML = `
      <div class="table-wrap"><table>
        <thead><tr>
          <th>심볼</th><th>판정</th><th class="right">점수</th><th class="right">기준</th><th class="right">여유</th><th>심각도</th><th>메모</th>
        </tr></thead>
        <tbody>
          ${sortedSymbolRows.map((r) => {
            const margin = guardMargin(r.metric_value, r.threshold_value);
            const tone = guardTone(r.guard_status);
            const note = r.reason_detail || (r.guard_status === "PASS" ? "기준 통과" : r.reason_code || "상세 사유 없음");
            return `<tr>
              <td><strong>${esc(r.guard_name)}</strong></td>
              <td>${guardChip(r.guard_status)}</td>
              <td class="right">${fmt(r.metric_value)}</td>
              <td class="right">${fmt(r.threshold_value)}</td>
              <td class="right guard-margin ${tone}">${margin == null ? "—" : (margin >= 0 ? "+" : "") + fmt(margin)}</td>
              <td>${esc(severityLabel(r.severity))}</td>
              <td style="font-size:12px;color:var(--color-text-secondary);">${esc(note)}</td>
            </tr>`;
          }).join("")}
        </tbody>
      </table></div>
      <div class="guard-table-note">PASS여도 여유 값이 작을수록 threshold 근처입니다. 실전 전환 시에는 FAIL/WARN보다 먼저 이 종목들을 재검토하는 편이 좋습니다.</div>`;
  }

  const classMap = { PASS: "pass", FAIL: "fail", WARN: "warn" };
  if (!otherRows.length) {
    grid.innerHTML = '<div class="empty-state" style="grid-column:1/-1;">현재는 심볼별 가드만 생성되어 별도 공통/계좌 가드는 없습니다.</div>';
    return;
  }
  grid.innerHTML = otherRows.map((r) => `
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

// ── 실계좌 현황 ────────────────────────────────────────
async function loadLiveAccount() {
  try {
    const res = await fetch("/api/us/live/account");
    if (!res.ok) return null;
    return await res.json();
  } catch { return null; }
}

function renderLiveAccount(data) {
  const content = document.getElementById("liveAcctContent");
  const acctNoEl = document.getElementById("liveAcctAccountNo");
  const fetchedEl = document.getElementById("liveAcctFetched");

  if (!data || !data.ok) {
    content.innerHTML = `<div class="live-acct-error">계좌 조회 실패: ${esc(data?.error || "연결 오류")}</div>`;
    return;
  }

  if (data.account_no) acctNoEl.textContent = `(${data.account_no})`;
  if (data.fetched_at) {
    const dt = new Date(data.fetched_at).toLocaleString("ko-KR", { timeZone: "Asia/Seoul" });
    fetchedEl.textContent = `갱신: ${dt.slice(0, 16)}`;
  }

  const cashUsd = data.cash_usd != null ? "$" + Number(data.cash_usd).toFixed(2) : "—";
  const totalKrw = data.total_eval_krw != null
    ? Number(data.total_eval_krw).toLocaleString("ko-KR") + "원" : "—";
  const pnlStr = data.total_pnl_usd != null ? fmtUsd(data.total_pnl_usd) : "—";
  const pnlClass = (data.total_pnl_usd ?? 0) >= 0 ? "pos" : "neg";

  const statsHtml = `
    <div class="live-acct-stats">
      <div class="live-acct-stat">
        <div class="live-acct-stat-label">가용 현금 (USD)</div>
        <div class="live-acct-stat-value">${esc(cashUsd)}</div>
      </div>
      <div class="live-acct-stat">
        <div class="live-acct-stat-label">보유 포지션</div>
        <div class="live-acct-stat-value">${data.position_count ?? 0}개</div>
      </div>
      <div class="live-acct-stat">
        <div class="live-acct-stat-label">총평가 / 손익</div>
        <div class="live-acct-stat-value" style="font-size:16px;">${esc(totalKrw)}</div>
        <div style="font-size:12px;margin-top:2px;" class="${pnlClass}">${esc(pnlStr)}</div>
      </div>
    </div>`;

  let posHtml = "";
  if (data.positions && data.positions.length > 0) {
    posHtml = `
      <div class="table-wrap"><table>
        <thead><tr>
          <th>심볼</th>
          <th class="right">수량</th>
          <th class="right">평균단가(USD)</th>
          <th class="right">현재가(USD)</th>
          <th class="right">평가금액(USD)</th>
          <th class="right">손익(USD)</th>
          <th class="right">수익률</th>
        </tr></thead>
        <tbody>
          ${data.positions.map((p) => `<tr>
            <td><strong>${esc(p.symbol)}</strong></td>
            <td class="right">${p.qty ?? "—"}</td>
            <td class="right">$${fmt(p.avg_price, 2)}</td>
            <td class="right">$${fmt(p.current_price, 2)}</td>
            <td class="right">${fmtMoney(p.market_value)}</td>
            <td class="right ${posneg(p.pnl)}">${fmtUsd(p.pnl)}</td>
            <td class="right ${posneg(p.pnl_pct)}">${fmtPct(p.pnl_pct)}</td>
          </tr>`).join("")}
        </tbody>
      </table></div>`;
  } else {
    posHtml = '<div style="font-size:12px;color:var(--color-text-secondary);padding:6px 0;">보유 포지션 없음</div>';
  }

  content.innerHTML = statsHtml + posHtml;
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

async function loadPaperOrders(date) {
  const params = date ? `?date=${encodeURIComponent(date)}` : "";
  try {
    const res = await fetch("/api/us/trading/orders" + params);
    if (!res.ok) return { rows: [], trade_date: null };
    return await res.json();
  } catch { return { rows: [], trade_date: null }; }
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
  const [buyData, sellData, orderData] = await Promise.all([loadBuyDecisions(date), loadSellDecisions(date), loadPaperOrders(date)]);
  renderBuyDecisions(buyData.rows || [], buyData.trade_date);
  renderSellDecisions(sellData.rows || [], sellData.trade_date);
  renderPaperOrders(orderData.rows || [], orderData.trade_date);
}

// ── 전체 초기화 ─────────────────────────────────────────
async function init() {
  const [summary, posData, guardData, liveAcct] = await Promise.all([
    loadSummary(),
    loadPositions(),
    loadGuards(),
    loadLiveAccount(),
  ]);

  state.summary = summary;
  state.positions = posData?.rows || [];
  state.guards = guardData?.rows || [];

  updateBanner(summary);
  renderOverview(summary);
  renderLiveAccount(liveAcct);
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
document.getElementById("btnRefreshLiveAcct")?.addEventListener("click", async () => {
  document.getElementById("liveAcctContent").innerHTML = '<div class="live-acct-error">불러오는 중…</div>';
  const data = await loadLiveAccount();
  renderLiveAccount(data);
});

document.addEventListener("DOMContentLoaded", init);
