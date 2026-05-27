const fmtNum = (value, digits = 0) => {
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  return n.toLocaleString("ko-KR", { minimumFractionDigits: digits, maximumFractionDigits: digits });
};

const fmtPct = (value, digits = 2) => {
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  return `${(n * 100).toFixed(digits)}%`;
};

const escapeHtml = (value) => String(value ?? "")
  .replace(/&/g, "&amp;")
  .replace(/</g, "&lt;")
  .replace(/>/g, "&gt;")
  .replace(/"/g, "&quot;")
  .replace(/'/g, "&#39;");

const getToneClass = (value) => {
  const n = Number(value);
  if (!Number.isFinite(n)) return "";
  return n > 0 ? "pos" : n < 0 ? "neg" : "";
};

const ACTION_LABELS = {
  BUY_NEW_COHORT:  "신규 코호트",
  HOLD_NEW_ENTRY:  "신규 보유",
  HOLD_ACTIVE:     "일반 보유",
  HOLD_REVIEW_SOON:"점검 임박",
  EXIT_REVIEW_SOON:"청산 임박",
  EXIT_HOLD_D20:   "20일 청산",
  POSITION_CLOSED: "청산 완료",
};

const STRATEGY_COLORS = { top5: "#3ddc97", top8: "#69a7ff", top10: "#f7b955" };

function toDate(value) {
  if (!value) return null;
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? null : d;
}

function isBusinessDay(date) {
  const day = date.getDay();
  return day !== 0 && day !== 6;
}

function diffBusinessDays(fromValue, toValue) {
  const from = toDate(fromValue);
  const to = toDate(toValue);
  if (!from || !to) return null;
  if (to < from) return 0;
  const cursor = new Date(from);
  let count = 0;
  while (cursor < to) {
    cursor.setDate(cursor.getDate() + 1);
    if (isBusinessDay(cursor)) count += 1;
  }
  return count;
}

function addBusinessDays(value, days) {
  const start = toDate(value);
  if (!start || !Number.isFinite(days)) return null;
  const cursor = new Date(start);
  let added = 0;
  while (added < days) {
    cursor.setDate(cursor.getDate() + 1);
    if (isBusinessDay(cursor)) added += 1;
  }
  return cursor.toISOString().slice(0, 10);
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

// ── NAV 라인 차트 ───────────────────────────────────────────────────────────

function buildNavChart(items) {
  const el = document.getElementById("navChart");
  if (!items.length) {
    el.innerHTML = '<div class="empty-state">NAV 데이터가 없습니다.</div>';
    return;
  }

  // 전략별 그룹 + 날짜 정렬
  const byStrategy = {};
  for (const row of items) {
    if (!byStrategy[row.strategy]) byStrategy[row.strategy] = [];
    byStrategy[row.strategy].push(row);
  }
  for (const rows of Object.values(byStrategy)) {
    rows.sort((a, b) => a.date.localeCompare(b.date));
  }

  const allDates = [...new Set(items.map((r) => r.date))].sort();
  const dateIndex = new Map(allDates.map((d, i) => [d, i]));

  const allReturns = items.map((r) => Number(r.cumulative_return)).filter(Number.isFinite);
  const rawMin = Math.min(0, ...allReturns);
  const rawMax = Math.max(0, ...allReturns);
  const pad = Math.max((rawMax - rawMin) * 0.12, 0.005);
  const yMin = rawMin - pad;
  const yMax = rawMax + pad;

  const W = 900, H = 240;
  const PAD = { t: 28, r: 70, b: 36, l: 54 };
  const cW = W - PAD.l - PAD.r;
  const cH = H - PAD.t - PAD.b;

  const xS = (i) => PAD.l + (i / Math.max(allDates.length - 1, 1)) * cW;
  const yS = (v) => PAD.t + (1 - (v - yMin) / (yMax - yMin)) * cH;

  // 그리드 + Y 축 레이블
  const tickCount = 5;
  const yTicks = Array.from({ length: tickCount + 1 }, (_, i) => {
    const v = yMin + (i / tickCount) * (yMax - yMin);
    const y = yS(v).toFixed(1);
    return `
      <line x1="${PAD.l}" y1="${y}" x2="${PAD.l + cW}" y2="${y}" stroke="#1e3050" stroke-width="1"/>
      <text x="${PAD.l - 6}" y="${(+y + 3.5).toFixed(1)}" fill="#6b8ab5" font-size="10" text-anchor="end">${(v * 100).toFixed(1)}%</text>
    `;
  }).join("");

  // X 축 레이블 (최대 8개)
  const step = Math.max(1, Math.floor(allDates.length / 8));
  const xLabels = allDates
    .map((d, i) => {
      if (i !== 0 && i !== allDates.length - 1 && i % step !== 0) return "";
      return `<text x="${xS(i).toFixed(1)}" y="${H - 6}" fill="#6b8ab5" font-size="10" text-anchor="middle">${d.slice(5)}</text>`;
    })
    .join("");

  // 기준선 (0%)
  const zeroY = yS(0).toFixed(1);
  const zeroLine = `<line x1="${PAD.l}" y1="${zeroY}" x2="${PAD.l + cW}" y2="${zeroY}" stroke="#3a5070" stroke-width="1.5" stroke-dasharray="5,4"/>`;

  // 전략별 라인
  const lines = Object.entries(byStrategy).map(([strategy, rows]) => {
    const clr = STRATEGY_COLORS[strategy] || "#fff";
    const pts = rows
      .map((r) => {
        const xi = dateIndex.get(r.date);
        const v = Number(r.cumulative_return);
        if (xi === undefined || !Number.isFinite(v)) return null;
        return [xS(xi).toFixed(1), yS(v).toFixed(1)];
      })
      .filter(Boolean);
    if (pts.length < 2) return "";
    const pathD = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p[0]},${p[1]}`).join(" ");
    const last = pts[pts.length - 1];
    const lastVal = ((Number(rows[rows.length - 1]?.cumulative_return) || 0) * 100).toFixed(2);
    const sign = +lastVal >= 0 ? "+" : "";
    return `
      <path d="${pathD}" fill="none" stroke="${clr}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" opacity="0.9"/>
      <circle cx="${last[0]}" cy="${last[1]}" r="3.5" fill="${clr}"/>
      <text x="${(+last[0] + 6).toFixed(1)}" y="${(+last[1] + 4).toFixed(1)}" fill="${clr}" font-size="11" font-weight="700">${sign}${lastVal}%</text>
    `;
  }).join("");

  // 범례
  const legend = Object.entries(STRATEGY_COLORS)
    .map(([name, clr], i) => {
      const x = PAD.l + i * 80;
      return `
        <line x1="${x}" y1="14" x2="${x + 18}" y2="14" stroke="${clr}" stroke-width="2.5"/>
        <circle cx="${x + 9}" cy="14" r="3" fill="${clr}"/>
        <text x="${x + 24}" y="18" fill="${clr}" font-size="11" font-weight="700">${name}</text>
      `;
    })
    .join("");

  el.innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto;display:block" preserveAspectRatio="xMidYMid meet">
      <g>${legend}</g>
      ${yTicks}
      ${zeroLine}
      ${xLabels}
      <rect x="${PAD.l}" y="${PAD.t}" width="${cW}" height="${cH}" fill="none" stroke="#1e3050" stroke-width="1"/>
      ${lines}
    </svg>
  `;
}

// ── 전략 요약 카드 ──────────────────────────────────────────────────────────

function buildMetaStrip(run) {
  const el = document.getElementById("metaStrip");
  if (!run) {
    el.innerHTML = '<div class="empty-state">run 정보가 없습니다.</div>';
    return;
  }
  el.innerHTML = `
    <article class="meta-card">
      <div class="card-label">기준일</div>
      <div class="card-value">${run.asof_date || "-"}</div>
      <div class="card-detail">hold_days ${fmtNum(run.hold_days)} · source ${run.source_mode || "-"}</div>
    </article>
    <article class="meta-card">
      <div class="card-label">Run</div>
      <div class="card-value">${run.run_tag || "-"}</div>
      <div class="card-detail">초기자금 ${fmtNum(run.initial_nav)}</div>
    </article>
    <article class="meta-card">
      <div class="card-label">포지션 / NAV rows</div>
      <div class="card-value">${fmtNum(run.positions_row_count)} / ${fmtNum(run.nav_row_count)}</div>
      <div class="card-detail">누적 데이터</div>
    </article>
    <article class="meta-card">
      <div class="card-label">구동 모드</div>
      <div class="card-value">${run.source_mode || "-"}</div>
      <div class="card-detail">run_id ${run.paper_run_id || "-"}</div>
    </article>
  `;
  document.getElementById("runInfo").textContent = `기준일 ${run.asof_date || "-"} · run_id ${run.paper_run_id || "-"}`;
}

function buildStrategyCards(strategies) {
  const el = document.getElementById("strategyGrid");
  if (!strategies.length) {
    el.innerHTML = '<div class="empty-state">전략 요약 데이터가 없습니다.</div>';
    return;
  }
  el.innerHTML = strategies.map((item) => {
    const clr = STRATEGY_COLORS[item.strategy] || "#fff";
    const retCls = getToneClass(item.cumulative_return);
    const ddCls = getToneClass(item.drawdown);
    return `
      <article class="strategy-card">
        <div class="strategy-head">
          <div>
            <h3 class="strategy-name" style="color:${clr}">${item.strategy}</h3>
            <div class="strategy-date">최신 ${item.latest_date || "-"}</div>
          </div>
          <div class="strategy-return ${retCls}">${fmtPct(item.cumulative_return)}</div>
        </div>
        <div class="strategy-metrics">
          <div class="metric-box">
            <div class="metric-box__label">NAV</div>
            <div class="metric-box__value">${fmtNum(item.nav, 0)}</div>
          </div>
          <div class="metric-box">
            <div class="metric-box__label">MDD</div>
            <div class="metric-box__value ${ddCls}">${fmtPct(item.drawdown)}</div>
          </div>
          <div class="metric-box">
            <div class="metric-box__label">보유 종목</div>
            <div class="metric-box__value">${fmtNum(item.open_position_count)}</div>
          </div>
          <div class="metric-box">
            <div class="metric-box__label">누적 청산</div>
            <div class="metric-box__value">${fmtNum(item.closed_trade_count)}</div>
          </div>
        </div>
      </article>
    `;
  }).join("");
}

function buildExplainGrid(summary, navItems, openItems, strategyLabel) {
  const el = document.getElementById("explainGrid");
  if (!el) return;
  const strategies = Array.isArray(summary?.strategies) ? summary.strategies : [];
  const ranked = strategies
    .slice()
    .sort((a, b) => Number(b.cumulative_return || 0) - Number(a.cumulative_return || 0));
  const best = ranked[0] || null;
  const latestNav = navItems.length ? navItems[navItems.length - 1] : null;
  const nearExitCount = openItems.filter((item) => Number.isFinite(item.remaining_days) && item.remaining_days <= 5).length;
  const reviewCount = openItems.filter((item) =>
    ["BLOCK", "FULL_SELL", "PARTIAL_SELL", "EXIT_REVIEW", "REVIEW"].includes(String(item.system_review_status || ""))
  ).length;
  const negativeCount = openItems.filter((item) => Number(item.current_return ?? item.net_return) < 0).length;
  const avgOpenReturn = openItems.length
    ? openItems.reduce((sum, item) => sum + (Number(item.current_return ?? item.net_return) || 0), 0) / openItems.length
    : null;
  const openedToday = Number(latestNav?.opened_today);
  const duplicateSkip = Number(latestNav?.duplicate_skip_count);
  const latestDate = latestNav?.date || summary?.run?.asof_date || "-";
  const bestRiskText = best
    ? `${best.strategy} · 수익 ${fmtPct(best.cumulative_return)} · MDD ${fmtPct(best.drawdown)}`
    : "-";
  const selectedText = latestNav
    ? `${strategyLabel} · ${latestDate} · 신규 ${fmtNum(openedToday)} / 중복 ${fmtNum(duplicateSkip)}`
    : `${strategyLabel} · NAV 없음`;

  el.innerHTML = `
    <article class="explain-card">
      <div class="card-label">성과 리더</div>
      <div class="card-value">${escapeHtml(best?.strategy || "-")}</div>
      <div class="card-detail">${escapeHtml(bestRiskText)}</div>
    </article>
    <article class="explain-card">
      <div class="card-label">선택 전략 흐름</div>
      <div class="card-value">${latestNav ? fmtPct(latestNav.daily_return, 2) : "-"}</div>
      <div class="card-detail">${escapeHtml(selectedText)}</div>
    </article>
    <article class="explain-card">
      <div class="card-label">보유 포지션 평균</div>
      <div class="card-value ${getToneClass(avgOpenReturn)}">${avgOpenReturn === null ? "-" : fmtPct(avgOpenReturn, 2)}</div>
      <div class="card-detail">열린 포지션 ${fmtNum(openItems.length)}개 · 손실권 ${fmtNum(negativeCount)}개</div>
    </article>
    <article class="explain-card">
      <div class="card-label">오늘 점검 우선</div>
      <div class="card-value">${fmtNum(Math.max(nearExitCount, reviewCount))}개</div>
      <div class="card-detail">청산 임박 ${fmtNum(nearExitCount)}개 · 시스템 점검 ${fmtNum(reviewCount)}개</div>
    </article>
  `;
}

// ── 현재 상태 (신규 / 보유 / 청산임박) ────────────────────────────────────

function computeFocusSummary(run, navItems, openItems) {
  const holdDays = Number(run?.hold_days) || 20;
  const latestEntryDate = openItems.reduce((acc, item) =>
    !acc || String(item.entry_date) > String(acc) ? item.entry_date : acc, null);
  const newItems = latestEntryDate ? openItems.filter((item) => item.entry_date === latestEntryDate) : [];
  const ongoingItems = latestEntryDate ? openItems.filter((item) => item.entry_date !== latestEntryDate) : openItems.slice();
  const nearExitItems = openItems.filter((item) => Number.isFinite(item.remaining_days) && item.remaining_days <= 5);
  const latestNav = navItems.length ? navItems[navItems.length - 1] : null;
  return { holdDays, latestEntryDate, newItems, ongoingItems, nearExitItems, latestNav };
}

function buildStatusStrip(summary, strategyLabel) {
  const el = document.getElementById("statusStrip");
  const activeText = summary.latestNav ? fmtNum(summary.latestNav.active_position_count) : "-";
  const dupText = summary.latestNav ? fmtNum(summary.latestNav.duplicate_skip_count) : "-";
  const makeList = (items) =>
    items.slice(0, 5).map((item) => `<li>${item.code} ${item.name}</li>`).join("") || "<li>없음</li>";
  const makeExitList = (items) =>
    items.slice(0, 5).map((item) => `<li>${item.code} ${item.name} · 잔여 ${fmtNum(item.remaining_days)}일</li>`).join("") || "<li>없음</li>";

  el.innerHTML = `
    <article class="status-card">
      <div class="card-label">신규 편입</div>
      <div class="card-value">${fmtNum(summary.newItems.length)}개</div>
      <div class="card-detail">${strategyLabel} 최근 진입일 ${summary.latestEntryDate || "-"}</div>
      <ul class="status-card__list">${makeList(summary.newItems)}</ul>
    </article>
    <article class="status-card">
      <div class="card-label">계속 보유</div>
      <div class="card-value">${fmtNum(summary.ongoingItems.length)}개</div>
      <div class="card-detail">활성 ${activeText}개 · 중복 스킵 ${dupText}건</div>
      <ul class="status-card__list">${makeList(summary.ongoingItems)}</ul>
    </article>
    <article class="status-card">
      <div class="card-label">청산 임박 (잔여 5일↓)</div>
      <div class="card-value">${fmtNum(summary.nearExitItems.length)}개</div>
      <div class="card-detail">20영업일 만기 임박 종목</div>
      <ul class="status-card__list">${makeExitList(summary.nearExitItems)}</ul>
    </article>
  `;
}

// ── 포지션 테이블 ──────────────────────────────────────────────────────────

function enrichPositions(items, run) {
  const asofDate = run?.asof_date || null;
  const holdDays = Number(run?.hold_days) || 20;
  return items.map((item) => {
    const heldDays = Number.isFinite(Number(item.holding_age_trading_days))
      ? Number(item.holding_age_trading_days)
      : diffBusinessDays(item.entry_date, asofDate);
    const remainingDays = Number.isFinite(Number(item.remaining_holding_days))
      ? Number(item.remaining_holding_days)
      : Number.isFinite(heldDays) ? Math.max(0, holdDays - heldDays) : null;
    const expectedExitDate = item.planned_exit_date || addBusinessDays(item.entry_date, holdDays);
    return { ...item, held_days: heldDays, remaining_days: remainingDays, expected_exit_date: expectedExitDate };
  });
}

function buildPositionsTable(items) {
  const tbody = document.getElementById("positionsTbody");
  if (!items.length) {
    tbody.innerHTML = '<tr><td colspan="9" class="center">포지션 데이터가 없습니다.</td></tr>';
    return;
  }
  const statusChip = (row) => {
    if (row.system_review_status === "BLOCK") return '<span class="chip is-loss">차단</span>';
    if (row.system_review_status === "FULL_SELL") return '<span class="chip is-loss">전량청산</span>';
    if (row.system_review_status === "PARTIAL_SELL") return '<span class="chip is-near">부분청산</span>';
    if (row.system_review_status === "EXIT_REVIEW") return '<span class="chip is-near">청산검토</span>';
    if (row.system_review_status === "REVIEW") return '<span class="chip is-watch">점검필요</span>';
    if (row.remaining_days !== null && row.remaining_days <= 5) return '<span class="chip is-near">곧 마감</span>';
    if (row.held_days !== null && row.held_days <= 1) return '<span class="chip is-new">신규</span>';
    return '<span class="chip is-hold">보유중</span>';
  };
  const sellDecisionHtml = (row) => {
    const code = row.current_action_code || row.holding_policy_code || "";
    const reason = row.current_action_reason || row.entry_action_reason || "";
    if (!code) return '-';
    const chipCls = ["FULL_SELL", "BLOCK"].includes(code) ? "is-loss"
      : ["PARTIAL_SELL", "EXIT_REVIEW"].includes(code) ? "is-near"
      : ["REVIEW", "REVIEW_REQUIRED"].includes(code) ? "is-watch"
      : "is-hold";
    const label = {
      FULL_SELL: "전량청산", PARTIAL_SELL: "부분청산", HOLD: "보유유지",
      REVIEW_REQUIRED: "점검필요", EXIT_REVIEW: "청산검토", BLOCK: "차단",
    }[code] || code;
    const title = reason ? ` title="${reason.replace(/"/g, "&quot;")}"` : "";
    return `<span class="chip ${chipCls}"${title}>${label}</span>`;
  };
  const clr = (s) => STRATEGY_COLORS[s] || "";
  tbody.innerHTML = items.map((row) => `
    <tr>
      <td>${statusChip(row)}</td>
      <td><span style="color:${clr(row.strategy)};font-weight:700">${row.strategy || "-"}</span></td>
      <td>
        <div>${row.name || row.code || "-"}</div>
        <div style="font-size:11px;color:#6b8ab5">${row.code || ""}</div>
      </td>
      <td>${row.entry_date || "-"}</td>
      <td class="right">
        <div>${fmtNum(row.held_days)}일</div>
        <div style="font-size:11px;color:#6b8ab5">잔여 ${fmtNum(row.remaining_days)}일</div>
      </td>
      <td class="right ${getToneClass(row.current_return || row.net_return)}">${fmtPct(row.current_return ?? row.net_return)}</td>
      <td class="right">${fmtNum(row.confidence_score, 1)}</td>
      <td class="right">${fmtNum(row.final_score, 1)}</td>
      <td>${sellDecisionHtml(row)}</td>
    </tr>
  `).join("");
}

// ── 메인 로드 ──────────────────────────────────────────────────────────────

async function loadPaperTrading() {
  const strategy = document.getElementById("strategyFilter").value.trim();
  const status   = document.getElementById("statusFilter").value.trim();
  const stateEl  = document.getElementById("pageState");
  const label    = strategy || "전체 전략";
  stateEl.textContent = "모의투자 데이터를 불러오는 중입니다.";

  try {
    const summary = await fetchJson("/api/paper-trading/summary");
    const navQs = new URLSearchParams();
    const posQs = new URLSearchParams();
    if (strategy) { navQs.set("strategy", strategy); posQs.set("strategy", strategy); }
    if (status)   posQs.set("status", status);

    const [nav, positions] = await Promise.all([
      fetchJson(`/api/paper-trading/nav?${navQs}`),
      fetchJson(`/api/paper-trading/positions?${posQs}`),
    ]);

    const openItems = enrichPositions(
      (positions.items || []).filter((item) => (item.status || "OPEN") === "OPEN"),
      summary.run
    );

    buildMetaStrip(summary.run);
    buildExplainGrid(summary, nav.items || [], openItems, label);
    buildStrategyCards(summary.strategies || []);
    buildNavChart(nav.items || []);
    buildStatusStrip(computeFocusSummary(summary.run, nav.items || [], openItems), label);
    buildPositionsTable(openItems);

    stateEl.textContent = `기준일 ${summary.run?.asof_date || "-"} · ${label} · open ${openItems.length}건`;
  } catch (error) {
    console.error(error);
    document.getElementById("metaStrip").innerHTML = '<div class="empty-state">데이터를 불러오지 못했습니다.</div>';
    document.getElementById("explainGrid").innerHTML = "";
    document.getElementById("strategyGrid").innerHTML = "";
    document.getElementById("navChart").innerHTML = '<div class="empty-state">차트를 불러오지 못했습니다.</div>';
    document.getElementById("statusStrip").innerHTML = "";
    buildPositionsTable([]);
    stateEl.textContent = `조회 실패: ${error.message}`;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("refreshBtn").addEventListener("click", () => loadPaperTrading().catch(console.error));
  document.getElementById("strategyFilter").addEventListener("change", () => loadPaperTrading().catch(console.error));
  document.getElementById("statusFilter").addEventListener("change", () => loadPaperTrading().catch(console.error));
  loadPaperTrading().catch(console.error);
});
