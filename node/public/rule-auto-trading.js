/* ── 포맷 헬퍼 ── */
const fmtNum = (v, digits = 0) => {
  const n = Number(v);
  if (!Number.isFinite(n)) return "-";
  return n.toLocaleString("ko-KR", { minimumFractionDigits: digits, maximumFractionDigits: digits });
};

const fmtPct = (v, digits = 2) => {
  if (v === null || v === undefined) return "-";
  const n = Number(v);
  if (!Number.isFinite(n)) return "-";
  return `${(n * 100).toFixed(digits)}%`;
};

const fmtWon = (v) => {
  const n = Number(v);
  if (!Number.isFinite(n)) return "-";
  if (Math.abs(n) >= 1e8) return `${(n / 1e8).toFixed(1)}억`;
  if (Math.abs(n) >= 1e4) return `${(n / 1e4).toFixed(0)}만`;
  return n.toLocaleString("ko-KR");
};

const esc = (v) =>
  String(v ?? "").replace(/[&<>"']/g, (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m]));

const signedClass = (v) => {
  const n = Number(v);
  if (!Number.isFinite(n) || n === 0) return "";
  return n > 0 ? "pos" : "neg";
};

const flagText = (v) => (v === true ? "ON" : v === false ? "OFF" : "-");

/* ── 칩 렌더러 ── */
const signalChip = (v) => {
  if (v === "strong_entry") return `<span class="chip ok">Strong</span>`;
  if (v === "entry")        return `<span class="chip info">Entry</span>`;
  return `<span class="chip warn">없음</span>`;
};

const actionChip = (v) => {
  const map = { buy: ["ok","매수"], hold: ["info","보유"], reduce: ["warn","축소"], exit: ["bad","청산"] };
  const [cls, label] = map[v] || ["warn", esc(v || "스킵")];
  return `<span class="chip ${cls}">${label}</span>`;
};

const orderStatusChip = (row) => {
  const s = row.order_status;
  if (s === "simulated_filled" || s === "filled")     return `<span class="chip ok">체결</span>`;
  if (s === "simulated_unfilled" || s === "unfilled") return `<span class="chip warn">미체결</span>`;
  if (s === "submitted")    return `<span class="chip ok">제출</span>`;
  if (s === "partial_filled") return `<span class="chip info">부분체결</span>`;
  if (s === "canceled")     return `<span class="chip warn">취소</span>`;
  if (s === "failed")       return `<span class="chip bad">실패</span>`;
  if (s === "blocked")      return `<span class="chip bad">차단</span>`;
  if (row.order_allowed)    return `<span class="chip ok">허용</span>`;
  if (row.side === "BUY" || row.side === "SELL") return `<span class="chip info">미리보기</span>`;
  return `<span class="chip bad">차단</span>`;
};

const sideText = (v) => {
  if (v === "BUY")  return `<span class="pos">매수</span>`;
  if (v === "SELL") return `<span class="neg">매도</span>`;
  return esc(v || "-");
};

/* ── API 호출 ── */
async function fetchJson(url) {
  const res = await fetch(url, { credentials: "same-origin" });
  if (!res.ok) throw new Error(`${url} HTTP ${res.status}`);
  return res.json();
}

/* ── 주 탭 전환 ── */
function initTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(`tab-${tab}`)?.classList.add("active");
      // 주문 탭 진입 시 토글 버튼 보임/숨김
      document.getElementById("buyOnlyToggle").style.display = tab === "orders" ? "" : "none";
    });
  });
}

/* ── 서브 탭 전환 ── */
function initSubTabs() {
  document.querySelectorAll(".sub-tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const sub = btn.dataset.sub;
      document.querySelectorAll(".sub-tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".sub-tab-pane").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(`sub-${sub}`)?.classList.add("active");
    });
  });
}

/* ── BUY only 토글 ── */
let _portfolioAllItems = [];
let _buyOnly = true;

function initBuyOnlyToggle() {
  const btn = document.getElementById("buyOnlyToggle");
  btn.style.display = "none"; // 주문 탭 외에는 숨김
  btn.addEventListener("click", () => {
    _buyOnly = !_buyOnly;
    btn.classList.toggle("active", _buyOnly);
    btn.textContent = _buyOnly ? "BUY/SELL만 보기" : "전체 보기";
    renderPortfolio(_portfolioAllItems);
  });
}

/* ── Safety 배너 ── */
function renderSafety(summary, diagnostics) {
  const banner = document.getElementById("safetyBanner");
  const text   = document.getElementById("safetyText");
  const detail = document.getElementById("safetyDetail");

  const ops      = summary.operations || {};
  const controls = ops.controls || {};
  const diag     = diagnostics?.summary || {};

  const killActive  = controls.global_kill_switch || controls.rule_kill_switch;
  const weeklyBlocked = diag.weekly_blocked;
  const liveReady   = diag.live_trade_ready;
  const runMode     = esc(diag.run_mode || summary.run_mode || "-");

  const flags = [
    `GLOBAL ${flagText(controls.global_kill_switch)}`,
    `RULE ${flagText(controls.rule_kill_switch)}`,
    `실행 ${flagText(controls.auto_trade_execute)}`,
    `BUY ${flagText(controls.auto_trade_allow_buy)}`,
  ].join("  ·  ");

  if (killActive) {
    banner.className = "safety-banner bad";
    text.textContent = "킬스위치 활성화 — 모든 주문 중단";
  } else if (weeklyBlocked) {
    banner.className = "safety-banner bad";
    text.textContent = "주간 손익 한도 초과 — 매수 차단 중";
  } else if (!liveReady && diag.run_mode) {
    banner.className = "safety-banner warn";
    text.textContent = `실매매 불가 상태 (${runMode})`;
  } else if (liveReady) {
    banner.className = "safety-banner ok";
    text.textContent = `정상 운영 중 — ${runMode}`;
  } else {
    banner.className = "safety-banner warn";
    text.textContent = "상태 정보 없음";
  }
  detail.textContent = flags;
}

/* ── 히어로 카드 ── */
function renderHero(summary) {
  const counts   = summary.counts || {};
  const account  = summary.account_state || summary.paper_state || {};
  const backtest = summary.backtest?.summary?.strong_entry_signal || {};
  const mode     = String(summary.account_mode || "paper").toLowerCase();
  const modeLabel = mode === "live" ? "RULE Live 계좌" : "RULE Paper 계좌";
  const asOf     = summary.account_as_of_date ? ` · ${esc(summary.account_as_of_date)}` : "";

  document.getElementById("heroGrid").innerHTML = `
    <div class="hero-card">
      <div class="hero-label">기준일</div>
      <div class="hero-value">${esc(summary.as_of_date || "-")}</div>
      <div class="hero-sub">${esc(summary.strategy_id || "-")} · ${esc(summary.run_mode || "-")}</div>
    </div>
    <div class="hero-card">
      <div class="hero-label">후보 종목</div>
      <div class="hero-value">${fmtNum(counts.total_candidates)}</div>
      <div class="hero-sub">Entry ${fmtNum(counts.entry_signal_count)} / Strong ${fmtNum(counts.strong_entry_count)}</div>
    </div>
    <div class="hero-card">
      <div class="hero-label">${modeLabel}</div>
      <div class="hero-value">${fmtWon(account.total_equity)}</div>
      <div class="hero-sub">현금 ${fmtWon(account.cash)} · 보유 ${fmtNum(counts.account_position_count || counts.paper_position_count)}종목${asOf}</div>
    </div>
    <div class="hero-card">
      <div class="hero-label">Strong D+20 수익</div>
      <div class="hero-value ${signedClass(backtest.avg_return_d20)}">${fmtPct(backtest.avg_return_d20)}</div>
      <div class="hero-sub">승률 ${fmtPct(backtest.win_rate_d20)} · ${fmtNum(backtest.trade_count)}건</div>
    </div>
  `;
}

/* ── 탭 뱃지 업데이트 ── */
function updateBadges(summary, signals, preview, execResults) {
  const counts = summary.counts || {};
  document.getElementById("badgeSignals").textContent =
    fmtNum((counts.entry_signal_count || 0) + (counts.strong_entry_count || 0));
  document.getElementById("badgeOrders").textContent =
    fmtNum(counts.preview_request_count || 0);
  document.getElementById("badgeAccount").textContent =
    fmtNum(counts.account_position_count || counts.paper_position_count || 0);

  // 서브탭 뱃지
  const portfolioCount = counts.total_candidates || 0;
  const draftCount     = (preview?.items || []).length;
  const execCount      = (execResults?.items || []).length;
  document.getElementById("badgePortfolio").textContent = portfolioCount || "-";
  document.getElementById("badgeDraft").textContent     = draftCount     || "-";
  document.getElementById("badgeExecution").textContent = execCount      || "-";
}

/* ── 개요 — Top 3 Strong 카드 ── */
function renderSignalCards(strongItems) {
  const el = document.getElementById("signalCardsGrid");
  if (!strongItems.length) {
    el.innerHTML = `<div class="empty" style="grid-column:1/-1">오늘 Strong 시그널이 없습니다.</div>`;
    return;
  }
  el.innerHTML = strongItems.slice(0, 3).map((item, idx) => {
    const rankLabel = ["1위", "2위", "3위"][idx] || `${idx+1}위`;
    return `
      <div class="signal-card">
        <div class="signal-card-top">
          <span class="chip ok" style="font-size:10px">${rankLabel}</span>
          <span class="signal-card-code">${esc(item.code)}</span>
          ${actionChip(item.portfolio_action || "hold")}
        </div>
        <div class="signal-card-name">${esc(item.name)}</div>
        <div class="signal-card-score ${signedClass(item.rule_score)}">${fmtNum(item.rule_score, 1)}</div>
        <div class="signal-card-meta">
          예상 갭 ${fmtPct(item.expected_gap)} &nbsp;·&nbsp; ${esc(item.sector || "-")}
        </div>
      </div>
    `;
  }).join("");
}

/* ── 시스템 안전 상태 ── */
function renderSystemStatus(diagnostics) {
  const s = diagnostics?.summary || {};
  document.getElementById("systemStatusKv").innerHTML = `
    <div class="kv-row"><span>실매매 가능 여부</span><strong class="${s.live_trade_ready ? "pos" : "neg"}">${s.live_trade_ready ? "가능" : "차단"}</strong></div>
    <div class="kv-row"><span>운용 모드</span><strong>${esc(s.run_mode || "-")}${s.debug_trade_mode ? " · DEBUG" : ""}</strong></div>
    <div class="kv-row"><span>현재 현금</span><strong>${fmtWon(s.current_cash)}</strong></div>
    <div class="kv-row"><span>주간 손익</span><strong class="${signedClass(s.weekly_total_pnl)}">${fmtWon(s.weekly_total_pnl)}</strong></div>
    <div class="kv-row"><span>주간 한도</span><strong>${fmtPct(s.weekly_limit)}</strong></div>
    <div class="kv-row"><span>주간 차단</span><strong class="${s.weekly_blocked ? "neg" : "pos"}">${s.weekly_blocked ? "차단" : "정상"}</strong></div>
    <div class="kv-row"><span>주 시작일</span><strong>${esc(s.week_start_date || "-")}</strong></div>
    <div class="kv-row"><span>Preview 생성 시각</span><strong>${esc(s.current_preview_generated_at || "-")}</strong></div>
  `;
}

/* ── 주문이 없었던 이유 ── */
function renderWhyNoTrade(diagnostics) {
  const s     = diagnostics?.summary || {};
  const items = Array.isArray(diagnostics?.diagnostics) ? diagnostics.diagnostics : [];
  const main  = s.main_block_detail || items[0] || null;

  document.getElementById("whyNoTradeKv").innerHTML = `
    <div class="kv-row"><span>주요 원인</span><strong>${esc(main?.user_message_ko || main?.message_ko || "차단 사유 없음")}</strong></div>
    <div class="kv-row"><span>원문 코드</span><strong>${esc(main?.raw_reason || s.main_block_reason || "-")}</strong></div>
    <div class="kv-row"><span>제출 가능 주문</span><strong>${fmtNum(s.submit_allowed_count)}건</strong></div>
    <div class="kv-row"><span>정책 차단</span><strong>${fmtNum(s.policy_blocked_count)}건</strong></div>
    <div class="kv-row"><span>API 실패</span><strong>${fmtNum(s.api_error_count)}건</strong></div>
    <div class="kv-row"><span>재생성 경고</span><strong class="${s.replay_warning ? "neg" : ""}">${s.replay_warning ? "있음" : "없음"}</strong></div>
  `;

  document.getElementById("diagnosticChips").innerHTML = items.map((item) => {
    const cls = item.severity === "ERROR" || item.severity === "BLOCKED" ? "bad" : "warn";
    return `<span class="chip ${cls}">${esc(item.type || "DIAGNOSTIC")}</span>`;
  }).join("");
}

/* ── 차단 사유 분포 ── */
function renderBlockReasons(summary) {
  const dist  = summary.distributions || {};
  const order = dist.order_block_reason || [];
  const value = dist.trading_value_block_reason || [];
  const gap   = dist.gap_risk_reason || [];
  const rows  = [];

  order.slice(0, 5).forEach((i) =>
    rows.push(`<div class="kv-row"><span>주문 차단 · ${esc(i.name)}</span><strong>${fmtNum(i.count)}건</strong></div>`));
  value.slice(0, 3).forEach((i) =>
    rows.push(`<div class="kv-row"><span>거래대금 · ${esc(i.name)}</span><strong>${fmtNum(i.count)}건</strong></div>`));
  gap.slice(0, 3).forEach((i) =>
    rows.push(`<div class="kv-row"><span>갭 리스크 · ${esc(i.name)}</span><strong>${fmtNum(i.count)}건</strong></div>`));

  document.getElementById("blockReasonKv").innerHTML =
    rows.length ? rows.join("") : `<div class="empty">차단 사유 데이터가 없습니다.</div>`;
}

/* ── 백테스트 요약 ── */
function renderBacktest(summary) {
  const strong = summary.backtest?.summary?.strong_entry_signal || {};
  const entry  = summary.backtest?.summary?.entry_signal || {};
  const curve  = summary.backtest?.portfolio_equity_curve?.strong_entry_signal || {};

  document.getElementById("backtestKv").innerHTML = `
    <div class="kv-row"><span>Entry D+20 평균수익</span><strong class="${signedClass(entry.avg_return_d20)}">${fmtPct(entry.avg_return_d20)}</strong></div>
    <div class="kv-row"><span>Strong D+20 평균수익</span><strong class="${signedClass(strong.avg_return_d20)}">${fmtPct(strong.avg_return_d20)}</strong></div>
    <div class="kv-row"><span>Strong D+60 평균수익</span><strong class="${signedClass(strong.avg_return_d60)}">${fmtPct(strong.avg_return_d60)}</strong></div>
    <div class="kv-row"><span>포트폴리오 MDD</span><strong class="${signedClass(curve.mdd_d20_portfolio_equity)}">${fmtPct(curve.mdd_d20_portfolio_equity)}</strong></div>
    <div class="kv-row"><span>포트폴리오 최종 수익</span><strong class="${signedClass(curve.final_return_d20_portfolio_equity)}">${fmtPct(curve.final_return_d20_portfolio_equity)}</strong></div>
  `;

  document.getElementById("backtestChips").innerHTML = `
    <span class="chip info">거래 ${fmtNum(strong.trade_count)}건</span>
    <span class="chip info">승률 ${fmtPct(strong.win_rate_d20)}</span>
    <span class="chip info">턴오버 ${fmtNum(summary.backtest?.summary?.turnover_strong_per_signal_date, 2)}</span>
  `;
}

const scoreClass = (v) => {
  const n = Number(v);
  if (!Number.isFinite(n)) return "";
  if (n >= 85) return "score-hi";
  if (n < 75)  return "score-lo";
  return "";
};

/* ── Strong / Entry 시그널 테이블 ── */
function renderStrongSignals(items) {
  const tbody = document.getElementById("strongSignalsTbody");
  if (!items.length) {
    tbody.innerHTML = `<tr><td colspan="9" class="center"><div class="empty">Strong 후보가 없습니다.</div></td></tr>`;
    return;
  }
  tbody.innerHTML = items.map((item, idx) => {
    const rank = idx + 1;
    return `
      <tr>
        <td><span class="rank-badge ${rank <= 3 ? "top" : ""}">${rank}</span></td>
        <td class="mono">${esc(item.code)}</td>
        <td>${esc(item.name)}</td>
        <td title="${esc(item.sector || "")}" style="max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(item.sector || "-")}</td>
        <td class="right ${scoreClass(item.rule_score)}">${fmtNum(item.rule_score, 2)}</td>
        <td class="right ${scoreClass(item.rule_score_v2)}">${fmtNum(item.rule_score_v2, 2)}</td>
        <td class="right">${fmtNum(item.expected_entry_price)}</td>
        <td class="right ${signedClass(item.expected_gap)}">${fmtPct(item.expected_gap)}</td>
        <td class="center">${actionChip(item.portfolio_action || "hold")}</td>
      </tr>
    `;
  }).join("");
}

function renderEntrySignals(items) {
  const tbody = document.getElementById("entrySignalsTbody");
  if (!items.length) {
    tbody.innerHTML = `<tr><td colspan="8" class="center"><div class="empty">Entry 후보가 없습니다.</div></td></tr>`;
    return;
  }
  tbody.innerHTML = items.map((item) => {
    const blockText = item.gap_risk_blocked
      ? item.gap_risk_reason
      : (!item.trading_value_pass ? item.trading_value_block_reason : "없음");
    return `
      <tr>
        <td class="center">${signalChip(item.signal_strength)}</td>
        <td class="mono">${esc(item.code)}</td>
        <td>${esc(item.name)}</td>
        <td title="${esc(item.sector || "")}" style="max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(item.sector || "-")}</td>
        <td class="right ${scoreClass(item.rule_score)}">${fmtNum(item.rule_score, 2)}</td>
        <td class="right ${scoreClass(item.rule_score_v2)}">${fmtNum(item.rule_score_v2, 2)}</td>
        <td class="right">${fmtWon(item.trading_value_ma_20)}</td>
        <td>${esc(blockText || "없음")}</td>
      </tr>
    `;
  }).join("");
}

/* ── 포트폴리오 계획 ── */
function renderPortfolio(items) {
  _portfolioAllItems = items;
  const filtered = _buyOnly
    ? items.filter((i) => ["buy","sell"].includes(String(i.portfolio_action || "").toLowerCase()))
    : items;

  const meta = document.getElementById("portfolioMeta");
  const totalBuy = items.filter((i) => String(i.portfolio_action || "").toLowerCase() === "buy").length;
  meta.textContent = `BUY ${totalBuy}건 / 전체 ${items.length}건${_buyOnly ? " (BUY/SELL만 표시)" : ""}`;

  const tbody = document.getElementById("portfolioTbody");
  if (!filtered.length) {
    tbody.innerHTML = `<tr><td colspan="8" class="center"><div class="empty">포트폴리오 계획이 없습니다.</div></td></tr>`;
    return;
  }
  tbody.innerHTML = filtered.map((item) => `
    <tr>
      <td>${actionChip(item.portfolio_action)}</td>
      <td class="mono">${esc(item.code)}</td>
      <td>${esc(item.name)}</td>
      <td style="max-width:110px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(item.sector||"")}">${esc(item.sector || "-")}</td>
      <td class="right">${fmtPct(item.target_weight)}</td>
      <td class="right">${fmtPct(item.current_weight)}</td>
      <td class="right">${fmtWon(item.target_amount)}</td>
      <td>${esc(item.portfolio_action_reason || "-")}</td>
    </tr>
  `).join("");
}

/* ── 주문 초안 ── */
function renderPreview(items) {
  const tbody = document.getElementById("previewTbody");
  if (!items.length) {
    tbody.innerHTML = `<tr><td colspan="8" class="center"><div class="empty">주문 초안이 없습니다.</div></td></tr>`;
    return;
  }
  tbody.innerHTML = items.slice(0, 50).map((item) => {
    const blockChips = (Array.isArray(item.block_reason_details) ? item.block_reason_details : [])
      .map((d) => {
        const cls = d.category === "API_ERROR" || d.severity === "ERROR" || d.severity === "BLOCKED" ? "bad" : "warn";
        const label = d.block_reason && d.block_reason !== "UNMAPPED" ? d.block_reason : (d.raw_reason || "-");
        const tip = d.user_message_ko ? esc(d.user_message_ko) : "";
        return `<span class="chip ${cls}" title="${tip}">${esc(label)}</span>`;
      }).join(" ") || esc(item.order_block_reason || "-");
    return `
      <tr>
        <td>${orderStatusChip(item)}</td>
        <td class="mono">${esc(item.code)}</td>
        <td>${esc(item.name)}</td>
        <td>${sideText(item.side)}</td>
        <td class="right">${fmtNum(item.expected_execution_price)}</td>
        <td class="right">${fmtWon(item.order_sizing?.base_order_amount ?? item.order_amount)}</td>
        <td class="right">${fmtNum(item.order_qty)}</td>
        <td>${blockChips}</td>
      </tr>
    `;
  }).join("");
}

/* ── 개장 전 실행 결과 ── */
function renderExecution(results) {
  const tbody = document.getElementById("executionTbody");
  const items = Array.isArray(results.items) ? results.items : [];
  if (!items.length) {
    const msg = results.order_run_aborted
      ? `실행 중단: ${esc(results.order_run_abort_reason || "원인 불명")}`
      : "실행 결과가 없습니다.";
    tbody.innerHTML = `<tr><td colspan="8" class="center"><div class="empty">${msg}</div></td></tr>`;
    return;
  }
  tbody.innerHTML = items.slice(0, 50).map((item) => {
    const blockChips = (Array.isArray(item.block_reason_details) ? item.block_reason_details : [])
      .map((d) => {
        const cls = d.category === "API_ERROR" || d.severity === "ERROR" || d.severity === "BLOCKED" ? "bad" : "warn";
        const label = d.block_reason && d.block_reason !== "UNMAPPED" ? d.block_reason : (d.raw_reason || "-");
        const tip = d.user_message_ko ? esc(d.user_message_ko) : "";
        return `<span class="chip ${cls}" title="${tip}">${esc(label)}</span>`;
      }).join(" ") || esc(item.order_block_reason || "-");
    return `
      <tr>
        <td>${orderStatusChip(item)}</td>
        <td class="mono">${esc(item.code)}</td>
        <td>${esc(item.name)}</td>
        <td>${sideText(item.side)}</td>
        <td class="right">${fmtNum(item.filled_qty)}</td>
        <td class="right">${fmtNum(item.avg_fill_price)}</td>
        <td class="right ${signedClass(item.actual_open_gap)}">${fmtPct(item.actual_open_gap)}</td>
        <td>${blockChips}</td>
      </tr>
    `;
  }).join("");
}

/* ── Paper 계좌 보유 종목 ── */
function renderPaperState(paperState) {
  const positions = Array.isArray(paperState.positions) ? paperState.positions : [];
  const tbody = document.getElementById("positionsTbody");

  if (!positions.length) {
    tbody.innerHTML = `<tr><td colspan="8" class="center"><div class="empty">보유 종목이 없습니다.</div></td></tr>`;
  } else {
    tbody.innerHTML = positions.map((item) => `
      <tr>
        <td class="mono">${esc(item.code)}</td>
        <td>${esc(item.name)}</td>
        <td>${esc(item.sector || "-")}</td>
        <td class="right">${fmtNum(item.qty)}</td>
        <td class="right">${fmtNum(item.entry_price)}</td>
        <td class="right">${fmtNum(item.last_price)}</td>
        <td class="right">${fmtWon(item.market_value)}</td>
        <td class="right">${fmtPct(item.weight)}</td>
      </tr>
    `).join("");
  }

  const trades = Array.isArray(paperState.recent_trades) ? paperState.recent_trades : [];
  document.getElementById("recentTrades").innerHTML = trades.length
    ? trades.slice(-6).reverse().map((t) =>
        `<span class="chip info">${esc(t.date)} · ${t.side === "BUY" ? "매수" : "매도"} · ${esc(t.code)} · ${fmtNum(t.qty)}주</span>`
      ).join("")
    : `<div class="empty">최근 거래 내역이 없습니다.</div>`;
}

/* ── 메인 로드 ── */
async function loadRuleDashboard() {
  const bar = document.getElementById("statusBar");
  bar.textContent = "RULE 대시보드 데이터를 불러오는 중…";

  try {
    const [executionResults, summary, signals, portfolio, preview, paperState, diagnostics] = await Promise.all([
      fetchJson("/api/rule/execution-results").catch(() => ({ items: [], summary: {} })),
      fetchJson("/api/rule/summary").catch(() => ({ counts: {}, operations: {} })),
      fetchJson("/api/rule/signals/latest?strength=all&limit=30").catch(() => ({ items: [] })),
      fetchJson("/api/rule/portfolio-plan").catch(() => ({ items: [] })),
      fetchJson("/api/rule/order-preview").catch(() => ({ items: [] })),
      fetchJson("/api/rule/paper-state").catch(() => ({ positions: [], recent_trades: [] })),
      fetchJson("/api/rule/trading-diagnostics").catch(() => ({ summary: {}, diagnostics: [] })),
    ]);

    renderSafety(summary, diagnostics);
    renderHero(summary);
    updateBadges(summary, signals, preview, executionResults);

    // 개요 탭
    const strongItems = (signals.items || [])
      .filter((i) => i.strong_entry_signal)
      .slice(0, 10)
      .map((i) => {
        const plan = (portfolio.items || []).find((r) => r.code === i.code);
        return { ...i, portfolio_action: plan?.portfolio_action || "hold" };
      });
    renderSignalCards(strongItems);
    renderSystemStatus(diagnostics);
    renderWhyNoTrade(diagnostics);
    renderBlockReasons(summary);
    renderBacktest(summary);

    // 시그널 탭
    renderStrongSignals(strongItems);
    renderEntrySignals((signals.items || []).filter((i) => i.entry_signal).slice(0, 20));

    // 주문 탭 (서브탭 3개)
    renderPortfolio(portfolio.items || []);
    renderPreview(preview.items || []);
    renderExecution(executionResults);

    // 계좌 탭
    renderPaperState(paperState);

    const counts = summary.counts || {};
    const aborted = executionResults.order_run_aborted
      ? ` · 중단: ${executionResults.order_run_abort_reason || "-"}`
      : "";
    bar.textContent = [
      `기준일 ${summary.as_of_date || "-"}`,
      `후보 ${fmtNum(counts.total_candidates)}`,
      `Strong ${fmtNum(counts.strong_entry_count)}`,
      `체결 ${fmtNum(counts.execution_filled_count)}`,
      `부분체결 ${fmtNum(counts.execution_partial_filled_count)}`,
      `제출 ${fmtNum(counts.execution_submitted_count)}`,
    ].join("  ·  ") + aborted;

  } catch (err) {
    console.error(err);
    bar.textContent = `RULE 대시보드 로드 실패: ${err.message}`;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initSubTabs();
  initBuyOnlyToggle();
  loadRuleDashboard().catch(console.error);
});
