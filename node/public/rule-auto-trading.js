const fmtNum = (value, digits = 0) => {
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  return n.toLocaleString("ko-KR", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
};

const fmtPct = (value, digits = 2) => {
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

const signedClass = (value) => {
  const n = Number(value);
  if (!Number.isFinite(n) || n === 0) return "";
  return n > 0 ? "pos" : "neg";
};

const opsToneClass = (value) => {
  const tone = String(value || "").toLowerCase();
  if (tone === "normal") return "good";
  if (tone === "risk" || tone === "stopped") return "bad";
  if (tone === "warning") return "warn";
  return "watch";
};

const flagText = (value) => {
  if (value === true) return "ON";
  if (value === false) return "OFF";
  return "-";
};

const signalChip = (value) => {
  if (value === "strong_entry") return `<span class="chip good">strong</span>`;
  if (value === "entry") return `<span class="chip watch">entry</span>`;
  return `<span class="chip warn">none</span>`;
};

const actionChip = (value) => {
  if (value === "buy") return `<span class="chip good">buy</span>`;
  if (value === "hold") return `<span class="chip watch">hold</span>`;
  if (value === "reduce") return `<span class="chip warn">reduce</span>`;
  if (value === "exit") return `<span class="chip bad">exit</span>`;
  return `<span class="chip warn">${escapeHtml(value || "skip")}</span>`;
};

const orderChip = (row) => {
  if (row.order_status === "simulated_filled") return `<span class="chip good">filled</span>`;
  if (row.order_status === "simulated_unfilled") return `<span class="chip warn">unfilled</span>`;
  if (row.order_status === "submitted") return `<span class="chip good">submitted</span>`;
  if (row.order_status === "filled") return `<span class="chip good">filled</span>`;
  if (row.order_status === "partial_filled") return `<span class="chip watch">partial</span>`;
  if (row.order_status === "unfilled") return `<span class="chip warn">unfilled</span>`;
  if (row.order_status === "canceled") return `<span class="chip warn">canceled</span>`;
  if (row.order_status === "failed") return `<span class="chip bad">failed</span>`;
  if (row.order_status === "blocked") return `<span class="chip bad">blocked</span>`;
  if (row.order_allowed) return `<span class="chip good">allowed</span>`;
  if (row.side === "BUY" || row.side === "SELL") return `<span class="chip watch">preview</span>`;
  return `<span class="chip bad">blocked</span>`;
};

async function fetchJson(url) {
  const res = await fetch(url, { credentials: "same-origin" });
  if (!res.ok) throw new Error(`${url} HTTP ${res.status}`);
  return res.json();
}

function renderHero(summary) {
  const counts = summary.counts || {};
  const accountState = summary.account_state || summary.paper_state || {};
  const accountMode = String(summary.account_mode || "paper").toLowerCase();
  const accountLabel = accountMode === "live" ? "RULE LIVE" : "RULE Paper";
  const accountAsOfDate = summary.account_as_of_date ? ` / ${escapeHtml(summary.account_as_of_date)}` : "";
  const backtest = summary.backtest?.summary?.strong_entry_signal || {};
  document.getElementById("heroGrid").innerHTML = `
    <article class="hero-card">
      <div class="card-label">As Of</div>
      <div class="card-value">${escapeHtml(summary.as_of_date || "-")}</div>
      <div class="card-detail">${escapeHtml(summary.strategy_id || "-")} / ${escapeHtml(summary.run_mode || "-")}</div>
    </article>
    <article class="hero-card">
      <div class="card-label">Candidates</div>
      <div class="card-value">${fmtNum(counts.total_candidates)}</div>
      <div class="card-detail">entry ${fmtNum(counts.entry_signal_count)} / strong ${fmtNum(counts.strong_entry_count)}</div>
    </article>
    <article class="hero-card">
      <div class="card-label">${accountLabel}</div>
      <div class="card-value">${fmtNum(accountState.total_equity)}</div>
      <div class="card-detail">cash ${fmtNum(accountState.cash)} / positions ${fmtNum(counts.account_position_count || counts.paper_position_count)}${accountAsOfDate}</div>
    </article>
    <article class="hero-card">
      <div class="card-label">Strong D+20</div>
      <div class="card-value ${signedClass(backtest.avg_return_d20)}">${fmtPct(backtest.avg_return_d20)}</div>
      <div class="card-detail">win ${fmtPct(backtest.win_rate_d20)} / trade ${fmtNum(backtest.trade_count)}</div>
    </article>
  `;
}
function renderStatus(summary) {
  const counts = summary.counts || {};
  const scoreDist = summary.backtest?.score_distribution || {};
  const tradingDist = summary.backtest?.trading_value_distribution || {};
  const executionDone =
    (counts.execution_filled_count || 0) +
    (counts.execution_partial_filled_count || 0) +
    (counts.execution_unfilled_count || 0) +
    (counts.execution_failed_count || 0);
  document.getElementById("statusGrid").innerHTML = `
    <article class="hero-card">
      <div class="card-label">?ы듃?대━???먮떒</div>
      <div class="card-value">${fmtNum(counts.hold_count)} / ${fmtNum(counts.buy_count)}</div>
      <div class="card-detail">hold / buy / reduce ${fmtNum(counts.reduce_count)} / exit ${fmtNum(counts.exit_count)}</div>
    </article>
    <article class="hero-card">
      <div class="card-label">Order / Execution</div>
      <div class="card-value">${fmtNum(counts.preview_request_count)} / ${fmtNum(executionDone || counts.execution_simulated_filled_count)}</div>
      <div class="card-detail">preview / filled ${fmtNum(counts.execution_filled_count)} / partial ${fmtNum(counts.execution_partial_filled_count)} / unfilled ${fmtNum(counts.execution_unfilled_count)} / failed ${fmtNum(counts.execution_failed_count)}</div>
    </article>
    <article class="hero-card">
      <div class="card-label">?먯닔 遺꾪룷</div>
      <div class="card-value">${fmtNum(scoreDist.rule_score_v2?.max, 2)}</div>
      <div class="card-detail">v2 max / p75 ${fmtNum(scoreDist.rule_score_v2?.p75, 2)} / 嫄곕옒?湲?5???듦낵??${fmtPct(tradingDist.thresholds?.[0]?.pass_rate, 0)}</div>
    </article>
  `;
}

function renderStatusV2(summary) {
  const counts = summary.counts || {};
  const scoreDist = summary.backtest?.score_distribution || {};
  const tradingDist = summary.backtest?.trading_value_distribution || {};
  const ops = summary.operations || {};
  const controls = ops.controls || {};
  const ruleOps = ops.rule || {};
  const opsCards = ops.cards || {};
  const executionDone =
    (counts.execution_filled_count || 0) +
    (counts.execution_partial_filled_count || 0) +
    (counts.execution_unfilled_count || 0) +
    (counts.execution_failed_count || 0);
  document.getElementById("statusGrid").innerHTML = `
    <article class="hero-card">
      <div class="card-label">Portfolio</div>
      <div class="card-value">${fmtNum(counts.hold_count)} / ${fmtNum(counts.buy_count)}</div>
      <div class="card-detail">hold / buy / reduce ${fmtNum(counts.reduce_count)} / exit ${fmtNum(counts.exit_count)}</div>
    </article>
    <article class="hero-card">
      <div class="card-label">Order / Execution</div>
      <div class="card-value">${fmtNum(counts.preview_request_count)} / ${fmtNum(executionDone || counts.execution_simulated_filled_count)}</div>
      <div class="card-detail">preview / filled ${fmtNum(counts.execution_filled_count)} / partial ${fmtNum(counts.execution_partial_filled_count)} / unfilled ${fmtNum(counts.execution_unfilled_count)} / failed ${fmtNum(counts.execution_failed_count)}</div>
    </article>
    <article class="hero-card">
      <div class="card-label">Score Dist</div>
      <div class="card-value">${fmtNum(scoreDist.rule_score_v2?.max, 2)}</div>
      <div class="card-detail">v2 max / p75 ${fmtNum(scoreDist.rule_score_v2?.p75, 2)} / value 5e8 pass ${fmtPct(tradingDist.thresholds?.[0]?.pass_rate, 0)}</div>
    </article>
    <article class="hero-card">
      <div class="card-label">RULE Ops</div>
      <div class="card-value ${opsToneClass(ops.summary?.overall_tone)}">${fmtNum(ruleOps.buy_candidate_count)}</div>
      <div class="card-detail">blocked ${fmtNum(ruleOps.buy_blocked_count)} / submitted ${fmtNum(ruleOps.submitted_count)} / filled ${fmtNum(ruleOps.filled_count)} / pre-open ${opsCards.rule_before_open?.today_success ? "OK" : "WAIT"} / after-open ${opsCards.rule_after_open?.today_success ? "OK" : "WAIT"}</div>
    </article>
    <article class="hero-card">
      <div class="card-label">Safety</div>
      <div class="card-value ${opsToneClass(ops.summary?.overall_tone)}">${controls.rule_kill_switch ? "STOP" : "RUN"}</div>
      <div class="card-detail">GLOBAL ${flagText(controls.global_kill_switch)} / RULE ${flagText(controls.rule_kill_switch)} / EXECUTE ${flagText(controls.auto_trade_execute)} / ALLOW_BUY ${flagText(controls.auto_trade_allow_buy)}</div>
    </article>
  `;
}

function renderBlockReasons(summary) {
  const root = document.getElementById("blockReasonKv");
  const orderReasons = summary.distributions?.order_block_reason || [];
  const tradingReasons = summary.distributions?.trading_value_block_reason || [];
  const gapReasons = summary.distributions?.gap_risk_reason || [];
  const rows = [];
  orderReasons.slice(0, 5).forEach((item) => {
    rows.push(`<div class="kv-row"><span>order block: ${escapeHtml(item.name)}</span><strong>${fmtNum(item.count)}</strong></div>`);
  });
  tradingReasons.slice(0, 3).forEach((item) => {
    rows.push(`<div class="kv-row"><span>trading value: ${escapeHtml(item.name)}</span><strong>${fmtNum(item.count)}</strong></div>`);
  });
  gapReasons.slice(0, 3).forEach((item) => {
    rows.push(`<div class="kv-row"><span>gap risk: ${escapeHtml(item.name)}</span><strong>${fmtNum(item.count)}</strong></div>`);
  });
  root.innerHTML = rows.length ? rows.join("") : `<div class="empty-state">李⑤떒 ?ъ쑀 ?곗씠?곌? ?놁뒿?덈떎.</div>`;
}

function renderSystemStatus(diagnostics) {
  const root = document.getElementById("systemStatusKv");
  const summary = diagnostics?.summary || {};
  root.innerHTML = `
    <div class="kv-row"><span>실매매 가능 여부</span><strong>${summary.live_trade_ready ? "가능" : "차단"}</strong></div>
    <div class="kv-row"><span>현재 모드</span><strong>${escapeHtml(summary.run_mode || "-")}${summary.debug_trade_mode ? " / DEBUG" : ""}</strong></div>
    <div class="kv-row"><span>현재 현금</span><strong>${fmtNum(summary.current_cash)}</strong></div>
    <div class="kv-row"><span>주간 손익</span><strong class="${signedClass(summary.weekly_total_pnl)}">${fmtNum(summary.weekly_total_pnl)}</strong></div>
    <div class="kv-row"><span>주간 제한값</span><strong>${fmtPct(summary.weekly_limit)}</strong></div>
    <div class="kv-row"><span>주간 차단 여부</span><strong>${summary.weekly_blocked ? "차단" : "정상"}</strong></div>
    <div class="kv-row"><span>주간 계산 기준</span><strong>${escapeHtml(summary.weekly_loss_source || "-")}</strong></div>
    <div class="kv-row"><span>주간 시작일</span><strong>${escapeHtml(summary.week_start_date || "-")}</strong></div>
    <div class="kv-row"><span>실패 기준 실행 시각</span><strong>${escapeHtml(summary.actual_failure_at || "-")}</strong></div>
    <div class="kv-row"><span>현재 Preview 시각</span><strong>${escapeHtml(summary.current_preview_generated_at || "-")}</strong></div>
    <div class="kv-row"><span>Timezone</span><strong>${escapeHtml(summary.timezone || "-")}</strong></div>
  `;
}

function renderWhyNoTrade(diagnostics) {
  const root = document.getElementById("whyNoTradeKv");
  const badges = document.getElementById("diagnosticBadges");
  const summary = diagnostics?.summary || {};
  const items = Array.isArray(diagnostics?.diagnostics) ? diagnostics.diagnostics : [];
  const mainDetail = summary.main_block_detail || items[0] || null;
  root.innerHTML = `
    <div class="kv-row"><span>주요 원인</span><strong>${escapeHtml(mainDetail?.user_message_ko || mainDetail?.message_ko || "차단 사유 없음")}</strong></div>
    <div class="kv-row"><span>원문 사유</span><strong>${escapeHtml(mainDetail?.raw_reason || summary.main_block_reason || "-")}</strong></div>
    <div class="kv-row"><span>실제 제출 가능 주문 수</span><strong>${fmtNum(summary.submit_allowed_count)}</strong></div>
    <div class="kv-row"><span>정책 차단 수</span><strong>${fmtNum(summary.policy_blocked_count)}</strong></div>
    <div class="kv-row"><span>API 실패 수</span><strong>${fmtNum(summary.api_error_count)}</strong></div>
    <div class="kv-row"><span>실제 실패 기준</span><strong>${escapeHtml(summary.actual_failure_at || "-")} / as_of ${escapeHtml(summary.actual_failure_as_of_date || "-")}</strong></div>
    <div class="kv-row"><span>현재 Preview 기준</span><strong>${escapeHtml(summary.current_preview_generated_at || "-")} / as_of ${escapeHtml(summary.current_preview_as_of_date || "-")}</strong></div>
    <div class="kv-row"><span>재생성 경고</span><strong>${summary.replay_warning ? "있음" : "없음"}</strong></div>
  `;
  badges.innerHTML = items.map((item) => {
    const tone = item.severity === "ERROR" ? "bad" : item.severity === "BLOCKED" ? "bad" : "watch";
    return `<span class="chip ${tone}">${escapeHtml(item.type || "DIAGNOSTIC")}</span>`;
  }).join("");
}

function renderBacktest(summary) {
  const strong = summary.backtest?.summary?.strong_entry_signal || {};
  const entry = summary.backtest?.summary?.entry_signal || {};
  const curve = summary.backtest?.portfolio_equity_curve?.strong_entry_signal || {};
  document.getElementById("backtestKv").innerHTML = `
    <div class="kv-row"><span>entry D+20 ?됯퇏</span><strong class="${signedClass(entry.avg_return_d20)}">${fmtPct(entry.avg_return_d20)}</strong></div>
    <div class="kv-row"><span>strong D+20 ?됯퇏</span><strong class="${signedClass(strong.avg_return_d20)}">${fmtPct(strong.avg_return_d20)}</strong></div>
    <div class="kv-row"><span>strong D+60 ?됯퇏</span><strong class="${signedClass(strong.avg_return_d60)}">${fmtPct(strong.avg_return_d60)}</strong></div>
    <div class="kv-row"><span>portfolio MDD</span><strong class="${signedClass(curve.mdd_d20_portfolio_equity)}">${fmtPct(curve.mdd_d20_portfolio_equity)}</strong></div>
    <div class="kv-row"><span>portfolio 理쒖쥌 ?섏씡</span><strong class="${signedClass(curve.final_return_d20_portfolio_equity)}">${fmtPct(curve.final_return_d20_portfolio_equity)}</strong></div>
  `;
  document.getElementById("backtestChips").innerHTML = `
    <span class="chip watch">trade ${fmtNum(strong.trade_count)}</span>
    <span class="chip watch">win ${fmtPct(strong.win_rate_d20)}</span>
    <span class="chip watch">turnover ${fmtNum(summary.backtest?.summary?.turnover_strong_per_signal_date, 2)}</span>
  `;
}

function renderSignals(tbodyId, items, entryMode = false) {
  const tbody = document.getElementById(tbodyId);
  if (!items.length) {
    tbody.innerHTML = `<tr><td colspan="8" class="center">?곗씠?곌? ?놁뒿?덈떎.</td></tr>`;
    return;
  }
  tbody.innerHTML = items.map((item) => {
    if (entryMode) {
      const blockText = item.gap_risk_blocked
        ? item.gap_risk_reason
        : (!item.trading_value_pass ? item.trading_value_block_reason : "none");
      return `
        <tr>
          <td>${signalChip(item.signal_strength)}</td>
          <td class="mono">${escapeHtml(item.code)}</td>
          <td>${escapeHtml(item.name)}</td>
          <td>${escapeHtml(item.sector || "-")}</td>
          <td class="right">${fmtNum(item.rule_score, 2)}</td>
          <td class="right">${fmtNum(item.rule_score_v2, 2)}</td>
          <td class="right">${fmtNum(item.trading_value_ma_20)}</td>
          <td>${escapeHtml(blockText || "none")}</td>
        </tr>
      `;
    }
    return `
      <tr>
        <td class="mono">${escapeHtml(item.code)}</td>
        <td>${escapeHtml(item.name)}</td>
        <td>${escapeHtml(item.sector || "-")}</td>
        <td class="right">${fmtNum(item.rule_score, 2)}</td>
        <td class="right">${fmtNum(item.rule_score_v2, 2)}</td>
        <td class="right">${fmtNum(item.expected_entry_price, 0)}</td>
        <td class="right">${fmtPct(item.expected_gap)}</td>
        <td>${actionChip(item.portfolio_action || "hold")}</td>
      </tr>
    `;
  }).join("");
}

function renderPortfolio(items) {
  const tbody = document.getElementById("portfolioTbody");
  if (!items.length) {
    tbody.innerHTML = `<tr><td colspan="8" class="center">?ы듃?대━??怨꾪쉷???놁뒿?덈떎.</td></tr>`;
    return;
  }
  tbody.innerHTML = items.slice(0, 40).map((item) => `
    <tr>
      <td>${actionChip(item.portfolio_action)}</td>
      <td class="mono">${escapeHtml(item.code)}</td>
      <td>${escapeHtml(item.name)}</td>
      <td>${escapeHtml(item.sector || "-")}</td>
      <td class="right">${fmtPct(item.target_weight)}</td>
      <td class="right">${fmtPct(item.current_weight)}</td>
      <td class="right">${fmtNum(item.target_amount)}</td>
      <td>${escapeHtml(item.portfolio_action_reason || "-")}</td>
    </tr>
  `).join("");
}

function renderPreview(items) {
  const tbody = document.getElementById("previewTbody");
  if (!items.length) {
    tbody.innerHTML = `<tr><td colspan="11" class="center">order preview媛 ?놁뒿?덈떎.</td></tr>`;
    return;
  }
  tbody.innerHTML = items.slice(0, 50).map((item) => `
    <tr>
      <td>${orderChip(item)}</td>
      <td class="mono">${escapeHtml(item.code)}</td>
      <td>${escapeHtml(item.name)}</td>
      <td>${escapeHtml(item.portfolio_action || "-")}</td>
      <td>${escapeHtml(item.side || "-")}</td>
      <td class="right">${fmtNum(item.expected_execution_price)}</td>
      <td class="right">${fmtNum(item.order_sizing?.base_order_amount ?? item.order_amount)}</td>
      <td class="right">${fmtNum(item.order_qty)}</td>
      <td class="right">${fmtNum(item.order_amount)}</td>
      <td>${(Array.isArray(item.block_reason_details) ? item.block_reason_details : []).map((detail) => {
        const tone = detail.category === "API_ERROR" || detail.severity === "ERROR" ? "bad" : detail.severity === "BLOCKED" ? "bad" : "watch";
        return `<span class="chip ${tone}">${escapeHtml(detail.block_reason || detail.raw_reason || "-")}</span>`;
      }).join(" ") || escapeHtml(item.order_block_reason || "-")}</td>
      <td>${item.submit_attempted ? "YES" : "NO"}</td>
    </tr>
  `).join("");
}

function renderExecution(results) {
  const tbody = document.getElementById("executionTbody");
  const items = Array.isArray(results.items) ? results.items : [];
  if (!items.length) {
    const reason = results.order_run_aborted
      ? escapeHtml(results.order_run_abort_reason || "execution_aborted")
      : "execution result unavailable";
    tbody.innerHTML = `<tr><td colspan="9" class="center">${reason}</td></tr>`;
    return;
  }
  tbody.innerHTML = items.slice(0, 50).map((item) => `
    <tr>
      <td>${orderChip(item)}</td>
      <td class="mono">${escapeHtml(item.code)}</td>
      <td>${escapeHtml(item.name)}</td>
      <td>${escapeHtml(item.side || "-")}</td>
      <td class="right">${fmtNum(item.filled_qty)}</td>
      <td class="right">${fmtNum(item.avg_fill_price)}</td>
      <td class="right">${fmtPct(item.actual_open_gap)}</td>
      <td>${(Array.isArray(item.block_reason_details) ? item.block_reason_details : []).map((detail) => {
        const tone = detail.category === "API_ERROR" || detail.severity === "ERROR" ? "bad" : detail.severity === "BLOCKED" ? "bad" : "watch";
        return `<span class="chip ${tone}">${escapeHtml(detail.block_reason || detail.raw_reason || "-")}</span>`;
      }).join(" ") || escapeHtml(item.order_block_reason || "-")}</td>
      <td>${item.submit_attempted ? "YES" : "NO"}</td>
    </tr>
  `).join("");
}

function renderPaperState(paperState) {
  const tbody = document.getElementById("positionsTbody");
  const positions = Array.isArray(paperState.positions) ? paperState.positions : [];
  if (!positions.length) {
    tbody.innerHTML = `<tr><td colspan="8" class="center">paper 蹂댁쑀 醫낅ぉ???놁뒿?덈떎.</td></tr>`;
  } else {
    tbody.innerHTML = positions.map((item) => `
      <tr>
        <td class="mono">${escapeHtml(item.code)}</td>
        <td>${escapeHtml(item.name)}</td>
        <td>${escapeHtml(item.sector || "-")}</td>
        <td class="right">${fmtNum(item.qty)}</td>
        <td class="right">${fmtNum(item.entry_price)}</td>
        <td class="right">${fmtNum(item.last_price)}</td>
        <td class="right">${fmtNum(item.market_value)}</td>
        <td class="right">${fmtPct(item.weight)}</td>
      </tr>
    `).join("");
  }
  const trades = Array.isArray(paperState.recent_trades) ? paperState.recent_trades : [];
  document.getElementById("recentTrades").innerHTML = trades.slice(-6).reverse().map((item) => `
    <span class="chip watch">${escapeHtml(item.date)} / ${escapeHtml(item.side)} / ${escapeHtml(item.code)} / ${fmtNum(item.qty)}二?/span>
  `).join("");
}

async function loadRuleDashboard() {
  const pageState = document.getElementById("pageState");
  pageState.textContent = "RULE ??쒕낫???곗씠?곕? 遺덈윭?ㅻ뒗 以묒엯?덈떎.";
  try {
    const [executionResults, summary, signals, portfolio, preview, paperState, diagnostics] = await Promise.all([
      fetchJson("/api/rule/execution-results").catch(() => ({ items: [], summary: {} })),
      fetchJson("/api/rule/summary"),
      fetchJson("/api/rule/signals/latest?strength=all&limit=30"),
      fetchJson("/api/rule/portfolio-plan"),
      fetchJson("/api/rule/order-preview"),
      fetchJson("/api/rule/paper-state").catch(() => ({ positions: [], recent_trades: [] })),
      fetchJson("/api/rule/trading-diagnostics").catch(() => ({ summary: {}, diagnostics: [] })),
    ]);

    renderHero(summary);
    renderStatusV2(summary);
    renderSystemStatus(diagnostics);
    renderWhyNoTrade(diagnostics);
    renderBlockReasons(summary);
    renderBacktest(summary);
    renderSignals(
      "strongSignalsTbody",
      (signals.items || []).filter((item) => item.strong_entry_signal).slice(0, 10).map((item) => {
        const plan = (portfolio.items || []).find((row) => row.code === item.code);
        return { ...item, portfolio_action: plan?.portfolio_action || "hold" };
      }),
      false
    );
    renderSignals("entrySignalsTbody", (signals.items || []).filter((item) => item.entry_signal).slice(0, 20), true);
    renderPortfolio(portfolio.items || []);
    renderPreview(preview.items || []);
    renderExecution(executionResults);
    renderPaperState(paperState);

    const abortedText = executionResults.order_run_aborted
      ? ` | aborted: ${executionResults.order_run_abort_reason || "-"}`
      : "";
    pageState.textContent = `湲곗???${summary.as_of_date || "-"} | ?꾨낫 ${fmtNum(summary.counts?.total_candidates)} | strong ${fmtNum(summary.counts?.strong_entry_count)} | filled ${fmtNum(summary.counts?.execution_filled_count)} | partial ${fmtNum(summary.counts?.execution_partial_filled_count)} | submitted ${fmtNum(summary.counts?.execution_submitted_count)}${abortedText}`;
  } catch (error) {
    console.error(error);
    pageState.textContent = `RULE ??쒕낫??濡쒕뱶 ?ㅽ뙣: ${error.message}`;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("refreshBtn").addEventListener("click", () => {
    loadRuleDashboard().catch((error) => console.error(error));
  });
  loadRuleDashboard().catch((error) => console.error(error));
});
