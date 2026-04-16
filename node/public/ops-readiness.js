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

const fmtBool = (value) => {
  if (value === true) return "예";
  if (value === false) return "아니오";
  return "-";
};

const escapeHtml = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (m) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[m]));

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function isIsoDate(value) {
  return /^\d{4}-\d{2}-\d{2}$/.test(String(value || "").trim());
}

function markStaleDate(value, referenceDate) {
  const date = String(value || "").trim();
  const ref = String(referenceDate || "").trim();
  if (!isIsoDate(date)) return date || "-";
  if (!isIsoDate(ref)) return date;
  if (date < ref) return `${date} (stale)`;
  if (date > ref) return `${date} (ahead)`;
  return date;
}

async function postJson(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function fetchTradingPolicySafe() {
  try {
    const res = await fetch("/api/trading-policy");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (error) {
    console.warn("trading policy unavailable", error);
    return null;
  }
}

function chipClass(status) {
  const s = String(status || "").toUpperCase();
  if (["READY", "GOOD", "BUY_ALLOWED", "PASS", "GO_CHECK"].includes(s)) return "good";
  if (["WATCH", "CONDITIONAL", "HOLD"].includes(s)) return "watch";
  if (["ALERT", "WAIT", "BLOCK", "NO_GO"].includes(s)) return "bad";
  return "info";
}

function renderHero(data) {
  const gate = data.gate || {};
  const readiness = data.readiness || {};
  const kpi = data.kpi || {};
  const outputs = data.outputs || {};
  const basis = data.execution_basis || {};
  document.getElementById("heroGrid").innerHTML = `
    <article class="card">
      <div class="eyebrow">기준일</div>
      <div class="big-value">${escapeHtml(data.asof_date || "-")}</div>
      <div class="muted">${escapeHtml(basis.label || "기준 정보 없음")} · 랭킹 최신일 ${escapeHtml(outputs.ranking_latest_date || "-")}</div>
    </article>
    <article class="card">
      <div class="eyebrow">60일 준비 상태</div>
      <div class="big-value">${escapeHtml(readiness.confidence_calibration_readiness_60d || "WAIT")}</div>
      <div class="muted">60일 성숙 스냅샷 ${fmtNum(readiness.matured_snapshot_count_60d)}</div>
    </article>
    <article class="card">
      <div class="eyebrow">매수 gate</div>
      <div class="big-value">${escapeHtml(gate.overall_status || "-")}</div>
      <div class="muted">워크포워드 ${escapeHtml(gate.walkforward_acceptance || "-")}</div>
    </article>
    <article class="card">
      <div class="eyebrow">KPI</div>
      <div class="big-value">${escapeHtml(kpi.overall_status || "-")}</div>
      <div class="muted">경고 ${fmtNum(kpi.alert_metric_count)} / 관찰 ${fmtNum(kpi.watch_metric_count)}</div>
    </article>
  `;
}

function renderKv(targetId, rows) {
  const el = document.getElementById(targetId);
  if (!el) return;
  el.innerHTML = rows
    .map(
      ([label, value]) => `
        <div class="kv-row">
          <span class="muted">${escapeHtml(label)}</span>
          <strong>${escapeHtml(value)}</strong>
        </div>
      `
    )
    .join("");
}

function renderText(targetId, value) {
  const el = document.getElementById(targetId);
  if (!el) return;
  el.textContent = value || "";
}

function renderChipRow(targetId, chips) {
  const el = document.getElementById(targetId);
  if (!el) return;
  el.innerHTML = (chips || [])
    .map((item) => `<span class="chip ${chipClass(item.kind || item.label)}">${escapeHtml(item.label)}</span>`)
    .join("");
}

function renderList(targetId, items, emptyText) {
  const el = document.getElementById(targetId);
  if (!el) return;
  if (!items || !items.length) {
    el.innerHTML = `<li>${escapeHtml(emptyText)}</li>`;
    return;
  }
  el.innerHTML = items.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
}

function renderMetricList(targetId, items, emptyText) {
  const el = document.getElementById(targetId);
  if (!el) return;
  if (!items || !items.length) {
    el.innerHTML = `<li>${escapeHtml(emptyText)}</li>`;
    return;
  }
  el.innerHTML = items
    .map((item) => {
      const metric = item.metric || "-";
      const value = Number.isFinite(Number(item.value)) ? fmtNum(item.value, 4) : String(item.value || "-");
      const status = item.status || "-";
      return `<li><strong>${escapeHtml(metric)}</strong> · ${escapeHtml(value)} · ${escapeHtml(status)}</li>`;
    })
    .join("");
}

function quoteSourceMeta(item) {
  const source = String(item?.intraday_quote?.source || "").trim().toLowerCase();
  if (source === "kis") return { label: "KIS", kind: "GOOD" };
  if (source === "naver_fallback") return { label: "NAVER", kind: "WATCH" };
  return { label: "NONE", kind: "ALERT" };
}

function fmtMoneyShort(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  if (n >= 1_0000_0000_0000) return `${(n / 1_0000_0000_0000).toFixed(1)}조`;
  if (n >= 1_0000_0000) return `${(n / 1_0000_0000).toFixed(1)}억`;
  if (n >= 1_0000) return `${(n / 1_0000).toFixed(1)}만`;
  return n.toLocaleString("ko-KR");
}

function renderCandidates(targetId, items) {
  const el = document.getElementById(targetId);
  if (!el) return;
  if (!items || !items.length) {
    el.innerHTML = `<div class="empty-state">오늘 바로 볼 후보가 아직 없습니다.</div>`;
    return;
  }
  el.innerHTML = items.slice(0, 6).map((item) => {
    const quoteSource = quoteSourceMeta(item);
    const reasons = (item.supporting_reasons || item.blocking_reasons || []).slice(0, 3);
    const score = item.live_score ?? item.final_score;
    const changePct = item?.intraday_quote?.change_pct;
    const tradingValue = item?.intraday_quote?.trading_value;
    const shadowRank = Number(item.shadow_quality_risk_guard_rank);
    const shadowDelta = Number(item.shadow_quality_risk_guard_rank_delta);
    const shadowPenalty = Number(item.shadow_quality_risk_guard_penalty);
    const shadowChip = Number.isFinite(shadowDelta) && shadowDelta > 0
      ? `<span class="chip info">shadow +${fmtNum(shadowDelta)}</span>`
      : Number.isFinite(shadowPenalty) && shadowPenalty > 0
      ? `<span class="chip watch">shadow penalty ${fmtNum(shadowPenalty)}</span>`
      : "";
    const shadowLine = Number.isFinite(shadowRank)
      ? `shadow guard rank ${fmtNum(shadowRank)}${Number.isFinite(shadowDelta) && shadowDelta > 0 ? ` · baseline 대비 ${fmtNum(shadowDelta)}계단 개선` : Number.isFinite(shadowPenalty) && shadowPenalty > 0 ? ` · penalty ${fmtNum(shadowPenalty)}` : ""}`
      : "shadow guard 정보 없음";
    return `
      <article class="candidate-card">
        <div class="candidate-top">
          <div>
            <h3 class="candidate-name">${escapeHtml(item.name || item.code || "-")}</h3>
            <div class="candidate-meta">${escapeHtml(item.code || "-")} · ${escapeHtml(item.sector || "-")} · ${escapeHtml(item.dominant_theme || "(none)")}</div>
          </div>
          <div class="candidate-score">${fmtNum(score, 1)}</div>
        </div>
        <div class="chip-row">
          <span class="chip ${chipClass(item.buyability_status)}">${escapeHtml(item.buyability_status || "-")}</span>
          <span class="chip ${chipClass(item.watchlist_tier)}">${escapeHtml(item.watchlist_tier || "-")}</span>
          <span class="chip ${chipClass(item.confidence_state_v2)}">${escapeHtml(item.confidence_state_v2 || "-")}</span>
          <span class="chip ${chipClass(quoteSource.kind)}">시세 ${escapeHtml(quoteSource.label)}</span>
          ${shadowChip}
        </div>
        <div class="state-line">${escapeHtml(shadowLine)}</div>
        <div class="state-line">장중 ${escapeHtml(quoteSource.label)} · 변동률 ${escapeHtml(fmtPct(changePct, 1))} · 거래대금 ${escapeHtml(fmtMoneyShort(tradingValue))}</div>
        <ul class="candidate-list">
          ${reasons.length ? reasons.map((reason) => `<li>${escapeHtml(reason)}</li>`).join("") : "<li>사유 정보 없음</li>"}
        </ul>
      </article>
    `;
  }).join("");
}

function renderIntradayOps(targetId, intraday) {
  const el = document.getElementById(targetId);
  if (!el) return;
  const groups = [
    {
      title: "승격",
      detail: "오후장 우선 검토로 올라온 종목",
      items: intraday?.promoted_to_priority || [],
      empty: "승격 없음",
      kind: "good",
    },
    {
      title: "제외",
      detail: "우선 검토에서 빠진 종목",
      items: intraday?.dropped_from_priority || [],
      empty: "제외 없음",
      kind: "watch",
    },
    {
      title: "장중 재확인 필요",
      detail: "장중 시세 연결이 약해 오후장 판단 전에 재확인이 필요한 종목",
      items: intraday?.missing_quotes || [],
      empty: "재확인 필요 없음",
      kind: "bad",
    },
    {
      title: "경계 강화",
      detail: "장중 이유로 주의가 커진 종목",
      items: intraday?.caution_escalations || [],
      empty: "추가 경계 없음",
      kind: "info",
    },
  ];
  el.innerHTML = groups.map((group) => `
    <article class="candidate-card">
      <div class="candidate-top">
        <div>
          <h3 class="candidate-name">${escapeHtml(group.title)}</h3>
          <div class="candidate-meta">${escapeHtml(group.detail)}</div>
        </div>
        <div class="candidate-score">${fmtNum(group.items.length)}</div>
      </div>
      <div class="chip-row"><span class="chip ${group.kind}">${escapeHtml(group.title)}</span></div>
      ${
        group.items.length
          ? `<ul class="candidate-list">${group.items.slice(0, 4).map((item) => `<li>${escapeHtml(item?.name || item?.code || "-")} ${item?.intraday_reasons?.[0] ? `- ${escapeHtml(item.intraday_reasons[0])}` : ""}</li>`).join("")}</ul>`
          : `<div class="state-line">${escapeHtml(group.empty)}</div>`
      }
    </article>
  `).join("");
}

function renderShadowCandidates(targetId, items) {
  const el = document.getElementById(targetId);
  if (!el) return;
  if (!items || !items.length) {
    el.innerHTML = `<div class="empty-state">지금은 baseline 대비 선명하게 개선되는 shadow 후보가 없습니다.</div>`;
    return;
  }
  el.innerHTML = items.slice(0, 5).map((item) => {
    const score = item.live_score ?? item.final_score;
    const shadowScore = Number(item.shadow_quality_risk_guard_score);
    const shadowRank = Number(item.shadow_quality_risk_guard_rank);
    const shadowDelta = Number(item.shadow_quality_risk_guard_rank_delta);
    const shadowPenalty = Number(item.shadow_quality_risk_guard_penalty);
    return `
      <article class="candidate-card">
        <div class="candidate-top">
          <div>
            <h3 class="candidate-name">${escapeHtml(item.name || item.code || "-")}</h3>
            <div class="candidate-meta">${escapeHtml(item.code || "-")} · ${escapeHtml(item.sector || "-")} · ${escapeHtml(item.dominant_theme || "(none)")}</div>
          </div>
          <div class="candidate-score">${fmtNum(score, 1)}</div>
        </div>
        <div class="chip-row">
          <span class="chip info">baseline rank ${fmtNum(item.live_rank)}</span>
          <span class="chip good">shadow rank ${fmtNum(shadowRank)}</span>
          <span class="chip good">+${fmtNum(shadowDelta)}</span>
          ${Number.isFinite(shadowPenalty) && shadowPenalty > 0 ? `<span class="chip watch">penalty ${fmtNum(shadowPenalty)}</span>` : ""}
        </div>
        <div class="state-line">
          quality/risk guard 적용 후 live_score ${fmtNum(score, 1)} 기준 후보가 shadow score ${Number.isFinite(shadowScore) ? fmtNum(shadowScore, 1) : "-"}로 재정렬됐습니다.
        </div>
        <ul class="candidate-list">
          <li>baseline rank ${fmtNum(item.live_rank)} -> shadow rank ${fmtNum(shadowRank)}</li>
          <li>confidence ${fmtNum(item.confidence_score, 1)} · qual ${fmtNum(item.qual_score, 1)} · risk penalty ${fmtNum(item.risk_penalty, 1)}</li>
        </ul>
      </article>
    `;
  }).join("");
}

function renderShadowRepeatability(shadow) {
  const repeatability = shadow?.repeatability || {};
  const summary = repeatability.summary || {};
  const judgment = String(summary.judgment || "insufficient_history");
  const usableDates = Array.isArray(repeatability.usable_dates) ? repeatability.usable_dates : [];
  const topRepeaters = Array.isArray(repeatability.top_repeaters) ? repeatability.top_repeaters : [];

  renderChipRow("shadowRepeatabilityChips", [
    {
      label: judgment,
      kind: judgment === "emerging_repeatability" ? "GOOD" : judgment === "no_repeaters_yet" ? "WATCH" : "ALERT",
    },
    { label: `사용 가능 ${fmtNum(summary.usable_snapshot_count)}`, kind: "info" },
    { label: `repeaters ${fmtNum(summary.repeated_candidate_count)}`, kind: "info" },
  ]);
  renderKv("shadowRepeatabilityKv", [
    ["latest asof", summary.latest_asof_date || "-"],
    ["전체 스냅샷", fmtNum(summary.total_snapshot_count)],
    ["사용 가능 스냅샷", fmtNum(summary.usable_snapshot_count)],
    ["repeated candidates", fmtNum(summary.repeated_candidate_count)],
    ["usable dates", usableDates.length ? usableDates.join(", ") : "-"],
  ]);

  let helpText = "shadow 반복 개선 후보가 며칠 연속 나오는지 확인합니다.";
  if (judgment === "insufficient_history") {
    helpText = "아직 archived ranking snapshot의 shadow 컬럼 이력이 부족해 반복성 해석을 보류합니다.";
  } else if (judgment === "no_repeaters_yet") {
    helpText = "usable snapshot은 쌓였지만 같은 종목이 반복적으로 개선되는 패턴은 아직 희박합니다.";
  } else if (judgment === "emerging_repeatability") {
    helpText = "같은 shadow 개선 후보가 여러 날짜에 반복 등장해 승격 판단의 보조 근거로 볼 수 있습니다.";
  }
  renderText("shadowRepeatabilityHelp", helpText);

  if (!topRepeaters.length) {
    renderList(
      "shadowRepeatabilityList",
      judgment === "insufficient_history"
        ? ["기존 archived snapshot에 shadow_quality_risk_guard_* 컬럼이 없어 usable snapshot이 아직 없습니다."]
        : ["아직 2일 이상 반복 등장한 shadow 개선 후보가 없습니다."],
      "반복성 정보가 없습니다."
    );
    return;
  }

  renderList(
    "shadowRepeatabilityList",
    topRepeaters.map((item) => {
      const appearance = fmtNum(item.appearance_days);
      const streak = fmtNum(item.consecutive_recent_days);
      const avgDelta = fmtNum(item.avg_rank_delta, 1);
      const latestDelta = fmtNum(item.latest_rank_delta);
      return `${item.name || item.code || "-"} (${item.code || "-"}) - ${appearance}일 등장, 최근 ${streak}일 연속, 평균 개선 ${avgDelta}, 최신 +${latestDelta}`;
    }),
    "반복성 정보가 없습니다."
  );
}

function renderTradingPolicy(policy) {
  if (!window.TradingPolicyUI) return;
  if (!policy) {
    window.TradingPolicyUI.renderStrip("policyStrip", []);
    window.TradingPolicyUI.renderRuleSection("opsPolicyRules", {
      title: "운영 정책 요약",
      note: "정책 정보를 불러오지 못해 운영 대시보드 본문만 표시합니다.",
      items: [],
    });
    return;
  }

  const opsRules = []
    .concat(policy.page_rules?.manual || [])
    .concat(policy.page_rules?.portfolio || []);

  window.TradingPolicyUI.renderStrip("policyStrip", policy.banner || []);
  window.TradingPolicyUI.renderRuleSection("opsPolicyRules", {
    title: "운영 정책 요약",
    note: "오늘 매수 가능 구간과 포트폴리오 한도를 운영자 화면에서 바로 확인합니다.",
    items: opsRules,
  });
}

function renderTrendChart(targetId, rows) {
  const el = document.getElementById(targetId);
  if (!el) return;
  if (!rows || !rows.length) {
    el.innerHTML = "";
    return;
  }
  const width = 640;
  const height = 220;
  const pad = { top: 18, right: 18, bottom: 34, left: 30 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const maxA = Math.max(...rows.map((r) => Number(r.snapshot_index) || 0), 1);
  const maxB = Math.max(...rows.map((r) => Number(r.matured_60d_cumulative) || 0), 1);
  const x = (idx) => pad.left + (rows.length === 1 ? plotW / 2 : (plotW * idx) / (rows.length - 1));
  const yA = (v) => pad.top + plotH - (plotH * (Number(v) || 0)) / maxA;
  const yB = (v) => pad.top + plotH - (plotH * (Number(v) || 0)) / maxB;
  const pathA = rows.map((r, idx) => `${idx === 0 ? "M" : "L"} ${x(idx).toFixed(1)} ${yA(r.snapshot_index).toFixed(1)}`).join(" ");
  const pathB = rows.map((r, idx) => `${idx === 0 ? "M" : "L"} ${x(idx).toFixed(1)} ${yB(r.matured_60d_cumulative).toFixed(1)}`).join(" ");
  const xLabels = rows.map((r, idx) => {
    if (!(idx === 0 || idx === rows.length - 1 || idx === Math.floor(rows.length / 2))) return "";
    return `<text x="${x(idx)}" y="${height - 10}" text-anchor="middle" class="chart-label">${String(r.as_of_date || "").slice(5)}</text>`;
  }).join("");
  const dotsA = rows.map((r, idx) => `<circle cx="${x(idx)}" cy="${yA(r.snapshot_index)}" r="3.5" class="chart-dot-a"></circle>`).join("");
  const dotsB = rows.map((r, idx) => `<circle cx="${x(idx)}" cy="${yB(r.matured_60d_cumulative)}" r="3.5" class="chart-dot-b"></circle>`).join("");
  el.innerHTML = `
    <rect x="${pad.left}" y="${pad.top}" width="${plotW}" height="${plotH}" rx="12" class="chart-bg"></rect>
    <line x1="${pad.left}" y1="${pad.top + plotH}" x2="${pad.left + plotW}" y2="${pad.top + plotH}" class="chart-axis"></line>
    <path d="${pathA}" class="chart-line-a"></path>
    <path d="${pathB}" class="chart-line-b"></path>
    ${dotsA}
    ${dotsB}
    <text x="${pad.left}" y="12" class="chart-label">스냅샷 지수</text>
    <text x="${pad.left + 140}" y="12" class="chart-label">matured 60d cumulative</text>
    ${xLabels}
  `;
}

function renderSeriesTrendChart(targetId, rows, series) {
  const el = document.getElementById(targetId);
  if (!el) return;
  if (!rows || !rows.length || !series || !series.length) {
    el.innerHTML = "";
    return;
  }
  const activeSeries = series.filter((item) => rows.some((row) => Number.isFinite(Number(row?.[item.key]))));
  if (!activeSeries.length) {
    el.innerHTML = "";
    return;
  }
  const width = 640;
  const height = 220;
  const pad = { top: 18, right: 18, bottom: 34, left: 30 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const x = (idx) => pad.left + (rows.length === 1 ? plotW / 2 : (plotW * idx) / (rows.length - 1));
  const paths = activeSeries.map((item) => {
    const maxValue = Math.max(...rows.map((row) => Number(row?.[item.key]) || 0), 1);
    const y = (value) => pad.top + plotH - (plotH * (Number(value) || 0)) / maxValue;
    const path = rows.map((row, idx) => `${idx === 0 ? "M" : "L"} ${x(idx).toFixed(1)} ${y(row?.[item.key]).toFixed(1)}`).join(" ");
    const dots = rows.map((row, idx) => `<circle cx="${x(idx)}" cy="${y(row?.[item.key]).toFixed(1)}" r="3.5" fill="${item.color}"></circle>`).join("");
    return { item, path, dots };
  });
  const xLabels = rows.map((row, idx) => {
    if (!(idx === 0 || idx === rows.length - 1 || idx === Math.floor(rows.length / 2))) return "";
    return `<text x="${x(idx)}" y="${height - 10}" text-anchor="middle" class="chart-label">${String(row.as_of_date || "").slice(5)}</text>`;
  }).join("");
  const legends = paths.map((entry, idx) => `<text x="${pad.left + idx * 180}" y="12" class="chart-label">${entry.item.label}</text>`).join("");
  el.innerHTML = `
    <rect x="${pad.left}" y="${pad.top}" width="${plotW}" height="${plotH}" rx="12" class="chart-bg"></rect>
    <line x1="${pad.left}" y1="${pad.top + plotH}" x2="${pad.left + plotW}" y2="${pad.top + plotH}" class="chart-axis"></line>
    ${paths.map((entry) => `<path d="${entry.path}" fill="none" stroke="${entry.item.color}" stroke-width="3"></path>`).join("")}
    ${paths.map((entry) => entry.dots).join("")}
    ${legends}
    ${xLabels}
  `;
}

function renderHistoryBadges(targetId, rows, options) {
  const labelKey = options?.labelKey || "overall_status";
  const detailKey = options?.detailKey || null;
  renderChipRow(targetId, (rows || []).map((row) => ({
    label: `${String(row.as_of_date || "").slice(5)} ${row?.[labelKey] || "-"}${detailKey && row?.[detailKey] !== undefined && row?.[detailKey] !== null ? ` ${row[detailKey]}` : ""}`,
    kind: row?.[labelKey] || "info",
  })));
}

function renderTransitionChecklist(items) {
  const total = Array.isArray(items) ? items.length : 0;
  const passed = Array.isArray(items) ? items.filter((item) => item.passed).length : 0;
  renderChipRow("transitionSummary", [
    { label: `PASS ${passed}/${total}`, kind: passed === total ? "GOOD" : passed >= Math.ceil(total / 2) ? "WATCH" : "ALERT" },
  ]);
  const el = document.getElementById("transitionChecklist");
  if (!el) return;
  if (!items || !items.length) {
    el.innerHTML = "<li>운영 전환 체크 정보가 없습니다.</li>";
    return;
  }
  el.innerHTML = items.map((item) => {
    const mark = item.passed ? "통과" : "대기";
    return `<li><strong>${escapeHtml(mark)}</strong> · ${escapeHtml(item.label || "-")} · ${escapeHtml(item.detail || "")}</li>`;
  }).join("");
}

function renderRecentStateBadges(items) {
  renderChipRow("recentStateBadges", (items || []).map((item) => ({
    label: `${item.label} ${item.detail}`,
    kind: item.kind || "info",
  })));
}

function renderVisitorAnalytics(analytics) {
  const summary = analytics || {};
  renderChipRow("visitorChipRow", [
    { label: `오늘 PV ${fmtNum(summary.today_pageviews)}`, kind: "INFO" },
    { label: `오늘 UV ${fmtNum(summary.today_unique_visitors)}`, kind: "INFO" },
    { label: summary.available === false ? "tracking unavailable" : "tracking active", kind: summary.available === false ? "ALERT" : "GOOD" },
  ]);
  renderKv("visitorKv", [
    ["today pageviews", fmtNum(summary.today_pageviews)],
    ["today unique visitors", fmtNum(summary.today_unique_visitors)],
    ["last 7d pageviews", fmtNum(summary.last_7d_pageviews)],
    ["last 7d unique visitors", fmtNum(summary.last_7d_unique_visitors)],
  ]);
  renderList(
    "visitorTopPages",
    (summary.top_pages_7d || []).map((item) => `${item.path || "/"} - PV ${fmtNum(item.pageviews)} / UV ${fmtNum(item.unique_visitors)}`),
    "최근 7일 상위 페이지가 없습니다."
  );
  renderSeriesTrendChart("visitorTrendChart", summary.trend_7d || [], [
    { key: "pageviews", label: "pageviews", color: "#38bdf8" },
    { key: "unique_visitors", label: "unique visitors", color: "#22c55e" },
  ]);
  renderHistoryBadges("visitorTrendBadges", summary.trend_7d || [], {
    labelKey: "pageviews",
    detailKey: "unique_visitors",
  });
}

function renderOperatorMemo(notes) {
  const textarea = document.getElementById("operatorMemo");
  const meta = document.getElementById("operatorMemoMeta");
  if (textarea) textarea.value = notes?.operator_memo || "";
  if (meta) {
    const updatedAt = notes?.last_updated_at || null;
    const updatedBy = notes?.last_updated_by || null;
    meta.textContent = updatedAt ? `마지막 저장 ${updatedAt}${updatedBy ? ` · ${updatedBy}` : ""}` : "아직 저장한 메모가 없습니다.";
  }
}

async function saveOperatorMemo() {
  const textarea = document.getElementById("operatorMemo");
  const state = document.getElementById("pageState");
  if (!textarea) return;
  state.textContent = "운영 메모를 저장하는 중입니다.";
  try {
    const payload = await postJson("/api/ops-readiness/notes", {
      operator_memo: textarea.value || "",
      updated_by: "web_operator",
    });
    renderOperatorMemo(payload.notes || {});
    state.textContent = "운영 메모를 저장했습니다.";
  } catch (error) {
    console.error(error);
    state.textContent = `메모 저장 실패: ${error.message}`;
  }
}

async function loadOpsReadiness() {
  const state = document.getElementById("pageState");
  state.textContent = "운영 readiness 대시보드를 불러오는 중입니다.";
  try {
    const [data, policy] = await Promise.all([
      fetchJson("/api/ops-readiness"),
      fetchTradingPolicySafe(),
    ]);
    renderTradingPolicy(policy);
    renderHero(data);

    const outputs = data.outputs || {};
    const latestRefDate = outputs.ranking_latest_date || data.asof_date || null;
    const intraday = data.manual?.intraday_summary || {};
    const visitorAnalytics = data.visitor_analytics || {};

    const interpretation = data.interpretation || {};
    const cardHelp = interpretation.cards || {};
    const basis = data.execution_basis || {};
    renderChipRow("interpretationChips", [
      { label: basis.label || "기준 미상", kind: basis.current_basis === "intraday" ? "WATCH" : "GOOD" },
      { label: data.go_no_go?.decision || "WAIT", kind: data.go_no_go?.decision || "WAIT" },
      { label: data.gate?.overall_status || "-", kind: data.gate?.overall_status || "-" },
      { label: data.kpi?.overall_status || "-", kind: data.kpi?.overall_status || "-" },
    ]);
    renderKv("interpretationKv", [
      ["판단 기준", basis.description || "-"],
      ["오늘 결론", interpretation.summary_decision || "-"],
      ["왜 이런가", interpretation.summary_reason || "-"],
      ["오늘 행동", interpretation.action_guide || "-"],
    ]);

    renderList("criticalReasons", interpretation.critical_reasons || [], "추가로 확인할 핵심 사유가 없습니다.");

    const readiness = data.readiness || {};
    renderKv("readinessKv", [
      ["스냅샷 수", fmtNum(readiness.snapshot_count)],
      ["20일 성숙", fmtNum(readiness.matured_snapshot_count_20d)],
      ["60일 성숙", fmtNum(readiness.matured_snapshot_count_60d)],
      ["90일 성숙", fmtNum(readiness.matured_snapshot_count_90d)],
      ["가장 오래된 스냅샷", readiness.oldest_snapshot_date || "-"],
      ["최신 스냅샷", readiness.latest_snapshot_date || "-"],
      ["calibration 60d", readiness.confidence_calibration_readiness_60d || "-"],
    ]);
    renderText("readinessHelp", cardHelp.readiness || "");

    const go = data.go_no_go || {};
    renderChipRow("goChipRow", [{ label: go.decision || "WAIT", kind: go.decision || "WAIT" }]);
    renderList("goReasons", go.reasons || [], "현재 운영 전환 판단 사유가 정리되지 않았습니다.");
    renderText("transitionHelp", cardHelp.transition || "");

    const gate = data.gate || {};
    renderChipRow("gateChipRow", [
      { label: gate.overall_status || "-", kind: gate.overall_status || "-" },
      { label: gate.walkforward_acceptance || "-", kind: gate.walkforward_acceptance || "-" },
    ]);
    renderKv("gateKv", [
      ["asof date", markStaleDate(gate.asof_date || data.asof_date || "-", latestRefDate)],
      ["matured benchmark dates", fmtNum(gate.matured_benchmark_dates)],
      ["trusted ratio top20", fmtPct(gate.trusted_ratio_top20)],
      ["buy_now", fmtNum(gate.buy_now_count)],
      ["watchlist", fmtNum(gate.watchlist_count)],
      ["blocked", fmtNum(gate.blocked_count)],
      ["paper_only", fmtNum(gate.paper_only_count)],
    ]);
    renderText("gateHelp", cardHelp.gate || "");

    const marketRegime = gate.market_regime || {};
    const marketRegimeInterpretation = gate.market_regime_interpretation || {};
    renderChipRow("marketRegimeChipRow", [
      { label: marketRegime.regime || "-", kind: marketRegime.regime === "bull" ? "GOOD" : marketRegime.regime === "defensive" ? "ALERT" : "WATCH" },
      { label: `true ${fmtNum(marketRegime.true_count)}/5`, kind: "info" },
      { label: marketRegimeInterpretation.tone || "INFO", kind: marketRegimeInterpretation.tone || "INFO" },
    ]);
    renderKv("marketRegimeKv", [
      ["latest date", markStaleDate(marketRegime.latest_date || "-", latestRefDate)],
      ["breadth 20d", fmtNum(marketRegime.breadth_20d, 3)],
      ["recent 20d return", fmtPct(marketRegime.recent_20d_return, 1)],
      ["volatility 5d", fmtNum(marketRegime.volatility_5d, 4)],
      ["close > ma20", fmtBool(marketRegime.close_gt_ma20)],
      ["ma20 > ma60", fmtBool(marketRegime.ma20_gt_ma60)],
      ["recent return > 3%", fmtBool(marketRegime.recent_20d_return_gt_0_03 ?? marketRegime["recent_20d_return_gt_0.03"])],
      ["breadth > 0.55", fmtBool(marketRegime.breadth_20d_gt_0_55 ?? marketRegime["breadth_20d_gt_0.55"])],
      ["volatility risk", fmtBool(marketRegime.volatility_risk_flag)],
    ]);
    renderText(
      "marketRegimeHelp",
      marketRegimeInterpretation.summary ||
      (Array.isArray(marketRegime.diagnosis) && marketRegime.diagnosis.length
        ? marketRegime.diagnosis.join(" ")
        : (marketRegime.reason || ""))
    );
    renderText("marketRegimeStance", marketRegimeInterpretation.stance || "");
    renderList(
      "marketRegimeActions",
      marketRegimeInterpretation.action_items || [],
      "지금은 추가로 볼 시장 국면 체크 포인트가 없습니다."
    );

    const kpi = data.kpi || {};
    renderChipRow("kpiChipRow", [{ label: kpi.overall_status || "-", kind: kpi.overall_status || "-" }]);
    renderKv("kpiKv", [
      ["latest date", markStaleDate(kpi.latest_date || "-", latestRefDate)],
      ["score formula", kpi.score_formula_version || "-"],
      ["top20 mean confidence", fmtNum(kpi.top20_mean_confidence_score, 2)],
      ["walkforward top20 avg return 60d", fmtNum(kpi.walkforward_top20_avg_return_60d, 4)],
      ["alert metrics", fmtNum(kpi.alert_metric_count)],
      ["watch metrics", fmtNum(kpi.watch_metric_count)],
    ]);
    renderText("kpiHelp", cardHelp.kpi || "");

    renderKv("outputsKv", [
      ["execution basis", basis.label || "-"],
      ["최근 산출물 refresh", basis.last_artifact_refresh_at || basis.last_refresh_at || "-"],
      ["최근 자동 스케줄 성공", basis.last_auto_success_at || "-"],
      ["ops asof", markStaleDate(data.asof_date || "-", latestRefDate)],
      ["ranking latest date", outputs.ranking_latest_date || "-"],
      ["gate asof", markStaleDate(outputs.gate_asof_date || "-", latestRefDate)],
      ["ranking row count", fmtNum(outputs.ranking_row_count)],
      ["daily recommendations date", markStaleDate(outputs.daily_recommendations_date || "-", latestRefDate)],
      ["우선 검토 후보", fmtNum(outputs.priority_candidate_count)],
      ["보수 검토 후보", fmtNum(outputs.caution_candidate_count)],
      ["intraday quotes", `${fmtNum(intraday.quote_success_count)}/${fmtNum((intraday.quote_success_count || 0) + (intraday.quote_failure_count || 0))}`],
      ["우선 검토 승격", fmtNum(intraday.promoted_to_priority_count)],
      ["우선 검토 제외", fmtNum(intraday.dropped_from_priority_count)],
    ]);

    const scheduler = data.scheduler || {};
    const schedulers = data.schedulers || {};
    renderChipRow("schedulerChipRow", [
      { label: scheduler.status || "-", kind: scheduler.status || "info" },
      { label: scheduler.current_label || basis.label || "-", kind: scheduler.current_role === "intraday" ? "WATCH" : "GOOD" },
      { label: scheduler.expected_daily_time || scheduler.configured_daily_time || "-", kind: "info" },
    ]);
    renderKv("schedulerKv", [
      ["실행 모드", scheduler.mode || "-"],
      ["시간대", scheduler.timezone || "-"],
      ["현재 기준", scheduler.current_label || basis.label || "-"],
      ["권장 마감 기준", schedulers.primary?.expected_daily_time || "-"],
      ["권장 장중 기준", schedulers.intraday?.expected_daily_time || "-"],
      ["배포된 마감 기준", schedulers.primary?.configured_daily_time || "-"],
      ["배포된 장중 기준", schedulers.intraday?.configured_daily_time || "-"],
      ["catch-up 생략", scheduler.skip_catchup_on_start === null || scheduler.skip_catchup_on_start === undefined ? "-" : (scheduler.skip_catchup_on_start ? "예" : "아니오")],
      ["최근 배포 반영", scheduler.last_success_at || "-"],
      ["최근 마감 반영", schedulers.primary?.last_success_at || "-"],
      ["최근 장중 반영", schedulers.intraday?.last_success_at || "-"],
      ["최근 실패", scheduler.last_failure_at || "-"],
      ["메모", scheduler.status_note || "-"],
    ]);
    if (Array.isArray(scheduler.config_warnings) && scheduler.config_warnings.length) {
      const current = document.getElementById("schedulerKv")?.innerHTML || "";
      document.getElementById("schedulerKv").innerHTML = `${current}
        <div class="kv-row">
          <span class="muted">config warning</span>
          <strong>${escapeHtml(scheduler.config_warnings[0])}</strong>
        </div>
      `;
    }

    renderMetricList("alertMetrics", (kpi.alert_metrics || []).concat(kpi.watch_metrics || []), "현재 표시할 경고/관찰 지표가 없습니다.");
    renderVisitorAnalytics(visitorAnalytics);
    renderList("dailyChecklist", [
      ...(intraday?.headline ? [`장중 변화: ${intraday.headline}`] : []),
      ...(data.manual?.checklist || []),
    ], "오늘 확인할 체크리스트가 없습니다.");
    renderIntradayOps("intradayOpsGrid", intraday);
    renderTrendChart("readinessTrendChart", data.trends?.readiness || []);
    renderRecentStateBadges(data.trends?.recent_state_badges || []);
    renderSeriesTrendChart("gateTrendChart", data.trends?.gate || [], [
      { key: "trusted_ratio_top20", label: "trusted ratio", color: "#22c55e" },
      { key: "matured_benchmark_dates", label: "matured dates", color: "#38bdf8" },
      { key: "watchlist_count", label: "watchlist", color: "#f59e0b" },
    ]);
    renderHistoryBadges("gateTrendBadges", data.trends?.gate || [], {
      labelKey: "overall_status",
      detailKey: "daily_cycle_status",
    });
    renderSeriesTrendChart("kpiTrendChart", data.trends?.kpi || [], [
      { key: "top20_mean_confidence_score", label: "mean confidence", color: "#38bdf8" },
      { key: "walkforward_top20_avg_return_60d", label: "wf avg ret 60d", color: "#22c55e" },
      { key: "alert_metric_count", label: "alert metrics", color: "#ef4444" },
    ]);
    renderHistoryBadges("kpiTrendBadges", data.trends?.kpi || [], {
      labelKey: "overall_status",
      detailKey: "alert_metric_count",
    });
    renderTransitionChecklist(data.transition_checklist || []);
    renderOperatorMemo(data.notes || {});
    renderCandidates("candidateGrid", data.manual?.priority_candidates || []);
    renderShadowCandidates("shadowCandidateGrid", data.shadow?.quality_risk_guard_candidates || []);
    renderShadowRepeatability(data.shadow || {});

    const staleNotes = [];
    if (isIsoDate(outputs.gate_asof_date) && isIsoDate(latestRefDate) && outputs.gate_asof_date < latestRefDate) staleNotes.push(`gate ${outputs.gate_asof_date}`);
    if (isIsoDate(outputs.daily_recommendations_date) && isIsoDate(latestRefDate) && outputs.daily_recommendations_date < latestRefDate) staleNotes.push(`daily ${outputs.daily_recommendations_date}`);
    if (isIsoDate(kpi.latest_date) && isIsoDate(latestRefDate) && kpi.latest_date < latestRefDate) staleNotes.push(`kpi ${kpi.latest_date}`);
    if (isIsoDate(marketRegime.latest_date) && isIsoDate(latestRefDate) && marketRegime.latest_date < latestRefDate) staleNotes.push(`market ${marketRegime.latest_date}`);
    state.textContent = `기준일 ${data.asof_date || "-"} · ${basis.label || "기준 미상"} 기준 운영 readiness 대시보드를 불러왔습니다.${staleNotes.length ? ` stale sources: ${staleNotes.join(", ")}` : ""}`;
  } catch (error) {
    console.error(error);
    renderTradingPolicy(null);
    document.getElementById("heroGrid").innerHTML = '<div class="empty-state">운영 대시보드를 불러오지 못했습니다.</div>';
    renderList("goReasons", [], "조회 실패");
    renderMetricList("alertMetrics", [], "조회 실패");
    renderList("dailyChecklist", [], "조회 실패");
    renderCandidates("candidateGrid", []);
    renderIntradayOps("intradayOpsGrid", {});
    renderShadowCandidates("shadowCandidateGrid", []);
    renderShadowRepeatability({});
    state.textContent = `조회 실패: ${error.message}`;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const pageActions = document.querySelector(".page-actions");
  if (pageActions && !pageActions.querySelector('[data-link="score-check"]')) {
    const scoreCheckBtn = document.createElement("button");
    scoreCheckBtn.type = "button";
    scoreCheckBtn.dataset.link = "score-check";
    scoreCheckBtn.textContent = "점수 검증";
    scoreCheckBtn.addEventListener("click", () => {
      window.location.href = "/score-check";
    });
    pageActions.insertBefore(scoreCheckBtn, pageActions.children[4] || null);
  }
  document.getElementById("saveOperatorMemoBtn")?.addEventListener("click", () => {
    saveOperatorMemo().catch((error) => console.error(error));
  });
  loadOpsReadiness().catch((error) => console.error(error));
});







