async function fetchRanking(dateValue, marketValue = "ALL", sectorValue = "ALL") {
  const params = new URLSearchParams();
  if (dateValue) params.set("date", dateValue);
  if (marketValue && marketValue !== "ALL") params.set("market", marketValue);
  if (sectorValue && sectorValue !== "ALL") params.set("sector", sectorValue);
  const res = await fetch("/api/ranking?" + params.toString());
  if (!res.ok) throw new Error("ranking API error");
  return res.json();
}

async function fetchTrendSafe(dateValue) {
  try {
    const params = new URLSearchParams();
    if (dateValue) params.set("date", dateValue);
    const res = await fetch("/api/ranking-trend?" + params.toString());
    if (!res.ok) throw new Error("trend API error");
    return await res.json();
  } catch (error) {
    console.warn("trend unavailable", error);
    return null;
  }
}

async function fetchManualSummarySafe() {
  try {
    const res = await fetch("/api/manual-trading/summary");
    if (!res.ok) throw new Error("manual summary API error");
    return await res.json();
  } catch (error) {
    console.warn("manual summary unavailable", error);
    return null;
  }
}


const EXPLAIN_LABELS = {
  high_ret_score: "기대수익 강점",
  high_probability_score: "확률 강점",
  high_tech_score: "기술 강점",
  strong_quality_profile: "퀄리티 강점",
  strong_safety_profile: "안정성 강점",
  healthy_liquidity_profile: "유동성 강점",
  elevated_risk_penalty: "리스크 패널티 높음",
  very_high_risk_penalty: "리스크 패널티 매우 높음",
  weak_quality_score: "퀄리티 약점",
  weak_tech_score: "기술 약점",
  weak_safety_score: "안정성 약점",
  weak_ret_score: "기대수익 약점",
  low_probability_score: "확률 약점",
  low_liquidity_score: "유동성 약점",
  low_confidence: "신뢰도 낮음",
  multiple_component_fallbacks: "대체값 사용 다수",
  partial_quality_coverage: "퀄리티 데이터 커버리지 부족",
};

const state = {
  rows: [],
  filteredRows: [],
  manualSummary: null,
  showOverlay: false,
  trendData: null,
  trendMap: new Map(),
};

let manualCandidateMeta = new Map();
let manualPriorityMap = new Map();
let manualCautionMap = new Map();

function toNum(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

function readOptionalNumber(id) {
  const raw = String(document.getElementById(id)?.value ?? "").trim();
  if (!raw) return null;
  return toNum(raw);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function fmtScore(value) {
  const num = toNum(value);
  return num === null ? "-" : num.toFixed(1);
}

function fmtPct(value, digits = 1) {
  const num = toNum(value);
  return num === null ? "-" : `${(num * 100).toFixed(digits)}%`;
}

function fmtPrice(value) {
  const num = toNum(value);
  return num === null ? "-" : num.toLocaleString("ko-KR");
}

function normalizeSentence(text, fallback) {
  const value = String(text || "").trim();
  return value || fallback;
}

function countValues(values) {
  return values.filter(Boolean).reduce((acc, key) => {
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});
}

function explainLabel(code) {
  if (!code) return "";
  return EXPLAIN_LABELS[code] || code;
}

function buyEligibilityLabel(status) {
  if (status === "BUY_ALLOWED") return "매수 가능";
  if (status === "WATCH") return "관찰";
  if (status === "BLOCK") return "제외";
  return null;
}

function getBuyEligibilityMeta(code) {
  return manualCandidateMeta.get(String(code || "").trim()) || null;
}

function getManualOverlay(code) {
  const key = String(code || "").trim();
  return manualPriorityMap.get(key) || manualCautionMap.get(key) || null;
}

function buildStatusMeta(code) {
  const meta = getBuyEligibilityMeta(code);
  if (!meta || !meta.status) {
    return { label: "", cls: "", title: "" };
  }
  const label = buyEligibilityLabel(meta.status) || meta.status;
  const cls = meta.status === "BUY_ALLOWED" ? "good" : meta.status === "WATCH" ? "warn" : "bad";
  const reasonText = Array.isArray(meta.reasons) && meta.reasons.length ? meta.reasons.slice(0, 3).join(" / ") : "";
  return { label, cls, title: reasonText };
}

function buildOverlayMeta(code) {
  const item = getManualOverlay(code);
  if (!item || !item.intraday_verdict) return null;
  const verdict = String(item.intraday_verdict || "").toUpperCase();
  if (verdict === "PRIORITY") {
    return { label: "수동매매 우선", cls: "info" };
  }
  if (verdict === "CAUTION") {
    return { label: "수동매매 보수", cls: "warn" };
  }
  return { label: `수동매매 ${verdict}`, cls: "neutral" };
}

function buildQualityRiskGuardText(row) {
  const shadowRank = toNum(row?.shadow_quality_risk_guard_rank);
  const shadowPenalty = toNum(row?.shadow_quality_risk_guard_penalty);
  const delta = toNum(row?.shadow_quality_risk_guard_rank_delta);
  if (shadowRank === null) return "quality/risk guard shadow 정보가 없습니다.";
  if (delta !== null && delta > 0) {
    return `shadow guard 적용 후 rank ${fmtScore(shadowRank)}로 ${fmtScore(delta)}단계 개선됐습니다.`;
  }
  if (delta !== null && delta < 0) {
    return `shadow guard 적용 후 rank ${fmtScore(shadowRank)}로 ${fmtScore(Math.abs(delta))}단계 밀렸습니다.`;
  }
  if (shadowPenalty !== null && shadowPenalty > 0) {
    return `shadow guard penalty ${fmtScore(shadowPenalty)}가 걸렸지만 현재 rank 변화는 없습니다.`;
  }
  return "shadow guard 기준으로도 현재 순위가 유지됩니다.";
}

function buildThesis(row) {
  const strengths = normalizeSentence(
    row.score_explain_strengths,
    row.score_explain_summary || "설명 가능한 강점 정보가 없습니다."
  );
  return strengths
    .split("/")
    .map((part) => part.trim())
    .filter(Boolean)
    .slice(0, 3)
    .join(" / ");
}

function buildWhyNow(row) {
  const drivers = [row.top_driver_1, row.top_driver_2, row.top_driver_3]
    .filter(Boolean)
    .map(explainLabel);
  if (drivers.length) return `${drivers.slice(0, 2).join(", ")} 신호가 동시에 보입니다.`;
  return normalizeSentence(row.score_explain_summary, "왜 지금 봐야 하는지에 대한 별도 설명은 없습니다.");
}

function buildRiskText(row) {
  const risks = [row.risk_factor_1, row.risk_factor_2].filter(Boolean).map(explainLabel);
  if (risks.length) return risks.join(" / ");
  return normalizeSentence(row.score_explain_risks, "주요 리스크 설명이 없습니다.");
}

function buildScoreDiagnostic(row) {
  const confidence = toNum(row.confidence_score);
  const penalty = toNum(row.risk_penalty);
  const flowScore = toNum(row.flow_score);
  const parts = [];
  if (confidence !== null) {
    if (confidence >= 85 && (penalty === null || penalty < 6)) parts.push("신뢰도·리스크 균형 양호");
    else if (confidence >= 75) parts.push("신뢰도 보통");
    else parts.push("신뢰도 낮음");
  }
  if (flowScore !== null) {
    if (flowScore >= 70) parts.push("수급 강세");
    else if (flowScore < 35) parts.push("수급 약세");
    else parts.push("수급 중립");
  }
  return parts.join(" · ") || normalizeSentence(row.action_note, "진단 정보 없음");
}

function calcRewardRiskRatio(row) {
  const ret = toNum(row.pred_return_60d);
  const mdd = Math.abs(toNum(row.pred_mdd_60d) || 0);
  if (ret === null || !mdd) return null;
  return ret / mdd;
}

function classifyBias(rows) {
  const top = rows.slice(0, 20);
  const driverCounts = countValues(top.flatMap((row) => [row.score_driver_1, row.score_driver_2, row.score_driver_3]));
  const trendCount = (driverCounts.high_ret_score || 0) + (driverCounts.high_probability_score || 0) + (driverCounts.high_tech_score || 0);
  const defensiveCount = (driverCounts.strong_safety_profile || 0) + (driverCounts.strong_quality_profile || 0) + (driverCounts.healthy_liquidity_profile || 0);
  if (trendCount >= defensiveCount + 4) return "공격 신호 우세";
  if (defensiveCount >= trendCount + 4) return "방어 신호 우세";
  return "균형";
}

function getRegimeCopy(row) {
  const regime = String(row?.regime || "").toLowerCase();
  if (regime === "bull") {
    return {
      label: "상승장",
      detail: "수익과 확률, 기술 점수 비중이 높게 해석되는 구간입니다.",
      cls: "good",
    };
  }
  if (regime === "neutral") {
    return {
      label: "중립장",
      detail: "수익, 확률, 기술, 안정성을 균형 있게 보는 구간입니다.",
      cls: "info",
    };
  }
  return {
    label: "방어장",
    detail: "안정성과 방어를 우선하며, 공격 신호는 더 엄격하게 해석됩니다.",
    cls: "warn",
  };
}

function statusCodeForRow(row) {
  return getBuyEligibilityMeta(row.code)?.status || "NO_STATUS";
}

function getSearchText(row) {
  return [row.name, row.code, row.sector, row.market, row.dominant_theme]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

function compareRows(a, b, sortValue) {
  const byNumDesc = (key) => (toNum(b[key]) ?? -Infinity) - (toNum(a[key]) ?? -Infinity);
  const byNumAsc = (key) => (toNum(a[key]) ?? Infinity) - (toNum(b[key]) ?? Infinity);

  if (sortValue === "expected_return_desc") return byNumDesc("pred_return_60d");
  if (sortValue === "confidence_desc") return byNumDesc("confidence_score");
  if (sortValue === "risk_penalty_asc") return byNumAsc("risk_penalty");
  if (sortValue === "mdd_asc") return byNumAsc("pred_mdd_60d");
  if (sortValue === "name_asc") return String(a.name || a.code || "").localeCompare(String(b.name || b.code || ""), "ko");
  return byNumDesc("live_score");
}

function csvEscape(value) {
  const text = String(value ?? "");
  if (text.includes('"') || text.includes(",") || text.includes("\n")) {
    return `"${text.replaceAll('"', '""')}"`;
  }
  return text;
}

function buildTrendItem(s, showPrev = true) {
  const deltaHtml = s.is_new
    ? `<div class="trend-delta delta-new">NEW</div>`
    : s.delta_5d > 0
    ? `<div class="trend-delta delta-up">▲${s.delta_5d}</div>`
    : s.delta_5d < 0
    ? `<div class="trend-delta delta-down">▼${Math.abs(s.delta_5d)}</div>`
    : `<div class="trend-delta delta-stable">—</div>`;
  const prevHtml = showPrev && !s.is_new && s.rank_5d
    ? `<div class="trend-rank-prev">${s.rank_5d}위→</div>`
    : "";
  const streakHtml = (s.streak_top20 >= 3)
    ? `<div class="rank-streak ${s.streak_top20 >= 7 ? "delta-up" : "delta-new"}" style="margin-top:2px">${s.streak_top20}일 연속 top20</div>`
    : "";
  const sparkHtml = s.history ? buildSparkline(s.history) : "";
  return `
    <div class="trend-item" onclick="location.href='./detail.html?code=${encodeURIComponent(s.code)}'">
      <div class="trend-rank-box">
        <div class="trend-rank-now">${s.rank_today}</div>
        ${prevHtml}
      </div>
      <div style="flex:1;min-width:0">
        <div class="trend-name">${escapeHtml(s.name || s.code)}</div>
        <div class="trend-name-sub">${escapeHtml(s.code)}</div>
        ${streakHtml}
        ${buildScoreDeltaLine(s.score_delta)}
      </div>
      <div style="display:flex;flex-direction:column;align-items:flex-end;gap:4px">
        ${deltaHtml}
        ${sparkHtml}
      </div>
    </div>
  `;
}

function renderTrend(data) {
  if (!data) {
    document.getElementById("trendMeta").textContent = "변화 데이터를 불러오지 못했습니다.";
    ["trendNew", "trendRising", "trendFalling"].forEach((id) => {
      document.getElementById(id).innerHTML = '<span class="trend-empty">데이터 없음</span>';
    });
    return;
  }

  const compare = data.compare_5d || "-";
  document.getElementById("trendMeta").textContent =
    `기준일 ${data.date?.slice(0, 10)} · 비교일 ${compare} (5영업일 전)`;

  const stocks = Array.isArray(data.stocks) ? data.stocks : [];

  // 신규 진입: 5일 전에 없었던 종목 (상위 30위 이내)
  const newEntries = stocks.filter((s) => s.is_new && s.rank_today <= 30);
  document.getElementById("trendNew").innerHTML = newEntries.length
    ? newEntries.map((s) => buildTrendItem(s, false)).join("")
    : '<span class="trend-empty">신규 진입 종목 없음</span>';

  // 급부상: 현재 상위 50위 이내이고 10계단 이상 상승
  const rising = stocks
    .filter((s) => !s.is_new && s.rank_today <= 50 && s.delta_5d !== null && s.delta_5d >= 10)
    .sort((a, b) => b.delta_5d - a.delta_5d)
    .slice(0, 5);
  document.getElementById("trendRising").innerHTML = rising.length
    ? rising.map((s) => buildTrendItem(s, true)).join("")
    : '<span class="trend-empty">급부상 종목 없음</span>';

  // 급하락: 5일 전 상위 30위였는데 10계단 이상 하락
  const falling = stocks
    .filter((s) => !s.is_new && s.rank_5d !== null && s.rank_5d <= 30 && s.delta_5d !== null && s.delta_5d <= -10)
    .sort((a, b) => a.delta_5d - b.delta_5d)
    .slice(0, 5);
  document.getElementById("trendFalling").innerHTML = falling.length
    ? falling.map((s) => buildTrendItem(s, true)).join("")
    : '<span class="trend-empty">급하락 종목 없음</span>';
}

function buildStatCard(label, value, detail, cls = "") {
  return `
    <article class="stat-card">
      <div class="stat-label">${escapeHtml(label)}</div>
      <div class="stat-value ${escapeHtml(cls)}">${escapeHtml(value)}</div>
      <div class="stat-detail">${escapeHtml(detail)}</div>
    </article>
  `;
}

function buildChip(label, cls = "neutral", title = "") {
  if (!label) return "";
  const titleAttr = title ? ` title="${escapeHtml(title)}"` : "";
  return `<span class="chip ${escapeHtml(cls)}"${titleAttr}>${escapeHtml(label)}</span>`;
}

function buildTopIdeaCard(row, index) {
  const statusMeta = buildStatusMeta(row.code);
  const overlayMeta = state.showOverlay ? buildOverlayMeta(row.code) : null;
  const ratio = calcRewardRiskRatio(row);
  const chips = [
    buildChip(statusMeta.label, statusMeta.cls, statusMeta.title),
    overlayMeta ? buildChip(overlayMeta.label, overlayMeta.cls) : "",
    row.score_driver_1 ? buildChip(explainLabel(row.score_driver_1), "driver") : "",
    row.score_drag_1 ? buildChip(explainLabel(row.score_drag_1), "drag") : "",
  ].filter(Boolean).join("");
  return `
    <article class="idea-card" onclick="location.href='./detail.html?code=${encodeURIComponent(row.code)}'">
      <div class="idea-head">
        <div>
          <div class="idea-name">${escapeHtml(`${index + 1}. ${row.name || row.code}`)}</div>
          <div class="idea-meta">${escapeHtml(`${(row.market || "").toUpperCase() || "-"} / ${row.sector || "-"}`)}</div>
        </div>
        <div class="idea-score">${fmtScore(row.live_score ?? row.final_score)}</div>
      </div>
      <div class="idea-copy">${escapeHtml(buildThesis(row))}</div>
      <div class="driver-wrap">${chips}</div>
      <div class="idea-rows">
        <div class="idea-row">
          <strong>주요 드라이버</strong>
          <div class="muted">${escapeHtml(buildWhyNow(row))}</div>
        </div>
        <div class="idea-row">
          <strong>점수 구성</strong>
          <div class="muted">기대수익 ${fmtScore(row.ret_score)} / 확률 ${fmtScore(row.prob_score)} / 기술 ${fmtScore(row.tech_score)} / 수급 ${fmtScore(row.flow_score)}</div>
        </div>
        <div class="idea-row">
          <strong>진단</strong>
          <div class="muted">${escapeHtml(buildScoreDiagnostic(row))}</div>
        </div>
      </div>
    </article>
  `;
}

function buildSparkline(history) {
  const MAX = 50, W = 52, H = 18;
  if (!Array.isArray(history)) return "";
  const points = [];
  history.forEach((rank, i) => {
    if (rank === null) return;
    const x = (i / Math.max(history.length - 1, 1)) * W;
    const y = (Math.min(rank, MAX) - 1) / (MAX - 1) * H;
    points.push([x.toFixed(1), y.toFixed(1)]);
  });
  if (points.length < 2) return "";
  const last = points[points.length - 1];
  const first = points[0];
  const clr = Number(last[1]) <= Number(first[1]) ? "var(--green)" : "var(--red)";
  return `<svg class="rank-spark" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" style="overflow:visible">
    <polyline points="${points.map((p) => p.join(",")).join(" ")}"
      fill="none" stroke="${clr}" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round" opacity="0.8"/>
    <circle cx="${last[0]}" cy="${last[1]}" r="2.5" fill="${clr}"/>
  </svg>`;
}

function buildStreakBadge(streak) {
  if (!streak || streak < 3) return "";
  const cls = streak >= 7 ? "delta-up" : streak >= 4 ? "delta-new" : "delta-stable";
  return `<div class="rank-streak ${cls}">${streak}일 연속 top20</div>`;
}

function buildScoreDeltaLine(scoreDelta) {
  if (!scoreDelta) return "";
  const parts = [
    ["flow", scoreDelta.flow_score],
    ["ret",  scoreDelta.ret_score],
    ["tech", scoreDelta.tech_score],
    ["prob", scoreDelta.prob_score],
  ]
    .filter(([, v]) => v !== null && Math.abs(v) >= 1)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 3)
    .map(([k, v]) => {
      const cls = v > 0 ? "delta-up" : "delta-down";
      return `<span class="score-delta-tag ${cls}">${k} ${v > 0 ? "+" : ""}${v}</span>`;
    });
  return parts.length ? `<div class="score-delta-line">${parts.join("")}</div>` : "";
}

function buildRankDeltaBadge(code) {
  const t = state.trendMap.get(code);
  if (!t) return "";
  if (t.is_new) return `<div class="rank-delta delta-new">NEW</div>`;
  if (t.delta_5d === null) return "";
  if (t.delta_5d > 0) return `<div class="rank-delta delta-up">▲${t.delta_5d}</div>`;
  if (t.delta_5d < 0) return `<div class="rank-delta delta-down">▼${Math.abs(t.delta_5d)}</div>`;
  return `<div class="rank-delta delta-stable">—</div>`;
}

function buildTableRow(row, index) {
  const statusMeta = buildStatusMeta(row.code);
  const overlayMeta = state.showOverlay ? buildOverlayMeta(row.code) : null;
  const driverChips = [
    buildChip(statusMeta.label, statusMeta.cls, statusMeta.title),
    overlayMeta ? buildChip(overlayMeta.label, overlayMeta.cls) : "",
    row.score_driver_1 ? buildChip(explainLabel(row.score_driver_1), "driver") : "",
    row.score_driver_2 ? buildChip(explainLabel(row.score_driver_2), "driver") : "",
    row.score_drag_1 ? buildChip(explainLabel(row.score_drag_1), "drag") : "",
  ].filter(Boolean).join("");
  const scoreDetail = [
    `현재 ${fmtScore(row.live_score ?? row.final_score)}`,
    `저장 ${fmtScore(row.final_score)}`,
    row.live_rank ? `live_rank ${fmtScore(row.live_rank)}` : "",
  ].filter(Boolean).join(" / ");
  const flowVal = toNum(row.flow_score);
  const flowTag = flowVal === null ? "" : flowVal >= 70 ? " ▲" : flowVal < 35 ? " ▼" : "";
  const confidenceText = [
    `신뢰 ${fmtScore(row.confidence_score)}`,
    `리스크 ${fmtScore(row.risk_penalty)}`,
    `수급 ${fmtScore(row.flow_score)}${flowTag}`,
  ].join(" / ");
  const memo = [
    normalizeSentence(row.score_explain_summary, buildWhyNow(row)),
    buildRiskText(row),
  ].filter(Boolean).join(" | ");
  return `
    <tr onclick="location.href='./detail.html?code=${encodeURIComponent(row.code)}'">
      <td>
        <div class="rank-cell">
          <div class="rank-left">
            <div class="rank-main">${escapeHtml(String(index + 1))}</div>
            ${buildRankDeltaBadge(row.code)}
            ${buildStreakBadge(state.trendMap.get(row.code)?.streak_top20 ?? 0)}
            <div class="rank-sub">${escapeHtml(row.code || "-")}</div>
          </div>
          ${buildSparkline(state.trendMap.get(row.code)?.history || null)}
        </div>
      </td>
      <td>
        <div class="name-cell">
          <div class="name-main">${escapeHtml(row.name || row.code)}</div>
          <div class="name-sub">${escapeHtml(`${(row.market || "").toUpperCase() || "-"} / ${row.sector || "-"}`)}</div>
        </div>
      </td>
      <td>
        <div class="metric-stack">
          <div class="score-value">${fmtScore(row.live_score ?? row.final_score)}</div>
          <div class="muted">${escapeHtml(scoreDetail)}</div>
        </div>
      </td>
      <td>
        <div class="metric-stack">
          <div>기대수익 ${fmtPct(row.pred_return_60d)}</div>
          <div>예상 MDD ${fmtPct(row.pred_mdd_60d)}</div>
          <div class="muted">수익대비위험 ${calcRewardRiskRatio(row) === null ? "-" : calcRewardRiskRatio(row).toFixed(2)}</div>
        </div>
      </td>
      <td>
        <div class="metric-stack">
          <div>${escapeHtml(confidenceText)}</div>
          <div class="muted">${escapeHtml(row.confidence_reason || buildQualityRiskGuardText(row))}</div>
        </div>
      </td>
      <td><div class="driver-wrap">${driverChips}</div></td>
      <td>
        <div class="metric-stack">
          <div>${escapeHtml(buildScoreDiagnostic(row))}</div>
          <div class="muted">${escapeHtml(memo)}</div>
        </div>
      </td>
    </tr>
  `;
}

function populateSelect(selectEl, values, keepValue = "ALL") {
  const currentValue = keepValue && values.includes(keepValue) ? keepValue : "ALL";
  const options = ['<option value="ALL">전체</option>']
    .concat(values.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`))
    .join("");
  selectEl.innerHTML = options;
  selectEl.value = currentValue;
}


function renderHero() {
  const rows = state.filteredRows.length ? state.filteredRows : state.rows;
  const first = rows[0] || {};
  const top = rows.slice(0, 20);
  const regimeMeta = getRegimeCopy(first);
  const bias = classifyBias(rows);
  const rewardAvg = top.reduce((sum, row) => sum + (toNum(row.pred_return_60d) || 0), 0) / Math.max(top.length, 1);
  const riskAvg = top.reduce((sum, row) => sum + Math.abs(toNum(row.pred_mdd_60d) || 0), 0) / Math.max(top.length, 1);
  const marketRegime = state.manualSummary?.market_regime || null;
  const regimeReason = marketRegime?.diagnosis?.[0] || marketRegime?.reason || first.regime_reason || "시장 국면 설명이 없습니다.";
  const regimeMetrics = [];
  if (toNum(marketRegime?.breadth_20d) !== null) regimeMetrics.push(`breadth ${fmtPct(marketRegime.breadth_20d)}`);
  if (toNum(marketRegime?.recent_20d_return) !== null) regimeMetrics.push(`20일 ${fmtPct(marketRegime.recent_20d_return)}`);
  if (toNum(marketRegime?.volatility_5d) !== null) regimeMetrics.push(`변동성 ${fmtPct(marketRegime.volatility_5d)}`);
  const regimeSummary = regimeMetrics.length ? `${regimeReason} (${regimeMetrics.join(" / ")})` : regimeReason;

  document.getElementById("heroDate").textContent = `기준일 ${first.date || "-"}`;
  const heroRegime = document.getElementById("heroRegime");
  heroRegime.textContent = `시장 국면 ${regimeMeta.label}`;
  heroRegime.className = `pill ${regimeMeta.cls}`;
  document.getElementById("heroBias").textContent = `상위 성향 ${bias}`;
  document.getElementById("heroOverlay").textContent = state.showOverlay ? "오버레이 표시 중" : "오버레이 꺼짐";
  const avgFlow = top.reduce((sum, row) => sum + (toNum(row.flow_score) || 0), 0) / Math.max(top.length, 1);
  document.getElementById("heroCopy").textContent =
    `${rows.length}개 종목 · 상위20 평균 기대수익 ${fmtPct(rewardAvg)} / 예상 MDD ${fmtPct(riskAvg)} / 수급 ${fmtScore(avgFlow)}`;
  document.getElementById("heroThesis").textContent = buildThesis(first);
  document.getElementById("heroAction").textContent = buildScoreDiagnostic(first);
  document.getElementById("heroRegimeSummary").textContent = regimeSummary;
  const buyCount2 = state.filteredRows.filter((row) => statusCodeForRow(row) === "BUY_ALLOWED").length;
  const watchCount2 = state.filteredRows.filter((row) => statusCodeForRow(row) === "WATCH").length;
  document.getElementById("heroDistribution").textContent =
    `신호 성향 ${bias} · 매수가능 ${buyCount2} / 관찰 ${watchCount2} · 기준일 ${first.date || "-"}`;
}

function renderStats() {
  const rows = state.filteredRows;
  const avg = (key) => {
    const values = rows.map((row) => toNum(row[key])).filter((value) => value !== null);
    if (!values.length) return null;
    return values.reduce((sum, value) => sum + value, 0) / values.length;
  };
  const buyCount = rows.filter((row) => statusCodeForRow(row) === "BUY_ALLOWED").length;
  const watchCount = rows.filter((row) => statusCodeForRow(row) === "WATCH").length;
  const blockCount = rows.filter((row) => statusCodeForRow(row) === "BLOCK").length;

  document.getElementById("statsGrid").innerHTML = [
    buildStatCard("남은 종목 수", String(rows.length), "현재 필터 기준 후보군 규모"),
    buildStatCard("평균 현재 점수", fmtScore(avg("live_score")), "live_score 또는 final_score 기준"),
    buildStatCard("평균 기대수익", fmtPct(avg("pred_return_60d")), "pred_return_60d 평균"),
    buildStatCard("평균 예상 MDD", fmtPct(avg("pred_mdd_60d")), "pred_mdd_60d 평균"),
    buildStatCard("매수 가능 / 관찰 / 제외", `${buyCount} / ${watchCount} / ${blockCount}`, "AI 매수 게이트 기준"),
    buildStatCard("평균 수급 점수", fmtScore(avg("flow_score")), "flow_score 기준 필터 후보군 평균"),
  ].join("");
}



function renderTable() {
  const rows = state.filteredRows;
  const body = document.getElementById("rankingTableBody");
  if (!rows.length) {
    body.innerHTML = '<tr><td colspan="8" class="table-empty">조건에 맞는 종목이 없습니다.</td></tr>';
    document.getElementById("tableMeta").textContent = "0개 종목";
    return;
  }
  body.innerHTML = rows.map((row, index) => buildTableRow(row, index)).join("");
  const first = rows[0] || {};
  document.getElementById("tableMeta").textContent =
    `총 ${rows.length}개 종목 · 기준일 ${first.date || "-"} · 상위 종목을 눌러 상세 페이지로 이동`;
}

function applyFilters() {
  const searchValue = String(document.getElementById("searchInput").value || "").trim().toLowerCase();
  const statusValue = document.getElementById("statusFilter").value;
  const sortValue = document.getElementById("sortSelect").value;
  const confidenceMin = readOptionalNumber("confidenceMinInput");
  const riskMax = readOptionalNumber("riskMaxInput");

  const filtered = state.rows
    .filter((row) => {
      if (statusValue !== "ALL" && statusCodeForRow(row) !== statusValue) return false;
      if (searchValue && !getSearchText(row).includes(searchValue)) return false;
      if (confidenceMin !== null && (toNum(row.confidence_score) ?? -Infinity) < confidenceMin) return false;
      if (riskMax !== null && (toNum(row.risk_penalty) ?? Infinity) > riskMax) return false;
      return true;
    })
    .sort((a, b) => compareRows(a, b, sortValue));

  state.filteredRows = filtered;
  renderHero();
  renderStats();
  renderTable();
}

function downloadCurrentRowsAsCsv() {
  const rows = state.filteredRows;
  if (!rows.length) {
    alert("다운로드할 종목이 없습니다.");
    return;
  }

  const header = [
    "date",
    "rank",
    "code",
    "name",
    "market",
    "sector",
    "status",
    "live_score",
    "final_score",
    "pred_return_60d",
    "pred_mdd_60d",
    "confidence_score",
    "risk_penalty",
    "score_driver_1",
    "score_driver_2",
    "score_drag_1",
    "action_note",
    "score_explain_summary",
  ];

  const lines = [header.join(",")];
  rows.forEach((row, index) => {
    const status = buildStatusMeta(row.code).label;
    const values = [
      row.date || "",
      index + 1,
      row.code || "",
      row.name || "",
      row.market || "",
      row.sector || "",
      status,
      fmtScore(row.live_score ?? row.final_score),
      fmtScore(row.final_score),
      fmtPct(row.pred_return_60d),
      fmtPct(row.pred_mdd_60d),
      fmtScore(row.confidence_score),
      fmtScore(row.risk_penalty),
      explainLabel(row.score_driver_1),
      explainLabel(row.score_driver_2),
      explainLabel(row.score_drag_1),
      normalizeSentence(row.action_note, ""),
      normalizeSentence(row.score_explain_summary, ""),
    ];
    lines.push(values.map(csvEscape).join(","));
  });

  const blob = new Blob(["\uFEFF" + lines.join("\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  const baseDate = rows[0]?.date || "ranking";
  link.href = url;
  link.download = `ranking_${baseDate}.csv`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function syncManualMaps(manualSummary) {
  const candidateItems = [
    ...((manualSummary && Array.isArray(manualSummary.priority_candidates)) ? manualSummary.priority_candidates : []),
    ...((manualSummary && Array.isArray(manualSummary.caution_candidates)) ? manualSummary.caution_candidates : []),
  ];
  manualPriorityMap = new Map(
    (((manualSummary && Array.isArray(manualSummary.priority_candidates)) ? manualSummary.priority_candidates : []) || []).map((item) => [
      String(item.code || "").trim(),
      item,
    ])
  );
  manualCautionMap = new Map(
    (((manualSummary && Array.isArray(manualSummary.caution_candidates)) ? manualSummary.caution_candidates : []) || []).map((item) => [
      String(item.code || "").trim(),
      item,
    ])
  );
  manualCandidateMeta = new Map(
    candidateItems.map((item) => [
      String(item.code || "").trim(),
      {
        status: item.buy_eligibility_status || null,
        score: toNum(item.buy_eligibility_score),
        reasons: [
          ...((Array.isArray(item.buy_eligibility_hard_block_reasons) ? item.buy_eligibility_hard_block_reasons : [])),
          ...((Array.isArray(item.buy_eligibility_caution_reasons) ? item.buy_eligibility_caution_reasons : [])),
        ],
      },
    ])
  );
}

function setOverlayButtonText() {
  const button = document.getElementById("toggleOverlayBtn");
  button.textContent = state.showOverlay ? "오버레이 끄기" : "오버레이 켜기";
}

async function loadAll() {
  const dateInput = document.getElementById("signalDate");
  const marketFilter = document.getElementById("marketFilter");
  const sectorFilter = document.getElementById("sectorFilter");
  const selectedMarket = marketFilter.value || "ALL";
  const selectedSector = sectorFilter.value || "ALL";
  const dateValue = dateInput.value || "";

  try {
    const rows = await fetchRanking(dateValue, selectedMarket, selectedSector);

    if (!Array.isArray(rows) || !rows.length) throw new Error("empty ranking");

    state.rows = rows.slice().sort((a, b) => compareRows(a, b, "live_score_desc"));

    const inferredDate = rows[0]?.date || "";
    if (!dateValue && inferredDate) dateInput.value = inferredDate;

    const markets = Array.from(new Set(rows.map((row) => String(row.market || "").trim()).filter(Boolean))).sort((a, b) => a.localeCompare(b, "ko"));
    const sectors = Array.from(new Set(rows.map((row) => String(row.sector || "").trim()).filter(Boolean))).sort((a, b) => a.localeCompare(b, "ko"));
    populateSelect(marketFilter, markets, selectedMarket);
    populateSelect(sectorFilter, sectors, selectedSector);

    applyFilters();
    setOverlayButtonText();

    const [manualSummary, trendData] = await Promise.all([
      fetchManualSummarySafe(),
      fetchTrendSafe(inferredDate || dateValue || ""),
    ]);

    state.manualSummary = manualSummary;
    state.trendData = trendData;
    state.trendMap = new Map(
      ((trendData?.stocks) || []).map((s) => [s.code, s])
    );
    syncManualMaps(manualSummary);
    renderTrend(trendData);
    applyFilters();
  } catch (error) {
    console.error(error);
    state.rows = [];
    state.filteredRows = [];
    manualCandidateMeta = new Map();
    manualPriorityMap = new Map();
    manualCautionMap = new Map();
    renderTrend(null);
    document.getElementById("rankingTableBody").innerHTML = '<tr><td colspan="8" class="table-empty">랭킹 데이터를 불러오지 못했습니다.</td></tr>';
    document.getElementById("tableMeta").textContent = "데이터 로드 실패";
  }
}

function resetFilters() {
  document.getElementById("searchInput").value = "";
  document.getElementById("marketFilter").value = "ALL";
  document.getElementById("sectorFilter").value = "ALL";
  document.getElementById("statusFilter").value = "ALL";
  document.getElementById("sortSelect").value = "live_score_desc";
  document.getElementById("confidenceMinInput").value = "";
  document.getElementById("riskMaxInput").value = "";
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("reloadBtn").addEventListener("click", () => loadAll());
  document.getElementById("signalDate").addEventListener("change", () => loadAll());
  document.getElementById("marketFilter").addEventListener("change", () => loadAll());
  document.getElementById("sectorFilter").addEventListener("change", () => loadAll());
  document.getElementById("searchInput").addEventListener("input", applyFilters);
  document.getElementById("statusFilter").addEventListener("change", applyFilters);
  document.getElementById("sortSelect").addEventListener("change", applyFilters);
  document.getElementById("confidenceMinInput").addEventListener("input", applyFilters);
  document.getElementById("riskMaxInput").addEventListener("input", applyFilters);
  document.getElementById("downloadCsvBtn").addEventListener("click", downloadCurrentRowsAsCsv);
  document.getElementById("resetBtn").addEventListener("click", () => {
    resetFilters();
    loadAll();
  });
  document.getElementById("toggleOverlayBtn").addEventListener("click", () => {
    state.showOverlay = !state.showOverlay;
    setOverlayButtonText();
    renderHero();
    renderStats();
    renderTopIdeas();
    renderTable();
  });
  loadAll().catch(() => {});
});
