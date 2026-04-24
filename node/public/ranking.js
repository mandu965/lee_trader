async function fetchRanking(dateValue, marketValue = "ALL", sectorValue = "ALL") {
  const params = new URLSearchParams();
  if (dateValue) params.set("date", dateValue);
  if (marketValue && marketValue !== "ALL") params.set("market", marketValue);
  if (sectorValue && sectorValue !== "ALL") params.set("sector", sectorValue);
  const res = await fetch("/api/ranking?" + params.toString());
  if (!res.ok) throw new Error("ranking API error");
  return res.json();
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

async function fetchTradingPolicySafe() {
  try {
    const res = await fetch("/api/trading-policy");
    if (!res.ok) throw new Error("trading policy API error");
    return await res.json();
  } catch (error) {
    console.warn("trading policy unavailable", error);
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
  policy: null,
  showOverlay: false,
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
    return { label: "상태 없음", cls: "neutral", title: "" };
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

function buildActionText(row) {
  const note = normalizeSentence(row.action_note, "추가 확인 후 판단");
  const confidence = toNum(row.confidence_score);
  const penalty = toNum(row.risk_penalty);
  if (confidence !== null && confidence >= 85 && penalty !== null && penalty < 6) {
    return `${note}. 상단 후보군에서 먼저 볼 만한 수준입니다.`;
  }
  if (confidence !== null && confidence >= 75) {
    return `${note}. 거래대금과 추세를 확인한 뒤 진입 검토가 가능합니다.`;
  }
  return `${note}. 바로 매수보다 추적 관찰이 우선입니다.`;
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
          <strong>왜 지금 보나</strong>
          <div class="muted">${escapeHtml(buildWhyNow(row))}</div>
        </div>
        <div class="idea-row">
          <strong>핵심 수치</strong>
          <div class="muted">기대수익 ${fmtPct(row.pred_return_60d)} / 예상 MDD ${fmtPct(row.pred_mdd_60d)} / 수익대비위험 ${ratio === null ? "-" : ratio.toFixed(2)}</div>
        </div>
        <div class="idea-row">
          <strong>행동 메모</strong>
          <div class="muted">${escapeHtml(buildActionText(row))}</div>
        </div>
      </div>
    </article>
  `;
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
  const confidenceText = [
    `신뢰 ${fmtScore(row.confidence_score)}`,
    `리스크 ${fmtScore(row.risk_penalty)}`,
  ].join(" / ");
  const memo = [
    normalizeSentence(row.score_explain_summary, buildWhyNow(row)),
    buildRiskText(row),
  ].filter(Boolean).join(" | ");
  return `
    <tr onclick="location.href='./detail.html?code=${encodeURIComponent(row.code)}'">
      <td>
        <div class="rank-cell">
          <div class="rank-main">${escapeHtml(String(index + 1))}</div>
          <div class="rank-sub">${escapeHtml(row.code || "-")}</div>
        </div>
      </td>
      <td>
        <div class="name-cell">
          <div class="name-main">${escapeHtml(row.name || row.code)}</div>
          <div class="name-sub">${escapeHtml(`${(row.market || "").toUpperCase() || "-"} / ${row.sector || "-"}`)}</div>
        </div>
      </td>
      <td>${buildChip(statusMeta.label, statusMeta.cls, statusMeta.title)}</td>
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
          <div>${escapeHtml(buildActionText(row))}</div>
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

function renderTradingPolicy(policy) {
  if (!window.TradingPolicyUI) return;
  try {
    if (!policy) {
      window.TradingPolicyUI.renderStrip("policyStrip", []);
      window.TradingPolicyUI.renderRuleSection("rankingRules", {
        title: "랭킹 해석 규칙",
        note: "정책 데이터를 불러오지 못해 본문만 표시합니다.",
        items: [],
      });
      return;
    }
    window.TradingPolicyUI.renderStrip("policyStrip", policy.banner || []);
    window.TradingPolicyUI.renderRuleSection("rankingRules", {
      title: "랭킹 해석 규칙",
      note: "추천 순위와 실제 매수 판단 사이에 공통으로 적용되는 정책입니다.",
      items: (policy.page_rules?.ranking || []).concat(policy.page_rules?.portfolio || []),
    });
  } catch (error) {
    console.warn("renderTradingPolicy failed", error);
  }
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
  document.getElementById("heroOverlay").textContent = state.showOverlay ? "수동매매 오버레이 표시 중" : "수동매매 오버레이 숨김";
  document.getElementById("heroCopy").textContent =
    `현재 조건에서 ${rows.length}개 종목을 보고 있습니다. 상위 20개 평균 기대수익은 ${fmtPct(rewardAvg)}, 평균 예상 MDD는 ${fmtPct(riskAvg)}입니다. 카드보다 아래 전체 랭킹 표에서 비교하는 것이 핵심입니다.`;
  document.getElementById("heroThesis").textContent = buildThesis(first);
  document.getElementById("heroAction").textContent = buildActionText(first);
  document.getElementById("heroRegimeSummary").textContent = regimeSummary;
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
  const overlayCount = rows.filter((row) => Boolean(getManualOverlay(row.code))).length;

  document.getElementById("statsGrid").innerHTML = [
    buildStatCard("남은 종목 수", String(rows.length), "현재 필터 기준 후보군 규모"),
    buildStatCard("평균 현재 점수", fmtScore(avg("live_score")), "live_score 또는 final_score 기준"),
    buildStatCard("평균 기대수익", fmtPct(avg("pred_return_60d")), "pred_return_60d 평균"),
    buildStatCard("평균 예상 MDD", fmtPct(avg("pred_mdd_60d")), "pred_mdd_60d 평균"),
    buildStatCard("매수 가능 / 관찰 / 제외", `${buyCount} / ${watchCount} / ${blockCount}`, "수동매매 상태 오버레이 기준"),
    buildStatCard("수동매매 중첩", String(overlayCount), state.showOverlay ? "표와 카드에 함께 표시 중" : "버튼으로 켤 수 있음"),
  ].join("");
}

function renderTopIdeas() {
  const rows = state.filteredRows.slice(0, 3);
  const el = document.getElementById("topIdeas");
  if (!rows.length) {
    el.innerHTML = '<div class="table-empty">현재 조건에서 보여줄 상위 종목이 없습니다.</div>';
    return;
  }
  el.innerHTML = rows.map((row, index) => buildTopIdeaCard(row, index)).join("");
}

function renderInsights() {
  const rows = state.filteredRows;
  const top = rows.slice(0, 20);
  const driverCounts = countValues(top.flatMap((row) => [row.score_driver_1, row.score_driver_2, row.score_driver_3]));
  const dragCounts = countValues(top.flatMap((row) => [row.score_drag_1, row.score_drag_2]));
  const topMean = (key) => {
    const values = top.map((row) => toNum(row[key])).filter((value) => value !== null);
    if (!values.length) return null;
    return values.reduce((a, b) => a + b, 0) / values.length;
  };
  document.getElementById("alignmentNotes").innerHTML = [
    `<li><strong>ret_score 평균</strong> ${fmtScore(topMean("ret_score"))}</li>`,
    `<li><strong>prob_score 평균</strong> ${fmtScore(topMean("prob_score"))}</li>`,
    `<li><strong>tech_score 평균</strong> ${fmtScore(topMean("tech_score"))}</li>`,
    `<li><strong>confidence 평균</strong> ${fmtScore(topMean("confidence_score"))}</li>`,
    `<li><strong>상위 20개 평균 기대수익</strong> ${fmtPct(topMean("pred_return_60d"))}</li>`,
    `<li><strong>상위 20개 평균 예상 MDD</strong> ${fmtPct(topMean("pred_mdd_60d"))}</li>`,
  ].join("");

  const sortedDrivers = Object.entries(driverCounts).sort((a, b) => b[1] - a[1]).slice(0, 4);
  const sortedDrags = Object.entries(dragCounts).sort((a, b) => b[1] - a[1]).slice(0, 4);
  const notes = [
    ...sortedDrivers.map(([key, value]) => `<li><strong>driver</strong> ${escapeHtml(explainLabel(key))} ${value}건</li>`),
    ...sortedDrags.map(([key, value]) => `<li><strong>drag</strong> ${escapeHtml(explainLabel(key))} ${value}건</li>`),
  ];
  document.getElementById("driverNotes").innerHTML = notes.length ? notes.join("") : "<li>driver / drag 분포 정보가 없습니다.</li>";

  const marketRegime = state.manualSummary?.market_regime || null;
  const marketSummary = [];
  if (marketRegime?.headline) marketSummary.push(marketRegime.headline);
  if (marketRegime?.reason) marketSummary.push(marketRegime.reason);
  if (!marketSummary.length && rows[0]?.regime_reason) marketSummary.push(rows[0].regime_reason);
  document.getElementById("marketSummary").textContent = marketSummary.join(" / ") || "별도 시장 요약 데이터가 없습니다.";
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
  renderTopIdeas();
  renderInsights();
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
  button.textContent = state.showOverlay ? "수동매매 오버레이 숨기기" : "수동매매 오버레이 보기";
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

    const [policy, manualSummary] = await Promise.all([
      fetchTradingPolicySafe(),
      fetchManualSummarySafe(),
    ]);

    state.policy = policy;
    state.manualSummary = manualSummary;
    syncManualMaps(manualSummary);
    renderTradingPolicy(policy);
    applyFilters();
  } catch (error) {
    console.error(error);
    state.rows = [];
    state.filteredRows = [];
    manualCandidateMeta = new Map();
    manualPriorityMap = new Map();
    manualCautionMap = new Map();
    renderTradingPolicy(null);
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
