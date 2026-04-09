async function fetchRanking(dateValue) {
  const params = new URLSearchParams();
  if (dateValue) params.set("date", dateValue);
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

let manualCandidateMeta = new Map();
let manualPriorityMap = new Map();
let manualCautionMap = new Map();

const EXPLAIN_LABELS = {
  high_ret_score: "기대수익 강점",
  high_probability_score: "확률 강점",
  high_tech_score: "기술 흐름 강점",
  strong_quality_profile: "퀄리티 강점",
  strong_safety_profile: "안정성 강점",
  healthy_liquidity_profile: "유동성 강점",
  elevated_risk_penalty: "리스크 감점 높음",
  very_high_risk_penalty: "리스크 감점 매우 높음",
  weak_quality_score: "퀄리티 약점",
  weak_tech_score: "기술 흐름 약점",
  weak_safety_score: "안정성 약점",
  weak_ret_score: "기대수익 약점",
  low_probability_score: "확률 약점",
  low_liquidity_score: "유동성 약점",
  low_confidence: "신뢰도 낮음",
  multiple_component_fallbacks: "대체값 사용 다수",
  partial_quality_coverage: "퀄리티 데이터 커버리지 부족",
};

function toNum(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
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

function buildIntradayOverlayChip(code) {
  const item = getManualOverlay(code);
  if (!item || !item.intraday_verdict) return "";
  const verdict = String(item.intraday_verdict || "").toUpperCase();
  const tone = verdict === "PRIORITY" ? "driver" : verdict === "CAUTION" ? "drag" : "info";
  const label = verdict === "PRIORITY" ? "장중 우선 검토" : verdict === "CAUTION" ? "장중 보수 검토" : `장중 ${verdict}`;
  return `<span class="chip ${tone}">${escapeHtml(label)}</span>`;
}

function buildBuyEligibilityChip(code) {
  const meta = getBuyEligibilityMeta(code);
  if (!meta || !meta.status) return "";
  const label = buyEligibilityLabel(meta.status);
  if (!label) return "";
  const cls = meta.status === "BLOCK" ? "drag" : "driver";
  const scoreText = Number.isFinite(meta.score) ? ` ${meta.score.toFixed(1)}` : "";
  const reasonText = Array.isArray(meta.reasons) && meta.reasons.length ? meta.reasons.slice(0, 2).join(" / ") : "";
  const titleAttr = reasonText ? ` title="${escapeHtml(reasonText)}"` : "";
  return `<span class="chip ${cls}"${titleAttr}>${escapeHtml(label + scoreText)}</span>`;
}

function buildScoreRoleText(code, finalScore) {
  const finalText = fmtScore(finalScore);
  const meta = getBuyEligibilityMeta(code);
  if (!meta || !meta.status) {
    return `추천 점수 ${finalText}는 상대순위입니다. 절대 매수 판단 정보는 아직 없습니다.`;
  }
  const label = buyEligibilityLabel(meta.status) || meta.status;
  const scoreText = Number.isFinite(meta.score) ? meta.score.toFixed(1) : "-";
  return `추천 점수 ${finalText}는 상대순위, 절대 판단 ${label} ${scoreText}는 실제 진입 기준입니다.`;
}

function buildQualityRiskGuardText(row) {
  const shadowRank = toNum(row?.shadow_quality_risk_guard_rank);
  const shadowPenalty = toNum(row?.shadow_quality_risk_guard_penalty);
  const delta = toNum(row?.shadow_quality_risk_guard_rank_delta);
  if (shadowRank === null) return "quality/risk guard shadow 정보가 아직 없습니다.";
  if (delta !== null && delta > 0) {
    return `shadow guard 적용 시 rank ${fmtScore(shadowRank)}로 ${fmtScore(delta)}계단 개선됩니다.`;
  }
  if (delta !== null && delta < 0) {
    return `shadow guard 적용 시 rank ${fmtScore(shadowRank)}로 ${fmtScore(Math.abs(delta))}계단 밀립니다.`;
  }
  if (shadowPenalty !== null && shadowPenalty > 0) {
    return `shadow guard penalty ${fmtScore(shadowPenalty)}가 걸리지만 현재 rank 변화는 없습니다.`;
  }
  return `shadow guard 기준으로도 현재 순위가 유지됩니다.`;
}

function buildQualityRiskGuardChip(row) {
  const delta = toNum(row?.shadow_quality_risk_guard_rank_delta);
  const penalty = toNum(row?.shadow_quality_risk_guard_penalty);
  if (delta !== null && delta > 0) {
    return `<span class="chip info" title="${escapeHtml(buildQualityRiskGuardText(row))}">shadow +${escapeHtml(String(delta.toFixed(0)))}</span>`;
  }
  if (penalty !== null && penalty > 0) {
    return `<span class="chip watch" title="${escapeHtml(buildQualityRiskGuardText(row))}">shadow penalty ${escapeHtml(String(penalty.toFixed(0)))}</span>`;
  }
  return "";
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

function dedupeByCode(rows) {
  const seen = new Set();
  return rows.filter((row) => {
    if (!row || seen.has(row.code)) return false;
    seen.add(row.code);
    return true;
  });
}

function hasAnyDriver(row, codes) {
  const drivers = [row?.score_driver_1, row?.score_driver_2, row?.score_driver_3];
  return drivers.some((code) => codes.includes(code));
}

function hasAnyDrag(row, codes) {
  const drags = [row?.score_drag_1, row?.score_drag_2];
  return drags.some((code) => codes.includes(code));
}

function pickUniqueRows(rows, predicate, limit, usedCodes) {
  const selected = [];
  for (const row of rows) {
    if (!row || usedCodes.has(row.code)) continue;
    if (!predicate(row)) continue;
    selected.push(row);
    usedCodes.add(row.code);
    if (selected.length >= limit) break;
  }
  return selected;
}

function buildRowLookup(rows) {
  return new Map(
    (Array.isArray(rows) ? rows : []).map((row) => [String(row?.code || "").trim(), row])
  );
}

function mapManualItemsToRows(items, rowLookup) {
  return dedupeByCode(
    (Array.isArray(items) ? items : [])
      .map((item) => rowLookup.get(String(item?.code || "").trim()))
      .filter(Boolean)
  );
}

function rowSectorKey(row) {
  return String(row?.sector || "").trim() || "__none__";
}

function rowThemeKey(row) {
  return String(row?.dominant_theme || "").trim() || "__none__";
}

function pickDiversifiedRows(rows, predicate, limit, usedCodes, options = {}) {
  const selected = [];
  const sectorCounts = new Map();
  const themeCounts = new Map();
  const maxPerSector = options.maxPerSector ?? 1;
  const maxPerTheme = options.maxPerTheme ?? 1;

  for (const row of rows) {
    if (!row || usedCodes.has(row.code)) continue;
    if (!predicate(row)) continue;

    const sectorKey = rowSectorKey(row);
    const themeKey = rowThemeKey(row);
    const sectorCount = sectorCounts.get(sectorKey) || 0;
    const themeCount = themeCounts.get(themeKey) || 0;

    if (sectorKey !== "__none__" && sectorCount >= maxPerSector) continue;
    if (themeKey !== "__none__" && themeCount >= maxPerTheme) continue;

    selected.push(row);
    usedCodes.add(row.code);
    if (sectorKey !== "__none__") sectorCounts.set(sectorKey, sectorCount + 1);
    if (themeKey !== "__none__") themeCounts.set(themeKey, themeCount + 1);
    if (selected.length >= limit) break;
  }
  return selected;
}

function manualPriorityRows(rows) {
  return mapManualItemsToRows(Array.from(manualPriorityMap.values()), buildRowLookup(rows));
}

function manualCautionRows(rows) {
  return mapManualItemsToRows(Array.from(manualCautionMap.values()), buildRowLookup(rows));
}

function classifyBias(rows) {
  const top = rows.slice(0, 20);
  const driverCounts = countValues(top.flatMap((row) => [row.score_driver_1, row.score_driver_2, row.score_driver_3]));
  const trendCount = (driverCounts.high_ret_score || 0) + (driverCounts.high_probability_score || 0) + (driverCounts.high_tech_score || 0);
  const defensiveCount = (driverCounts.strong_safety_profile || 0) + (driverCounts.strong_quality_profile || 0) + (driverCounts.healthy_liquidity_profile || 0);
  if (trendCount >= defensiveCount + 4) return "상승 선호 우세";
  if (defensiveCount >= trendCount + 4) return "안정 선호 우세";
  return "균형";
}

function getRegimeCopy(row) {
  const regime = String(row?.regime || "").toLowerCase();
  if (regime === "bull") {
    return {
      label: "bull",
      detail: "상승 추세가 강해 수익, 확률, 기술 점수를 조금 더 공격적으로 해석할 수 있는 구간입니다.",
      pillClass: "good",
    };
  }
  if (regime === "neutral") {
    return {
      label: "neutral",
      detail: "수익, 확률, 기술, 안정성을 균형 있게 해석하는 중립 구간입니다.",
      pillClass: "info",
    };
  }
  return {
    label: "defensive",
    detail: "안정성과 방어를 우선으로 보되 상승 신호를 완전히 버리지는 않는 방어 구간입니다.",
    pillClass: "warn",
  };
}

function buildThesis(row) {
  const strengths = normalizeSentence(row.score_explain_strengths, row.score_explain_summary || "뚜렷한 강점 정보가 없습니다.");
  return strengths.split("/").map((part) => part.trim()).filter(Boolean).slice(0, 3).join(" · ");
}

function buildWhyNow(row) {
  const drivers = [row.top_driver_1, row.top_driver_2, row.top_driver_3].filter(Boolean).map(explainLabel);
  if (drivers.length) return `${drivers.slice(0, 2).join(", ")} 신호가 동시에 올라온 구간입니다.`;
  return normalizeSentence(row.score_explain_summary, "현재 시점의 변화 신호가 충분히 요약되지 않았습니다.");
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
    return `${note}. 오늘 보드에서 실행 우선순위가 높습니다.`;
  }
  if (confidence !== null && confidence >= 75) {
    return `${note}. 거래대금과 눌림 구간 확인 후 진입 검토가 가능합니다.`;
  }
  return `${note}. 바로 매수보다 추적 관찰이 우선입니다.`;
}

function getRewardRiskTone(row) {
  const ret = toNum(row.pred_return_60d);
  const mdd = Math.abs(toNum(row.pred_mdd_60d) || 0);
  if (ret !== null && ret >= 0.18 && mdd <= 0.12) return "good";
  if (ret !== null && ret >= 0.1) return "info";
  return "warn";
}

function calcRewardRiskRatio(row) {
  const ret = toNum(row.pred_return_60d);
  const mdd = Math.abs(toNum(row.pred_mdd_60d) || 0);
  if (ret === null || !mdd) return null;
  return ret / mdd;
}

function calcTargetPrice(row) {
  const close = toNum(row.close);
  const ret = toNum(row.pred_return_60d);
  if (close === null || ret === null) return null;
  return close * (1 + ret);
}

function buildInvalidationText(row) {
  const ma20 = toNum(row.ma_20);
  if (ma20 !== null) return `20일선 ${fmtPrice(ma20)} 이탈 시 흐름 약화`;
  const risk = row.risk_factor_1 || row.score_drag_1;
  if (risk) return `${explainLabel(risk)} 강도가 커지면 재검토 필요`;
  return "리스크 요인 재평가 필요";
}

function buildMetricPills(row) {
  return `
    <div class="metric-chip-row">
      <span class="metric-chip">기대수익 ${fmtPct(row.pred_return_60d)}</span>
      <span class="metric-chip">예상 MDD ${fmtPct(row.pred_mdd_60d)}</span>
      <span class="metric-chip">신뢰도 ${fmtScore(row.confidence_score)}</span>
    </div>
  `;
}

function buildMiniCard(row) {
  return `
    <article class="mini-card" onclick="location.href='./detail.html?code=${encodeURIComponent(row.code)}'">
      <div class="mini-card-title">
        <div>
          <div class="mini-card-name">${escapeHtml(row.name || row.code)}</div>
          <div class="mini-card-sub">${escapeHtml((row.market || "").toUpperCase() || "-")} / ${escapeHtml(row.sector || "-")}</div>
        </div>
        <div class="mini-card-score">${fmtScore(row.final_score)}</div>
      </div>
      <div class="mini-card-meta">${escapeHtml(buildScoreRoleText(row.code, row.final_score))}</div>
      <div class="mini-card-meta">${escapeHtml(buildQualityRiskGuardText(row))}</div>
      <div class="mini-card-thesis">${escapeHtml(buildThesis(row))}</div>
      <div class="chip-row">${buildBuyEligibilityChip(row.code)}${buildIntradayOverlayChip(row.code)}${buildQualityRiskGuardChip(row)}</div>
      ${buildMetricPills(row)}
      <div class="mini-card-meta">${escapeHtml(buildActionText(row))}</div>
    </article>
  `;
}

function buildCandidateCard(row) {
  const tone = getRewardRiskTone(row);
  const ratio = calcRewardRiskRatio(row);
  const targetPrice = calcTargetPrice(row);
  return `
    <article class="candidate-card candidate-card--${tone}" onclick="location.href='./detail.html?code=${encodeURIComponent(row.code)}'">
      <div class="candidate-title">
        <div>
          <div class="candidate-name">${escapeHtml(row.name || row.code)}</div>
          <div class="candidate-meta">${escapeHtml((row.market || "").toUpperCase() || "-")} / ${escapeHtml(row.sector || "-")}</div>
        </div>
        <div class="candidate-score-wrap">
          <div class="candidate-score-label">final_score</div>
          <div class="candidate-score">${fmtScore(row.final_score)}</div>
        </div>
      </div>
      <div class="section-sub">${escapeHtml(buildScoreRoleText(row.code, row.final_score))}</div>
      <div class="section-sub">${escapeHtml(buildQualityRiskGuardText(row))}</div>
      <div class="thesis-block">
        <div class="thesis-label">투자 논리</div>
        <div class="candidate-thesis">${escapeHtml(buildThesis(row))}</div>
      </div>
      <div class="candidate-grid-2">
        <div class="candidate-panel">
          <div class="candidate-panel-label">지금 보는 이유</div>
          <div class="candidate-panel-value">${escapeHtml(buildWhyNow(row))}</div>
        </div>
        <div class="candidate-panel">
          <div class="candidate-panel-label">행동 가이드</div>
          <div class="candidate-panel-value">${escapeHtml(buildActionText(row))}</div>
        </div>
      </div>
      <div class="reward-risk-bar reward-risk-bar--${tone}">
        <div class="reward-risk-item"><span class="reward-risk-label">기대수익</span><strong>${fmtPct(row.pred_return_60d)}</strong></div>
        <div class="reward-risk-item"><span class="reward-risk-label">예상 MDD</span><strong>${fmtPct(row.pred_mdd_60d)}</strong></div>
        <div class="reward-risk-item"><span class="reward-risk-label">수익비</span><strong>${ratio === null ? "-" : ratio.toFixed(2)}</strong></div>
        <div class="reward-risk-item"><span class="reward-risk-label">신뢰도</span><strong>${fmtScore(row.confidence_score)}</strong></div>
      </div>
      <div class="decision-strip">
        <div class="decision-box"><div class="decision-label">목표가 가정</div><div class="decision-value">${targetPrice === null ? "-" : fmtPrice(targetPrice)}</div></div>
        <div class="decision-box"><div class="decision-label">무효화 조건</div><div class="decision-value decision-value--text">${escapeHtml(buildInvalidationText(row))}</div></div>
      </div>
      <div class="chip-row">
        ${buildBuyEligibilityChip(row.code)}
        ${buildIntradayOverlayChip(row.code)}
        ${buildQualityRiskGuardChip(row)}
        ${row.score_driver_1 ? `<span class="chip driver">${escapeHtml(explainLabel(row.score_driver_1))}</span>` : ""}
        ${row.score_driver_2 ? `<span class="chip driver">${escapeHtml(explainLabel(row.score_driver_2))}</span>` : ""}
        ${row.score_drag_1 ? `<span class="chip drag">${escapeHtml(explainLabel(row.score_drag_1))}</span>` : ""}
      </div>
      <div class="candidate-footer">
        <div class="metric-box"><div class="metric-label">주요 강점</div><div class="metric-value metric-value--text">${escapeHtml(normalizeSentence(row.score_explain_strengths, "강점 정보가 없습니다."))}</div></div>
        <div class="metric-box"><div class="metric-label">주의 요인</div><div class="metric-value metric-value--text">${escapeHtml(buildRiskText(row))}</div></div>
      </div>
    </article>
  `;
}

function buildTopIdeaCard(row, index) {
  const ratio = calcRewardRiskRatio(row);
  return `
    <article class="idea-card" onclick="location.href='./detail.html?code=${encodeURIComponent(row.code)}'">
      <div class="idea-rank">0${index + 1}</div>
      <div class="idea-body">
        <div class="idea-header">
          <div>
            <div class="idea-name">${escapeHtml(row.name || row.code)}</div>
            <div class="idea-meta">${escapeHtml((row.market || "").toUpperCase() || "-")} / ${escapeHtml(row.sector || "-")}</div>
        </div>
        <div class="idea-score">${fmtScore(row.final_score)}</div>
      </div>
      <div class="section-sub">${escapeHtml(buildScoreRoleText(row.code, row.final_score))}</div>
      <div class="section-sub">${escapeHtml(buildQualityRiskGuardText(row))}</div>
      <div class="idea-thesis">${escapeHtml(buildThesis(row))}</div>
      <div class="chip-row">${buildBuyEligibilityChip(row.code)}${buildIntradayOverlayChip(row.code)}${buildQualityRiskGuardChip(row)}</div>
        <div class="idea-stats">
          <span>기대수익 ${fmtPct(row.pred_return_60d)}</span>
          <span>예상 MDD ${fmtPct(row.pred_mdd_60d)}</span>
          <span>수익비 ${ratio === null ? "-" : ratio.toFixed(2)}</span>
        </div>
        <div class="idea-action">${escapeHtml(buildActionText(row))}</div>
      </div>
    </article>
  `;
}

function buildIntradaySummaryCard(title, detail, items, chipClass) {
  return `
    <article class="candidate-card candidate-card--info">
      <div class="candidate-title">
        <div>
          <div class="candidate-name">${escapeHtml(title)}</div>
          <div class="candidate-meta">${escapeHtml(detail)}</div>
        </div>
        <div class="candidate-score-wrap">
          <div class="candidate-score-label">count</div>
          <div class="candidate-score">${fmtScore((items || []).length)}</div>
        </div>
      </div>
      ${
        items && items.length
          ? `<div class="chip-row">${items.slice(0, 4).map((item) => `<span class="chip ${chipClass}">${escapeHtml(item?.name || item?.code || "-")}</span>`).join("")}</div>`
          : `<div class="section-sub">해당 종목이 없습니다.</div>`
      }
    </article>
  `;
}

function renderIntradayBoard(manualSummary) {
  const intraday = manualSummary?.intraday_summary || {};
  const el = document.getElementById("intradayBoard");
  if (!el) return;
  el.innerHTML = [
    buildIntradaySummaryCard("우선 검토 승격", "오후장 기준으로 우선순위가 올라온 종목", intraday.promoted_to_priority || [], "driver"),
    buildIntradaySummaryCard("우선 검토 제외", "마감 기준 우선 검토에서 빠진 종목", intraday.dropped_from_priority || [], "drag"),
    buildIntradaySummaryCard("장중 재확인 필요", "장중 시세 연결이 약해 오후장 전에 다시 볼 종목", intraday.missing_quotes || [], "drag"),
  ].join("");
}

function renderTradingPolicy(policy) {
  if (!window.TradingPolicyUI) return;
  if (!policy) {
    window.TradingPolicyUI.renderStrip("policyStrip", []);
    window.TradingPolicyUI.renderRuleSection("rankingRules", {
      title: "추천종목 해석 규칙",
      note: "전략 정책을 불러오지 못해 추천 본문만 표시합니다.",
      items: [],
    });
    return;
  }
  window.TradingPolicyUI.renderStrip("policyStrip", policy.banner || []);
  window.TradingPolicyUI.renderRuleSection("rankingRules", {
    title: "추천종목 해석 규칙",
    note: "추천 순위와 실제 매수 판단을 함께 볼 때 적용하는 공통 기준입니다.",
    items: (policy.page_rules?.ranking || []).concat(policy.page_rules?.portfolio || []),
  });
}

function renderTopIdeas(rows) {
  const top = rows.slice(0, 30);
  const usedCodes = new Set();
  const ideas = [];
  const priority = manualPriorityRows(rows)
    .filter((row) => top.some((topRow) => topRow.code === row.code))
    .sort((a, b) => (toNum(getBuyEligibilityMeta(b.code)?.score) || 0) - (toNum(getBuyEligibilityMeta(a.code)?.score) || 0));

  for (const row of priority) {
    if (ideas.length >= 3) break;
    ideas.push(row);
    usedCodes.add(row.code);
  }

  ideas.push(
    ...pickDiversifiedRows(
      top,
      (row) => toNum(row.confidence_score) >= 75 && toNum(row.risk_penalty) < 10,
      3 - ideas.length,
      usedCodes,
      { maxPerSector: 1, maxPerTheme: 1 }
    )
  );
  document.getElementById("topIdeas").innerHTML = ideas.length
    ? ideas.map((row, index) => buildTopIdeaCard(row, index)).join("")
    : '<div class="empty">오늘 강하게 볼 만한 매수 후보가 아직 선명하지 않습니다.</div>';
}

function renderActionBoard(rows) {
  const top = rows.slice(0, 30);
  const usedCodes = new Set();
  const goNow = manualPriorityRows(rows).slice(0, 4);
  goNow.forEach((row) => usedCodes.add(row.code));
  const caution = manualCautionRows(rows).slice(0, 4);
  caution.forEach((row) => usedCodes.add(row.code));
  const trend = pickDiversifiedRows(
    top,
    (row) => hasAnyDriver(row, ["high_ret_score", "high_tech_score", "high_probability_score"]) && toNum(row.confidence_score) >= 70,
    4,
    usedCodes,
    { maxPerSector: 1, maxPerTheme: 1 }
  );
  const defensive = pickDiversifiedRows(
    top,
    (row) => hasAnyDriver(row, ["strong_safety_profile", "strong_quality_profile", "healthy_liquidity_profile"]) && toNum(row.risk_penalty) < 10,
    4,
    usedCodes,
    { maxPerSector: 1, maxPerTheme: 1 }
  );
  [
    ["goNowList", goNow, "지금 바로 검토할 후보가 없습니다."],
    ["trendList", trend, "상승 우선 후보가 없습니다."],
    ["defensiveList", defensive, "안정 대안 후보가 없습니다."],
    ["cautionList", caution, "리스크 체크가 필요한 후보가 없습니다."],
  ].forEach(([id, items, emptyText]) => {
    const el = document.getElementById(id);
    el.innerHTML = items.length ? items.map(buildMiniCard).join("") : `<div class="empty">${escapeHtml(emptyText)}</div>`;
  });
}

function renderCandidateGrid(rows) {
  const top = rows.slice(0, 30);
  const usedCodes = new Set();
  const picks = [];

  for (const row of manualPriorityRows(rows).slice(0, 3)) {
    if (!usedCodes.has(row.code)) {
      picks.push(row);
      usedCodes.add(row.code);
    }
  }

  picks.push(
    ...pickDiversifiedRows(
      top,
      (row) =>
        !manualPriorityMap.has(row.code) &&
        !manualCautionMap.has(row.code) &&
        toNum(row.confidence_score) >= 70 &&
        toNum(row.risk_penalty) < 10,
      2,
      usedCodes,
      { maxPerSector: 1, maxPerTheme: 1 }
    )
  );

  for (const row of manualCautionRows(rows).slice(0, 1)) {
    if (!usedCodes.has(row.code)) {
      picks.push(row);
      usedCodes.add(row.code);
    }
  }

  if (picks.length < 6) {
    picks.push(...pickDiversifiedRows(top, () => true, 6 - picks.length, usedCodes, { maxPerSector: 2, maxPerTheme: 1 }));
  }
  document.getElementById("candidateGrid").innerHTML = picks.length
    ? picks.map(buildCandidateCard).join("")
    : '<div class="empty">비교해 볼 만한 후보를 찾지 못했습니다.</div>';
}

function renderHero(rows, dateValue, manualSummary) {
  const first = manualPriorityRows(rows)[0] || rows[0] || {};
  const regimeMeta = getRegimeCopy(first);
  const marketRegime = manualSummary?.market_regime || null;
  const intradaySummary = manualSummary?.intraday_summary || null;
  const top = rows.slice(0, 20);
  const confidenceAvg = top.reduce((sum, row) => sum + (toNum(row.confidence_score) || 0), 0) / Math.max(top.length, 1);
  const bias = classifyBias(rows);
  const rewardAvg = top.reduce((sum, row) => sum + (toNum(row.pred_return_60d) || 0), 0) / Math.max(top.length, 1);
  const riskAvg = top.reduce((sum, row) => sum + Math.abs(toNum(row.pred_mdd_60d) || 0), 0) / Math.max(top.length, 1);
  const regimeReason = marketRegime?.diagnosis?.[0] || marketRegime?.reason || first.regime_reason || "시장 레짐 설명이 없습니다.";
  const regimeMetrics = [];
  if (toNum(marketRegime?.true_count) !== null) regimeMetrics.push(`true_count ${marketRegime.true_count}`);
  if (toNum(marketRegime?.breadth_20d) !== null) regimeMetrics.push(`breadth ${fmtPct(marketRegime.breadth_20d, 1)}`);
  if (toNum(marketRegime?.recent_20d_return) !== null) regimeMetrics.push(`20일 ${fmtPct(marketRegime.recent_20d_return, 1)}`);
  if (toNum(marketRegime?.volatility_5d) !== null) regimeMetrics.push(`변동성 ${fmtPct(marketRegime.volatility_5d, 1)}`);
  const regimeReasonText = regimeMetrics.length ? `${regimeReason} (${regimeMetrics.join(" / ")})` : regimeReason;
  document.getElementById("heroDate").textContent = `기준일 ${dateValue || first.date || "-"}`;
  const heroRegime = document.getElementById("heroRegime");
  heroRegime.textContent = `시장 톤 ${regimeMeta.label}`;
  heroRegime.className = `pill ${regimeMeta.pillClass}`;
  document.getElementById("heroProfile").textContent = `모델 구성 ${first.weight_profile || "-"}`;
  document.getElementById("heroBias").textContent = bias;
  document.getElementById("heroCopy").textContent = `오늘 상위 후보의 평균 기대수익은 ${fmtPct(rewardAvg)}, 평균 예상 MDD는 ${fmtPct(-riskAvg)} 수준입니다. 수동매매 우선 검토 ${manualPriorityMap.size}개와 보수 검토 ${manualCautionMap.size}개를 중심으로 섹터와 테마 중복을 줄여 정리했습니다. ${regimeReasonText}${intradaySummary?.is_active ? ` / ${intradaySummary.headline}` : ""}`;
  document.getElementById("regimeValue").textContent = regimeMeta.label;
  document.getElementById("regimeReason").textContent = regimeReasonText;
  document.getElementById("profileValue").textContent = first.weight_profile || "-";
  document.getElementById("profileDetail").textContent = regimeMeta.detail;
  document.getElementById("confidenceValue").textContent = `${fmtScore(confidenceAvg)} / ${first.confidence_grade || "-"}`;
  document.getElementById("confidenceDetail").textContent = first.confidence_reason || "신뢰 해석 정보가 없습니다.";
  document.getElementById("driverBiasValue").textContent = bias;
  document.getElementById("driverBiasDetail").textContent = "상위 20개 점수 구성을 기준으로 읽은 해석입니다.";
  document.getElementById("heroThesis").textContent = buildThesis(first);
  document.getElementById("heroAction").textContent = buildActionText(first);
}

function renderInsights(rows) {
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
    `<li><strong>예상 MDD 평균</strong> ${fmtPct(topMean("pred_mdd_60d"))}</li>`,
    `<li><strong>confidence 평균</strong> ${fmtScore(topMean("confidence_score"))}</li>`,
    `<li><strong>행동 메모</strong> ${escapeHtml(rows[0]?.action_note || "추가 확인 후 판단")}</li>`,
  ].join("");
  const sortedDrivers = Object.entries(driverCounts).sort((a, b) => b[1] - a[1]).slice(0, 5);
  const sortedDrags = Object.entries(dragCounts).sort((a, b) => b[1] - a[1]).slice(0, 4);
  const notes = [
    ...sortedDrivers.map(([key, value]) => `<li><strong>driver</strong> ${escapeHtml(explainLabel(key))} ${value}건</li>`),
    ...sortedDrags.map(([key, value]) => `<li><strong>drag</strong> ${escapeHtml(explainLabel(key))} ${value}건</li>`),
  ];
  document.getElementById("driverNotes").innerHTML = notes.length ? notes.join("") : "<li>driver / drag 분포 정보가 없습니다.</li>";
}

async function loadAll() {
  const dateInput = document.getElementById("signalDate");
  const dateValue = dateInput.value || "";
  try {
    const [rows, policy, manualSummary] = await Promise.all([
      fetchRanking(dateValue),
      fetchTradingPolicySafe(),
      fetchManualSummarySafe(),
    ]);
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
    renderTradingPolicy(policy);
    renderIntradayBoard(manualSummary);
    if (!Array.isArray(rows) || !rows.length) throw new Error("empty ranking");
    const inferredDate = rows[0]?.date || "";
    if (!dateValue && inferredDate) dateInput.value = inferredDate;
    renderHero(rows, inferredDate, manualSummary);
    renderTopIdeas(rows);
    renderActionBoard(rows);
    renderCandidateGrid(rows);
    renderInsights(rows);
  } catch (error) {
    console.error(error);
    manualCandidateMeta = new Map();
    manualPriorityMap = new Map();
    manualCautionMap = new Map();
    renderTradingPolicy(null);
    alert("추천 데이터를 불러오지 못했습니다.");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("reloadBtn").addEventListener("click", loadAll);
  document.getElementById("signalDate").addEventListener("change", loadAll);
  loadAll().catch(() => {});
});
