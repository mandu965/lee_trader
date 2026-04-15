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

async function fetchTradingPolicySafe() {
  try {
    return await fetchJson("/api/trading-policy");
  } catch (error) {
    console.warn("trading policy load skipped", error);
    return null;
  }
}

function chipClass(kind) {
  if (kind === "BUY_ALLOWED") return "is-good";
  if (kind === "BUY_NOW" || kind === "PROMOTION_READY" || kind === "TRUSTED") return "is-good";
  if (kind === "WATCHLIST" || kind === "WATCH" || kind === "MONITOR") return "is-watch";
  if (kind === "PAPER_ONLY" || kind === "CONDITIONAL" || kind === "HOLD") return "is-warn";
  return "is-bad";
}

function buyEligibilityLabel(status) {
  if (status === "BUY_ALLOWED") return "매수 가능";
  if (status === "WATCH") return "관찰";
  if (status === "BLOCK") return "차단";
  return "미정";
}

function buildBuyEligibilityLine(item) {
  const score = Number(item.buy_eligibility_score);
  const blocked = Array.isArray(item.buy_eligibility_hard_block_reasons)
    ? item.buy_eligibility_hard_block_reasons
    : [];
  const caution = Array.isArray(item.buy_eligibility_caution_reasons)
    ? item.buy_eligibility_caution_reasons
    : [];
  const reasons = (blocked.length ? blocked : caution).slice(0, 2);
  return {
    label: buyEligibilityLabel(item.buy_eligibility_status),
    scoreText: Number.isFinite(score) ? score.toFixed(1) : "-",
    reasonText: reasons.length ? reasons.join(" / ") : "추가 사유 없음",
  };
}

function buildScoreRoleLine(item) {
  const finalText = fmtNum(item.final_score, 1);
  const label = buyEligibilityLabel(item.buy_eligibility_status);
  const eligibilityScore = Number(item.buy_eligibility_score);
  const eligibilityText = Number.isFinite(eligibilityScore) ? eligibilityScore.toFixed(1) : "-";
  return `추천 점수 ${finalText}는 상대순위, 절대 판단 ${label || "미정"} ${eligibilityText}는 실제 진입 기준입니다.`;
}

function quoteSourceMeta(item) {
  const source = String(item?.intraday_quote?.source || "").trim().toLowerCase();
  if (source === "kis") return { label: "KIS", cls: "is-good" };
  if (source === "naver_fallback") return { label: "NAVER", cls: "is-watch" };
  return { label: "NONE", cls: "is-bad" };
}

function buildRecentSurgeChip(item) {
  if (!item?.recent_surge_label) return "";
  const cls = item.recent_surge_hard_flag ? "is-bad" : "is-warn";
  const detail = item.recent_surge_detail ? ` title="${escapeHtml(item.recent_surge_detail)}"` : "";
  return `<span class="chip ${cls}"${detail}>${escapeHtml(item.recent_surge_label)}</span>`;
}

function fmtMoneyShort(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  if (n >= 1_0000_0000_0000) return `${(n / 1_0000_0000_0000).toFixed(1)}조`;
  if (n >= 1_0000_0000) return `${(n / 1_0000_0000).toFixed(1)}억`;
  if (n >= 1_0000) return `${(n / 1_0000).toFixed(1)}만`;
  return n.toLocaleString("ko-KR");
}

function intradayNames(items) {
  return (Array.isArray(items) ? items : [])
    .slice(0, 3)
    .map((item) => item?.name || item?.code || "-")
    .join(", ");
}

function renderIntradayWarning(data) {
  const panel = document.getElementById("intradayWarningPanel");
  if (!panel) return;
  const basis = data.execution_basis || {};
  const intraday = data.intraday_summary || {};
  const sourceStatus = String(data.source_status || "").toLowerCase();
  const useIntraday = basis.label === "장중 기준";
  if (!useIntraday) {
    panel.style.display = "none";
    panel.innerHTML = "";
    return;
  }
  const tone = sourceStatus === "fallback" ? "is-bad" : "is-watch";
  const quoteTotal = (intraday.quote_success_count || 0) + (intraday.quote_failure_count || 0);
  const note = sourceStatus === "fallback"
    ? "장중 후보는 생성됐지만 시세 연결이 불안정해 보수 해석이 필요합니다."
    : "장중 시세 기반 오후장 후보입니다. 시세 출처와 장중 거래대금을 함께 보십시오.";
  const headline = intraday.headline || note;
  panel.style.display = "block";
  panel.innerHTML = `
    <div class="check-card">
      <div class="candidate-meta">
        <span class="chip ${tone}">${escapeHtml(basis.label || "장중 기준")}</span>
        <span class="chip ${tone}">source ${escapeHtml((data.source_status || "current").toUpperCase())}</span>
        <span class="chip ${tone}">priority ${escapeHtml(fmtNum(intraday.priority_count))}</span>
        <span class="chip ${tone}">caution ${escapeHtml(fmtNum(intraday.caution_count))}</span>
        <span class="chip ${tone}">quotes ${escapeHtml(fmtNum(intraday.quote_success_count))}/${escapeHtml(fmtNum(quoteTotal))}</span>
      </div>
      <div class="card-detail">${escapeHtml(headline)}</div>
      <div class="card-detail">${escapeHtml(intraday.action_guide || note)}</div>
      <div class="card-detail">승격 ${escapeHtml(fmtNum(intraday.promoted_to_priority_count))} / 제외 ${escapeHtml(fmtNum(intraday.dropped_from_priority_count))} / 경계 강화 ${escapeHtml(fmtNum(intraday.caution_escalation_count))}</div>
    </div>
  `;
}

function renderIntradayChanges(data) {
  const panel = document.getElementById("intradayChangePanel");
  const grid = document.getElementById("intradayChangeGrid");
  if (!panel || !grid) return;
  const intraday = data.intraday_summary || {};
  const basis = data.execution_basis || {};
  if (basis.label !== "장중 기준") {
    panel.style.display = "none";
    grid.innerHTML = "";
    return;
  }

  const groups = [
    {
      title: "우선 검토 승격",
      note: "마감 기준보다 오후장 우선순위가 올라온 종목",
      items: intraday.promoted_to_priority || [],
      empty: "승격된 종목이 없습니다.",
      tone: "is-good",
    },
    {
      title: "우선 검토 제외",
      note: "마감 기준 우선 검토에서 빠진 종목",
      items: intraday.dropped_from_priority || [],
      empty: "제외된 종목이 없습니다.",
      tone: "is-warn",
    },
    {
      title: "장중 재확인 필요",
      note: "장중 시세를 못 받아 오후장 판단이 약한 종목",
      items: intraday.missing_quotes || [],
      empty: "장중 재확인 필요 종목이 없습니다.",
      tone: "is-bad",
    },
    {
      title: "장중 경계 강화",
      note: "장중 이유로 보수 검토가 강화된 종목",
      items: intraday.caution_escalations || [],
      empty: "추가 경계 종목이 없습니다.",
      tone: "is-watch",
    },
  ];

  panel.style.display = "block";
  grid.innerHTML = groups.map((group) => `
    <article class="candidate-card">
      <div class="candidate-head">
        <div class="candidate-head-main">
          <h3 class="candidate-name">${escapeHtml(group.title)}</h3>
          <div class="candidate-code">${escapeHtml(group.note)}</div>
        </div>
        <div class="candidate-score">${fmtNum(group.items.length)}</div>
      </div>
      <div class="candidate-meta">
        <span class="chip ${group.tone}">${escapeHtml(group.title)}</span>
      </div>
      ${
        group.items.length
          ? `<ul class="candidate-list">${group.items.slice(0, 4).map((item) => `<li>${escapeHtml(item?.name || item?.code || "-")} ${item?.intraday_reasons?.[0] ? `- ${escapeHtml(item.intraday_reasons[0])}` : ""}</li>`).join("")}</ul>`
          : `<div class="card-detail">${escapeHtml(group.empty)}</div>`
      }
    </article>
  `).join("");
}

function renderHero(data) {
  const mode = data.manual_mode || {};
  const marketRegime = data.market_regime || {};
  const basis = data.execution_basis || {};
  const regimeReason =
    Array.isArray(marketRegime.diagnosis) && marketRegime.diagnosis.length
      ? marketRegime.diagnosis[0]
      : (marketRegime.reason || "시장 레짐 설명이 없습니다.");

  document.getElementById("heroGrid").innerHTML = `
    <article class="hero-card">
      <div class="hero-label">기준일</div>
      <div class="hero-value">${escapeHtml(data.asof_date || "-")}</div>
      <div class="hero-detail">${escapeHtml(basis.label || "기준 미상")} | 생성 시각 ${escapeHtml(data.generated_at || "-")}</div>
    </article>
    <article class="hero-card">
      <div class="hero-label">운영 Gate</div>
      <div class="hero-value">${escapeHtml(data.gate_status || "-")}</div>
      <div class="hero-detail">walk-forward ${escapeHtml(data.walkforward_acceptance_status || "-")}</div>
    </article>
    <article class="hero-card">
      <div class="hero-label">오늘 운영 모드</div>
      <div class="hero-value">${escapeHtml(mode.label || "-")}</div>
      <div class="hero-detail">${escapeHtml(mode.note || "설명 정보가 없습니다.")} ${basis.note ? `| ${escapeHtml(basis.note)}` : ""}</div>
    </article>
    <article class="hero-card">
      <div class="hero-label">시장 레짐</div>
      <div class="hero-value">${escapeHtml(marketRegime.regime || "-")}</div>
      <div class="hero-detail">
        true ${escapeHtml(fmtNum(marketRegime.true_count))}/5 |
        breadth ${escapeHtml(fmtNum(marketRegime.breadth_20d, 3))} |
        recent ${escapeHtml(fmtPct(marketRegime.recent_20d_return, 1))} |
        vol ${escapeHtml(fmtNum(marketRegime.volatility_5d, 4))}<br />
        ${escapeHtml(regimeReason)}
      </div>
    </article>
  `;
}

function renderChecklist(items) {
  document.getElementById("checkList").innerHTML = (items || [])
    .map((item) => `<li>${escapeHtml(item)}</li>`)
    .join("");
}

function renderTradingPolicy(policy) {
  if (!window.TradingPolicyUI) return;
  if (!policy) {
    window.TradingPolicyUI.renderStrip("policyStrip", []);
    window.TradingPolicyUI.renderRuleSection("manualRules", {
      title: "수동매매 공통 규칙",
      note: "전략 정책을 불러오지 못했습니다.",
      items: [],
    });
    return;
  }

  window.TradingPolicyUI.renderStrip("policyStrip", policy.banner || []);
  window.TradingPolicyUI.renderRuleSection("manualRules", {
    title: "수동매매 공통 규칙",
    note: "오늘 후보를 볼 때 공통적으로 적용할 기준입니다.",
    items: (policy.page_rules?.manual || []).concat(policy.page_rules?.portfolio || []),
  });
}

function renderCandidates(targetId, items, emptyText) {
  const wrap = document.getElementById(targetId);
  if (!wrap) return;
  if (!items || !items.length) {
    wrap.innerHTML = `<div class="empty-state">${escapeHtml(emptyText)}</div>`;
    return;
  }

  wrap.innerHTML = items.map((item) => {
    const eligibility = buildBuyEligibilityLine(item);
    const quoteSource = quoteSourceMeta(item);
    const changePct = item?.intraday_quote?.change_pct;
    const tradingValue = item?.intraday_quote?.trading_value;
    const reasons = (
      item.supporting_reasons && item.supporting_reasons.length
        ? item.supporting_reasons
        : item.blocking_reasons || []
    ).slice(0, 4);
    const link = `./detail.html?code=${encodeURIComponent(item.code || "")}`;

    return `
      <article
        class="candidate-card"
        tabindex="0"
        role="link"
        aria-label="${escapeHtml(item.name || item.code || "종목")} 상세 보기"
        title="클릭하면 상세 화면으로 이동"
        onclick="location.href='${link}'"
        onkeydown="if(event.key==='Enter'||event.key===' '){event.preventDefault();location.href='${link}';}"
      >
        <div class="candidate-head">
          <div class="candidate-head-main">
            <h3 class="candidate-name">${escapeHtml(item.name || item.code || "-")}</h3>
            <div class="candidate-code">${escapeHtml(item.code || "-")} | ${escapeHtml(item.sector || "-")} | ${escapeHtml(item.dominant_theme || "(none)")}</div>
            <div class="candidate-link-hint">
              <span class="candidate-link-icon">→</span>
              <span>클릭하면 상세 화면으로 이동</span>
            </div>
          </div>
          <div class="candidate-score">${fmtNum(item.final_score, 1)}</div>
        </div>
        <div class="candidate-meta">
          <span class="chip ${chipClass(item.buy_eligibility_status)}">${escapeHtml(eligibility.label)}</span>
          <span class="chip ${chipClass(item.buyability_status)}">${escapeHtml(item.buyability_status || "-")}</span>
          <span class="chip ${chipClass(item.watchlist_tier)}">${escapeHtml(item.watchlist_tier || "-")}</span>
          <span class="chip ${chipClass(item.confidence_state_v2)}">${escapeHtml(item.confidence_state_v2 || "-")}</span>
          ${buildRecentSurgeChip(item)}
          <span class="chip ${quoteSource.cls}">시세 ${escapeHtml(quoteSource.label)}</span>
        </div>
        <div class="card-detail">${escapeHtml(buildScoreRoleLine(item))}</div>
        <div class="card-detail">buy_rank ${fmtNum(item.buy_rank)} | eligibility ${escapeHtml(eligibility.scoreText)} | readiness ${fmtNum(item.promotion_readiness_score, 2)} | raw_confidence_v2 ${fmtNum(item.raw_confidence_v2, 2)}</div>
        <div class="card-detail">장중 ${escapeHtml(quoteSource.label)} | 변동률 ${escapeHtml(fmtPct(changePct, 1))} | 거래대금 ${escapeHtml(fmtMoneyShort(tradingValue))}</div>
        <div class="card-detail">절대 매수 판단: ${escapeHtml(eligibility.reasonText)}</div>
        <ul class="candidate-list">
          ${reasons.length ? reasons.map((reason) => `<li>${escapeHtml(reason)}</li>`).join("") : "<li>사유 정보 없음</li>"}
        </ul>
      </article>
    `;
  }).join("");
}

async function loadManualTradingSummary() {
  const state = document.getElementById("pageState");
  state.textContent = "수동매매 요약을 불러오는 중입니다.";

  try {
    const [data, policy] = await Promise.all([
      fetchJson("/api/manual-trading/summary"),
      fetchTradingPolicySafe(),
    ]);

    renderTradingPolicy(policy);
    renderHero(data);
    renderIntradayWarning(data);
    renderIntradayChanges(data);
    renderChecklist(data.checklist || []);
    renderCandidates("priorityGrid", data.priority_candidates || [], "우선 검토 후보가 없습니다.");
    renderCandidates("cautionGrid", data.caution_candidates || [], "보수 검토 후보가 없습니다.");
    state.textContent = `기준일 ${data.asof_date || "-"} 기준으로 수동매매 요약을 불러왔습니다.`;
  } catch (error) {
    console.error(error);
    renderTradingPolicy(null);
    renderIntradayWarning({});
    renderIntradayChanges({});
    document.getElementById("heroGrid").innerHTML = '<div class="empty-state">수동매매 요약을 불러오지 못했습니다.</div>';
    document.getElementById("checkList").innerHTML = "";
    renderCandidates("priorityGrid", [], "우선 검토 후보를 불러오지 못했습니다.");
    renderCandidates("cautionGrid", [], "보수 검토 후보를 불러오지 못했습니다.");
    state.textContent = `조회 실패: ${error.message}`;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  loadManualTradingSummary().catch((error) => console.error(error));
});
