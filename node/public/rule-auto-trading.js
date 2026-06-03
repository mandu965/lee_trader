/* ── 포맷 헬퍼 ── */
const fmtNum = (v, digits = 0) => {
  const n = Number(v);
  if (!Number.isFinite(n)) return "-";
  return n.toLocaleString("ko-KR", { minimumFractionDigits: digits, maximumFractionDigits: digits });
};

const fmtPct = (v, digits = 1) => {
  const n = Number(v);
  if (!Number.isFinite(n)) return "-";
  const sign = n > 0 ? "+" : "";
  return `${sign}${n.toFixed(digits)}%`;
};

const fmtWon = (v) => {
  const n = Number(v);
  if (!Number.isFinite(n)) return "-";
  const abs = Math.abs(n);
  const sign = n < 0 ? "-" : n > 0 ? "+" : "";
  if (abs >= 100_000_000) return `${sign}${(abs / 100_000_000).toFixed(2)}억`;
  if (abs >= 10_000) return `${sign}${Math.round(abs / 10_000)}만`;
  return `${sign}${Math.round(abs).toLocaleString("ko-KR")}원`;
};

const fmtWonFull = (v) => {
  const n = Number(v);
  if (!Number.isFinite(n)) return "-";
  const sign = n < 0 ? "-" : n > 0 ? "+" : "";
  return `${sign}${Math.round(Math.abs(n)).toLocaleString("ko-KR")}원`;
};

const esc = (v) =>
  String(v ?? "").replace(/[&<>"']/g, (m) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m]));

const signedClass = (v) => {
  const n = Number(v);
  if (!Number.isFinite(n) || n === 0) return "";
  return n > 0 ? "pos" : "neg";
};

/* ── API 호출 ── */
async function fetchJson(url) {
  const res = await fetch(url, { credentials: "same-origin" });
  if (!res.ok) throw new Error(`${url} HTTP ${res.status}`);
  return res.json();
}

/* ── 탭 전환 ── */
function initTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(`tab-${tab}`)?.classList.add("active");
    });
  });
}

/* ── 검토 이유 코드 → 한글 레이블 ── */
const REASON_LABEL = {
  loss_below_minus_8pct:     "손실 -8%",
  final_score_weak:          "최종점수 약화",
  score_delta_down:          "점수 하락",
  ret_score_weak:            "ret_score 약화",
  prob_score_weak:           "prob_score 약화",
  confidence_low:            "신뢰도 낮음",
  risk_penalty_high:         "risk_penalty 높음",
  hold_day_60_reached:       "60일 보유 점검",
  hold_day_20_reached:       "20일 보유 점검",
  profit_above_15pct:        "+15% 수익 점검",
  latest_price_missing:      "가격 누락",
  new_position:              "신규 3일 유예",
  holding_support_maintained:"보유 근거 유지",
};

function reasonLabel(code) {
  const displayLabels = {
    loss_below_minus_8pct: "손실 -8%",
    final_score_weak: "최종점수 약화",
    score_delta_down: "점수 하락",
    ret_score_weak: "수익 기대 약화",
    prob_score_weak: "상위권 확률 약화",
    confidence_low: "신뢰도 낮음",
    risk_penalty_high: "변동성 리스크 높음",
    hold_day_60_reached: "60일 보유 점검",
    hold_day_20_reached: "20일 보유 점검",
    profit_above_15pct: "+15% 수익 점검",
    latest_price_missing: "가격 누락",
    new_position: "신규 3일 유예",
    holding_support_maintained: "보유 근거 유지",
  };
  return displayLabels[code] || REASON_LABEL[code] || code;
}

function reasonChip(code) {
  const label = reasonLabel(code);
  if (code === "holding_support_maintained") return `<span class="chip ok">${esc(label)}</span>`;
  if (code === "new_position") return `<span class="chip info">${esc(label)}</span>`;
  if (code === "loss_below_minus_8pct" || code === "final_score_weak") return `<span class="chip bad">${esc(label)}</span>`;
  return `<span class="chip warn">${esc(label)}</span>`;
}

const fmtWonPlain = (v) => {
  const n = Number(v);
  if (!Number.isFinite(n)) return "-";
  return Math.round(n).toLocaleString("ko-KR") + "원";
};

/* ── 히어로 카드 업데이트 ── */
function renderHero(data) {
  const { count, total_cost, total_value, cash_balance, total_account_value, synced_at, items } = data;
  const tc = Number(total_cost) || 0;
  const tv = Number(total_value) || 0;
  const cashNum = Number(cash_balance);
  const acctNum = Number(total_account_value);
  const unrealizedPnl = tc > 0 ? tv - tc : null;
  const pnlPct = tc > 0 && unrealizedPnl != null ? (unrealizedPnl / tc) * 100 : null;

  const dates = (items || [])
    .map((h) => h.latest_rank_date)
    .filter(Boolean)
    .sort()
    .reverse();
  const latestDate = dates[0] || "-";

  document.getElementById("heroDate").textContent = synced_at ? synced_at.slice(0, 10) : latestDate;
  document.getElementById("heroDateSub").textContent = synced_at ? `계좌 동기화: ${synced_at}` : "마지막 랭킹 갱신일";

  document.getElementById("heroCount").textContent = fmtNum(count);
  const exit = (items || []).filter((h) => h.system_review_status === "EXIT_REVIEW").length;
  const review = (items || []).filter((h) => h.system_review_status === "REVIEW").length;
  document.getElementById("heroCountSub").textContent =
    exit > 0 ? `즉시검토 ${exit}종목` : review > 0 ? `점검필요 ${review}종목` : "전체 보유 유지";

  const valEl = document.getElementById("heroValue");
  if (tc > 0) {
    valEl.textContent = fmtWonPlain(tv);
  } else if (Number.isFinite(acctNum) && acctNum > 0) {
    valEl.textContent = fmtWonPlain(acctNum);
  } else {
    valEl.textContent = "-";
  }
  valEl.className = "hero-value";
  document.getElementById("heroValueSub").textContent =
    tc > 0 ? `매수금액 ${fmtWonPlain(tc)}` :
    (Number.isFinite(cashNum) && cashNum > 0 ? `예수금 ${fmtWonPlain(cashNum)}` : "보유 없음");

  const pnlEl = document.getElementById("heroPnl");
  pnlEl.textContent = unrealizedPnl != null ? fmtWonFull(unrealizedPnl) : "-";
  pnlEl.className = `hero-value ${unrealizedPnl != null ? signedClass(unrealizedPnl) : ""}`;
  document.getElementById("heroPnlSub").textContent =
    Number.isFinite(pnlPct) ? `수익률 ${fmtPct(pnlPct)}` : "-";
}

/* ── Safety 배너 ── */
function renderBanner(data) {
  const el = document.getElementById("safetyBanner");
  const textEl = document.getElementById("safetyText");
  const detailEl = document.getElementById("safetyDetail");

  const { count, items } = data;
  const sorted = (items || [])
    .slice()
    .sort((a, b) => (Number(b.sell_priority_score) || 0) - (Number(a.sell_priority_score) || 0));
  const topRisk = sorted[0] || null;
  const exit = (items || []).filter((h) => h.system_review_status === "EXIT_REVIEW").length;
  const review = (items || []).filter((h) => h.system_review_status === "REVIEW").length;
  const topReasons = (topRisk?.system_review_reasons || []).slice(0, 2).map(reasonLabel).join(" · ");

  if (!count) {
    el.className = "safety-banner warn";
    textEl.textContent = "보유 종목 없음 — KIS 계좌(44****02)에 보유 종목이 없거나 아직 동기화 전입니다.";
    detailEl.textContent = "종가 배치(18:10) 또는 스케줄러 실행 시 자동 동기화됩니다.";
    return;
  }
  if (exit > 0) {
    el.className = "safety-banner bad";
    textEl.textContent = topRisk
      ? `${topRisk.name || topRisk.code} 매도검토 필요 · 우선순위 ${fmtNum(topRisk.sell_priority_score)}`
      : `즉시 검토 필요 — ${exit}종목 EXIT_REVIEW 상태`;
    detailEl.innerHTML = `${esc(topReasons || `${count}종목 보유 중`)}${topRisk?.code ? ` <a href="/detail.html?code=${esc(topRisk.code)}">상세</a><a href="/trade-history.html">매매이력</a>` : ""}`;
    return;
  }
  if (review > 0) {
    el.className = "safety-banner warn";
    textEl.textContent = topRisk
      ? `${topRisk.name || topRisk.code} 점검 필요 · 우선순위 ${fmtNum(topRisk.sell_priority_score)}`
      : `점검 필요 — ${review}종목 REVIEW 상태`;
    detailEl.innerHTML = `${esc(topReasons || `${count}종목 보유 중`)}${topRisk?.code ? ` <a href="/detail.html?code=${esc(topRisk.code)}">상세</a><a href="/trade-history.html">매매이력</a>` : ""}`;
    return;
  }
  el.className = "safety-banner ok";
  textEl.textContent = `정상 보유 중 — ${count}종목 전체 KEEP`;
  detailEl.textContent = "";
}

function buildDecisionSummary(h, pct, priorityValue) {
  const reasons = (h.system_review_reasons || []).slice(0, 4).map(reasonLabel);
  const reasonText = reasons.length ? reasons.join(" + ") : "특이 사유 없음";
  const score = Number(h.final_score);
  const confidence = Number(h.confidence_score);
  const risk = Number(h.risk_penalty);
  const priceText = Number.isFinite(pct)
    ? (Math.abs(pct) < 1 ? "가격 손익은 아직 작음" : pct < 0 ? "가격 손실 구간" : "가격 수익 구간")
    : "가격 손익 확인 필요";
  const modelParts = [];
  if (Number.isFinite(score)) modelParts.push(`모델 ${score.toFixed(1)}`);
  if (Number.isFinite(confidence)) modelParts.push(`신뢰도 ${Math.round(confidence)}`);
  if (Number.isFinite(risk)) modelParts.push(`리스크 ${risk.toFixed(1)}`);
  const modelText = modelParts.length ? modelParts.join(" · ") : "모델 지표 부족";
  const actionText = priorityValue >= 80
    ? "가격보다 모델 훼손 기준으로 우선 검토합니다."
    : priorityValue >= 55
      ? "보유 유지 전 점검이 필요합니다."
      : "현재는 보유 추적 중심입니다.";
  return `
    <div class="holding-judgement">
      <strong>판단 요약</strong> ${esc(reasonText)}<br>
      <strong>해석</strong> ${esc(priceText)} · ${esc(modelText)} · ${esc(actionText)}
    </div>
  `;
}

function buildCompactDecisionSummary(h, pct, priorityValue) {
  const reasons = (h.system_review_reasons || []).slice(0, 3).map(reasonLabel);
  const reasonText = reasons.length ? reasons.join(", ") : "보유 근거 유지";
  const score = Number(h.final_score);
  const confidence = Number(h.confidence_score);
  const risk = Number(h.risk_penalty);
  const status = h.system_review_status || "KEEP";
  const statusText =
    status === "EXIT_REVIEW" ? "매도검토" :
    status === "REVIEW" ? "점검필요" : "계속보유";
  const priceText = Number.isFinite(pct)
    ? pct >= 15 ? "수익 보호 여부 확인" :
      pct > 0 ? "수익 구간" :
      pct <= -8 ? "손실 관리 구간" : "가격 변동 제한적"
    : "가격 확인 필요";
  const modelText = [
    Number.isFinite(score) ? `모델 ${score.toFixed(1)}` : null,
    Number.isFinite(confidence) ? `신뢰도 ${Math.round(confidence)}` : null,
    Number.isFinite(risk) ? `리스크 ${risk.toFixed(1)}` : null,
  ].filter(Boolean).join(" · ");
  const actionText = priorityValue >= 80
    ? "수익률과 별개로 모델/리스크 기준 우선 검토"
    : priorityValue >= 55
      ? "보유 유지 전 점검 필요"
      : "현재는 보유 추적 중심";

  return `
    <div class="holding-judgement">
      <strong>${esc(statusText)}</strong> ${esc(reasonText)}
      <span style="color:var(--color-text-secondary);"> · ${esc(priceText)} · ${esc(modelText)} · ${esc(actionText)}</span>
    </div>
  `;
}

/* ── 개요 탭: 운영 판단 요약 ── */
function renderDecisionSummary(data) {
  const el = document.getElementById("decisionSummaryGrid");
  if (!el) return;

  const items = data.items || [];
  const exitItems = items.filter((h) => h.system_review_status === "EXIT_REVIEW");
  const reviewItems = items.filter((h) => h.system_review_status === "REVIEW");
  const sortedRisk = items
    .slice()
    .sort((a, b) => (Number(b.sell_priority_score) || 0) - (Number(a.sell_priority_score) || 0));
  const topRisk = sortedRisk[0] || null;

  let actionLabel = "보유 유지";
  let actionColor = "#86efac";
  let actionDetail = "즉시 검토 대상이 없습니다.";
  if (exitItems.length) {
    actionLabel = `${exitItems.length}종목 매도검토`;
    actionColor = "#fca5a5";
    actionDetail = topRisk
      ? `${topRisk.name || topRisk.code} · 우선순위 ${fmtNum(topRisk.sell_priority_score)} · ${topRisk.system_action_note || "상세 검토 필요"}`
      : "EXIT_REVIEW 종목 상세 확인 필요";
  } else if (reviewItems.length) {
    actionLabel = `${reviewItems.length}종목 점검`;
    actionColor = "#fde68a";
    actionDetail = topRisk
      ? `${topRisk.name || topRisk.code} · 우선순위 ${fmtNum(topRisk.sell_priority_score)}`
      : "REVIEW 종목 상세 확인 필요";
  }

  const latestRankDates = items.map((h) => h.latest_rank_date).filter(Boolean).sort();
  const latestRankDate = latestRankDates[latestRankDates.length - 1] || "-";
  const syncedAt = data.synced_at || "-";
  const freshnessDetail = `계좌 ${syncedAt} · 랭킹 ${latestRankDate} · source ${data.source || "-"}`;

  const cash = Number(data.cash_balance);
  const stockValue = Number(data.total_value);
  const accountValue = Number(data.total_account_value);
  const expected = (Number.isFinite(cash) ? cash : 0) + (Number.isFinite(stockValue) ? stockValue : 0);
  const diff = Number.isFinite(accountValue) ? accountValue - expected : null;
  const diffAbs = Math.abs(Number(diff));
  const tolerance = Math.max(1000, expected * 0.01);
  const accountNeedsCheck = Number.isFinite(diff) && expected > 0 && diffAbs > tolerance;
  const accountLabel = accountNeedsCheck ? "합계 확인 필요" : "합계 정상 범위";
  const accountColor = accountNeedsCheck ? "#fde68a" : "#86efac";
  const accountDetail = Number.isFinite(diff)
    ? `계좌총액 ${fmtWonPlain(accountValue)} · 예수금+주식 ${fmtWonPlain(expected)} · 차이 ${fmtWonFull(diff)}`
    : "계좌 총액 정보가 없습니다.";

  el.innerHTML = `
    <div class="status-summary-card">
      <div class="status-summary-label">오늘 조치</div>
      <div class="status-summary-value" style="color:${actionColor};">${esc(actionLabel)}</div>
      <div class="status-summary-sub">${esc(actionDetail)}</div>
    </div>
    <div class="status-summary-card">
      <div class="status-summary-label">데이터 최신성</div>
      <div class="status-summary-value" style="color:#93c5fd;">${esc(latestRankDate)}</div>
      <div class="status-summary-sub">${esc(freshnessDetail)}</div>
    </div>
    <div class="status-summary-card">
      <div class="status-summary-label">계좌 합계</div>
      <div class="status-summary-value" style="color:${accountColor};">${esc(accountLabel)}</div>
      <div class="status-summary-sub">${esc(accountDetail)}</div>
    </div>
  `;
}

/* ── 개요 탭: 포트폴리오 손익 ── */
function renderOverviewPnl(data) {
  const el = document.getElementById("overviewPnlKv");
  if (!el) return;
  const { total_cost, total_value, items } = data;
  const unrealized = total_value - total_cost;
  const unrealizedPct = total_cost > 0 ? (unrealized / total_cost) * 100 : null;
  const realized = (items || []).reduce((s, h) => s + (Number(h.realized_pnl) || 0), 0);

  const cashBalance = Number(data.cash_balance);
  const acctTotal = Number(data.total_account_value);
  el.innerHTML = `
    <div class="kv-row"><span>총 계좌 평가금액</span><strong>${Number.isFinite(acctTotal) && acctTotal > 0 ? fmtWonPlain(acctTotal) : "-"}</strong></div>
    <div class="kv-row"><span>예수금</span><strong>${Number.isFinite(cashBalance) && cashBalance > 0 ? fmtWonPlain(cashBalance) : "-"}</strong></div>
    <div class="kv-row"><span>총 매수금액 (주식)</span><strong>${Number.isFinite(total_cost) && total_cost > 0 ? fmtWonPlain(total_cost) : "-"}</strong></div>
    <div class="kv-row"><span>총 평가금액 (주식)</span><strong>${Number.isFinite(total_value) && total_value > 0 ? fmtWonPlain(total_value) : "-"}</strong></div>
    <div class="kv-row"><span>미실현 손익</span><strong class="${signedClass(unrealized)}">${total_cost > 0 ? fmtWonFull(unrealized) + (Number.isFinite(unrealizedPct) ? ` (${fmtPct(unrealizedPct)})` : "") : "-"}</strong></div>
    <div class="kv-row"><span>실현 손익 누계</span><strong class="${signedClass(realized)}">${fmtWonFull(realized)}</strong></div>
  `;
}

/* ── 개요 탭: 점검 상태 분포 ── */
function renderOverviewStatus(data) {
  const el = document.getElementById("overviewStatusGrid");
  if (!el) return;
  const items = data.items || [];
  const keep = items.filter((h) => h.system_review_status === "KEEP").length;
  const review = items.filter((h) => h.system_review_status === "REVIEW").length;
  const exitReview = items.filter((h) => h.system_review_status === "EXIT_REVIEW").length;

  el.innerHTML = `
    <div class="status-summary-card">
      <div class="status-summary-label">계속 보유</div>
      <div class="status-summary-value" style="color:#86efac;">${keep}</div>
      <div class="status-summary-sub">종목</div>
    </div>
    <div class="status-summary-card">
      <div class="status-summary-label">점검 필요</div>
      <div class="status-summary-value" style="color:#fde68a;">${review}</div>
      <div class="status-summary-sub">종목</div>
    </div>
    <div class="status-summary-card">
      <div class="status-summary-label">즉시 검토</div>
      <div class="status-summary-value" style="color:#fca5a5;">${exitReview}</div>
      <div class="status-summary-sub">종목</div>
    </div>
  `;
}

/* ── 개요 탭: 주의 종목 요약 ── */
function renderOverviewAlerts(data) {
  const el = document.getElementById("overviewAlertList");
  if (!el) return;
  const items = (data.items || []).filter(
    (h) => h.system_review_status === "EXIT_REVIEW" || h.system_review_status === "REVIEW"
  );
  if (!items.length) {
    el.innerHTML = `<div class="empty">검토 필요 종목이 없습니다.</div>`;
    return;
  }
  el.innerHTML = items.map((h) => {
    const pct = Number(h.unrealized_pnl_pct);
    const pctStr = Number.isFinite(pct) ? fmtPct(pct) : "-";
    const statusLabel = h.system_review_status === "EXIT_REVIEW" ? "즉시검토" : "점검필요";
    const statusCls = h.system_review_status === "EXIT_REVIEW" ? "bad" : "warn";
    return `
      <div style="display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px solid rgba(30,41,59,.5);flex-wrap:wrap;">
        <span class="chip ${statusCls}">${statusLabel}</span>
        <span style="font-weight:700;font-size:14px;">${esc(h.name || h.code)}</span>
        <span style="font-size:12px;color:var(--color-text-secondary);">${esc(h.code)}</span>
        <span class="${signedClass(pct)}" style="font-size:13px;">${pctStr}</span>
        <span style="font-size:12px;color:var(--color-text-secondary);flex:1;min-width:160px;">${esc(h.system_action_note || "")}</span>
        <a href="/detail.html?code=${esc(h.code)}" style="font-size:11px;color:#93c5fd;text-decoration:none;">상세 →</a>
      </div>`;
  }).join("");
}

/* ── 계좌 탭: 보유종목 카드 리스트 ── */
function renderAccountSummaryBar(data) {
  const el = document.getElementById("accountSummaryBar");
  if (!el) return;

  const items = data.items || [];
  const exit = items.filter((h) => h.system_review_status === "EXIT_REVIEW").length;
  const review = items.filter((h) => h.system_review_status === "REVIEW").length;
  const pnl = Number(data.total_unrealized_pnl);
  const totalValue = Number(data.total_value);
  const accountValue = Number(data.total_account_value);

  el.innerHTML = `
    <div class="account-summary-item">
      <div class="account-summary-label">보유</div>
      <div class="account-summary-value">${fmtNum(items.length)}종목</div>
    </div>
    <div class="account-summary-item">
      <div class="account-summary-label">매도검토</div>
      <div class="account-summary-value neg">${fmtNum(exit)}</div>
    </div>
    <div class="account-summary-item">
      <div class="account-summary-label">점검필요</div>
      <div class="account-summary-value" style="color:#fde68a;">${fmtNum(review)}</div>
    </div>
    <div class="account-summary-item">
      <div class="account-summary-label">평가손익</div>
      <div class="account-summary-value ${signedClass(pnl)}">${Number.isFinite(pnl) ? fmtWonFull(pnl) : "-"}</div>
    </div>
    <div class="account-summary-item">
      <div class="account-summary-label">계좌총액</div>
      <div class="account-summary-value">${Number.isFinite(accountValue) ? fmtWonPlain(accountValue) : fmtWonPlain(totalValue)}</div>
    </div>
  `;
}

function renderHoldingsList(data) {
  const el = document.getElementById("holdingsList");
  const badge = document.getElementById("badgeAccount");
  if (!el) return;

  const items = data.items || [];
  badge.textContent = items.length;

  if (!items.length) {
    el.innerHTML = `<div class="empty">보유 종목이 없습니다. 종목 상세 페이지에서 거래를 입력하거나 매매 이력을 등록하세요.</div>`;
    return;
  }

  el.innerHTML = items.map((h) => {
    const pct = Number(h.unrealized_pnl_pct);
    const pnl = Number(h.unrealized_pnl);
    const score = Number(h.final_score);
    const priority = Number(h.sell_priority_score);
    const statusCls = h.system_review_status || "KEEP";

    // 우선순위 바 색상
    const priorityValue = Number.isFinite(priority) ? priority : 0;
    const barColor = priorityValue >= 80 ? "#f87171" : priorityValue >= 55 ? "#facc15" : "#4ade80";
    const barPct = Math.max(0, Math.min(100, priorityValue));
    const reasons = (h.system_review_reasons || []);
    const statusLabel =
      statusCls === "EXIT_REVIEW" ? "매도검토" :
      statusCls === "REVIEW" ? "점검필요" : "계속보유";
    const currentValue = Number(h.current_value);
    const totalValue = Number(data.total_value);
    const weightPct = Number.isFinite(currentValue) && Number.isFinite(totalValue) && totalValue > 0
      ? (currentValue / totalValue) * 100
      : null;

    // 이유 칩
    const reasonsHtml = reasons.map(reasonChip).join("");

    return `
      <div class="holding-card" onclick="location.href='/detail.html?code=${esc(h.code)}'">
        <div class="holding-card-top">
          <div class="holding-stock">
            <div class="holding-stock-name">${esc(h.name || h.code)}</div>
            <div class="holding-stock-meta">${esc(h.code)} · ${esc(h.market || "")} · ${esc(h.sector || "")} · ${fmtNum(h.current_qty)}주</div>
            <div class="holding-badges">
              <span class="review-badge ${statusCls}">${esc(statusLabel)}</span>
              ${Number.isFinite(score) ? `<span class="score-badge">모델 ${score.toFixed(1)}</span>` : ""}
              ${Number.isFinite(h.confidence_score) ? `<span class="chip info">신뢰도 ${Math.round(h.confidence_score)}</span>` : ""}
              ${Number.isFinite(h.score_delta) && h.score_delta !== 0 ? `<span class="chip ${h.score_delta > 0 ? "ok" : "warn"}">점수 변화 ${h.score_delta > 0 ? "+" : ""}${h.score_delta.toFixed(1)}</span>` : ""}
            </div>
          </div>
          <div class="holding-metrics">
            <div class="metric">
              <div class="metric-label">수익률</div>
              <div class="metric-value ${signedClass(pct)}">${Number.isFinite(pct) ? fmtPct(pct) : "-"}</div>
            </div>
            <div class="metric">
              <div class="metric-label">평가손익</div>
              <div class="metric-value ${signedClass(pnl)}">${Number.isFinite(pnl) ? fmtWonFull(pnl) : "-"}</div>
            </div>
            <div class="metric">
              <div class="metric-label">현재가</div>
              <div class="metric-value">${Number.isFinite(h.current_price) ? fmtNum(h.current_price) : "-"}</div>
            </div>
            <div class="metric">
              <div class="metric-label">평균단가</div>
              <div class="metric-value">${Number.isFinite(h.avg_buy_price) ? fmtNum(h.avg_buy_price) : "-"}</div>
            </div>
            <div class="metric">
              <div class="metric-label">평가금액</div>
              <div class="metric-value">${Number.isFinite(currentValue) ? fmtWonPlain(currentValue) : "-"}</div>
            </div>
            <div class="metric">
              <div class="metric-label">비중</div>
              <div class="metric-value">${Number.isFinite(weightPct) ? fmtPct(weightPct) : "-"}</div>
            </div>
          </div>
          <div class="holding-side">
            <div class="priority-bar">
              <div class="priority-head">
                <span>매도 우선순위</span>
                <strong class="priority-score">${Number.isFinite(priority) ? fmtNum(priority) : "-"}</strong>
              </div>
              <div class="priority-fill">
                <div class="priority-fill-inner" style="width:${barPct}%;background:${barColor};"></div>
              </div>
              <div class="priority-segments" aria-hidden="true">
                <span>보유</span><span>점검</span><span>매도검토</span>
              </div>
            </div>
            <a class="detail-link" href="/detail.html?code=${esc(h.code)}" onclick="event.stopPropagation()">상세 보기</a>
          </div>
        </div>
        <div class="holding-reasons">${reasonsHtml}</div>
        ${buildCompactDecisionSummary(h, pct, priorityValue)}
        ${h.system_action_note ? `<div class="holding-action">${esc(h.system_action_note)}</div>` : ""}
      </div>`;
  }).join("");
}

/* ── 분석 탭: 수급 추이 ── */
function renderFlowHistory(flowData, holdingItems) {
  const el = document.getElementById("flowHistoryPanel");
  if (!el) return;

  const byCode = {};
  const rows = Array.isArray(flowData?.items) ? flowData.items : Array.isArray(flowData) ? flowData : [];
  rows.forEach((r) => {
    if (!byCode[r.code]) byCode[r.code] = [];
    byCode[r.code].push(r);
  });

  const codes = (holdingItems || []).map((h) => h.code);
  if (!codes.length) {
    el.innerHTML = `<div class="empty">보유 종목이 없습니다.</div>`;
    return;
  }

  const fmtShares = (v) => {
    const n = Number(v);
    if (!Number.isFinite(n)) return "-";
    if (Math.abs(n) >= 10000) return `${(n / 10000).toFixed(1)}만주`;
    return `${Math.round(n).toLocaleString()}주`;
  };

  const panels = codes.map((code) => {
    const holding = (holdingItems || []).find((h) => h.code === code);
    const name = holding?.name || code;
    const history = (byCode[code] || []).sort((a, b) => (a.date > b.date ? 1 : -1)).slice(-20);

    if (!history.length) {
      return `<div style="margin-bottom:16px;padding:10px;border:1px dashed var(--border);border-radius:8px;font-size:12px;color:var(--color-text-secondary);">${esc(name)} (${esc(code)}) — 수급 데이터 없음</div>`;
    }

    const latest = history[history.length - 1];
    const fNet5 = Number(latest?.flow_foreign_net_5d);
    const iNet5 = Number(latest?.flow_inst_net_5d);

    const rowsHtml = history.map((r) => {
      const fVal = Number(r.flow_foreign_net_5d);
      const iVal = Number(r.flow_inst_net_5d);
      return `<div style="display:flex;gap:8px;font-size:12px;padding:3px 0;border-bottom:1px solid rgba(30,41,59,.4);">
        <span style="width:88px;color:var(--color-text-secondary);">${esc(r.date)}</span>
        <span style="width:80px;text-align:right;color:${fVal >= 0 ? "#4ade80" : "#f87171"};">${fmtShares(fVal)}</span>
        <span style="width:80px;text-align:right;color:${iVal >= 0 ? "#60a5fa" : "#fbbf24"};">${fmtShares(iVal)}</span>
      </div>`;
    }).join("");

    return `<div style="margin-bottom:20px;">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;">
        <strong style="font-size:13px;">${esc(code)} ${esc(name)}</strong>
        <span style="font-size:11px;color:var(--color-text-secondary);">
          5일 외국인 <span style="color:${fNet5 >= 0 ? "#4ade80" : "#f87171"}">${fmtShares(fNet5)}</span>
          · 기관 <span style="color:${iNet5 >= 0 ? "#60a5fa" : "#fbbf24"}">${fmtShares(iNet5)}</span>
        </span>
      </div>
      <div style="font-size:10px;color:var(--color-text-secondary);margin-bottom:4px;">날짜 / 외국인(초록=매수·빨강=매도) / 기관(파랑=매수·노랑=매도)</div>
      ${rowsHtml}
    </div>`;
  }).join("");

  el.innerHTML = panels || `<div class="empty">수급 데이터 없음</div>`;
}

/* ── 분석 탭: 레짐 히스토리 ── */
function renderRegimeHistory(regimeData) {
  const el = document.getElementById("regimeHistoryPanel");
  if (!el) return;
  const items = Array.isArray(regimeData?.items) ? regimeData.items
    : Array.isArray(regimeData) ? regimeData : [];
  if (!items.length) {
    el.innerHTML = `<div class="empty">레짐 히스토리 데이터가 없습니다.</div>`;
    return;
  }
  const regimeColor = (r) =>
    r === "RISK_ON" ? "#4ade80" : r === "RISK_OFF" ? "#f87171" : "#94a3b8";
  const rows = items.slice(0, 30).map((r) => {
    const col = regimeColor(r.macro_status);
    return `<div style="display:flex;align-items:center;gap:10px;padding:4px 0;border-bottom:1px solid rgba(30,41,59,.5);font-size:13px;">
      <span style="width:90px;color:var(--color-text-secondary);">${esc(r.date)}</span>
      <span style="font-weight:700;color:${col};">${esc(r.macro_status)}</span>
      <span style="color:var(--color-text-secondary);font-size:11px;">${fmtNum(r.stock_count)}종목 적용</span>
    </div>`;
  }).join("");
  el.innerHTML = rows;
}

/* ── 메인 로드 ── */
async function loadDashboard() {
  const bar = document.getElementById("statusBar");
  bar.textContent = "데이터 불러오는 중…";

  try {
    // KIS RULE 계좌(KIS_RULE_CANO) 잔고 기반 보유종목
    // 종가 배치 후 sync_rule_account_holdings.py 가 data/rule_account_holdings.csv 를 갱신함
    const holdingsData = await fetchJson("/api/rule-holdings");

    renderBanner(holdingsData);
    renderHero(holdingsData);
    renderDecisionSummary(holdingsData);
    renderOverviewPnl(holdingsData);
    renderOverviewStatus(holdingsData);
    renderOverviewAlerts(holdingsData);
    renderAccountSummaryBar(holdingsData);
    renderHoldingsList(holdingsData);

    const holdingItems = holdingsData.items || [];
    const codes = holdingItems.map((h) => h.code).join(",");

    if (codes) {
      fetchJson(`/api/flow-history?codes=${codes}&days=20`)
        .then((fd) => renderFlowHistory(fd, holdingItems))
        .catch(() => renderFlowHistory(null, holdingItems));
    } else {
      renderFlowHistory(null, holdingItems);
    }

    fetchJson("/api/regime-history?days=30")
      .then((rd) => renderRegimeHistory(rd))
      .catch(() => renderRegimeHistory(null));

    const count = holdingsData.count || 0;
    const total_value = holdingsData.total_value || 0;
    const total_cost = holdingsData.total_cost || 0;
    const pnlPct = total_cost > 0 ? ((total_value - total_cost) / total_cost * 100).toFixed(2) : "-";
    bar.textContent = `보유 ${count}종목 · 평가금액 ${fmtWon(total_value)} · 손익률 ${pnlPct}%`;

  } catch (err) {
    console.error(err);
    document.getElementById("statusBar").textContent = `데이터 로드 실패: ${err.message}`;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  loadDashboard().catch(console.error);
});
