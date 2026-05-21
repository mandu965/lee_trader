const state = {
  rows: [],
  tradeDate: null,
  totalCount: 0,
  gradeCounts: {},
  dataStatusCounts: {},
  okCount: 0,
  partialCount: 0,
  excludeCount: 0,
  source: "",
  expectedSource: "",
  sourceFallbackUsed: false,
  availableSources: [],
  universeLabel: "",
  activeGrade: "",
  loading: false,
};

function esc(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function fmt(value, digits = 1) {
  if (value == null || !Number.isFinite(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function fmtPct(value, digits = 1) {
  if (value == null || !Number.isFinite(Number(value))) return "-";
  const n = Number(value) * 100;
  return (n >= 0 ? "+" : "") + n.toFixed(digits) + "%";
}

function fmtProb(value) {
  if (value == null || !Number.isFinite(Number(value))) return "-";
  return (Number(value) * 100).toFixed(1) + "%";
}

function fmtMarketCap(value) {
  if (value == null || !Number.isFinite(Number(value))) return "-";
  const n = Number(value);
  if (n >= 1e12) return `${(n / 1e12).toFixed(1)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  return n.toLocaleString();
}

function gradeClass(grade) {
  const map = {
    STRONG_BUY: "grade-STRONG_BUY",
    BUY: "grade-BUY",
    WATCH: "grade-WATCH",
    HOLD: "grade-HOLD",
    EXCLUDE: "grade-EXCLUDE",
  };
  return map[grade] || "grade-EXCLUDE";
}

function scoreBarColor(score) {
  if (score >= 80) return "green";
  if (score >= 60) return "";
  return "amber";
}

function isMlSource() {
  return Boolean(state.source && state.source.startsWith("ml_"));
}

function getColCount() {
  return isMlSource() ? 11 : 13;
}

function updateTableHeaders() {
  const thead = document.querySelector("#rankTable thead tr");
  if (!thead) return;
  if (isMlSource()) {
    thead.innerHTML = `
      <th></th>
      <th>Rank</th>
      <th>Symbol</th>
      <th>Company</th>
      <th>Sector</th>
      <th>Grade</th>
      <th>ML Score</th>
      <th>Pred 20d</th>
      <th>Pred 60d</th>
      <th>P(Top20/20d)</th>
      <th>P(Top20/60d)</th>
    `;
  } else {
    thead.innerHTML = `
      <th></th>
      <th>Rank</th>
      <th>Symbol</th>
      <th>Company</th>
      <th>Sector</th>
      <th>Grade</th>
      <th>Total</th>
      <th>Mom</th>
      <th>RS</th>
      <th>Fund</th>
      <th>Growth</th>
      <th>Val</th>
      <th>Risk</th>
    `;
  }
}

function ensureHealthBox() {
  let box = document.getElementById("healthBox");
  if (box) return box;
  const errorBox = document.getElementById("errorBox");
  if (!errorBox || !errorBox.parentElement) return null;
  box = document.createElement("div");
  box.id = "healthBox";
  box.className = "error-msg";
  box.style.display = "none";
  errorBox.parentElement.insertBefore(box, errorBox);
  return box;
}

function renderTable(rows) {
  const tbody = document.getElementById("rankBody");
  if (!tbody) return;
  const cols = getColCount();

  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="${cols}" class="empty">해당 조건의 종목이 없습니다.</td></tr>`;
    return;
  }

  const ml = isMlSource();

  const html = rows.map((row, index) => {
    const score = row.total_score ?? 0;
    const pct = Math.min(100, Math.max(0, score));
    const barColor = scoreBarColor(score);
    const detailId = `detail-${index}`;

    const detail = (() => {
      if (!row.score_detail) return {};
      if (typeof row.score_detail === "string") {
        try { return JSON.parse(row.score_detail); } catch { return {}; }
      }
      return row.score_detail;
    })();

    let scoreCells;
    if (ml) {
      const mlScore = detail.ml_score ?? score;
      const pred20 = detail.pred_ret_20d;
      const pred60 = detail.pred_ret_60d;
      const prob20 = detail.prob_top20_20d;
      const prob60 = detail.prob_top20_60d;
      const mlPct = Math.min(100, Math.max(0, mlScore));
      const mlColor = scoreBarColor(mlScore);

      const pred20Color = pred20 == null ? "" : (Number(pred20) >= 0 ? "color:var(--green)" : "color:var(--red)");
      const pred60Color = pred60 == null ? "" : (Number(pred60) >= 0 ? "color:var(--green)" : "color:var(--red)");

      scoreCells = `
        <td class="score">
          <div class="score-bar">
            <span>${fmt(mlScore)}</span>
            <div class="score-bar-track"><div class="score-bar-fill ${mlColor}" style="width:${mlPct}%"></div></div>
          </div>
        </td>
        <td class="score-sub" style="${pred20Color}">${fmtPct(pred20)}</td>
        <td class="score-sub" style="${pred60Color}">${fmtPct(pred60)}</td>
        <td class="score-sub">${fmtProb(prob20)}</td>
        <td class="score-sub">${fmtProb(prob60)}</td>`;
    } else {
      const riskVal = row.risk_score ?? 0;
      scoreCells = `
        <td class="score">
          <div class="score-bar">
            <span>${fmt(score)}</span>
            <div class="score-bar-track"><div class="score-bar-fill ${barColor}" style="width:${pct}%"></div></div>
          </div>
        </td>
        <td class="score-sub">${fmt(row.momentum_score)}</td>
        <td class="score-sub">${fmt(row.rs_score)}</td>
        <td class="score-sub">${fmt(row.fundamental_score)}</td>
        <td class="score-sub">${fmt(row.growth_score)}</td>
        <td class="score-sub">${fmt(row.valuation_score)}</td>
        <td class="${riskVal < 0 ? "score-neg" : "score-sub"}">${fmt(riskVal)}</td>`;
    }

    const mainRow = `
      <tr>
        <td><button class="expand-btn" onclick="toggleDetail('${detailId}')" title="상세">▶</button></td>
        <td class="rank">${row.rank_no ?? "-"}</td>
        <td class="symbol">${esc(row.symbol)}</td>
        <td class="name" title="${esc(row.company_name)}">${esc(row.company_name)}</td>
        <td class="sector" title="${esc(row.sector || "-")}">${esc(row.sector || "-")}</td>
        <td><span class="grade ${gradeClass(row.grade)}">${esc(row.grade || "-")}</span></td>
        ${scoreCells}
      </tr>`;

    let detailItems;
    if (ml) {
      const mlScore = detail.ml_score ?? score;
      const pred20 = detail.pred_ret_20d;
      const pred60 = detail.pred_ret_60d;
      const prob20 = detail.prob_top20_20d;
      const prob60 = detail.prob_top20_60d;
      const pred20Color = pred20 != null && Number(pred20) < 0 ? "color:var(--red)" : "color:var(--green)";
      const pred60Color = pred60 != null && Number(pred60) < 0 ? "color:var(--red)" : "color:var(--green)";
      detailItems = `
        <div class="detail-item"><div class="lbl">Source</div><div class="val" style="font-size:14px;">${esc(row.source || "-")}</div></div>
        <div class="detail-item"><div class="lbl">Universe</div><div class="val" style="font-size:12px;line-height:1.3;">${esc(row.universe_group || "-")}</div></div>
        <div class="detail-item"><div class="lbl">Data Status</div><div class="val" style="font-size:12px;line-height:1.3;">${esc(row.data_status || "-")}</div></div>
        <div class="detail-item"><div class="lbl">Market Cap</div><div class="val" style="font-size:14px;">${fmtMarketCap(row.market_cap)}</div></div>
        <div class="detail-item"><div class="lbl">Industry</div><div class="val" style="font-size:12px;line-height:1.3;">${esc(row.industry || "-")}</div></div>
        <div class="detail-item"><div class="lbl">ML Score</div><div class="val">${fmt(mlScore)}</div></div>
        <div class="detail-item"><div class="lbl">Pred Ret 20d</div><div class="val" style="${pred20Color}">${fmtPct(pred20)}</div></div>
        <div class="detail-item"><div class="lbl">Pred Ret 60d</div><div class="val" style="${pred60Color}">${fmtPct(pred60)}</div></div>
        <div class="detail-item"><div class="lbl">P(Top20/20d)</div><div class="val">${fmtProb(prob20)}</div></div>
        <div class="detail-item"><div class="lbl">P(Top20/60d)</div><div class="val">${fmtProb(prob60)}</div></div>
        <div class="detail-item"><div class="lbl">Data Quality</div><div class="val" style="font-size:14px;">${fmt(row.feature_quality_score)}</div></div>
        <div class="detail-item"><div class="lbl">Exclude Reason</div><div class="val" style="font-size:12px;line-height:1.3;">${esc(row.exclude_reason || "-")}</div></div>`;
    } else {
      const riskVal = row.risk_score ?? 0;
      detailItems = `
        <div class="detail-item"><div class="lbl">Source</div><div class="val" style="font-size:14px;">${esc(row.source || "-")}</div></div>
        <div class="detail-item"><div class="lbl">Universe</div><div class="val" style="font-size:12px;line-height:1.3;">${esc(row.universe_group || "-")}</div></div>
        <div class="detail-item"><div class="lbl">Data Status</div><div class="val" style="font-size:12px;line-height:1.3;">${esc(row.data_status || "-")}</div></div>
        <div class="detail-item"><div class="lbl">Market Cap</div><div class="val" style="font-size:14px;">${fmtMarketCap(row.market_cap)}</div></div>
        <div class="detail-item"><div class="lbl">Industry</div><div class="val" style="font-size:12px;line-height:1.3;">${esc(row.industry || "-")}</div></div>
        <div class="detail-item"><div class="lbl">Momentum</div><div class="val">${fmt(row.momentum_score)}</div></div>
        <div class="detail-item"><div class="lbl">Rel. Strength</div><div class="val">${fmt(row.rs_score)}</div></div>
        <div class="detail-item"><div class="lbl">Fundamental</div><div class="val">${fmt(row.fundamental_score)}</div></div>
        <div class="detail-item"><div class="lbl">Growth</div><div class="val">${fmt(row.growth_score)}</div></div>
        <div class="detail-item"><div class="lbl">Valuation</div><div class="val">${fmt(row.valuation_score)}</div></div>
        <div class="detail-item"><div class="lbl">Risk</div><div class="val" style="color:${riskVal < 0 ? "var(--red)" : "inherit"}">${fmt(riskVal)}</div></div>
        <div class="detail-item"><div class="lbl">Data Quality</div><div class="val" style="font-size:14px;">${fmt(row.feature_quality_score)}</div></div>
        <div class="detail-item"><div class="lbl">Exclude Reason</div><div class="val" style="font-size:12px;line-height:1.3;">${esc(row.exclude_reason || "-")}</div></div>`;
    }

    const detailRow = `
      <tr id="${detailId}" class="detail-row">
        <td colspan="${cols}" class="detail-cell">
          <div style="font-size:12px;color:var(--muted);margin-bottom:6px;">${esc(row.reason_summary || "")}</div>
          <div class="detail-grid">${detailItems}</div>
        </td>
      </tr>`;

    return mainRow + detailRow;
  }).join("");

  tbody.innerHTML = html;
}

function toggleDetail(id) {
  const row = document.getElementById(id);
  if (!row) return;
  const isOpen = row.classList.contains("is-open");
  row.classList.toggle("is-open", !isOpen);
  const button = row.previousElementSibling?.querySelector(".expand-btn");
  if (button) button.textContent = isOpen ? "▶" : "▼";
}

function renderHealthSummary() {
  const eyebrow = document.querySelector(".eyebrow");
  const heroSub = document.getElementById("heroSub");
  const modeLabel = document.getElementById("modeLabel");
  const healthBox = ensureHealthBox();
  const statSBLabel = document.getElementById("statSBLabel");
  const statBuyLabel = document.getElementById("statBuyLabel");

  const ml = isMlSource();

  if (eyebrow) {
    eyebrow.textContent = `US Stock Module · ${state.source || "unknown"}`;
  }

  if (heroSub) {
    const universe = state.universeLabel || "US universe";
    if (ml) {
      heroSub.textContent = `${universe} 기준 AI 모델 예측 점수. OK ${state.okCount} · partial ${state.partialCount} · exclude ${state.excludeCount}`;
    } else {
      heroSub.textContent = `${universe} 기준 모멘텀·상대강도·펀더멘털 종합 점수. OK ${state.okCount} · partial ${state.partialCount} · exclude ${state.excludeCount}`;
    }
  }

  if (modeLabel) {
    const parts = [];
    if (state.source) parts.push(`active=${state.source}`);
    if (state.expectedSource) parts.push(`expected=${state.expectedSource}`);
    modeLabel.textContent = parts.join(" · ") || "랭킹 메타 없음";
  }

  if (statSBLabel) {
    statSBLabel.textContent = ml ? "상위 20% (20d 예측)" : "80점 이상";
  }
  if (statBuyLabel) {
    statBuyLabel.textContent = ml ? "상위 20% (60d 예측)" : "70~79점";
  }

  if (!healthBox) return;

  const messages = [];
  if (state.sourceFallbackUsed && state.expectedSource && state.source && state.expectedSource !== state.source) {
    messages.push(`기대 source ${state.expectedSource}를 찾지 못해 ${state.source}를 사용 중입니다.`);
  }
  if (state.totalCount > 0 && state.okCount < state.totalCount) {
    messages.push(`전체 ${state.totalCount}개 중 완전 데이터(OK)는 ${state.okCount}개입니다.`);
  }
  const missingPrice = state.dataStatusCounts.MISSING_PRICE_FEATURE || 0;
  if (missingPrice > 0) {
    messages.push(`MISSING_PRICE_FEATURE ${missingPrice}개가 포함돼 있습니다.`);
  }

  if (!messages.length) {
    healthBox.style.display = "none";
    healthBox.textContent = "";
    return;
  }

  healthBox.style.display = "block";
  healthBox.textContent = messages.join(" ");
}

function updateHero() {
  const statDate = document.getElementById("statDate");
  const statTotal = document.getElementById("statTotal");
  const statStrongBuy = document.getElementById("statSB");
  const statBuy = document.getElementById("statBuy");

  if (statDate) statDate.textContent = state.tradeDate || "-";
  if (statTotal) statTotal.textContent = (state.totalCount || 0).toLocaleString();
  if (statStrongBuy) statStrongBuy.textContent = String(state.gradeCounts.STRONG_BUY || 0);
  if (statBuy) statBuy.textContent = String(state.gradeCounts.BUY || 0);
  updateTableHeaders();
  renderHealthSummary();
}

function getFilters() {
  return {
    date: document.getElementById("filterDate")?.value || "",
    sector: document.getElementById("filterSector")?.value || "",
    topN: document.getElementById("filterTopN")?.value || "50",
    grade: state.activeGrade,
  };
}

async function loadRanking() {
  if (state.loading) return;
  state.loading = true;

  const errorBox = document.getElementById("errorBox");
  if (errorBox) errorBox.style.display = "none";

  const tbody = document.getElementById("rankBody");
  if (tbody) {
    const cols = getColCount();
    tbody.innerHTML = `<tr><td colspan="${cols}" class="empty">로딩 중입니다.</td></tr>`;
  }

  const { date, sector, topN, grade } = getFilters();
  const params = new URLSearchParams();
  if (date) params.set("date", date);
  if (sector) params.set("sector", sector);
  if (grade) params.set("grade", grade);
  params.set("top_n", topN);

  try {
    const response = await fetch(`/api/us/ranking?${params.toString()}`);
    if (!response.ok) {
      const errorPayload = await response.json().catch(() => ({ error: response.statusText }));
      throw new Error(errorPayload.error || response.statusText);
    }

    const data = await response.json();
    state.rows = data.rows || [];
    state.tradeDate = data.trade_date || null;
    state.totalCount = data.total_count || 0;
    state.gradeCounts = data.grade_counts || {};
    state.dataStatusCounts = data.data_status_counts || {};
    state.okCount = data.ok_count || 0;
    state.partialCount = data.partial_count || 0;
    state.excludeCount = data.exclude_count || 0;
    state.source = data.source || "";
    state.expectedSource = data.expected_source || "";
    state.sourceFallbackUsed = Boolean(data.source_fallback_used);
    state.availableSources = data.available_sources || [];
    state.universeLabel = data.universe_label || "";

    updateHero();
    renderTable(state.rows);
  } catch (error) {
    if (errorBox) {
      errorBox.textContent = `데이터 로드 실패: ${error.message}`;
      errorBox.style.display = "block";
    }
    if (tbody) {
      const cols = getColCount();
      tbody.innerHTML = `<tr><td colspan="${cols}" class="empty">데이터를 불러오지 못했습니다.</td></tr>`;
    }
  } finally {
    state.loading = false;
  }
}

async function loadDates() {
  try {
    const response = await fetch("/api/us/ranking/dates");
    if (!response.ok) return;
    const data = await response.json();
    const select = document.getElementById("filterDate");
    if (!select) return;
    select.innerHTML =
      '<option value="">최신 날짜</option>' +
      (data.dates || []).map((date) => `<option value="${esc(date)}">${esc(date)}</option>`).join("");
  } catch {}
}

async function loadSectors() {
  try {
    const response = await fetch("/api/us/ranking/sectors");
    if (!response.ok) return;
    const data = await response.json();
    const select = document.getElementById("filterSector");
    if (!select) return;
    (data.sectors || []).forEach((sector) => {
      const option = document.createElement("option");
      option.value = sector;
      option.textContent = sector;
      select.appendChild(option);
    });
  } catch {}
}

function initGradePills() {
  const root = document.getElementById("gradePills");
  if (!root) return;
  root.addEventListener("click", (event) => {
    const pill = event.target.closest(".grade-pill");
    if (!pill) return;
    document.querySelectorAll(".grade-pill").forEach((item) => item.classList.remove("active"));
    pill.classList.add("active");
    state.activeGrade = pill.dataset.grade || "";
    loadRanking();
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  ensureHealthBox();
  await Promise.all([loadDates(), loadSectors()]);
  initGradePills();

  document.getElementById("btnRefresh")?.addEventListener("click", loadRanking);
  document.getElementById("filterDate")?.addEventListener("change", loadRanking);
  document.getElementById("filterSector")?.addEventListener("change", loadRanking);
  document.getElementById("filterTopN")?.addEventListener("change", loadRanking);

  loadRanking();
});
