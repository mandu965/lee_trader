const state = {
  rows: [],
  tradeDate: null,
  previousTradeDate: null,
  source: null,
  totalCount: 0,
  gradeCounts: {},
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

function fmtMarketCap(value) {
  if (value == null || !Number.isFinite(Number(value))) return "-";
  const number = Number(value);
  if (number >= 1e12) return `${(number / 1e12).toFixed(1)}T`;
  if (number >= 1e9) return `${(number / 1e9).toFixed(1)}B`;
  if (number >= 1e6) return `${(number / 1e6).toFixed(1)}M`;
  return number.toLocaleString();
}

function gradeClass(grade) {
  const mapping = {
    STRONG_BUY: "grade-STRONG_BUY",
    BUY: "grade-BUY",
    WATCH: "grade-WATCH",
    HOLD: "grade-HOLD",
    EXCLUDE: "grade-EXCLUDE",
  };
  return mapping[grade] || "grade-EXCLUDE";
}

function scoreBarColor(score) {
  if (score >= 80) return "green";
  if (score >= 60) return "";
  return "amber";
}

function buildTrendBadge(row) {
  const delta = Number(row?.rank_delta);
  const previousRank = Number(row?.previous_rank_no);
  if (!Number.isFinite(delta) || !Number.isFinite(previousRank)) {
    return '<span class="trend-badge flat">NEW</span>';
  }
  if (delta > 0) return `<span class="trend-badge up">▲${Math.abs(delta)}</span>`;
  if (delta < 0) return `<span class="trend-badge down">▼${Math.abs(delta)}</span>`;
  return '<span class="trend-badge flat">—</span>';
}

function renderTable(rows) {
  const tbody = document.getElementById("rankBody");
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="15" class="empty">조건에 맞는 종목이 없습니다.</td></tr>';
    return;
  }

  const html = rows
    .map((row, index) => {
      const score = row.total_score ?? 0;
      const pct = Math.min(100, Math.max(0, (score / 100) * 100));
      const barColor = scoreBarColor(score);
      const riskVal = row.risk_score ?? 0;
      const detailId = `detail-${index}`;
      const detail = row.score_detail && typeof row.score_detail === "object" ? row.score_detail : null;
      const probability = row.probability ?? detail?.probability ?? null;
      const probabilityText =
        probability != null && Number.isFinite(Number(probability))
          ? `${fmt(Number(probability) * 100, 1)}%`
          : "-";

      const mainRow = `
        <tr>
          <td><button class="expand-btn" onclick="toggleDetail('${detailId}')" title="상세">▶</button></td>
          <td class="rank">${row.rank_no ?? "-"}</td>
          <td class="trend">${buildTrendBadge(row)}</td>
          <td class="symbol">${esc(row.symbol)}</td>
          <td class="name" title="${esc(row.company_name)}">${esc(row.company_name)}</td>
          <td class="sector" title="${esc(row.sector)}">${esc(row.sector || "-")}</td>
          <td><span class="grade ${gradeClass(row.grade)}">${esc(row.grade || "-")}</span></td>
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
          <td class="${riskVal < 0 ? "score-neg" : "score-sub"}">${fmt(riskVal)}</td>
          <td class="score-sub">${probabilityText}</td>
        </tr>`;

      const detailRow = `
        <tr id="${detailId}" class="detail-row">
          <td colspan="15" class="detail-cell">
            <div style="font-size:12px;color:var(--muted);margin-bottom:6px;">${esc(row.reason_summary || "")}</div>
            <div class="detail-grid">
              <div class="detail-item"><div class="lbl">Previous Rank</div><div class="val">${row.previous_rank_no ?? "-"}</div></div>
              <div class="detail-item"><div class="lbl">Market Cap</div><div class="val" style="font-size:14px;">${fmtMarketCap(row.market_cap)}</div></div>
              <div class="detail-item"><div class="lbl">Industry</div><div class="val" style="font-size:12px;line-height:1.3;">${esc(row.industry || "-")}</div></div>
              <div class="detail-item"><div class="lbl">Probability</div><div class="val">${probabilityText}</div></div>
              <div class="detail-item"><div class="lbl">Momentum</div><div class="val">${fmt(row.momentum_score)}</div></div>
              <div class="detail-item"><div class="lbl">Rel. Strength</div><div class="val">${fmt(row.rs_score)}</div></div>
              <div class="detail-item"><div class="lbl">Fundamental</div><div class="val">${fmt(row.fundamental_score)}</div></div>
              <div class="detail-item"><div class="lbl">Growth</div><div class="val">${fmt(row.growth_score)}</div></div>
              <div class="detail-item"><div class="lbl">Valuation</div><div class="val">${fmt(row.valuation_score)}</div></div>
              <div class="detail-item"><div class="lbl">Risk</div><div class="val" style="color:${riskVal < 0 ? "var(--red)" : "inherit"}">${fmt(riskVal)}</div></div>
              <div class="detail-item"><div class="lbl">Data Quality</div><div class="val" style="font-size:14px;">${fmt(row.feature_quality_score)}</div></div>
              <div class="detail-item"><div class="lbl">Data Status</div><div class="val" style="font-size:12px;">${esc(row.data_status || "-")}</div></div>
            </div>
          </td>
        </tr>`;

      return mainRow + detailRow;
    })
    .join("");

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

function updateHero() {
  const tradeDateEl = document.getElementById("statDate");
  const totalEl = document.getElementById("statTotal");
  const strongBuyEl = document.getElementById("statSB");
  const buyEl = document.getElementById("statBuy");
  const modeLabelEl = document.getElementById("modeLabel");

  if (tradeDateEl) tradeDateEl.textContent = state.tradeDate || "-";
  if (totalEl) totalEl.textContent = (state.totalCount || 0).toLocaleString();
  if (strongBuyEl) strongBuyEl.textContent = String(state.gradeCounts.STRONG_BUY || 0);
  if (buyEl) buyEl.textContent = String(state.gradeCounts.BUY || 0);
  if (modeLabelEl) {
    const source = state.source || "unknown";
    const previousDate = state.previousTradeDate || "이전 동일 source 없음";
    modeLabelEl.textContent = `source=${source} · 비교 기준=${previousDate}`;
  }
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
  document.getElementById("errorBox").style.display = "none";
  document.getElementById("rankBody").innerHTML = '<tr><td colspan="15" class="empty">로딩 중...</td></tr>';

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
    state.previousTradeDate = data.previous_trade_date || null;
    state.source = data.source || null;
    state.totalCount = data.total_count || 0;
    state.gradeCounts = data.grade_counts || {};
    updateHero();
    renderTable(state.rows);
  } catch (error) {
    const box = document.getElementById("errorBox");
    box.textContent = `데이터 로드 실패: ${error.message}`;
    box.style.display = "block";
    document.getElementById("rankBody").innerHTML = '<tr><td colspan="15" class="empty">데이터를 불러오지 못했습니다.</td></tr>';
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
  } catch {
    // ignore
  }
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
  } catch {
    // ignore
  }
}

function initGradePills() {
  document.getElementById("gradePills")?.addEventListener("click", (event) => {
    const pill = event.target.closest(".grade-pill");
    if (!pill) return;
    document.querySelectorAll(".grade-pill").forEach((item) => item.classList.remove("active"));
    pill.classList.add("active");
    state.activeGrade = pill.dataset.grade || "";
    loadRanking();
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  await Promise.all([loadDates(), loadSectors()]);
  initGradePills();

  document.getElementById("btnRefresh")?.addEventListener("click", loadRanking);
  document.getElementById("filterDate")?.addEventListener("change", loadRanking);
  document.getElementById("filterSector")?.addEventListener("change", loadRanking);
  document.getElementById("filterTopN")?.addEventListener("change", loadRanking);

  loadRanking();
});
