const state = {
  rows: [],
  tradeDate: null,
  totalCount: 0,
  gradeCounts: {},
  activeGrade: "",
  loading: false,
};

function esc(v) {
  return String(v ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function fmt(v, digits = 1) {
  if (v == null || !Number.isFinite(Number(v))) return "—";
  return Number(v).toFixed(digits);
}

function fmtMarketCap(v) {
  if (v == null || !Number.isFinite(Number(v))) return "—";
  const n = Number(v);
  if (n >= 1e12) return (n / 1e12).toFixed(1) + "T";
  if (n >= 1e9) return (n / 1e9).toFixed(1) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  return n.toLocaleString();
}

function gradeClass(g) {
  const map = { STRONG_BUY: "grade-STRONG_BUY", BUY: "grade-BUY", WATCH: "grade-WATCH", HOLD: "grade-HOLD", EXCLUDE: "grade-EXCLUDE" };
  return map[g] || "grade-EXCLUDE";
}

function scoreBarColor(score) {
  if (score >= 80) return "green";
  if (score >= 60) return "";
  return "amber";
}

function renderTable(rows) {
  const tbody = document.getElementById("rankBody");
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="13" class="empty">해당 조건의 종목이 없습니다.</td></tr>';
    return;
  }

  const html = rows.map((r, idx) => {
    const score = r.total_score ?? 0;
    const pct = Math.min(100, Math.max(0, score / 100 * 100));
    const barColor = scoreBarColor(score);
    const riskVal = r.risk_score ?? 0;
    const detailId = `detail-${idx}`;

    const mainRow = `
      <tr>
        <td><button class="expand-btn" onclick="toggleDetail('${detailId}')" title="상세">▸</button></td>
        <td class="rank">${r.rank_no ?? "—"}</td>
        <td class="symbol">${esc(r.symbol)}</td>
        <td class="name" title="${esc(r.company_name)}">${esc(r.company_name)}</td>
        <td class="sector" title="${esc(r.sector)}">${esc(r.sector || "—")}</td>
        <td><span class="grade ${gradeClass(r.grade)}">${esc(r.grade || "—")}</span></td>
        <td class="score">
          <div class="score-bar">
            <span>${fmt(score)}</span>
            <div class="score-bar-track"><div class="score-bar-fill ${barColor}" style="width:${pct}%"></div></div>
          </div>
        </td>
        <td class="score-sub">${fmt(r.momentum_score)}</td>
        <td class="score-sub">${fmt(r.rs_score)}</td>
        <td class="score-sub">${fmt(r.fundamental_score)}</td>
        <td class="score-sub">${fmt(r.growth_score)}</td>
        <td class="score-sub">${fmt(r.valuation_score)}</td>
        <td class="${riskVal < 0 ? "score-neg" : "score-sub"}">${fmt(riskVal)}</td>
      </tr>`;

    const detailRow = `
      <tr id="${detailId}" class="detail-row">
        <td colspan="13" class="detail-cell">
          <div style="font-size:12px;color:var(--muted);margin-bottom:6px;">${esc(r.reason_summary || "")}</div>
          <div class="detail-grid">
            <div class="detail-item"><div class="lbl">Market Cap</div><div class="val" style="font-size:14px;">${fmtMarketCap(r.market_cap)}</div></div>
            <div class="detail-item"><div class="lbl">Industry</div><div class="val" style="font-size:12px;line-height:1.3;">${esc(r.industry || "—")}</div></div>
            <div class="detail-item"><div class="lbl">Momentum</div><div class="val">${fmt(r.momentum_score)}</div></div>
            <div class="detail-item"><div class="lbl">Rel. Strength</div><div class="val">${fmt(r.rs_score)}</div></div>
            <div class="detail-item"><div class="lbl">Fundamental</div><div class="val">${fmt(r.fundamental_score)}</div></div>
            <div class="detail-item"><div class="lbl">Growth</div><div class="val">${fmt(r.growth_score)}</div></div>
            <div class="detail-item"><div class="lbl">Valuation</div><div class="val">${fmt(r.valuation_score)}</div></div>
            <div class="detail-item"><div class="lbl">Risk</div><div class="val" style="color:${riskVal < 0 ? "var(--red)" : "inherit"}">${fmt(riskVal)}</div></div>
            <div class="detail-item"><div class="lbl">Data Quality</div><div class="val" style="font-size:14px;">${fmt(r.feature_quality_score)}</div></div>
            <div class="detail-item"><div class="lbl">Data Status</div><div class="val" style="font-size:12px;">${esc(r.data_status || "—")}</div></div>
          </div>
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
  const btn = row.previousElementSibling?.querySelector(".expand-btn");
  if (btn) btn.textContent = isOpen ? "▸" : "▾";
}

function updateHero() {
  const d = document.getElementById("statDate");
  const t = document.getElementById("statTotal");
  const sb = document.getElementById("statSB");
  const buy = document.getElementById("statBuy");

  if (d) d.textContent = state.tradeDate || "—";
  if (t) t.textContent = (state.totalCount || 0).toLocaleString();
  if (sb) sb.textContent = (state.gradeCounts["STRONG_BUY"] || 0).toString();
  if (buy) buy.textContent = (state.gradeCounts["BUY"] || 0).toString();
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
  document.getElementById("rankBody").innerHTML = '<tr><td colspan="13" class="empty">로딩 중…</td></tr>';

  const { date, sector, topN, grade } = getFilters();
  const params = new URLSearchParams();
  if (date) params.set("date", date);
  if (sector) params.set("sector", sector);
  if (grade) params.set("grade", grade);
  params.set("top_n", topN);

  try {
    const res = await fetch("/api/us/ranking?" + params.toString());
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || res.statusText);
    }
    const data = await res.json();
    state.rows = data.rows || [];
    state.tradeDate = data.trade_date || null;
    state.totalCount = data.total_count || 0;
    state.gradeCounts = data.grade_counts || {};
    updateHero();
    renderTable(state.rows);
  } catch (e) {
    const box = document.getElementById("errorBox");
    box.textContent = "데이터 로드 실패: " + e.message;
    box.style.display = "block";
    document.getElementById("rankBody").innerHTML = '<tr><td colspan="13" class="empty">—</td></tr>';
  } finally {
    state.loading = false;
  }
}

async function loadDates() {
  try {
    const res = await fetch("/api/us/ranking/dates");
    if (!res.ok) return;
    const data = await res.json();
    const select = document.getElementById("filterDate");
    if (!select) return;
    select.innerHTML = '<option value="">최신 날짜</option>' +
      (data.dates || []).map((d) => `<option value="${esc(d)}">${esc(d)}</option>`).join("");
  } catch {}
}

async function loadSectors() {
  try {
    const res = await fetch("/api/us/ranking/sectors");
    if (!res.ok) return;
    const data = await res.json();
    const select = document.getElementById("filterSector");
    if (!select) return;
    (data.sectors || []).forEach((s) => {
      const opt = document.createElement("option");
      opt.value = s;
      opt.textContent = s;
      select.appendChild(opt);
    });
  } catch {}
}

function initGradePills() {
  document.getElementById("gradePills")?.addEventListener("click", (e) => {
    const pill = e.target.closest(".grade-pill");
    if (!pill) return;
    document.querySelectorAll(".grade-pill").forEach((p) => p.classList.remove("active"));
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
