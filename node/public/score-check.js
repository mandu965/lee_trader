const fmtNum = (value, digits = 0) => {
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  return n.toLocaleString("ko-KR", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
};

const fmtSigned = (value, digits = 4) => {
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  return `${n > 0 ? "+" : ""}${n.toFixed(digits)}`;
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

function chipClassByDiff(value) {
  const n = Math.abs(Number(value));
  if (!Number.isFinite(n)) return "info";
  if (n <= 0.01) return "good";
  if (n <= 0.05) return "watch";
  return "bad";
}

function diffClass(value) {
  const cls = chipClassByDiff(value);
  if (cls === "good") return "diff-good";
  if (cls === "watch") return "diff-watch";
  return "diff-bad";
}

function renderHero(summary, date) {
  const heroGrid = document.getElementById("heroGrid");
  heroGrid.innerHTML = `
    <article class="card">
      <div class="eyebrow">기준일</div>
      <div class="big-value">${escapeHtml(date || "-")}</div>
      <div class="muted">현재 보고 있는 daily_ranking 기준일</div>
    </article>
    <article class="card">
      <div class="eyebrow">종목 수</div>
      <div class="big-value">${fmtNum(summary.row_count)}</div>
      <div class="muted">최신 산출 행 수</div>
    </article>
    <article class="card">
      <div class="eyebrow">Top 20</div>
      <div class="big-value">${fmtNum(summary.top20_count)}</div>
      <div class="muted">랭킹 상위 점검 대상</div>
    </article>
    <article class="card">
      <div class="eyebrow">최대 오차</div>
      <div class="big-value ${diffClass(summary.max_abs_diff)}">${fmtNum(summary.max_abs_diff, 4)}</div>
      <div class="muted">저장값 vs 재계산값 절대차</div>
    </article>
    <article class="card">
      <div class="eyebrow">평균 오차</div>
      <div class="big-value ${diffClass(summary.mean_abs_diff)}">${fmtNum(summary.mean_abs_diff, 4)}</div>
      <div class="muted">전 종목 평균 절대차</div>
    </article>
    <article class="card">
      <div class="eyebrow">Fallback 행</div>
      <div class="big-value">${fmtNum(summary.rows_with_any_fallback)}</div>
      <div class="muted">핵심/보조 점수 대체 사용 행 수</div>
    </article>
  `;
}

function renderKv(summary) {
  const kv = document.getElementById("summaryKv");
  kv.innerHTML = [
    ["정확 일치 행", fmtNum(summary.exact_match_count)],
    ["오차 > 0.01", fmtNum(summary.diff_gt_001_count)],
    ["오차 > 0.05", fmtNum(summary.diff_gt_005_count)],
    ["live_score override", fmtNum(summary.rows_with_live_override)],
    ["핵심 점수 누락 행", fmtNum(summary.rows_with_missing_core_scores)],
    ["Top20 fallback 행", fmtNum(summary.top20_fallback_rows)],
  ].map(([label, value]) => `
    <div class="kv-row">
      <span class="muted">${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
    </div>
  `).join("");

  const chips = document.getElementById("summaryChips");
  chips.innerHTML = [
    { label: `max diff ${fmtNum(summary.max_abs_diff, 4)}`, kind: chipClassByDiff(summary.max_abs_diff) },
    { label: `mean diff ${fmtNum(summary.mean_abs_diff, 4)}`, kind: chipClassByDiff(summary.mean_abs_diff) },
    { label: `fallback ${fmtNum(summary.rows_with_any_fallback)}`, kind: summary.rows_with_any_fallback > 0 ? "watch" : "good" },
    { label: `missing ${fmtNum(summary.rows_with_missing_core_scores)}`, kind: summary.rows_with_missing_core_scores > 0 ? "bad" : "good" },
    { label: `live override ${fmtNum(summary.rows_with_live_override)}`, kind: summary.rows_with_live_override > 0 ? "info" : "good" },
  ].map((item) => `<span class="chip ${item.kind}">${escapeHtml(item.label)}</span>`).join("");
}

function renderNotes(summary) {
  const notes = [];
  if (Number(summary.max_abs_diff) <= 0.01) {
    notes.push("저장 final_score와 재계산값 차이가 거의 없습니다.");
  } else if (Number(summary.max_abs_diff) <= 0.05) {
    notes.push("재계산 차이가 작지만, 상위 오차 행은 한 번 확인하는 편이 안전합니다.");
  } else {
    notes.push("공식 재계산 차이가 큰 행이 있어 ranking_builder 후속 보정 또는 live score 경로를 확인해야 합니다.");
  }
  if (Number(summary.rows_with_any_fallback) > 0) {
    notes.push("fallback 사용 행이 있으므로 원천 점수 결측이 실제로 발생한 종목이 있습니다.");
  } else {
    notes.push("fallback 사용 행이 없어 핵심 점수는 비교적 안정적입니다.");
  }
  if (Number(summary.rows_with_live_override) > 0) {
    notes.push("운영 해석은 final_score가 아니라 live_score_source도 같이 봐야 합니다.");
  }
  if (Number(summary.top20_fallback_rows) > 0) {
    notes.push("Top20 안에 fallback 종목이 있어 상위권 해석이 왜곡될 수 있습니다.");
  }
  document.getElementById("checkNotes").innerHTML = notes.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
}

function renderCountChips(targetId, counts, formatter) {
  const el = document.getElementById(targetId);
  const entries = Object.entries(counts || {});
  if (!entries.length) {
    el.innerHTML = `<span class="chip info">데이터 없음</span>`;
    return;
  }
  el.innerHTML = entries
    .sort((a, b) => b[1] - a[1])
    .map(([key, value]) => `<span class="chip info">${escapeHtml(formatter ? formatter(key, value) : `${key} ${value}`)}</span>`)
    .join("");
}

function buildTable(columns, rows, emptyText) {
  if (!Array.isArray(rows) || !rows.length) {
    return `<div class="empty-state">${escapeHtml(emptyText)}</div>`;
  }
  return `
    <table>
      <thead>
        <tr>${columns.map((col) => `<th>${escapeHtml(col.label)}</th>`).join("")}</tr>
      </thead>
      <tbody>
        ${rows.map((row) => `
          <tr>
            ${columns.map((col) => `<td class="${escapeHtml(col.className || "")}">${col.render(row)}</td>`).join("")}
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function renderTables(payload) {
  const top20Columns = [
    { label: "순위", render: (row) => escapeHtml(fmtNum(row.rank_final)) },
    { label: "종목", render: (row) => `<strong>${escapeHtml(row.name || row.code || "-")}</strong><div class="muted mono">${escapeHtml(row.code || "-")}</div>` },
    { label: "final", render: (row) => escapeHtml(fmtNum(row.final_score, 2)) },
    { label: "recomputed", render: (row) => escapeHtml(fmtNum(row.recomputed_final_score, 2)) },
    { label: "diff", render: (row) => `<span class="${escapeHtml(diffClass(row.final_diff))} mono">${escapeHtml(fmtSigned(row.final_diff, 4))}</span>` },
    { label: "live", render: (row) => `${escapeHtml(fmtNum(row.live_score, 2))}<div class="muted mono">${escapeHtml(row.live_score_source || "-")}</div>` },
    { label: "regime", render: (row) => `${escapeHtml(row.regime || "-")}<div class="muted mono">${escapeHtml(row.weight_profile || "-")}</div>` },
    { label: "ret / prob", render: (row) => `${escapeHtml(fmtNum(row.ret_score, 1))} / ${escapeHtml(fmtNum(row.prob_score, 1))}` },
    { label: "tech / qual", render: (row) => `${escapeHtml(fmtNum(row.tech_score, 1))} / ${escapeHtml(fmtNum(row.qual_score, 1))}` },
    { label: "valuation / risk", render: (row) => `${escapeHtml(fmtNum(row.valuation_score, 1))} / ${escapeHtml(fmtNum(row.risk_penalty, 1))}` },
    { label: "fallback", render: (row) => escapeHtml((row.fallback_flags || []).join(", ") || "-") },
  ];
  document.getElementById("top20Table").innerHTML = buildTable(top20Columns, payload.top20, "Top20 데이터가 없습니다.");

  const diffColumns = [
    { label: "종목", render: (row) => `<strong>${escapeHtml(row.name || row.code || "-")}</strong><div class="muted mono">${escapeHtml(row.code || "-")}</div>` },
    { label: "순위", render: (row) => escapeHtml(fmtNum(row.rank_final)) },
    { label: "final", render: (row) => escapeHtml(fmtNum(row.final_score, 4)) },
    { label: "recomputed", render: (row) => escapeHtml(fmtNum(row.recomputed_final_score, 4)) },
    { label: "diff", render: (row) => `<span class="${escapeHtml(diffClass(row.final_diff))} mono">${escapeHtml(fmtSigned(row.final_diff, 4))}</span>` },
    { label: "source", render: (row) => escapeHtml(row.live_score_source || "-") },
  ];
  document.getElementById("diffTable").innerHTML = buildTable(diffColumns, payload.biggest_diffs, "오차 데이터가 없습니다.");

  const flaggedColumns = [
    { label: "종목", render: (row) => `<strong>${escapeHtml(row.name || row.code || "-")}</strong><div class="muted mono">${escapeHtml(row.code || "-")}</div>` },
    { label: "순위", render: (row) => escapeHtml(fmtNum(row.rank_final)) },
    { label: "diff", render: (row) => `<span class="${escapeHtml(diffClass(row.final_diff))} mono">${escapeHtml(fmtSigned(row.final_diff, 4))}</span>` },
    { label: "fallback", render: (row) => escapeHtml((row.fallback_flags || []).join(", ") || "-") },
    { label: "missing", render: (row) => escapeHtml((row.missing_flags || []).join(", ") || "-") },
    { label: "source", render: (row) => escapeHtml(row.live_score_source || "-") },
  ];
  document.getElementById("flaggedTable").innerHTML = buildTable(flaggedColumns, payload.flagged_rows, "경고 행이 없습니다.");
}

async function loadScoreCheck() {
  const statusText = document.getElementById("statusText");
  const emptyState = document.getElementById("emptyState");
  const dateValue = document.getElementById("dateInput").value;
  try {
    statusText.textContent = "불러오는 중...";
    emptyState.hidden = true;
    const params = new URLSearchParams();
    if (dateValue) params.set("date", dateValue);
    const payload = await fetchJson(`/api/score-check${params.toString() ? `?${params.toString()}` : ""}`);
    if (payload?.date) {
      document.getElementById("dateInput").value = payload.date;
    }
    renderHero(payload.summary || {}, payload.date);
    renderKv(payload.summary || {});
    renderNotes(payload.summary || {});
    renderCountChips("regimeChips", payload.regime_counts, (key, value) => `regime ${key} ${value}`);
    renderCountChips("formulaChips", payload.formula_counts, (key, value) => `${key} ${value}`);
    renderTables(payload);
    statusText.textContent = `기준일 ${payload.date || "-"} 로드 완료`;
  } catch (error) {
    console.error("score check load failed", error);
    statusText.textContent = `로드 실패: ${error.message}`;
    emptyState.hidden = false;
  }
}

document.getElementById("refreshBtn").addEventListener("click", () => {
  void loadScoreCheck();
});

document.getElementById("dateInput").addEventListener("change", () => {
  void loadScoreCheck();
});

void loadScoreCheck();
