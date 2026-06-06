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

function httpError(url, status) {
  const error = new Error(`${url} HTTP ${status}`);
  error.status = status;
  error.url = url;
  return error;
}

async function fetchJson(url) {
  const res = await fetch(url, { credentials: "same-origin" });
  if (!res.ok) throw httpError(url, res.status);
  return res.json();
}

async function fetchJsonMaybe(url) {
  const res = await fetch(url, { credentials: "same-origin" });
  if (res.status === 404) return null;
  if (!res.ok) throw httpError(url, res.status);
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
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  if (!res.ok) throw httpError(url, res.status);
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
    <div class="hero-card">
      <div class="hero-label">기준일</div>
      <div class="hero-value">${escapeHtml(data.asof_date || "-")}</div>
      <div class="hero-sub">${escapeHtml(basis.label || "기준 정보 없음")} · 랭킹 최신일 ${escapeHtml(outputs.ranking_latest_date || "-")}</div>
    </div>
    <div class="hero-card">
      <div class="hero-label">60일 준비 상태</div>
      <div class="hero-value">${escapeHtml(readiness.confidence_calibration_readiness_60d || "WAIT")}</div>
      <div class="hero-sub">60일 성숙 스냅샷 ${fmtNum(readiness.matured_snapshot_count_60d)}</div>
    </div>
    <div class="hero-card">
      <div class="hero-label">매수 gate</div>
      <div class="hero-value">${escapeHtml(gate.overall_status || "-")}</div>
      <div class="hero-sub">워크포워드 ${escapeHtml(gate.walkforward_acceptance || "-")}</div>
    </div>
    <div class="hero-card">
      <div class="hero-label">KPI</div>
      <div class="hero-value">${escapeHtml(kpi.overall_status || "-")}</div>
      <div class="hero-sub">경고 ${fmtNum(kpi.alert_metric_count)} / 관찰 ${fmtNum(kpi.watch_metric_count)}</div>
    </div>
  `;
}

function renderSafetyBanner(data) {
  const banner = document.getElementById("safetyBanner");
  const text = document.getElementById("safetyText");
  const detail = document.getElementById("safetyDetail");
  if (!banner) return;
  const decision = data.go_no_go?.decision || "WAIT";
  const gateStatus = data.gate?.overall_status || "-";
  const kpiStatus = data.kpi?.overall_status || "-";
  let cls, msg;
  if (decision === "GO") { cls = "ok"; msg = "오늘 매수 진행 가능 (GO)"; }
  else if (decision === "NO_GO") { cls = "bad"; msg = "오늘 매수 차단 (NO_GO)"; }
  else { cls = "warn"; msg = `대기 중 (${decision})`; }
  banner.className = `safety-banner ${cls}`;
  text.textContent = msg;
  detail.textContent = `Gate: ${gateStatus} · KPI: ${kpiStatus}`;
}

function updateBadgeToday(data) {
  const badge = document.getElementById("badgeToday");
  const summary = document.getElementById("todayQueueSummary");
  if (!badge) return;
  const kpi = data.kpi || {};
  const alertMetricCount = Number(kpi.alert_metric_count || 0);
  const watchMetricCount = Number(kpi.watch_metric_count || 0);
  const alertCount = alertMetricCount + watchMetricCount;
  const critCount = (data.interpretation?.critical_reasons || []).length;
  const checklistCount = (data.manual?.checklist || []).length + (data.manual?.intraday_summary?.headline ? 1 : 0);
  const total = alertCount + critCount;
  if (total > 0) {
    badge.textContent = total;
    badge.className = "tab-badge bad";
    badge.title = `핵심 사유 ${critCount}건 · KPI 경고 ${alertMetricCount}건 · KPI 관찰 ${watchMetricCount}건`;
  } else {
    badge.textContent = "";
    badge.className = "tab-badge";
    badge.title = "";
  }
  if (summary) {
    if (total > 0 || checklistCount > 0) {
      summary.textContent = `핵심 사유 ${critCount}건 · KPI 경고 ${alertMetricCount}건 · KPI 관찰 ${watchMetricCount}건 · 체크리스트 ${checklistCount}건`;
    } else {
      summary.textContent = "오늘 바로 확인할 핵심 사유나 KPI 경고가 없습니다.";
    }
  }
}

function renderTodayTopActions(data, runtime, liveDiagnostics, ruleOps, usTradingSummary) {
  const el = document.getElementById("todayTopActions");
  if (!el) return;

  const actions = [];
  const criticalReasons = data?.interpretation?.critical_reasons || [];
  const kpi = data?.kpi || {};
  const runtimePairs = [
    ["종가 배치", runtime?.close_scheduler],
    ["AI auto_buy", runtime?.auto_buy_scheduler],
    ["live sync", runtime?.live_account_sync_scheduler],
    ["시장 급락 감시", runtime?.market_guard_scheduler],
  ];
  const failedRuntime = runtimePairs.find(([, row]) => ["failed", "error"].includes(String(row?.status || "").toLowerCase()));
  if (failedRuntime) {
    actions.push({
      title: `${failedRuntime[0]} 실패 확인`,
      detail: runtimeIssueCell(failedRuntime[1]) || "최근 실패 이력을 확인하세요.",
      href: "/ops-readiness.html",
      cta: "스케줄 확인",
    });
  }

  if (criticalReasons.length || Number(kpi.alert_metric_count || 0) > 0) {
    actions.push({
      title: "Gate / KPI 차단 사유 확인",
      detail: criticalReasons[0] || `KPI 경고 ${fmtNum(kpi.alert_metric_count)}건, 관찰 ${fmtNum(kpi.watch_metric_count)}건이 있습니다.`,
      href: "/ops-readiness.html",
      cta: "시스템 확인",
    });
  }

  const liveDiagSummary = liveDiagnostics?.summary || {};
  const firstLiveDiag = Array.isArray(liveDiagnostics?.diagnostics) ? liveDiagnostics.diagnostics[0] : null;
  if (firstLiveDiag || Number(liveDiagSummary.order_candidate_count || 0) > 0) {
    actions.push({
      title: "AI 주문 preview 확인",
      detail: firstLiveDiag?.message_ko || `주문 후보 ${fmtNum(liveDiagSummary.order_candidate_count)}건 · 제출 가능 ${fmtNum(liveDiagSummary.submit_allowed_count)}건`,
      href: "/live-auto-trading.html",
      cta: "AI 상세",
    });
  }

  const partialQueue = Array.isArray(ruleOps?.partial_fill_queue) ? ruleOps.partial_fill_queue : [];
  const rulePositionCount = Number(ruleOps?.account?.position_count ?? (Array.isArray(ruleOps?.positions) ? ruleOps.positions.length : 0));
  if (partialQueue.length > 0 || rulePositionCount > 0) {
    actions.push({
      title: "수동매매 계좌 동기화 점검",
      detail: `보유 ${fmtNum(rulePositionCount)}개 · 동기화 대기 ${fmtNum(partialQueue.length)}건`,
      href: "/manual-trading.html",
      cta: "수동매매 상세",
    });
  }

  const topActions = actions.slice(0, 3);
  if (!topActions.length) {
    el.innerHTML = `<div class="empty-state">지금 바로 확인할 우선 항목이 없습니다.</div>`;
    return;
  }
  el.innerHTML = topActions.map((item) => `
    <a class="summary-link-card" href="${item.href}">
      <h3>${escapeHtml(item.title)}</h3>
      <p>${escapeHtml(item.detail)}</p>
      <span class="state-link">${escapeHtml(item.cta)} →</span>
    </a>
  `).join("");
}

function initTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tab;
      tabBtns.forEach((b) => b.classList.remove("active"));
      tabPanes.forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      const pane = document.getElementById(`tab-${target}`);
      if (pane) pane.classList.add("active");
      if (target === "blog") loadBlogDrafts().catch(console.error);
    });
  });
}

// ── 블로그 초안 ─────────────────────────────────────────────────────────────

const TYPE_META = {
  A: { label: "TOP5 종목 분석", cls: "blog-type-a" },
  B: { label: "순위 변동 추적", cls: "blog-type-b" },
  C: { label: "시장 지표 분석", cls: "blog-type-c" },
};

let blogDraftsLoaded = false;

async function loadBlogDrafts() {
  if (blogDraftsLoaded) return;
  const list = document.getElementById("blogDraftList");
  const meta = document.getElementById("blogDraftMeta");
  if (!list) return;
  list.innerHTML = '<div class="blog-empty">불러오는 중…</div>';
  try {
    const res = await fetch("/api/ops-readiness/blog-drafts", { credentials: "include" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (meta && data.asof_date) meta.textContent = `기준일 ${data.asof_date} · ${data.drafts?.length || 0}건`;
    renderBlogDrafts(list, data.drafts || []);
    blogDraftsLoaded = true;
  } catch (e) {
    list.innerHTML = `<div class="blog-empty">블로그 초안 조회 실패: ${e.message}</div>`;
  }
}

function renderBlogDrafts(container, drafts) {
  if (!drafts.length) {
    container.innerHTML = '<div class="blog-empty">오늘 생성된 블로그 초안이 없습니다.<br>종가 배치 완료 후 자동 생성됩니다.</div>';
    return;
  }
  container.innerHTML = drafts.map((d) => {
    const tm = TYPE_META[d.type] || { label: `TYPE ${d.type}`, cls: "blog-type-a" };
    const naver = d.naver || null;
    const tistory = d.tistory || null;
    const title = naver?.title || tistory?.title || "";
    return `
      <div class="blog-card">
        <div class="blog-card-header">
          <div class="blog-type-badge ${tm.cls}">TYPE_${d.type} <span class="blog-type-sub">${tm.label}</span></div>
        </div>
        ${title ? `<div class="blog-card-title">${escHtml(title)}</div>` : ""}
        ${naver   ? blogPlatformRow("N", "네이버",   "blog-platform-n", naver,   `${d.type}_naver`)   : ""}
        ${tistory ? blogPlatformRow("T", "티스토리", "blog-platform-t", tistory, `${d.type}_tistory`) : ""}
      </div>`;
  }).join("");
}

function blogPlatformRow(abbr, name, cls, plat, key) {
  const ext = key.endsWith("_tistory") ? "md" : "txt";
  return `
    <div class="blog-platform-row">
      <span class="blog-platform-label ${cls}">${abbr} ${name}</span>
      <span class="blog-char-count">${plat.char_count?.toLocaleString()}자</span>
      <div class="blog-actions">
        <button class="blog-btn blog-btn-copy" onclick="blogCopy(this, '${key}')">복사</button>
        <button class="blog-btn blog-btn-dl" onclick="blogDownload('${key}', '${ext}')">．${ext}↓</button>
      </div>
    </div>`;
}

function escHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

const _blogContentCache = {};

async function _ensureBlogContent(key) {
  if (_blogContentCache[key]) return _blogContentCache[key];
  const res = await fetch(`/api/ops-readiness/blog-drafts?key=${encodeURIComponent(key)}`, { credentials: "include" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  const [type, platform] = key.split("_");
  const content = data.drafts?.find(d => d.type === type)?.[platform]?.content || "";
  _blogContentCache[key] = content;
  return content;
}

async function blogCopy(btn, key) {
  try {
    const text = await _ensureBlogContent(key);
    await navigator.clipboard.writeText(text);
    btn.textContent = "복사됨 ✓";
    btn.classList.add("copied");
    setTimeout(() => { btn.textContent = "복사"; btn.classList.remove("copied"); }, 2000);
  } catch (e) {
    alert("복사 실패: " + e.message);
  }
}

async function blogDownload(key, ext) {
  try {
    const text = await _ensureBlogContent(key);
    const [type, platform] = key.split("_");
    const today = new Date().toISOString().slice(0, 10);
    const filename = `${today}_type${type}_${platform}.${ext}`;
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = filename; a.click();
    URL.revokeObjectURL(url);
  } catch (e) {
    alert("다운로드 실패: " + e.message);
  }
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
  el.innerHTML = items.map((item) => {
    const text = String(item || "");
    const parts = text.split(/:\s(.+)/, 2);
    if (parts.length === 2) {
      return `<li><span class="list-keyword">${escapeHtml(parts[0])}</span>: <span class="list-value">${escapeHtml(parts[1])}</span></li>`;
    }
    return `<li>${escapeHtml(text)}</li>`;
  }).join("");
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
      return `<li><span class="list-keyword">${escapeHtml(metric)}</span> <span class="list-value">${escapeHtml(value)}</span> <span class="muted">(${escapeHtml(status)})</span></li>`;
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
  return n.toLocaleString("ko-KR");
}

function fmtUsd(value, digits = 0) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(n);
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
    { label: `${summary.timezone || "Asia/Seoul"} ${summary.day_boundary || "00:00"} 기준`, kind: "INFO" },
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

function fmtRuntimeDate(value) {
  if (!value) return "-";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return String(value);
  return parsed.toLocaleDateString("ko-KR");
}

function fmtRuntimeDateTime(value) {
  if (!value) return "-";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return String(value);
  return parsed.toLocaleString("ko-KR", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function summarizeRuntimeError(message) {
  const text = String(message || "").trim();
  if (!text) return "-";
  const exitMatch = text.match(/\(exit=\d+\)\s*$/);
  const exitSuffix = exitMatch ? ` ${exitMatch[0]}` : "";
  const commandMatch = text.match(/\/([^/\s]+\.py)'?/);
  if (commandMatch) return `${commandMatch[1]} failed${exitSuffix}`.trim();
  return text;
}

function schedulerRuntimeChip(row) {
  const status = String(row?.status || "").toLowerCase();
  if (status === "warning") return `<span class="chip watch">주의</span>`;
  if (status === "idle" && row?.last_warning_details?.error_message) return `<span class="chip watch">주의</span>`;
  if (status === "idle") return `<span class="chip good">대기</span>`;
  if (status === "running") return `<span class="chip watch">실행중</span>`;
  if (status === "failed" || status === "error") return `<span class="chip bad">오류</span>`;
  return `<span class="chip info">${escapeHtml(status || "unknown")}</span>`;
}

function runtimeIssueCell(row) {
  const failure = row?.last_failure_details || {};
  const warning = row?.last_warning_details || {};
  const primary = failure.error_message ? failure : (warning.error_message ? warning : null);
  if (!primary) return runtimeErrorCell(row);
  const occurredAt = fmtRuntimeDateTime(primary.occurred_at || row?.last_failure_at || row?.last_warning_at);
  const step = primary.step_name || "-";
  const script = primary.script_name || "-";
  const exitCode = primary.exit_code ?? "-";
  const exceptionType = primary.exception_type || "-";
  const parts = [
    `${script}: ${primary.error_message || "-"}`,
    `step ${step}`,
    `exit=${exitCode}`,
    exceptionType !== "-" ? exceptionType : null,
    primary.hint ? `hint ${primary.hint}` : null,
    occurredAt !== "-" ? `at ${occurredAt}` : null,
  ].filter(Boolean);
  return parts.join(" / ");
}

function runtimeErrorCell(row) {
  const errorText = summarizeRuntimeError(row?.last_error);
  const failedAt = fmtRuntimeDateTime(row?.last_failure_at);
  const successAtRaw = row?.last_success_at;
  const failureAtRaw = row?.last_failure_at;
  const successTs = successAtRaw ? Date.parse(successAtRaw) : NaN;
  const failureTs = failureAtRaw ? Date.parse(failureAtRaw) : NaN;
  const recoveredAfterFailure = Number.isFinite(successTs) && Number.isFinite(failureTs) && successTs > failureTs;

  if (errorText === "-" && failedAt === "-") return "실패 이력 없음";
  if (failedAt === "-") return errorText === "-" ? "실패 이력 없음" : errorText;
  if (errorText === "-") return recoveredAfterFailure ? `과거 실패 ${failedAt} / 현재 정상` : `실패 시각 ${failedAt}`;
  return recoveredAfterFailure
    ? `${errorText} / 과거 실패 ${failedAt} / 현재 정상`
    : `${errorText} / 실패 ${failedAt}`;
}

function schedulerActionLinks(key, row) {
  const links = [];
  const failed = ["failed", "error"].includes(String(row?.status || "").toLowerCase());
  if (failed) {
    links.push({ href: "/alerts.html", label: "알림 로그" });
  }
  if (key === "auto_buy") links.push({ href: "/live-auto-trading.html", label: "AI 상세" });
  if (key === "live_sync") links.push({ href: "/holdings.html", label: "보유종목" });
  if (key === "close" || key === "intraday") links.push({ href: "/trade-history.html", label: "매매기록" });
  if (key === "market_guard") links.push({ href: "/alerts.html", label: "알림 로그" });
  if (!links.length) links.push({ href: "/ops-readiness.html", label: "운영자" });
  return links
    .map((item) => `<a href="${item.href}" class="state-link">${escapeHtml(item.label)}</a>`)
    .join(" · ");
}

function opsToneToChipClass(value) {
  const tone = String(value || "").toLowerCase();
  if (tone === "normal") return "good";
  if (tone === "risk" || tone === "stopped") return "bad";
  if (tone === "warning") return "watch";
  return "info";
}

function opsFlagText(value) {
  if (value === true) return "ON";
  if (value === false) return "OFF";
  return "-";
}

function renderOperationsRuntime(runtime) {
  const operations = runtime?.operations || {};
  const controls = operations.controls || {};
  const cards = operations.cards || {};
  const ai = operations.ai || {};
  const rule = operations.rule || {};
  const summary = operations.summary || {};
  const hasPayload = Object.keys(operations).length > 0;

  const cardList = [
    {
      title: "운영 안전장치",
      tone: controls.global_kill_switch ? "stopped" : (controls.auto_trade_allow_buy ? "normal" : "warning"),
      chips: [
        { label: `GLOBAL_KILL_SWITCH ${opsFlagText(controls.global_kill_switch)}`, kind: controls.global_kill_switch ? "bad" : "good" },
        { label: `RULE_KILL_SWITCH ${opsFlagText(controls.rule_kill_switch)}`, kind: controls.rule_kill_switch ? "bad" : "good" },
        { label: `EXECUTE ${opsFlagText(controls.auto_trade_execute)}`, kind: controls.auto_trade_execute ? "good" : "watch" },
        { label: `ALLOW_BUY ${opsFlagText(controls.auto_trade_allow_buy)}`, kind: controls.auto_trade_allow_buy ? "good" : "watch" },
      ],
      kv: [
        ["마지막 성공", fmtRuntimeDateTime(summary.latest_success_at)],
        ["마지막 실패", fmtRuntimeDateTime(summary.latest_failure_at)],
        ["마지막 에러", summarizeRuntimeError(summary.latest_error_message)],
      ],
    },
    {
      title: "오늘 실행 상태",
      tone: summary.today_close_batch_success && summary.today_live_account_sync_success ? "normal" : "warning",
      chips: [
        { label: `종가배치 ${summary.today_close_batch_success ? "OK" : "WAIT"}`, kind: summary.today_close_batch_success ? "good" : "watch" },
        { label: `AI auto BUY ${summary.today_ai_auto_buy_success ? "OK" : "WAIT"}`, kind: summary.today_ai_auto_buy_success ? "good" : "watch" },
        { label: `live sync ${summary.today_live_account_sync_success ? "OK" : "WAIT"}`, kind: summary.today_live_account_sync_success ? "good" : "watch" },
      ],
      kv: [
        ["AI BUY 후보", fmtNum(ai.buy_candidate_count)],
        ["AI BUY 차단", fmtNum(ai.buy_blocked_count)],
      ],
    },
  ];

  if (!hasPayload) {
    return `<div class="empty-state">auto_trading_ops_status payload가 아직 없습니다.</div>`;
  }

  return `
    <div class="grid-2" style="margin-bottom:12px;">
      ${cardList.map((card) => `
        <article class="card compact-card">
          <div class="eyebrow">Ops Runtime</div>
          <h3>${escapeHtml(card.title)}</h3>
          <div class="chip-row">
            <span class="chip ${opsToneToChipClass(card.tone)}">${escapeHtml(String(card.tone || "warning").toUpperCase())}</span>
            ${card.chips.map((chip) => `<span class="chip ${chip.kind}">${escapeHtml(chip.label)}</span>`).join("")}
          </div>
          <div class="kv">
            ${card.kv.map(([label, value]) => `
              <div class="kv-row">
                <span class="muted">${escapeHtml(label)}</span>
                <strong>${escapeHtml(String(value ?? "-"))}</strong>
              </div>
            `).join("")}
          </div>
        </article>
      `).join("")}
    </div>
  `;
}

function renderSchedulerRuntime(runtime) {
  const wrap = document.getElementById("runtimeTableWrap");
  if (!wrap) return;

  const rows = [
    ["close", "close (18:10)", runtime?.close_scheduler],
    ["intraday", "intraday / recovery (12:00)", runtime?.intraday_scheduler],
    ["auto_buy", "auto_buy (09:30)", runtime?.auto_buy_scheduler],
    ["live_sync", "live_sync (10:00/14:00/18:00)", runtime?.live_account_sync_scheduler],
    ["market_guard", "market_guard (09:10/14:00)", runtime?.market_guard_scheduler],
  ].filter(([, , payload]) => payload);

  if (!rows.length) {
    wrap.innerHTML = `${renderOperationsRuntime(runtime)}<div class="empty-state">scheduler runtime status 산출물이 아직 없습니다.</div>`;
    return;
  }

  wrap.innerHTML = `
    ${renderOperationsRuntime(runtime)}
    <table class="runtime-table">
      <thead>
        <tr>
          <th>스케줄러</th>
          <th>상태</th>
          <th>마지막 성공일</th>
          <th>마지막 성공시각</th>
          <th>최근 실패 이력</th>
          <th>바로가기</th>
        </tr>
      </thead>
      <tbody>
        ${rows.map(([key, label, row]) => `
          <tr>
            <td>${escapeHtml(label)}</td>
            <td>${schedulerRuntimeChip(row)}</td>
            <td>${escapeHtml(fmtRuntimeDate(row.last_success_date || row.last_success_at))}</td>
            <td>${escapeHtml(fmtRuntimeDateTime(row.last_success_at))}</td>
            <td>${escapeHtml(runtimeIssueCell(row))}</td>
            <td>${schedulerActionLinks(key, row)}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
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

function fmtPnl(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  const sign = n >= 0 ? "+" : "";
  return `${sign}${(n * 100).toFixed(1)}%`;
}

function pnlClass(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "muted";
  return n >= 0 ? "good" : "bad";
}

function renderAccountOverview(liveAccountSummary, ruleOps, usTradingSummary) {
  const liveSummary = liveAccountSummary?.summary || {};
  const liveMetrics = liveSummary.derived_metrics || {};
  const liveGeneratedAt = liveSummary.generated_at || liveAccountSummary?.last_execution_at || "-";
  renderChipRow("aiAccountChips", [
    { label: liveAccountSummary?.preview_gate_display_status || "gate 없음", kind: liveAccountSummary?.preview_gate_source_status || liveAccountSummary?.preview_gate_display_status || "INFO" },
    { label: `보유 ${fmtNum(liveAccountSummary?.holding_count)}`, kind: Number(liveAccountSummary?.holding_count || 0) > 0 ? "INFO" : "WATCH" },
    { label: `실행 ${fmtNum(liveAccountSummary?.order_execution_count)}`, kind: Number(liveAccountSummary?.order_execution_count || 0) > 0 ? "GOOD" : "WATCH" },
  ]);
  renderKv("aiAccountKv", [
    ["기준시각", liveGeneratedAt],
    ["현금", fmtMoneyShort(liveMetrics.cash_amount)],
    ["총자산", fmtMoneyShort(liveMetrics.total_assets)],
    ["보유평가", fmtMoneyShort(liveMetrics.holding_eval_amount)],
    ["주간 손익", fmtMoneyShort(liveMetrics.weekly_total_pnl)],
  ]);
  renderText("aiAccountNote", liveAccountSummary
    ? `preview ${fmtNum(liveAccountSummary.order_preview_count)}건 · 최근 제출 ${fmtNum(liveAccountSummary.last_execution_submitted_count)}건`
    : "AI 실계좌 요약을 불러오지 못했습니다.");

  const ruleAccount = ruleOps?.account || {};
  const rulePositions = Array.isArray(ruleOps?.positions) ? ruleOps.positions : [];
  const partialQueue = Array.isArray(ruleOps?.partial_fill_queue) ? ruleOps.partial_fill_queue : [];
  renderChipRow("ruleAccountOverviewChips", [
    { label: `포지션 ${fmtNum(ruleAccount.position_count ?? rulePositions.length)}`, kind: Number(ruleAccount.position_count ?? rulePositions.length ?? 0) > 0 ? "INFO" : "WATCH" },
    { label: partialQueue.length > 0 ? `동기화 대기 ${partialQueue.length}` : "동기화 대기 없음", kind: partialQueue.length > 0 ? "WATCH" : "GOOD" },
    { label: `기준 ${ruleOps?.as_of_date || "-"}`, kind: "INFO" },
  ]);
  renderKv("ruleAccountOverviewKv", [
    ["기준일", ruleOps?.as_of_date || "-"],
    ["현금", fmtMoneyShort(ruleAccount.cash)],
    ["총자산", fmtMoneyShort(ruleAccount.total_equity)],
    ["보유평가", fmtMoneyShort(ruleAccount.position_market_value)],
    ["포지션 수", fmtNum(ruleAccount.position_count ?? rulePositions.length)],
  ]);
  renderText("ruleAccountOverviewNote", ruleOps
    ? `운영자가 직접 매매한 뒤 계좌 동기화 상태를 추적합니다. 반영 대기 ${partialQueue.length}건 · fill sync ${fmtNum(ruleOps?.fill_sync?.partial_filled_count)}건`
    : "수동매매 계좌 요약을 불러오지 못했습니다.");

  renderKv("accountComparisonKv", [
    ["AI 기준시각", liveGeneratedAt],
    ["수동 기준일", ruleOps?.as_of_date || "-"],
    ["AI 보유 수", fmtNum(liveAccountSummary?.holding_count)],
    ["수동 보유 수", fmtNum(ruleAccount.position_count ?? rulePositions.length)],
    ["AI gate", liveAccountSummary?.preview_gate_display_status || "-"],
    ["수동 계좌 현금", fmtMoneyShort(ruleAccount.cash)],
  ]);
}

function renderStrategyOverview(liveAccountSummary, ruleOps, usTradingSummary) {
  renderChipRow("aiStrategyChips", [
    { label: liveAccountSummary?.preview_gate_display_status || "-", kind: liveAccountSummary?.preview_gate_source_status || liveAccountSummary?.preview_gate_display_status || "INFO" },
    { label: `preview ${fmtNum(liveAccountSummary?.order_preview_count)}`, kind: Number(liveAccountSummary?.order_preview_count || 0) > 0 ? "INFO" : "WATCH" },
    { label: `제출 ${fmtNum(liveAccountSummary?.last_execution_submitted_count)}`, kind: Number(liveAccountSummary?.last_execution_submitted_count || 0) > 0 ? "GOOD" : "WATCH" },
  ]);
  renderKv("aiStrategyKv", [
    ["주문 후보", fmtNum(liveAccountSummary?.order_preview_count)],
    ["실행 로그", fmtNum(liveAccountSummary?.order_execution_count)],
    ["최근 실행", liveAccountSummary?.last_execution_at || "-"],
    ["최근 제출", fmtNum(liveAccountSummary?.last_execution_submitted_count)],
  ]);
  renderText("aiStrategyNote", liveAccountSummary?.preview_gate_runtime_status
    ? `runtime ${liveAccountSummary.preview_gate_runtime_status} · source ${liveAccountSummary.preview_gate_source_status || "-"}`
    : "AI 전략 실행 요약을 불러오지 못했습니다.");

  const ruleExec = ruleOps?.execution || {};
  renderChipRow("ruleStrategyChips", [
    { label: ruleOps?.as_of_date ? `기준 ${ruleOps.as_of_date}` : "기준일 없음", kind: ruleOps?.as_of_date ? "INFO" : "WATCH" },
    { label: `보유 ${fmtNum(ruleOps?.account?.position_count ?? 0)}`, kind: Number(ruleOps?.account?.position_count || 0) > 0 ? "GOOD" : "WATCH" },
    { label: Array.isArray(ruleOps?.partial_fill_queue) && ruleOps.partial_fill_queue.length > 0 ? `반영 대기 ${fmtNum(ruleOps.partial_fill_queue.length)}` : "동기화 정상", kind: Array.isArray(ruleOps?.partial_fill_queue) && ruleOps.partial_fill_queue.length > 0 ? "WATCH" : "GOOD" },
  ]);
  renderKv("ruleStrategyKv", [
    ["기준일", ruleExec.as_of_date || ruleOps?.as_of_date || "-"],
    ["보유 종목 수", fmtNum(ruleOps?.account?.position_count ?? 0)],
    ["현금", fmtMoneyShort(ruleOps?.account?.cash)],
    ["총자산", fmtMoneyShort(ruleOps?.account?.total_equity)],
    ["반영 대기", fmtNum(Array.isArray(ruleOps?.partial_fill_queue) ? ruleOps.partial_fill_queue.length : 0)],
  ]);
  renderText("ruleStrategyNote", ruleOps
    ? "수동매매는 운영자가 직접 체결하고, 이 화면에서는 계좌와 보유 상태가 시스템에 정상 반영됐는지 확인합니다."
    : "수동매매 동기화 요약을 불러오지 못했습니다.");
}

function renderStrategyDetails(liveDiagnostics, usTradingSummary) {
  const aiItems = [];
  const diagSummary = liveDiagnostics?.summary || {};
  const diagnostics = Array.isArray(liveDiagnostics?.diagnostics) ? liveDiagnostics.diagnostics : [];
  if (diagSummary.market_status_ko) aiItems.push(`시장 상태: ${diagSummary.market_status_ko}`);
  if (diagSummary.order_candidate_count != null) aiItems.push(`주문 후보: ${fmtNum(diagSummary.order_candidate_count)}건`);
  if (diagSummary.submit_allowed_count != null) aiItems.push(`제출 가능: ${fmtNum(diagSummary.submit_allowed_count)}건`);
  if (diagSummary.broker_rejected_count) aiItems.push(`증권사 거절: ${fmtNum(diagSummary.broker_rejected_count)}건`);
  if (diagnostics[0]?.message_ko) aiItems.push(`주요 진단: ${diagnostics[0].message_ko}`);
  renderList("aiStrategyDetails", aiItems, "AI 상세 진단 정보가 없습니다.");
}

function renderRuleOps(ruleOps) {
  if (!ruleOps) return;

  const exec = ruleOps.execution || {};
  const account = ruleOps.account || {};
  const positions = ruleOps.positions || [];
  const partialQ = ruleOps.partial_fill_queue || [];
  const fillSync = ruleOps.fill_sync || {};

  // ── 최근 동기화 / 체결 반영 ──
  const aborted = exec.order_run_aborted;
  const execTone = aborted ? "ALERT" : exec.filled_count > 0 ? "GOOD" : partialQ.length > 0 ? "WATCH" : "info";
  renderChipRow("ruleExecChipRow", [
    { label: exec.run_mode || "sync", kind: exec.run_mode === "live" ? "GOOD" : "INFO" },
    { label: aborted ? "동기화 점검 필요" : exec.filled_count > 0 ? "최근 체결 반영" : "최근 체결 없음", kind: execTone },
    ...(exec.reconciliation_blocked ? [{ label: "계좌 조정 점검", kind: "ALERT" }] : []),
  ]);
  renderKv("ruleExecKv", [
    ["기준일", exec.as_of_date || ruleOps.as_of_date || "-"],
    ["체결", fmtNum(exec.filled_count)],
    ["부분 체결", fmtNum(exec.partial_filled_count)],
    ["반영 대기", fmtNum(partialQ.length)],
    ["현금", fmtMoneyShort(account.cash)],
    ["총자산", fmtMoneyShort(account.total_equity)],
    ...(aborted ? [["점검 사유", exec.abort_reason || "-"]] : []),
  ]);
  const execItemEl = document.getElementById("ruleExecItems");
  if (execItemEl) {
    const items = (exec.items || []).slice(0, 8);
    if (!items.length) {
      execItemEl.innerHTML = "<li>최근 반영된 체결 내역 없음</li>";
    } else {
      execItemEl.innerHTML = items.map((item) => {
        const statusChip = item.order_status === "filled" ? "good" : item.order_status === "partial_filled" ? "watch" : item.order_status === "failed" ? "bad" : "info";
        return `<li><span class="chip ${statusChip}" style="font-size:10px;padding:1px 7px;margin-right:6px;">${escapeHtml(item.side || "-")} ${escapeHtml(item.order_status || "-")}</span><span class="list-keyword">${escapeHtml(item.code)}</span>${item.filled_qty ? ` <span class="muted">${fmtNum(item.filled_qty)}주 @${fmtNum(item.avg_fill_price)}</span>` : ""}${item.order_block_reason ? ` <span class="muted">(${escapeHtml(item.order_block_reason)})</span>` : ""}</li>`;
      }).join("");
    }
  }

  // ── 현재 보유 포지션 ──
  renderChipRow("ruleAccountChipRow", [
    { label: `포지션 ${fmtNum(account.position_count)}`, kind: account.position_count > 0 ? "info" : "WATCH" },
    { label: `현금 ${fmtMoneyShort(account.cash)}`, kind: "info" },
    { label: `총자산 ${fmtMoneyShort(account.total_equity)}`, kind: "info" },
  ]);
  const posEl = document.getElementById("rulePositionList");
  if (posEl) {
    if (!positions.length) {
        posEl.innerHTML = "<li>보유 포지션 없음</li>";
    } else {
      posEl.innerHTML = positions.map((p) => {
        const pnlStr = fmtPnl(p.pnl_pct);
        const pnlCls = pnlClass(p.pnl_pct);
        const weightStr = p.weight != null ? `${(p.weight * 100).toFixed(1)}%` : "-";
        return `<li><span class="list-keyword">${escapeHtml(p.name || p.code)}</span> <span class="muted">(${escapeHtml(p.code)})</span> · <span class="${pnlCls}">${escapeHtml(pnlStr)}</span> · ${escapeHtml(weightStr)} · ${fmtNum(p.qty)}주</li>`;
      }).join("");
    }
  }

  // ── 반영 대기 / 확인 항목 ──
  renderChipRow("rulePartialFillChipRow", [
    { label: partialQ.length > 0 ? `대기 ${partialQ.length}건` : "대기 없음", kind: partialQ.length > 0 ? "WATCH" : "GOOD" },
    ...(fillSync.partial_filled_count > 0 ? [{ label: `fill sync ${fmtNum(fillSync.partial_filled_count)}건`, kind: "WATCH" }] : []),
  ]);
  const pfEl = document.getElementById("rulePartialFillList");
  if (pfEl) {
    if (!partialQ.length) {
      pfEl.innerHTML = "<li>추가 확인이 필요한 반영 대기 없음</li>";
    } else {
      pfEl.innerHTML = partialQ.map((item) => {
        const ratio = item.order_qty ? `${fmtNum(item.filled_qty)}/${fmtNum(item.order_qty)}주` : `체결 ${fmtNum(item.filled_qty)}주`;
        return `<li><span class="list-keyword">${escapeHtml(item.code)}</span> · ${escapeHtml(ratio)} · 미체결 ${fmtNum(item.unfilled_qty)}주${item.avg_fill_price ? ` · @${fmtNum(item.avg_fill_price)}` : ""}</li>`;
      }).join("");
    }
  }
  const helpEl = document.getElementById("rulePartialFillHelp");
  if (helpEl) {
    const enabled = partialQ.length > 0;
    helpEl.textContent = enabled
      ? "부분 체결이나 반영 지연이 있어 다음 동기화 사이클에서 다시 확인이 필요합니다."
      : "현재 수동매매 계좌는 시스템 기준으로 안정적으로 동기화된 상태입니다.";
  }
}

async function renderRecentAlerts() {
  const el = document.getElementById("recentAlertsLog");
  if (!el) return;
  try {
    const data = await fetchJsonMaybe("/api/alerts?limit=10");
    if (!data) {
      el.innerHTML = '<div class="state-line">알림 로그를 불러오지 못했습니다.</div>';
      return;
    }
    const entries = (data.entries || []).filter((e) => e.level === "CRITICAL" || e.level === "WARNING");
    if (!entries.length) {
      el.innerHTML = '<div class="state-line" style="color:#86efac;">최근 CRITICAL/WARNING 알림 없음</div>';
      return;
    }
    el.innerHTML = entries.slice(0, 5).map((e) => {
      const levelClass = e.level === "CRITICAL" ? "bad" : "watch";
      const time = String(e.timestamp || "").replace("T", " ").slice(0, 16);
      return `
        <div style="display:flex;gap:8px;align-items:flex-start;padding:7px 0;border-bottom:1px solid rgba(51,65,85,0.4);">
          <span class="chip ${levelClass}" style="flex-shrink:0;margin-top:1px;">${escapeHtml(e.level)}</span>
          <div style="min-width:0;">
            <div style="font-size:13px;font-weight:700;color:#f8fafc;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${escapeHtml(e.title || "(제목 없음)")}</div>
            ${e.message ? `<div style="font-size:11px;color:#94a3b8;margin-top:2px;">${escapeHtml(e.message)}</div>` : ""}
            <div style="font-size:11px;color:#475569;margin-top:2px;">${escapeHtml(time)}</div>
          </div>
        </div>`;
    }).join("");
  } catch {
    el.innerHTML = '<div class="state-line">알림 조회 실패</div>';
  }
}

function macroStatusKind(status) {
  const s = String(status || "").toUpperCase();
  if (s === "RISK_OFF") return "bad";
  if (s === "RISK_ON") return "good";
  if (s === "DATA_INCOMPLETE") return "watch";
  return "info";
}

async function loadUsMacroOverlay() {
  try {
    const data = await fetchJsonMaybe("/api/us-macro-overlay");
    if (!data) {
      document.getElementById("usMacroSummary").textContent = "US Macro 테이블 없음 — 마이그레이션 필요";
      return;
    }

    // Update mode labels based on server-reported config
    const cfg = data.config || {};
    const isRealApply = cfg.enabled && !cfg.shadow_mode && cfg.allow_real_apply;
    const modeLabel = !cfg.enabled
      ? "비활성화"
      : isRealApply
        ? "실반영 모드 — RULE 자동매매 적용 중"
        : "Shadow Mode — 실매매 미반영";
    const modeColor = isRealApply ? "#4ade80" : "#fbbf24";
    const headerNote = document.getElementById("usMacroHeaderNote");
    if (headerNote) {
      headerNote.innerHTML = `미국 ETF/지수 기반 매크로 상태 · <span style="color:${modeColor};font-weight:600">${modeLabel}</span>`;
    }
    const overlayNote = document.getElementById("usMacroOverlayNote");
    if (overlayNote) {
      overlayNote.innerHTML = isRealApply
        ? `<span style="color:#4ade80;font-weight:600">실반영 ON</span> — RULE 매수 차단 실제 적용`
        : `<span style="color:#fbbf24">Shadow Mode</span> — 실제 주문/점수에 미반영`;
    }

    const latest = (data.macro_history || [])[0] || {};
    const status = latest.macro_status || "NO_DATA";
    const kind = macroStatusKind(status);

    // Status chips
    const chips = [{ label: status, kind }];
    if (latest.risk_off_flag) chips.push({ label: "RISK_OFF", kind: "bad" });
    if (latest.vix_spike_flag) chips.push({ label: "VIX SPIKE", kind: "bad" });
    if (latest.risk_on_flag) chips.push({ label: "RISK_ON", kind: "good" });
    renderChipRow("usMacroStatusChips", chips);

    const fmtR = (v) => v != null ? `${(Number(v) * 100).toFixed(2)}%` : "-";
    renderKv("usMacroKv", [
      ["US 거래일", latest.us_trade_date || "-"],
      ["KR 적용일", latest.kr_apply_date || "-"],
      ["SPY 1d", fmtR(latest.spy_ret_1d)],
      ["QQQ 1d", fmtR(latest.qqq_ret_1d)],
      ["VIX 1d", fmtR(latest.vix_ret_1d)],
      ["SMH 1d", fmtR(latest.semiconductor_ret_1d)],
      ["섹터 breadth", latest.sector_breadth != null ? `${(Number(latest.sector_breadth) * 100).toFixed(0)}%` : "-"],
      ["top 섹터", latest.top_sector || "-"],
      ["missing tickers", latest.missing_tickers ? latest.missing_tickers.join(", ") : "없음"],
    ]);
    document.getElementById("usMacroSummary").textContent = latest.macro_summary || "";

    // Overlay summary (prefer the latest run_date group; flag if it lags macro apply date)
    const overlaySummary = data.overlay_summary || [];
    const latestOverlayRunDate = overlaySummary[0]?.run_date || null;
    const todayOverlay = latestOverlayRunDate
      ? overlaySummary.filter((row) => row.run_date === latestOverlayRunDate)
      : [];
    const totalRows = todayOverlay.reduce((s, r) => s + (r.total || 0), 0);
    const blockedRows = todayOverlay.reduce((s, r) => s + (r.blocked || 0), 0);
    const penalizedRows = todayOverlay.reduce((s, r) => s + (r.penalized || 0), 0);
    const boostedRows = todayOverlay.reduce((s, r) => s + (r.boosted || 0), 0);
    const overlayStale = isIsoDate(latestOverlayRunDate) && isIsoDate(latest.kr_apply_date) && latestOverlayRunDate < latest.kr_apply_date;
    const overKind = blockedRows > 0 ? "bad" : penalizedRows > 0 ? "watch" : "good";
    renderChipRow("usMacroOverlayChips", [
      { label: `후보 ${totalRows}개`, kind: "info" },
      { label: blockedRows > 0 ? `차단 ${blockedRows}` : "차단 없음", kind: blockedRows > 0 ? "bad" : "good" },
      { label: penalizedRows > 0 ? `감점 ${penalizedRows}` : "감점 없음", kind: penalizedRows > 0 ? "watch" : "good" },
      ...(overlayStale ? [{ label: "overlay stale", kind: "WATCH" }] : []),
    ]);
    renderKv("usMacroOverlayKv", [
      ["총 후보", `AI ${todayOverlay.find(r => r.engine_type === "ai")?.total || 0} + RULE ${todayOverlay.find(r => r.engine_type === "rule")?.total || 0}`],
      ["차단 (buy_blocked)", `${blockedRows}개`],
      ["감점 (score -)", `${penalizedRows}개`],
      ["가산 (score +)", `${boostedRows}개`],
      ["run_date", overlayStale ? `${latestOverlayRunDate} (stale)` : (latestOverlayRunDate || "-")],
    ]);

    // Affected list
    const affected = overlayStale ? [] : (data.top_affected || []);
    const affectedEl = document.getElementById("usMacroAffectedList");
    if (!affected.length) {
      affectedEl.innerHTML = '<li>영향받은 종목 없음</li>';
    } else {
      affectedEl.innerHTML = affected.map(r => {
        const adj = Number(r.macro_adjustment) || 0;
        const adjtxt = adj >= 0 ? `+${adj.toFixed(1)}` : adj.toFixed(1);
        const badge = r.buy_blocked_flag ? '<span style="color:#fca5a5">[차단]</span>' : `<span style="color:#fde68a">[${adjtxt}점]</span>`;
        const engine = `<span style="color:#94a3b8;font-size:11px">[${(r.engine_type || "").toUpperCase()}]</span>`;
        return `<li>${engine} ${badge} <strong>${escapeHtml(r.name || r.code)}</strong> (${escapeHtml(r.code)}) <span style="color:#94a3b8;font-size:11px">${escapeHtml((r.overlay_reason || "").slice(0, 60))}</span></li>`;
      }).join("");
    }
  } catch (e) {
    console.warn("us-macro-overlay load failed", e);
    document.getElementById("usMacroSummary").textContent = `조회 실패: ${e.message}`;
  }
}

async function loadOpsReadiness() {
  const state = document.getElementById("pageState");
  state.textContent = "운영 readiness 대시보드를 불러오는 중입니다.";
  try {
    const [data, runtime, liveAccountSummary, usTradingSummary, liveDiagnostics] = await Promise.all([
      fetchJson("/api/ops-readiness"),
      fetchJsonMaybe("/api/auto-trading/runtime-status"),
      fetchJsonMaybe("/api/live-account/summary"),
      fetchJsonMaybe("/api/us/trading/summary"),
      fetchJsonMaybe("/api/live-auto-trading-diagnostics"),
    ]);
    renderRecentAlerts().catch(() => {});
    renderRuleOps(data.rule_ops || null);
    renderAccountOverview(liveAccountSummary, data.rule_ops || null, usTradingSummary);
    renderStrategyOverview(liveAccountSummary, data.rule_ops || null, usTradingSummary);
    renderStrategyDetails(liveDiagnostics, usTradingSummary);
    renderTodayTopActions(data, runtime, liveDiagnostics, data.rule_ops || null, usTradingSummary);
    renderHero(data);
    renderSafetyBanner(data);
    updateBadgeToday(data);
    renderSchedulerRuntime(runtime);

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
    renderText("transitionOpsHint", cardHelp.transition || "");

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
    renderIntradayOps("opsIntradayGrid", intraday);
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
    renderShadowRepeatability(data.shadow || {});

    const staleNotes = [];
    if (isIsoDate(outputs.gate_asof_date) && isIsoDate(latestRefDate) && outputs.gate_asof_date < latestRefDate) staleNotes.push(`gate ${outputs.gate_asof_date}`);
    if (isIsoDate(outputs.daily_recommendations_date) && isIsoDate(latestRefDate) && outputs.daily_recommendations_date < latestRefDate) staleNotes.push(`daily ${outputs.daily_recommendations_date}`);
    if (isIsoDate(kpi.latest_date) && isIsoDate(latestRefDate) && kpi.latest_date < latestRefDate) staleNotes.push(`kpi ${kpi.latest_date}`);
    if (isIsoDate(marketRegime.latest_date) && isIsoDate(latestRefDate) && marketRegime.latest_date < latestRefDate) staleNotes.push(`market ${marketRegime.latest_date}`);
    state.textContent = `기준일 ${data.asof_date || "-"} · ${basis.label || "기준 미상"} 기준 운영 readiness 대시보드를 불러왔습니다.${staleNotes.length ? ` stale sources: ${staleNotes.join(", ")}` : ""}`;
  } catch (error) {
    console.error(error);
    if (Number(error?.status) === 401) {
      const loginUrl = `/operator-login?next=${encodeURIComponent(location.pathname + location.search)}`;
      document.getElementById("heroGrid").innerHTML = `
        <div class="empty-state" style="grid-column:1/-1;">
          운영자 인증이 필요합니다. <a class="state-link" href="${loginUrl}">운영자 로그인 →</a>
        </div>
      `;
      renderList("goReasons", [], "운영자 인증 후 확인할 수 있습니다.");
      renderMetricList("alertMetrics", [], "운영자 인증 후 확인할 수 있습니다.");
      renderList("dailyChecklist", [], "운영자 인증 후 확인할 수 있습니다.");
      renderIntradayOps("opsIntradayGrid", {});
      renderShadowRepeatability({});
      renderRuleOps(null);
      renderSchedulerRuntime(null);
      state.textContent = "운영자 인증이 필요합니다.";
      return;
    }
    document.getElementById("heroGrid").innerHTML = '<div class="empty-state" style="grid-column:1/-1;">운영 대시보드를 불러오지 못했습니다.</div>';
    renderList("goReasons", [], "조회 실패");
    renderMetricList("alertMetrics", [], "조회 실패");
    renderList("dailyChecklist", [], "조회 실패");
    renderIntradayOps("opsIntradayGrid", {});
    renderShadowRepeatability({});
    renderRuleOps(null);
    renderSchedulerRuntime(null);
    state.textContent = `조회 실패: ${error.message}`;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  document.getElementById("saveOperatorMemoBtn")?.addEventListener("click", () => {
    saveOperatorMemo().catch((error) => console.error(error));
  });
  loadOpsReadiness().catch((error) => console.error(error));
});
