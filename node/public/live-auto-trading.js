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

const signedClass = (value) => {
  const n = Number(value);
  if (!Number.isFinite(n) || n === 0) return "";
  return n > 0 ? "pos" : "neg";
};

const metricHtml = (value, formatter, digits = 0) => {
  const cls = signedClass(value);
  const rendered = escapeHtml(formatter(value, digits));
  return cls ? `<span class="${cls}">${rendered}</span>` : rendered;
};

const escapeHtml = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (m) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[m]));

const fmtRuntimeDate = (value) => {
  if (!value) return "-";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return String(value);
  return parsed.toLocaleDateString("ko-KR");
};

const fmtRuntimeDateTime = (value) => {
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
};

function summarizeRuntimeError(message) {
  const text = String(message || "").trim();
  if (!text) return "-";
  const exitMatch = text.match(/\(exit=\d+\)\s*$/);
  const exitSuffix = exitMatch ? ` ${exitMatch[0]}` : "";
  const commandMatch = text.match(/\/([^/\s]+\.py)'?/);
  if (commandMatch) return `${commandMatch[1]} failed${exitSuffix}`.trim();
  return text;
}

function runtimeErrorCell(row) {
  const errorText = summarizeRuntimeError(row?.last_error);
  const failedAt = fmtRuntimeDateTime(row?.last_failure_at);
  if (errorText === "-" && failedAt === "-") return "-";
  if (failedAt === "-") return errorText;
  if (errorText === "-") return `실패 시각 ${failedAt}`;
  return `${errorText} · 실패 ${failedAt}`;
}

async function fetchJsonMaybe(url) {
  const res = await fetch(url, { credentials: "same-origin" });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`${url} HTTP ${res.status}`);
  return res.json();
}

function toneClass(kind) {
  const value = String(kind || "").toUpperCase();
  if (["BUY", "GOOD", "OPEN", "READY", "EXECUTABLE", "SELL"].includes(value)) return "good";
  if (["REVIEW", "WATCH", "PILOT", "TRIM", "HOLD"].includes(value)) return "watch";
  if (["BLOCK", "BAD", "ERROR", "EXIT"].includes(value)) return "bad";
  return "warn";
}

function opsToneClass(kind) {
  const value = String(kind || "").toLowerCase();
  if (value === "normal") return "good";
  if (value === "risk" || value === "stopped") return "bad";
  if (value === "warning") return "warn";
  return "watch";
}

function flagText(value) {
  if (value === true) return "ON";
  if (value === false) return "OFF";
  return "-";
}

function orderStateChip(row) {
  if (row.executable_now) return `<span class="chip good">제출 가능</span>`;
  if (row.blocked_reason) return `<span class="chip bad">차단됨</span>`;
  return `<span class="chip warn">초안</span>`;
}

function intentStateChip(row) {
  if (row.executable) return `<span class="chip good">실행 후보</span>`;
  if (String(row.intent_type || "").toUpperCase() === "REVIEW") return `<span class="chip watch">검토용</span>`;
  return `<span class="chip warn">설명용</span>`;
}

function holdingStateChip(row) {
  const pnlPct = Number(row.pnl_pct);
  if (String(row.status || "").toUpperCase() !== "OPEN") return `<span class="chip warn">${escapeHtml(row.status || "closed")}</span>`;
  if (Number.isFinite(pnlPct) && pnlPct > 0) return `<span class="chip good">수익</span>`;
  if (Number.isFinite(pnlPct) && pnlPct < 0) return `<span class="chip bad">손실</span>`;
  return `<span class="chip watch">보유중</span>`;
}

function executionStateChip(row) {
  const status = String(row.submission_status || "").toLowerCase();
  if (status === "submitted") return `<span class="chip good">제출됨</span>`;
  if (status === "failed") return `<span class="chip bad">실패</span>`;
  if (status === "skipped") return `<span class="chip watch">건너뜀</span>`;
  return `<span class="chip warn">미상</span>`;
}

function describeExecutionReason(reason, row, runtime) {
  const key = String(reason || "").trim();
  if (!key) return "-";
  const side = String(row?.side || "").toUpperCase();
  const buyApprovalRequired = !!runtime?.policy?.buy_approval_required;
  if (key.startsWith("policy_blocked:")) {
    return row?.user_message_ko || "정책 기준으로 주문이 차단되었습니다.";
  }
  switch (key) {
    case "buy_approval_required":
      return buyApprovalRequired
        ? "매수 승인 목록에 없는 요청이라 실주문이 보류되었습니다."
        : "매수 승인 조건 때문에 실주문이 보류되었습니다.";
    case "buy_requires_allow_buy":
      return "BUY 실주문 스위치가 꺼져 있어 매수 제출이 보류되었습니다.";
    case "duplicate_request_id":
      return "이미 성공 처리된 요청 ID라 중복 제출을 건너뛰었습니다.";
    case "invalid_final_request_qty":
      return "최종 주문 수량이 0 이하라 제출하지 않았습니다.";
    case "missing_request_id":
      return "요청 ID가 없어 제출할 수 없었습니다.";
    case "unsupported_side":
      return `현재 제출기는 ${side || "해당"} 주문 유형을 지원하지 않습니다.`;
    case "holding_qty_missing":
      return "실계좌 보유수량을 찾지 못해 매도 주문을 만들지 못했습니다.";
    case "buy_qty_zero":
      return "매수 가능 수량 또는 계산 수량이 0이라 매수 주문이 보류되었습니다.";
    case "buy_context_unavailable":
      return "실계좌 조회 문맥이 없어 매수 주문 계산을 진행하지 못했습니다.";
    default:
      if (key === "execution blocked: --confirm-text LIVE_ORDER is required") {
        return "실주문 확인 문구가 맞지 않아 제출이 차단되었습니다.";
      }
      if (key.includes("market_closed")) {
        return "장 운영 시간이 아니어서 주문이 제출되지 않았습니다.";
      }
      return key;
  }
}

function summarizeExecutionFlow(preview, execution, runtime) {
  const previewRows = preview?.items || [];
  const executionRows = execution?.items || [];
  const executablePreview = previewRows.filter((row) => row.executable_now);
  const buyPreview = executablePreview.filter((row) => String(row.side || "").toUpperCase() === "BUY");
  const skippedRows = executionRows.filter((row) => String(row.submission_status || "").toLowerCase() === "skipped");
  const failedRows = executionRows.filter((row) => String(row.submission_status || "").toLowerCase() === "failed");
  const submittedRows = executionRows.filter((row) => String(row.submission_status || "").toLowerCase() === "submitted");
  const skipCounts = new Map();
  skippedRows.forEach((row) => {
    const key = String(row.skip_reason || "unknown").trim() || "unknown";
    skipCounts.set(key, (skipCounts.get(key) || 0) + 1);
  });
  const topSkip = [...skipCounts.entries()].sort((a, b) => b[1] - a[1])[0] || null;
  const topSkipText = topSkip ? `${describeExecutionReason(topSkip[0], { side: "BUY" }, runtime)} (${fmtNum(topSkip[1])}건)` : "-";

  if (!executionRows.length) {
    if (!runtime?.policy?.auto_trade_execute) {
      return {
        tone: "warn",
        title: "실주문 미시도",
        detail: `제출 가능 주문 ${fmtNum(executablePreview.length)}건이 있어도 execute 스위치가 OFF라 실제 주문은 나가지 않습니다.`,
      };
    }
    if (buyPreview.length) {
      return {
        tone: "warn",
        title: "실주문 결과 없음",
        detail: `매수 후보 ${fmtNum(buyPreview.length)}건이 있었지만 execution 산출물이 없어 제출 시도 여부를 아직 확인할 수 없습니다.`,
      };
    }
    return {
      tone: "warn",
      title: "실주문 산출물 없음",
      detail: "아직 execution 파일이 없어 실제 제출 결과를 확인할 수 없습니다.",
    };
  }

  if (submittedRows.length) {
    return {
      tone: "primary",
      title: `실제 제출 ${fmtNum(submittedRows.length)}건`,
      detail: failedRows.length
        ? `일부는 제출되었고 실패 ${fmtNum(failedRows.length)}건이 함께 있었습니다.`
        : "가장 최근 사이클에서 브로커 주문 제출이 발생했습니다.",
    };
  }

  if (skippedRows.length) {
    return {
      tone: "warn",
      title: `주문 보류 ${fmtNum(skippedRows.length)}건`,
      detail: topSkipText,
    };
  }

  if (failedRows.length) {
    return {
      tone: "bad",
      title: `주문 실패 ${fmtNum(failedRows.length)}건`,
      detail: describeExecutionReason(failedRows[0]?.skip_reason, failedRows[0], runtime),
    };
  }

  return {
    tone: "warn",
    title: "실주문 결과 확인 필요",
    detail: "execution 산출물은 있으나 제출/보류/실패 상태를 명확히 읽지 못했습니다.",
  };
}

function describePreviewExecutionRisk(row, runtime) {
  const expectedHoldReason = String(row?.expected_hold_reason || "").trim();
  if (expectedHoldReason) {
    return expectedHoldReason;
  }
  const blockedReason = String(row?.blocked_reason || "").trim();
  if (blockedReason) {
    return describeExecutionReason(blockedReason, row, runtime);
  }

  const side = String(row?.side || "").toUpperCase();
  const qty = Number(row?.final_request_qty);
  if (!Number.isFinite(qty) || qty <= 0) {
    return "최종 주문 수량이 0이라 실제 제출이 진행되지 않습니다.";
  }
  if (side === "BUY" && !runtime?.policy?.auto_trade_execute) {
    return "현재 execute 스위치가 OFF라 실주문은 시도되지 않습니다.";
  }
  if (side === "BUY" && !runtime?.policy?.auto_trade_allow_buy) {
    return "현재 BUY 실주문 스위치가 OFF라 매수는 제출 직전 보류됩니다.";
  }
  if (side === "BUY" && runtime?.policy?.buy_approval_required) {
    return "승인 파일에 request ID가 없으면 제출 단계에서 매수가 보류됩니다.";
  }
  if (!runtime?.policy?.auto_trade_execute) {
    return "현재 execute 스위치가 OFF라 실주문은 시도되지 않습니다.";
  }
  return "현재 preview 기준으로는 제출 가능 상태입니다.";
}

function ensurePreviewHeaderColumn() {
  const headerRow = document.querySelector("#previewWrap thead tr");
  if (!headerRow) return;
  if (headerRow.children.length >= 11) return;
  const th = document.createElement("th");
  th.textContent = "예상 보류 사유";
  const reasonHeader = headerRow.children[headerRow.children.length - 1];
  headerRow.insertBefore(th, reasonHeader);
}

function schedulerStateChip(row) {
  const status = String(row?.status || "").toLowerCase();
  if (status === "idle") return `<span class="chip good">대기</span>`;
  if (status === "running") return `<span class="chip watch">실행중</span>`;
  if (status === "failed" || status === "error") return `<span class="chip bad">오류</span>`;
  return `<span class="chip warn">${escapeHtml(status || "unknown")}</span>`;
}

function renderDecisionBanner(summary, intents, preview, execution, runtime) {
  const root = document.getElementById("decisionBanner");
  const gate = String(intents?.gate_status || preview?.gate_status || "").toUpperCase();
  const submittedCount = Number(execution?.summary?.submitted_count || 0);
  const skippedCount = Number(execution?.summary?.skipped_count || 0);
  const executableCount = Number((preview?.items || []).filter((item) => item.executable_now).length || 0);
  const blockedCount = Number((preview?.items || []).filter((item) => item.blocked_reason).length || 0);
  const executeOn = !!runtime?.policy?.auto_trade_execute;
  const buyOn = !!runtime?.policy?.auto_trade_allow_buy;
  const accountSyncedAt = summary?.summary?.generated_at || runtime?.live_account_sync_scheduler?.last_success_at || "-";
  const executionFlow = summarizeExecutionFlow(preview, execution, runtime);
  let headline = "현재 상태를 판정할 수 없습니다";
  let headlineTone = "warn";
  let headlineDetail = "필수 산출물이 아직 부족합니다.";

  if (!executeOn) {
    headline = "실주문 비활성";
    headlineTone = "warn";
    headlineDetail = "지금은 주문 초안과 제출 후보만 확인하는 상태입니다. 실제 주문은 실행 스위치가 꺼져 있습니다.";
  } else if (gate === "BLOCK") {
    headline = "신규 진입 차단";
    headlineTone = "bad";
    headlineDetail = "gate가 BLOCK이라 신규 진입은 막혀 있습니다. 기존 보유 축소/정리 중심으로 해석해야 합니다.";
  } else if (gate === "WATCH") {
    headline = "소액 제한 진입 구간";
    headlineTone = "warn";
    headlineDetail = "WATCH 상태에서는 제한된 신규 진입만 허용됩니다. 종목 수와 노출 비중이 작게 제한됩니다.";
  } else if (gate === "PILOT") {
    headline = "PILOT 제한 실운용";
    headlineTone = "primary";
    headlineDetail = "PILOT 상태에서는 WATCH보다 넓은 제한 진입이 허용되지만, BUY_ALLOWED처럼 풀 비중 운용과 교체매매는 아직 보류됩니다.";
  } else if (submittedCount > 0) {
    headline = `최근 주문 제출 ${submittedCount}건`;
    headlineTone = "primary";
    headlineDetail = "가장 최근 사이클에서 실제 주문 제출이 있었습니다. 상세 결과와 계좌 반영 상태를 같이 확인하세요.";
  } else if (skippedCount > 0) {
    headline = `주문 보류 ${fmtNum(skippedCount)}건`;
    headlineTone = executionFlow.tone || "warn";
    headlineDetail = executionFlow.detail;
  } else if (executableCount > 0) {
    headline = `제출 가능 주문 ${executableCount}건`;
    headlineTone = "primary";
    headlineDetail = "차단되지 않은 주문 초안이 존재합니다. 실행 스위치와 승인 조건을 함께 확인하세요.";
  }

  root.innerHTML = `
    <article class="decision-card ${headlineTone}">
      <h2 class="decision-title">현재 한줄 결론</h2>
      <div class="decision-value">${escapeHtml(headline)}</div>
      <div class="decision-detail">${escapeHtml(headlineDetail)}</div>
    </article>
    <article class="decision-card">
      <h2 class="decision-title">주문 판단</h2>
      <div class="decision-value">${escapeHtml(gate || "-")}</div>
      <div class="decision-detail">실행 가능 ${fmtNum(executableCount)}건 · 차단 ${fmtNum(blockedCount)}건 · 매수 허용 ${buyOn ? "ON" : "OFF"}</div>
    </article>
    <article class="decision-card">
      <h2 class="decision-title">계좌 최신화</h2>
      <div class="decision-value">${escapeHtml(String(accountSyncedAt).slice(0, 16) || "-")}</div>
      <div class="decision-detail">실계좌 보유/현금 기준 시각입니다. 주문 결과와 시차가 있는지 먼저 확인하세요.</div>
    </article>
  `;
}

function renderHero(summary, intents, preview, holdings, execution) {
  const summaryInfo = summary?.summary || {};
  const heroGrid = document.getElementById("heroGrid");
  const intentRows = Array.isArray(intents?.intents) ? intents.intents : [];
  const previewRows = Array.isArray(preview?.items) ? preview.items : [];
  const executionRows = Array.isArray(execution?.items) ? execution.items : [];
  const gate = intents?.gate_status || preview?.gate_status || summary?.preview_gate_status || "-";
  const gateSource = intents?.gate_source_status || preview?.gate_source_status || summary?.preview_gate_source_status || gate || "-";
  const gateRuntime = intents?.gate_runtime_status || preview?.gate_runtime_status || summary?.preview_gate_runtime_status || gate || "-";
  const executableIntentCount = intentRows.filter((item) => item.executable).length;
  const blockedPreviewCount = previewRows.filter((item) => item.blocked_reason).length;
  const submittedCount = Number(execution?.summary?.submitted_count || 0);
  const submittedBuyCount = executionRows.filter((row) =>
    String(row.submission_status || "").toLowerCase() === "submitted" &&
    String(row.side || "").toUpperCase() === "BUY"
  ).length;
  heroGrid.innerHTML = `
    <article class="hero-card">
      <div class="card-label">기준일</div>
      <div class="card-value">${escapeHtml(intents?.asof_date || preview?.asof_date || "-")}</div>
      <div class="card-detail">주문 판단 생성 ${escapeHtml(intents?.generated_at || "-")}</div>
    </article>
    <article class="hero-card">
      <div class="card-label">운영 모드</div>
      <div class="card-value">${escapeHtml(gateChipText(gate))}</div>
      <div class="card-detail">${escapeHtml(`source ${gateSource} | runtime ${gateRuntime}`)}</div>
      <div class="card-detail">${escapeHtml(intents?.gate_guidance || "현재 신규 진입 허용 범위를 나타냅니다.")}</div>
    </article>
    <article class="hero-card">
      <div class="card-label">판단 / 주문초안</div>
      <div class="card-value">${fmtNum(intents?.intent_count)} / ${fmtNum(preview?.summary?.request_count)}</div>
      <div class="card-detail">실행 후보 ${fmtNum(executableIntentCount)}건 | 차단 ${fmtNum(blockedPreviewCount)}건 | 제출 ${fmtNum(submittedCount)}건</div>
    </article>
    <article class="hero-card">
      <div class="card-label">실계좌 보유 / BUY 제출</div>
      <div class="card-value">${fmtNum(summary?.holding_count ?? holdings?.count)}</div>
      <div class="card-detail">BUY 제출 ${fmtNum(submittedBuyCount)}건 | 계좌 요약 ${summaryInfo ? "연결" : "없음"} | holdings csv ${holdings?.count ? "존재" : "없음"}</div>
    </article>
  `;
}

function renderStatus(summary, intents, preview, execution, runtime) {
  const summaryInfo = summary?.summary || {};
  const cash = summaryInfo?.cash_summary || {};
  const executionFlow = summarizeExecutionFlow(preview, execution, runtime);
  const cards = [
    {
      label: "계좌 요약",
      value: summaryInfo?.tot_evlu_amt ?? summaryInfo?.total_evaluation_amount,
      detail: `예수금 ${fmtNum(cash?.dnca_tot_amt ?? cash?.ord_psbl_cash)} | 평가손익 ${fmtNum(summaryInfo?.evlu_pfls_smtl_amt ?? summaryInfo?.pnl_amount)}`,
    },
    {
      label: "Intent 분포",
      value: (intents?.intents || []).length,
      detail: `BUY ${fmtNum((intents?.intents || []).filter((item) => item.intent_type === "BUY").length)} | TRIM ${fmtNum((intents?.intents || []).filter((item) => item.intent_type === "TRIM").length)} | REVIEW ${fmtNum((intents?.intents || []).filter((item) => item.intent_type === "REVIEW").length)}`,
    },
    {
      label: "주문 초안 상태",
      value: preview?.summary?.request_count,
      detail: `실행 가능 ${fmtNum((preview?.items || []).filter((item) => item.executable_now).length)} | 차단 ${fmtNum((preview?.items || []).filter((item) => item.blocked_reason).length)}`,
    },
    {
      label: "최근 제출 결과",
      value: execution?.summary?.submitted_count ?? 0,
        detail: `실패 ${fmtNum(execution?.summary?.failed_count)} | 건너뜀 ${fmtNum(execution?.summary?.skipped_count)}`,
    },
  ];

  document.getElementById("statusGrid").innerHTML = cards.map((item) => `
    <article class="hero-card">
      <div class="card-label">${escapeHtml(item.label)}</div>
      <div class="card-value">${fmtNum(item.value)}</div>
      <div class="card-detail">${escapeHtml(item.detail)}</div>
    </article>
  `).join("");
}

function renderStatusV2(summary, intents, preview, execution, runtime) {
  const summaryInfo = summary?.summary || {};
  const cash = summaryInfo?.cash_summary || {};
  const executionFlow = summarizeExecutionFlow(preview, execution, runtime);
  const cards = [
    {
      label: "계좌 요약",
      value: summaryInfo?.tot_evlu_amt ?? summaryInfo?.total_evaluation_amount,
      detail: `예수금 ${fmtNum(cash?.dnca_tot_amt ?? cash?.ord_psbl_cash)} | 평가손익 ${fmtNum(summaryInfo?.evlu_pfls_smtl_amt ?? summaryInfo?.pnl_amount)}`,
    },
    {
      label: "Intent 분포",
      value: (intents?.intents || []).length,
      detail: `BUY ${fmtNum((intents?.intents || []).filter((item) => item.intent_type === "BUY").length)} | TRIM ${fmtNum((intents?.intents || []).filter((item) => item.intent_type === "TRIM").length)} | REVIEW ${fmtNum((intents?.intents || []).filter((item) => item.intent_type === "REVIEW").length)}`,
    },
    {
      label: "주문 초안 상태",
      value: preview?.summary?.request_count,
      detail: `실행 가능 ${fmtNum((preview?.items || []).filter((item) => item.executable_now).length)} | 차단 ${fmtNum((preview?.items || []).filter((item) => item.blocked_reason).length)}`,
    },
    {
      label: "실주문 결과",
      value: execution?.summary?.submitted_count ?? 0,
      detail: `${executionFlow.title} | 실패 ${fmtNum(execution?.summary?.failed_count)} | 보류 ${fmtNum(execution?.summary?.skipped_count)}`,
    },
  ];

  document.getElementById("statusGrid").innerHTML = cards.map((item) => `
    <article class="hero-card">
      <div class="card-label">${escapeHtml(item.label)}</div>
      <div class="card-value">${fmtNum(item.value)}</div>
      <div class="card-detail">${escapeHtml(item.detail)}</div>
    </article>
  `).join("");
}

function renderStatusV3(summary, intents, preview, execution, runtime) {
  const summaryInfo = summary?.summary || {};
  const cash = summaryInfo?.cash_summary || {};
  const executionFlow = summarizeExecutionFlow(preview, execution, runtime);
  const ops = runtime?.operations || {};
  const controls = ops.controls || {};
  const opsAi = ops.ai || {};
  const opsCards = ops.cards || {};
  const statusCards = [
    {
      label: "Account",
      value: summaryInfo?.tot_evlu_amt ?? summaryInfo?.total_evaluation_amount,
      detail: `cash ${fmtNum(cash?.dnca_tot_amt ?? cash?.ord_psbl_cash)} | pnl ${fmtNum(summaryInfo?.evlu_pfls_smtl_amt ?? summaryInfo?.pnl_amount)}`,
    },
    {
      label: "Intents",
      value: (intents?.intents || []).length,
      detail: `BUY ${fmtNum((intents?.intents || []).filter((item) => item.intent_type === "BUY").length)} | TRIM ${fmtNum((intents?.intents || []).filter((item) => item.intent_type === "TRIM").length)} | REVIEW ${fmtNum((intents?.intents || []).filter((item) => item.intent_type === "REVIEW").length)}`,
    },
    {
      label: "Preview",
      value: preview?.summary?.request_count,
      detail: `executable ${fmtNum((preview?.items || []).filter((item) => item.executable_now).length)} | blocked ${fmtNum((preview?.items || []).filter((item) => item.blocked_reason).length)}`,
    },
    {
      label: "Execution",
      value: execution?.summary?.submitted_count ?? 0,
      detail: `${executionFlow.title} | failed ${fmtNum(execution?.summary?.failed_count)} | skipped ${fmtNum(execution?.summary?.skipped_count)}`,
    },
    {
      label: "AI Ops",
      value: opsAi.buy_candidate_count ?? 0,
      detail: `blocked ${fmtNum(opsAi.buy_blocked_count)} | submitted ${fmtNum(opsAi.submitted_count)} | filled ${fmtNum(opsAi.filled_count)} | close ${opsCards.close_batch?.today_success ? "OK" : "WAIT"} / sync ${opsCards.live_account_sync?.today_success ? "OK" : "WAIT"}`,
    },
    {
      label: "Safety",
      value: controls.global_kill_switch ? 1 : 0,
      detail: `GLOBAL ${flagText(controls.global_kill_switch)} | EXECUTE ${flagText(controls.auto_trade_execute)} | ALLOW_BUY ${flagText(controls.auto_trade_allow_buy)} | ${String(ops.summary?.overall_tone || "warning").toUpperCase()}`,
    },
  ];

  document.getElementById("statusGrid").innerHTML = statusCards.map((item) => `
    <article class="hero-card">
      <div class="card-label">${escapeHtml(item.label)}</div>
      <div class="card-value ${item.label === "Safety" ? opsToneClass(ops.summary?.overall_tone) : ""}">${fmtNum(item.value)}</div>
      <div class="card-detail">${escapeHtml(item.detail)}</div>
    </article>
  `).join("");
}

function renderAccountDetails(summary, runtime) {
  const info = summary?.summary || {};
  const raw = info?.summary_row || {};
  const derived = info?.derived_metrics || {};
  const policy = runtime?.policy || {};
  const items = [
    ["예수금", derived.cash_amount ?? raw.dnca_tot_amt, `D+1 ${fmtNum(raw.nxdy_excc_amt)} | 전일정산 ${fmtNum(raw.prvs_rcdl_excc_amt)}`],
    ["증권평가", raw.scts_evlu_amt, `매입원가 ${fmtNum(raw.pchs_amt_smtl_amt)} | 평가금액 ${fmtNum(raw.evlu_amt_smtl_amt)}`],
    ["총자산", derived.total_assets ?? raw.tot_evlu_amt, `전일총자산 ${fmtNum(raw.bfdy_tot_asst_evlu_amt)} | 자산증감 ${fmtNum(raw.asst_icdc_amt)}`],
    ["평가손익", raw.evlu_pfls_smtl_amt, `보유합산 ${fmtNum(derived.holding_pnl_amount)} | 자산증감률 ${fmtPct(raw.asst_icdc_erng_rt, 2)}`],
    ["현금 비중", derived.cash_ratio, `투자 비중 ${fmtPct(derived.invested_ratio, 1)} | 평균 보유비중 ${fmtPct(derived.avg_position_weight, 1)}`],
    ["자동주문 정책", null, `execute ${policy.auto_trade_execute ? "ON" : "OFF"} | buy ${policy.auto_trade_allow_buy ? "ALLOW" : "BLOCK"} | buy approval ${policy.buy_approval_required ? "REQ" : "FREE"}`],
  ];
  document.getElementById("accountDetailGrid").innerHTML = items.map(([label, value, detail]) => `
    <article class="hero-card">
      <div class="card-label">${escapeHtml(label)}</div>
      <div class="card-value">${value === null ? "-" : (label === "현금 비중" ? escapeHtml(fmtPct(value, 1)) : escapeHtml(fmtNum(value)))}</div>
      <div class="card-detail">${escapeHtml(detail)}</div>
    </article>
  `).join("");
}

function renderRunSummary(intents, preview, holdings, runtime) {
  const kv = document.getElementById("runSummaryKv");
  const chips = document.getElementById("runSummaryChips");
  const help = document.getElementById("runSummaryHelp");
  const blockedOrders = (preview?.items || []).filter((item) => item.blocked_reason);
  const missingHoldingQty = blockedOrders.filter((item) => item.blocked_reason === "holding_qty_missing").length;
  const policy = runtime?.policy || {};
  const gateSource = preview?.gate_source_status || intents?.gate_source_status || preview?.gate_status || intents?.gate_status || "-";
  const gateRuntime = preview?.gate_runtime_status || intents?.gate_runtime_status || preview?.gate_status || intents?.gate_status || "-";
  kv.innerHTML = [
    ["정책 버전", intents?.policy_version || "-"],
    ["보유 데이터 기준", intents?.holdings_source || "-"],
    ["주문 초안 gate", preview?.gate_status || "-"],
    ["실행 가능 intent", fmtNum((intents?.intents || []).filter((item) => item.executable).length)],
    ["차단된 주문 초안", fmtNum(blockedOrders.length)],
    ["주문 초안 생성시각", preview?.generated_at || "-"],
  ].map(([label, value]) => `
    <div class="kv-row">
      <span class="muted">${escapeHtml(label)}</span>
      <strong>${escapeHtml(String(value))}</strong>
    </div>
  `).join("");
  chips.innerHTML = `
    <span class="chip ${toneClass(intents?.gate_status)}">${escapeHtml(intents?.gate_status || "gate unknown")}</span>
    <span class="chip ${blockedOrders.length ? "bad" : "good"}">차단된 초안 ${fmtNum(blockedOrders.length)}</span>
    <span class="chip ${holdings?.count ? "good" : "bad"}">보유 ${fmtNum(holdings?.count)}종목</span>
    <span class="chip ${policy.auto_trade_execute ? "bad" : "watch"}">실주문 ${policy.auto_trade_execute ? "ON" : "OFF"}</span>
    <span class="chip ${policy.auto_trade_allow_buy ? "watch" : "good"}">매수 ${policy.auto_trade_allow_buy ? "허용" : "차단"}</span>
    <span class="chip ${policy.buy_approval_required ? "good" : "warn"}">매수 승인 ${policy.buy_approval_required ? "필수" : "자유"}</span>
  `;
  help.textContent = missingHoldingQty
    ? `현재 주문 초안 중 ${missingHoldingQty}건이 holding_qty_missing 상태입니다. 실계좌 보유 CSV와 intent 대상 코드 정합성을 먼저 맞춰야 합니다.`
    : "intent, preview, holdings 간의 기본 연결은 확인된 상태입니다.";
}

function renderFocus(intents, preview, holdings) {
  const focusKv = document.getElementById("focusKv");
  const focusChips = document.getElementById("focusChips");
  const focusHelp = document.getElementById("focusHelp");
  const topIntent = (intents?.intents || []).slice().sort((a, b) => (Number(b.priority) || 0) - (Number(a.priority) || 0))[0];
  const topHolding = (holdings?.items || []).slice().sort((a, b) => (Number(b.weight) || 0) - (Number(a.weight) || 0))[0];
  const topPreview = (preview?.items || []).slice().sort((a, b) => (Number(b.priority) || 0) - (Number(a.priority) || 0))[0];
  focusKv.innerHTML = [
    ["최우선 intent", topIntent ? `${topIntent.code} ${topIntent.name || ""}`.trim() : "-"],
    ["최우선 주문초안", topPreview ? `${topPreview.side} ${topPreview.code}` : "-"],
    ["최대 보유비중", topHolding ? `${topHolding.code} ${fmtPct(topHolding.weight, 1)}` : "-"],
  ].map(([label, value]) => `
    <div class="kv-row">
      <span class="muted">${escapeHtml(label)}</span>
      <strong>${escapeHtml(String(value))}</strong>
    </div>
  `).join("");
  focusChips.innerHTML = `
    ${topIntent ? `<span class="chip ${toneClass(topIntent.intent_type)}">${escapeHtml(topIntent.intent_type)}</span>` : ""}
    ${topPreview ? `<span class="chip ${toneClass(topPreview.side)}">${escapeHtml(topPreview.side)}</span>` : ""}
    ${topHolding ? `<span class="chip watch">최대 보유 ${escapeHtml(topHolding.code)}</span>` : ""}
  `;
  focusHelp.textContent = topIntent
    ? `${topIntent.code} ${topIntent.name || ""} intent의 우선순위가 가장 높습니다. 다만 preview 단계에서 blocked면 실제 주문 엔진으로 넘기면 안 됩니다.`
    : "아직 생성된 intent가 없습니다.";
}

function renderOperationalExplain(intents, preview, runtime, holdings) {
  const root = document.getElementById("operationalExplain");
  if (!root) return;
  const gate = String(intents?.gate_status || preview?.gate_status || "").toUpperCase();
  const intentRows = intents?.intents || [];
  const buyRows = intentRows.filter((row) => String(row.intent_type || "").toUpperCase() === "BUY");
  const sellRows = intentRows.filter((row) => ["EXIT", "TRIM"].includes(String(row.intent_type || "").toUpperCase()));
  const hasHoldings = Number(holdings?.count || 0) > 0;
  const executeOn = !!runtime?.policy?.auto_trade_execute;

  let title = "현재는 실운영 정리 화면입니다";
  let body = "아래 표는 실제 운영 산출물 기준입니다. 연구용 후보와 섞어 읽지 않는 것이 맞습니다.";
  if (gate === "BLOCK") {
    title = "현재 gate가 BLOCK이라 신규 매수가 보이지 않습니다";
    body = hasHoldings
      ? `실계좌 보유가 ${fmtNum(holdings?.count)}종목 있고 gate가 BLOCK이라 신규 진입보다 보유 축소나 정리 후보가 먼저 나옵니다. 지금 보이는 EXIT/TRIM은 정상 결과입니다.`
      : "빈 계좌여도 BLOCK에서는 신규 매수 후보를 실운영 intent로 올리지 않습니다. 지금 화면은 매수 후보 누락이 아니라 정책상 차단 상태입니다.";
  } else if (gate === "WATCH") {
    title = "현재는 WATCH 제한 진입 구간입니다";
    body = "실운영에서도 소액 제한 진입만 허용됩니다. 현재 화면은 실제 운영 산출물만 보여주며, 연구용 가정 후보와 섞어 보지 않습니다.";
  } else if (gate === "PILOT") {
    title = "현재는 PILOT 제한 실운용 구간입니다";
    body = "실운영에서 제한된 신규 진입이 실제로 허용되는 단계입니다. WATCH보다 넓게 진입할 수 있지만, BUY_ALLOWED처럼 전면 자동매수로 해석하면 안 됩니다.";
  } else if (buyRows.length) {
    title = `실운영 신규 매수 후보 ${fmtNum(buyRows.length)}건이 보입니다`;
    body = "현재 gate와 정책 기준에서 실제 BUY intent가 올라온 상태입니다. 아래 Trade Intents와 Preview 표를 그대로 읽으면 됩니다.";
  } else if (sellRows.length) {
    title = `현재는 매수보다 정리 후보 ${fmtNum(sellRows.length)}건이 우선입니다`;
    body = "정책상 신규 진입보다 리스크 축소가 먼저 선택된 상태입니다. 프런트 미반영으로 볼 상황은 아닙니다.";
  }
  if (!executeOn) {
    body += " 현재 실주문 스위치가 꺼져 있다면 이 화면은 주문 제출기보다 운영 모니터 역할에 가깝습니다.";
  }

  root.innerHTML = `
    <h3 class="explain-title">${escapeHtml(title)}</h3>
    <div class="explain-body">${escapeHtml(body)}</div>
  `;
}

function renderIntents(intents) {
  const tbody = document.getElementById("intentsTbody");
  const rows = intents?.intents || [];
  if (!rows.length) {
    document.getElementById("intentsWrap").innerHTML = `<div class="empty-state">trade intents 산출물이 아직 없습니다.</div>`;
    return;
  }
  tbody.innerHTML = rows.map((row) => `
    <tr>
      <td>${intentStateChip(row)}</td>
      <td class="mono">${escapeHtml(row.code || "-")}</td>
      <td>${escapeHtml(row.name || "-")}</td>
      <td>${escapeHtml(row.intent_type || "-")}</td>
      <td class="right">${fmtNum(row.priority)}</td>
      <td class="right">${fmtPct(row.target_weight, 1)}</td>
      <td>${escapeHtml(row.reason || "-")}</td>
    </tr>
  `).join("");
}

function renderPreview(preview, runtime) {
  const tbody = document.getElementById("previewTbody");
  const rows = preview?.items || [];
  ensurePreviewHeaderColumn();
  if (!rows.length) {
    document.getElementById("previewWrap").innerHTML = `<div class="empty-state">order requests preview 산출물이 아직 없습니다.</div>`;
    return;
  }
  tbody.innerHTML = rows.map((row) => `
    <tr>
      <td>${orderStateChip(row)}</td>
      <td class="mono">${escapeHtml(row.request_id || "-")}</td>
      <td class="mono">${escapeHtml(row.code || "-")}</td>
      <td>${escapeHtml(row.name || "-")}</td>
      <td>${escapeHtml(row.side || "-")}</td>
      <td>${escapeHtml(row.intent_type || "-")}</td>
      <td class="right">${fmtNum(row.final_request_qty)}</td>
      <td class="right">${fmtNum(row.allowed_qty)}</td>
      <td>${escapeHtml(row.blocked_reason || "-")}</td>
      <td>${escapeHtml(describePreviewExecutionRisk(row, runtime))}</td>
      <td>${escapeHtml(row.reason || "-")}</td>
    </tr>
  `).join("");
}

function renderExecution(execution, preview, runtime) {
  const tbody = document.getElementById("executionTbody");
  const rows = execution?.items || [];
  if (!rows.length) {
    const summary = summarizeExecutionFlow(preview, execution, runtime);
    document.getElementById("executionWrap").innerHTML = `<div class="empty-state">order requests execution 산출물이 아직 없습니다.</div>`;
    return;
  }
  tbody.innerHTML = rows.map((row) => `
    <tr>
      <td>${executionStateChip(row)}</td>
      <td class="mono">${escapeHtml(row.request_id || "-")}</td>
      <td class="mono">${escapeHtml(row.code || "-")}</td>
      <td>${escapeHtml(row.side || "-")}</td>
      <td class="right">${fmtNum(row.final_request_qty)}</td>
      <td class="mono">${escapeHtml(row.broker_order_id || "-")}</td>
      <td>${escapeHtml(describeExecutionReason(row.skip_reason, row, runtime))}</td>
    </tr>
  `).join("");
}

function renderExecutionV2(execution, preview, runtime) {
  const wrap = document.getElementById("executionWrap");
  const tbody = document.getElementById("executionTbody");
  const rows = execution?.items || [];
  if (!rows.length) {
    const summary = summarizeExecutionFlow(preview, execution, runtime);
    wrap.innerHTML = `<div class="empty-state"><strong>${escapeHtml(summary.title)}</strong><br>${escapeHtml(summary.detail)}</div>`;
    return;
  }
  tbody.innerHTML = rows.map((row) => `
    <tr>
      <td>${executionStateChip(row)}</td>
      <td class="mono">${escapeHtml(row.request_id || "-")}</td>
      <td class="mono">${escapeHtml(row.code || "-")}</td>
      <td>${escapeHtml(row.side || "-")}</td>
      <td class="right">${fmtNum(row.final_request_qty)}</td>
      <td class="mono">${escapeHtml(row.broker_order_id || "-")}</td>
      <td>${escapeHtml(describeExecutionReason(row.skip_reason, row, runtime))}</td>
    </tr>
  `).join("");
}

function renderHoldings(holdings) {
  const tbody = document.getElementById("holdingsTbody");
  const rows = holdings?.items || [];
  if (!rows.length) {
    document.getElementById("holdingsWrap").innerHTML = `<div class="empty-state">실계좌 보유 CSV가 아직 없습니다.</div>`;
    return;
  }
  tbody.innerHTML = rows.map((row) => `
    <tr>
      <td>${holdingStateChip(row)}</td>
      <td class="mono">${escapeHtml(row.code || "-")}</td>
      <td>${escapeHtml(row.name || "-")}</td>
      <td class="right">${fmtNum(row.qty)}</td>
      <td class="right">${fmtNum(row.avg_price, 2)}</td>
      <td class="right">${fmtNum(row.current_price, 2)}</td>
      <td class="right">${fmtNum(row.eval_amount)}</td>
      <td class="right ${Number(row.pnl_amount) >= 0 ? "pos" : "neg"}">${fmtNum(row.pnl_amount)}</td>
      <td class="right ${Number(row.pnl_pct) >= 0 ? "pos" : "neg"}">${fmtPct(row.pnl_pct, 2)}</td>
      <td class="right">${fmtPct(row.weight, 1)}</td>
    </tr>
  `).join("");
}

function renderAccountDetailsV2(summary, runtime) {
  const info = summary?.summary || {};
  const raw = info?.summary_row || {};
  const derived = info?.derived_metrics || {};
  const policy = runtime?.policy || {};
  const items = [
    {
      label: "예수금",
      valueClass: "",
      valueHtml: escapeHtml(fmtNum(derived.cash_amount ?? raw.dnca_tot_amt)),
      detailHtml: `D+1 ${escapeHtml(fmtNum(raw.nxdy_excc_amt))} | 전일정산 ${escapeHtml(fmtNum(raw.prvs_rcdl_excc_amt))}`,
    },
    {
      label: "증권평가",
      valueClass: "",
      valueHtml: escapeHtml(fmtNum(raw.scts_evlu_amt)),
      detailHtml: `매입원가 ${escapeHtml(fmtNum(raw.pchs_amt_smtl_amt))} | 평가금액 ${escapeHtml(fmtNum(raw.evlu_amt_smtl_amt))}`,
    },
    {
      label: "총자산",
      valueClass: "",
      valueHtml: escapeHtml(fmtNum(derived.total_assets ?? raw.tot_evlu_amt)),
      detailHtml: `전일총자산 ${escapeHtml(fmtNum(raw.bfdy_tot_asst_evlu_amt))} | 자산증감 ${metricHtml(raw.asst_icdc_amt, fmtNum)}`,
    },
    {
      label: "평가손익",
      valueClass: signedClass(raw.evlu_pfls_smtl_amt),
      valueHtml: metricHtml(raw.evlu_pfls_smtl_amt, fmtNum),
      detailHtml: `보유합산 ${metricHtml(derived.holding_pnl_amount, fmtNum)} | 자산증감률 ${metricHtml(raw.asst_icdc_erng_rt, fmtPct, 2)}`,
    },
    {
      label: "현금 비중",
      valueClass: "",
      valueHtml: escapeHtml(fmtPct(derived.cash_ratio, 1)),
      detailHtml: `투자 비중 ${escapeHtml(fmtPct(derived.invested_ratio, 1))} | 평균 보유비중 ${escapeHtml(fmtPct(derived.avg_position_weight, 1))}`,
    },
    {
      label: "자동주문 정책",
      valueClass: "",
      valueHtml: "-",
      detailHtml: `execute ${escapeHtml(policy.auto_trade_execute ? "ON" : "OFF")} | buy ${escapeHtml(policy.auto_trade_allow_buy ? "ALLOW" : "BLOCK")} | buy approval ${escapeHtml(policy.buy_approval_required ? "REQ" : "FREE")}`,
    },
  ];
  document.getElementById("accountDetailGrid").innerHTML = items.map((item) => `
    <article class="hero-card">
      <div class="card-label">${escapeHtml(item.label)}</div>
      <div class="card-value ${escapeHtml(item.valueClass)}">${item.valueHtml}</div>
      <div class="card-detail">${item.detailHtml}</div>
    </article>
  `).join("");
}

function renderAccountDetailsV3(summary, runtime) {
  const info = summary?.summary || {};
  const raw = info?.summary_row || {};
  const derived = info?.derived_metrics || {};
  const policy = runtime?.policy || {};
  const items = [
    {
      label: "예수금",
      valueClass: "",
      valueHtml: escapeHtml(fmtNum(derived.cash_amount ?? raw.dnca_tot_amt)),
      detailHtml: `D+1 ${escapeHtml(fmtNum(raw.nxdy_excc_amt))} | 전일정산 ${escapeHtml(fmtNum(raw.prvs_rcdl_excc_amt))}`,
    },
    {
      label: "증권평가",
      valueClass: "",
      valueHtml: escapeHtml(fmtNum(raw.scts_evlu_amt)),
      detailHtml: `매입원가 ${escapeHtml(fmtNum(raw.pchs_amt_smtl_amt))} | 평가금액 ${escapeHtml(fmtNum(raw.evlu_amt_smtl_amt))}`,
    },
    {
      label: "총자산",
      valueClass: "",
      valueHtml: escapeHtml(fmtNum(derived.total_assets ?? raw.tot_evlu_amt)),
      detailHtml: `전일총자산 ${escapeHtml(fmtNum(raw.bfdy_tot_asst_evlu_amt))} | 자산증감 ${metricHtml(raw.asst_icdc_amt, fmtNum)}`,
    },
    {
      label: "평가손익",
      valueClass: signedClass(raw.evlu_pfls_smtl_amt),
      valueHtml: metricHtml(raw.evlu_pfls_smtl_amt, fmtNum),
      detailHtml: `보유합산 ${metricHtml(derived.holding_pnl_amount, fmtNum)} | 자산증감률 ${metricHtml(raw.asst_icdc_erng_rt, fmtPct, 2)}`,
    },
    {
      label: "현금 비중",
      valueClass: "",
      valueHtml: escapeHtml(fmtPct(derived.cash_ratio, 1)),
      detailHtml: `투자 비중 ${escapeHtml(fmtPct(derived.invested_ratio, 1))} | 평균 보유비중 ${escapeHtml(fmtPct(derived.avg_position_weight, 1))}`,
    },
    {
      label: "자동주문 정책",
      valueClass: "",
      valueHtml: "-",
      detailHtml: `execute ${escapeHtml(policy.auto_trade_execute ? "ON" : "OFF")} | buy ${escapeHtml(policy.auto_trade_allow_buy ? "ALLOW" : "BLOCK")} | buy approval ${escapeHtml(policy.buy_approval_required ? "REQ" : "FREE")}`,
    },
  ];
  document.getElementById("accountDetailGrid").innerHTML = items.map((item) => `
    <article class="hero-card">
      <div class="card-label">${escapeHtml(item.label)}</div>
      <div class="card-value ${escapeHtml(item.valueClass)}">${item.valueHtml}</div>
      <div class="card-detail">${item.detailHtml}</div>
    </article>
  `).join("");
}

function renderConsistency(consistency) {
  const root = document.getElementById("consistencyPanel");
  if (!root) return;
  if (!consistency || !Object.keys(consistency).length) {
    root.innerHTML = `<div class="empty-state">live_trade_consistency_report 산출물이 아직 없습니다.</div>`;
    return;
  }

  const counts = consistency.counts || {};
  const warnings = Array.isArray(consistency.warnings) ? consistency.warnings : [];
  const missingFills = Array.isArray(consistency.submitted_without_fill) ? consistency.submitted_without_fill : [];
  const requestsWithoutExecution = Array.isArray(consistency.request_without_execution) ? consistency.request_without_execution : [];
  const warningCount = Number(consistency.warning_count || warnings.length || 0);
  const tone = warningCount > 0 ? "bad" : "good";
  const fillCount = Number(counts.filled_count || 0);
  const submittedCount = Number(counts.submitted_count || 0);
  const fillDetail = submittedCount > 0
    ? `${fmtNum(fillCount)} / ${fmtNum(submittedCount)}`
    : fmtNum(fillCount);

  const warningHtml = warnings.length
    ? warnings.slice(0, 4).map((item) => `<div class="state-line">${escapeHtml(item)}</div>`).join("")
    : `<div class="state-line">정합성 경고가 없습니다.</div>`;

  const missingHtml = missingFills.length
    ? `
      <div class="table-wrap" style="margin-top:12px;">
        <table class="status-table">
          <thead>
            <tr>
              <th>request_id</th>
              <th>주문번호</th>
              <th>종목</th>
              <th>구분</th>
              <th>제출시각</th>
            </tr>
          </thead>
          <tbody>
            ${missingFills.slice(0, 8).map((row) => `
              <tr>
                <td class="mono">${escapeHtml(row.request_id || "")}</td>
                <td class="mono">${escapeHtml(row.broker_order_id || "")}</td>
                <td>${escapeHtml(row.code || "")}</td>
                <td>${escapeHtml(row.side || "")}</td>
                <td>${escapeHtml(fmtRuntimeDateTime(row.submitted_at))}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    `
    : "";

  root.innerHTML = `
    <div class="kv">
      <div class="kv-row"><span>기준일</span><strong>${escapeHtml(consistency.as_of_date || "-")}</strong></div>
      <div class="kv-row"><span>Intent / Preview / Execution</span><strong>${fmtNum(counts.intent_count)} / ${fmtNum(counts.request_count)} / ${fmtNum(counts.execution_count)}</strong></div>
      <div class="kv-row"><span>Submitted / Filled</span><strong>${escapeHtml(fillDetail)}</strong></div>
      <div class="kv-row"><span>Request without execution</span><strong>${fmtNum(requestsWithoutExecution.length)}</strong></div>
    </div>
    <div class="chip-row">
      <span class="chip ${tone}">warnings ${fmtNum(warningCount)}</span>
      <span class="chip ${missingFills.length ? "bad" : "good"}">missing fill ${fmtNum(missingFills.length)}</span>
      <span class="chip ${requestsWithoutExecution.length ? "warn" : "good"}">missing execution ${fmtNum(requestsWithoutExecution.length)}</span>
    </div>
    ${warningHtml}
    ${missingHtml}
  `;
}

function extractSignedReturn(item) {
  const returns = item?.returns || {};
  const order = ["d10", "d5", "d3", "d1", "d0"];
  for (const key of order) {
    const row = returns[key];
    const value = Number(row?.signed_return);
    if (Number.isFinite(value)) return { horizon: key.toUpperCase(), value };
  }
  const match = String(item?.review_note || "").match(/(d\d+)_signed_return=([-+]?\d+(?:\.\d+)?)%/i);
  if (match) return { horizon: match[1].toUpperCase(), value: Number(match[2]) / 100 };
  return { horizon: "-", value: null };
}

function reviewOutcomeChip(outcome) {
  const value = String(outcome || "").toLowerCase();
  if (value.includes("positive") || value.includes("good")) return "good";
  if (value.includes("pending")) return "warn";
  if (value.includes("bad") || value.includes("negative")) return "bad";
  return "watch";
}

function renderReviewSummaryRows(title, rows, keyName) {
  const values = Array.isArray(rows) ? rows.slice(0, 4) : [];
  if (!values.length) return "";
  return `
    <div class="table-wrap" style="margin-top:12px;">
      <table class="status-table">
        <thead>
          <tr>
            <th>${escapeHtml(title)}</th>
            <th class="right">건수</th>
            <th class="right">관찰</th>
            <th class="right">평균</th>
            <th class="right">승률</th>
          </tr>
        </thead>
        <tbody>
          ${values.map((row) => `
            <tr>
              <td>${escapeHtml(row[keyName] || "-")}</td>
              <td class="right">${fmtNum(row.count)}</td>
              <td class="right">${fmtNum(row.observed_count)}</td>
              <td class="right ${signedClass(row.avg_signed_return)}">${escapeHtml(fmtPct(row.avg_signed_return, 2))}</td>
              <td class="right">${escapeHtml(fmtPct(row.win_rate, 1))}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderReviewRecommendations(summary) {
  const recommendations = Array.isArray(summary?.recommendations) ? summary.recommendations.slice(0, 3) : [];
  if (!recommendations.length) return "";
  return `
    <div class="chip-row">
      ${recommendations.map((item) => `
        <span class="chip ${item.level === "watch" ? "warn" : "watch"}">${escapeHtml(item.topic || "review")}</span>
      `).join("")}
    </div>
    ${recommendations.map((item) => `
      <div class="state-line">${escapeHtml([item.group, item.message].filter(Boolean).join(": "))}</div>
    `).join("")}
  `;
}

function renderTradeReview(review, summary) {
  const root = document.getElementById("reviewPanel");
  if (!root) return;
  if (!review || !Object.keys(review).length) {
    root.innerHTML = `<div class="empty-state">live_trade_review_report 산출물이 아직 없습니다.</div>`;
    return;
  }

  const items = Array.isArray(review.items) ? review.items : [];
  const outcomeCounts = Array.isArray(review.outcome_counts) ? review.outcome_counts : [];
  const overview = summary?.overview || {};
  const rows = items.slice(0, 10);
  const countsHtml = outcomeCounts.length
    ? outcomeCounts.map((row) => `
      <span class="chip ${reviewOutcomeChip(row.outcome_label)}">${escapeHtml(row.outcome_label || "-")} ${fmtNum(row.count)}</span>
    `).join("")
    : `<span class="chip warn">no outcome summary</span>`;

  const tableHtml = rows.length
    ? `
      <div class="table-wrap" style="margin-top:12px;">
        <table class="status-table">
          <thead>
            <tr>
              <th>request_id</th>
              <th>코드</th>
              <th>종목명</th>
              <th>매수/매도</th>
              <th>Intent</th>
              <th class="right">체결가</th>
              <th class="right">성과</th>
              <th>판정</th>
            </tr>
          </thead>
          <tbody>
            ${rows.map((item) => {
              const ret = extractSignedReturn(item);
              return `
                <tr>
                  <td class="mono">${escapeHtml(item.request_id || "")}</td>
                  <td class="mono">${escapeHtml(item.code || "")}</td>
                  <td>${escapeHtml(item.name || "")}</td>
                  <td>${escapeHtml(item.side || "")}</td>
                  <td>${escapeHtml(item.intent_type || "")}</td>
                  <td class="right">${escapeHtml(fmtNum(item.filled_price, 0))}</td>
                  <td class="right ${signedClass(ret.value)}">${escapeHtml(ret.value === null ? "-" : `${ret.horizon} ${fmtPct(ret.value, 2)}`)}</td>
                  <td><span class="chip ${reviewOutcomeChip(item.outcome_label)}">${escapeHtml(item.outcome_label || "-")}</span></td>
                </tr>
              `;
            }).join("")}
          </tbody>
        </table>
      </div>
    `
    : `<div class="empty-state" style="margin-top:12px;">리뷰 대상 체결이 없습니다.</div>`;

  root.innerHTML = `
    <div class="kv">
      <div class="kv-row"><span>기준일 / 리뷰일</span><strong>${escapeHtml(review.as_of_date || "-")} / ${escapeHtml(review.review_date || "-")}</strong></div>
      <div class="kv-row"><span>가격 최신일</span><strong>${escapeHtml(review.price_latest_date || "-")}</strong></div>
      <div class="kv-row"><span>리뷰 건수</span><strong>${fmtNum(review.reviewed_count || items.length)}</strong></div>
      <div class="kv-row"><span>누적 평균 / 승률</span><strong>${escapeHtml(fmtPct(overview.avg_signed_return, 2))} / ${escapeHtml(fmtPct(overview.win_rate, 1))}</strong></div>
    </div>
    <div class="chip-row">${countsHtml}</div>
    ${renderReviewRecommendations(summary)}
    ${renderReviewSummaryRows("Intent", summary?.by_intent, "intent_type")}
    ${renderReviewSummaryRows("Rank", summary?.by_rank_bucket, "rank_bucket")}
    ${tableHtml}
  `;
}

function sampleTone(status) {
  const value = String(status || "").toUpperCase();
  if (value === "ACTIONABLE") return "good";
  if (value === "MONITOR_ONLY" || value === "REVIEW_READY") return "watch";
  if (value === "PROMOTE_CANDIDATE") return "good";
  if (value === "REJECT") return "bad";
  return "warn";
}

function gateStatusChip(ok, label, value, failTone = "warn") {
  return `<span class="chip ${ok ? "good" : failTone}">${escapeHtml(label)} ${escapeHtml(value)}</span>`;
}

function helpTip(text) {
  return `<span class="help-tip" title="${escapeHtml(text)}">?</span>`;
}

function metricLabel(label, help) {
  return `${escapeHtml(label)}${help ? helpTip(help) : ""}`;
}

function statusText(value) {
  const key = String(value || "").toUpperCase();
  const labels = {
    ACTIONABLE: "운영 참고 가능",
    MONITOR_ONLY: "관찰 전용",
    REVIEW_READY: "검토 가능",
    PROMOTE_CANDIDATE: "운영 반영 후보",
    KEEP_SHADOW: "데이터 축적 중",
    REJECT: "반영 보류",
    PASS: "검증 통과",
    FAIL: "검증 실패",
  };
  return labels[key] || key || "-";
}

function statusChip(value) {
  const raw = String(value || "-").toUpperCase();
  const label = statusText(raw);
  const suffix = raw && raw !== label && raw !== "-" ? ` · ${raw}` : "";
  return `<span class="chip ${sampleTone(raw)}">${escapeHtml(label + suffix)}</span>`;
}

function validationChip(value) {
  const raw = String(value || "MISSING").toUpperCase();
  const ok = raw === "PASS";
  return `<span class="chip ${ok ? "good" : "bad"}">${escapeHtml(statusText(raw))}${raw !== statusText(raw) ? ` · ${escapeHtml(raw)}` : ""}</span>`;
}

function gateLabel(value) {
  const key = String(value || "").toUpperCase();
  const labels = {
    BUY_ALLOWED: "매수 허용",
    PILOT: "제한 실운용",
    WATCH: "관찰 진입",
    BLOCK: "신규 진입 차단",
  };
  return labels[key] || key || "-";
}

function gateChipText(value) {
  const raw = String(value || "").toUpperCase();
  const label = gateLabel(raw);
  return raw && raw !== label && raw !== "-" ? `${label} · ${raw}` : label;
}

function helpDetails(items) {
  return `
    <details class="details-help">
      <summary>용어와 판단 기준</summary>
      <dl class="help-list">
        ${items.map((item) => `
          <div>
            <dt>${escapeHtml(item.term)}</dt>
            <dd>${escapeHtml(item.desc)}</dd>
          </div>
        `).join("")}
      </dl>
    </details>
  `;
}

function translatePromotionBlocker(text) {
  const value = String(text || "");
  if (value.includes("D+5 observed_count is below 30")) return "D+5 관찰 표본이 30건 미만입니다.";
  if (value.includes("Guard-applied observed_count is below 30")) return "Guard 적용군 관찰 표본이 30건 미만입니다.";
  if (value.includes("Guard-not-applied observed_count is below 30")) return "Guard 미적용군 관찰 표본이 30건 미만입니다.";
  if (value.includes("Production top20 vs shadow top20 comparison is not available")) return "Production Top20과 Shadow Top20 비교가 아직 불가능합니다.";
  if (value.includes("Closed-trade report is not available")) return "Closed Trade 리포트가 아직 없습니다.";
  if (value.includes("Closed-trade observed_count is below 30")) return "Closed Trade 관찰 표본이 30건 미만입니다.";
  if (value.includes("position snapshot avg_price fallback")) return "스냅샷 평균단가를 보조 원가로 사용해 lot 단위 손익이 근사값입니다.";
  if (value.includes("unmatched SELL basis")) return "일부 매도 체결의 매수 원가 기준을 찾지 못했습니다.";
  return value;
}

function latestHorizonRow(rows, horizon) {
  return (Array.isArray(rows) ? rows : []).find((row) => Number(row.horizon) === horizon) || {};
}

function guardAppliedRow(rows, applied, horizon = 5) {
  return (Array.isArray(rows) ? rows : []).find((row) =>
    Number(row.horizon) === horizon && Boolean(row.shadow_quality_risk_guard_applied) === applied
  ) || {};
}

function renderClosedQualityGuardTable(rows) {
  const values = Array.isArray(rows) ? rows.slice(0, 6) : [];
  if (!values.length) return "";
  return `
    <div class="table-wrap" style="margin-top:12px;">
      <table class="status-table">
        <thead>
          <tr>
            <th>Penalty</th>
            <th>Shadow Delta</th>
            <th class="right">건수</th>
            <th class="right">관찰</th>
            <th class="right">실현손익</th>
            <th class="right">평균</th>
            <th class="right">승률</th>
          </tr>
        </thead>
        <tbody>
          ${values.map((row) => `
            <tr>
              <td>${escapeHtml(row.guard_penalty_bucket || "-")}</td>
              <td>${escapeHtml(row.shadow_rank_delta_bucket || "-")}</td>
              <td class="right">${fmtNum(row.count)}</td>
              <td class="right">${fmtNum(row.observed_count)}</td>
              <td class="right ${signedClass(row.realized_net_pnl)}">${escapeHtml(fmtNum(row.realized_net_pnl))}</td>
              <td class="right ${signedClass(row.avg_realized_return)}">${escapeHtml(fmtPct(row.avg_realized_return, 2))}</td>
              <td class="right">${escapeHtml(fmtPct(row.win_rate, 1))}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderLiveKpiDaily(report) {
  const root = document.getElementById("liveKpiPanel");
  if (!root) return;
  if (!report || !Object.keys(report).length) {
    root.innerHTML = `<div class="empty-state">live_kpi_daily_report 산출물이 아직 없습니다.</div>`;
    return;
  }
  const overview = report.overview || {};
  const account = report.account || {};
  const today = report.today_counts || {};
  const d0 = latestHorizonRow(report.horizon_summary, 0);
  const d5 = latestHorizonRow(report.horizon_summary, 5);
  const warnings = Array.isArray(report.warnings) ? report.warnings.slice(0, 3) : [];
  root.innerHTML = `
    <p class="card-guide">오늘 자동매매 산출물이 정상적으로 쌓였는지와, 체결 후 수익률 표본이 충분한지 보는 카드입니다.</p>
    <div class="kv">
      <div class="kv-row"><span>${metricLabel("기준일", "이 리포트가 계산된 거래 기준일입니다.")}</span><strong>${escapeHtml(report.as_of_date || "-")}</strong></div>
      <div class="kv-row"><span>${metricLabel("표본 상태", "성과 판단에 필요한 관찰 건수가 충분한지 나타냅니다.")}</span><strong>${statusChip(report.sample_status)}</strong></div>
      <div class="kv-row"><span>${metricLabel("총자산 / 현금비중", "실계좌 요약 기준 총자산과 현금 비중입니다.")}</span><strong>${escapeHtml(fmtNum(account.total_assets))} / ${escapeHtml(fmtPct(account.cash_ratio, 1))}</strong></div>
      <div class="kv-row"><span>${metricLabel("판단 / 요청 / 제출 / 체결", "전략 판단부터 실제 체결까지 오늘 생성된 건수 흐름입니다.")}</span><strong>${fmtNum(today.decision_count)} / ${fmtNum(today.request_count)} / ${fmtNum(today.execution_count)} / ${fmtNum(today.fill_count)}</strong></div>
      <div class="kv-row"><span>${metricLabel("D0 평균 / 관찰", "체결 당일 기준 수익률 평균과 계산 가능한 표본 수입니다.")}</span><strong class="${signedClass(d0.avg_return)}">${escapeHtml(fmtPct(d0.avg_return, 2))} / ${fmtNum(d0.observed_count)}</strong></div>
      <div class="kv-row"><span>${metricLabel("D+5 평균 / 관찰", "체결 후 5거래일 기준 수익률 평균과 계산 가능한 표본 수입니다. 승격 판단의 핵심 표본입니다.")}</span><strong class="${signedClass(d5.avg_return)}">${escapeHtml(fmtPct(d5.avg_return, 2))} / ${fmtNum(d5.observed_count)}</strong></div>
    </div>
    <div class="chip-row">
      <span class="chip ${Number(overview.missing_ranking_context_count || 0) ? "warn" : "good"}">ranking missing ${fmtNum(overview.missing_ranking_context_count)}</span>
      <span class="chip ${Number(report?.consistency?.warning_count || 0) ? "warn" : "good"}">consistency warnings ${fmtNum(report?.consistency?.warning_count)}</span>
    </div>
    ${warnings.map((item) => `<div class="state-line">${escapeHtml(item)}</div>`).join("")}
    ${helpDetails([
      { term: "표본 상태", desc: "ACTIONABLE이면 참고 가능, MONITOR_ONLY면 아직 데이터 축적 단계입니다." },
      { term: "D0 / D+5", desc: "D0는 체결 당일, D+5는 체결 후 5거래일 성과입니다." },
      { term: "ranking missing", desc: "체결 또는 판단 데이터와 리서치 랭킹 문맥이 연결되지 않은 건수입니다." },
      { term: "consistency warnings", desc: "Intent, 주문 초안, 제출, 체결 산출물 사이에 확인이 필요한 항목 수입니다." },
    ])}
  `;
}

function renderQualityGuardReview(report, validation) {
  const root = document.getElementById("qualityGuardPanel");
  if (!root) return;
  if (!report || !Object.keys(report).length) {
    root.innerHTML = `<div class="empty-state">quality_risk_guard_live_review 산출물이 아직 없습니다.</div>`;
    return;
  }
  const overview = report.overview || {};
  const closed = report.closed_trade_summary || {};
  const d0 = latestHorizonRow(report.horizon_summary, 0);
  const d5 = latestHorizonRow(report.horizon_summary, 5);
  const appliedD5 = guardAppliedRow(report.by_guard_applied, true, 5);
  const notAppliedD5 = guardAppliedRow(report.by_guard_applied, false, 5);
  const blockers = Array.isArray(report.promotion_blockers) ? report.promotion_blockers.slice(0, 6) : [];
  const productionBlocked = String(report.promotion_status || "").toUpperCase() !== "PROMOTE_CANDIDATE";
  const closedQualityRows = closed.by_quality_guard || [];
  const validationStatus = String(validation?.validation_status || "").toUpperCase();
  const validationOk = validationStatus === "PASS";
  root.innerHTML = `
    <p class="card-guide">quality_risk_guard를 실제 운영 점수에 반영해도 되는지 보는 카드입니다. 현재는 표본이 부족하면 자동으로 데이터 축적 상태로 남습니다.</p>
    <div class="kv">
      <div class="kv-row"><span>${metricLabel("승격 상태", "Quality Guard를 production 로직에 반영할 수 있는지의 최종 판단입니다.")}</span><strong>${statusChip(report.promotion_status)}</strong></div>
      <div class="kv-row"><span>${metricLabel("산출물 검증", "리포트 JSON 구조, 필수 값, 차단 조건이 깨지지 않았는지 검사한 결과입니다.")}</span><strong>${validationChip(validationStatus)}</strong></div>
      <div class="kv-row"><span>${metricLabel("표본 상태", "성과 비교에 필요한 표본이 충분한지 나타냅니다.")}</span><strong>${statusChip(report.sample_status)}</strong></div>
      <div class="kv-row"><span>${metricLabel("Guard 적용 / 미적용", "Shadow 계산에서 품질 가드 페널티가 걸린 후보와 걸리지 않은 후보 수입니다.")}</span><strong>${fmtNum(overview.guard_applied_count)} / ${fmtNum(overview.guard_not_applied_count)}</strong></div>
      <div class="kv-row"><span>${metricLabel("Production / Shadow Top20", "현재 운영 랭킹 Top20과 Quality Guard 적용 가정 Top20의 후보 수입니다.")}</span><strong>${fmtNum(overview.production_top20_count)} / ${fmtNum(overview.shadow_top20_count)}</strong></div>
      <div class="kv-row"><span>${metricLabel("D0 평균 / 관찰", "체결 당일 성과입니다. 방향성만 참고하고 승격 판단은 D+5와 청산 표본을 더 봅니다.")}</span><strong class="${signedClass(d0.avg_return)}">${escapeHtml(fmtPct(d0.avg_return, 2))} / ${fmtNum(d0.observed_count)}</strong></div>
      <div class="kv-row"><span>${metricLabel("D+5 평균 / 관찰", "체결 후 5거래일 성과입니다. 최소 30건 이상 쌓여야 비교가 의미 있습니다.")}</span><strong class="${signedClass(d5.avg_return)}">${escapeHtml(fmtPct(d5.avg_return, 2))} / ${fmtNum(d5.observed_count)}</strong></div>
      <div class="kv-row"><span>${metricLabel("Closed PnL / 관찰", "실제로 매도까지 끝난 거래의 실현손익과 표본 수입니다.")}</span><strong class="${signedClass(closed.realized_net_pnl)}">${escapeHtml(fmtNum(closed.realized_net_pnl))} / ${fmtNum(closed.observed_count)}</strong></div>
      <div class="kv-row"><span>${metricLabel("Closed 평균 / fallback", "청산 거래 평균 수익률과 스냅샷 평균단가로 원가를 보조 계산한 건수입니다.")}</span><strong class="${signedClass(closed.avg_realized_return)}">${escapeHtml(fmtPct(closed.avg_realized_return, 2))} / ${fmtNum(closed.snapshot_fallback_count)}</strong></div>
    </div>
    <div class="chip-row">
      <span class="chip ${productionBlocked ? "bad" : "good"}">${productionBlocked ? "운영 반영 보류" : "운영 반영 후보"}</span>
      <span class="chip ${Number(d5.observed_count || 0) >= 30 ? "good" : "warn"}">D+5 observed ${fmtNum(d5.observed_count)}</span>
      ${gateStatusChip(Number(appliedD5.observed_count || 0) >= 30, "applied", fmtNum(appliedD5.observed_count))}
      ${gateStatusChip(Number(notAppliedD5.observed_count || 0) >= 30, "not-applied", fmtNum(notAppliedD5.observed_count))}
      <span class="chip ${overview.has_production_shadow_top20_comparison ? "good" : "warn"}">top20 compare ${overview.has_production_shadow_top20_comparison ? "YES" : "NO"}</span>
      <span class="chip ${Number(closed.observed_count || 0) >= 30 ? "good" : "warn"}">closed observed ${fmtNum(closed.observed_count)}</span>
      <span class="chip ${Number(closed.snapshot_fallback_count || 0) ? "warn" : "good"}">snapshot fallback ${fmtNum(closed.snapshot_fallback_count)}</span>
      <span class="chip ${validationOk ? "good" : "bad"}">validation ${escapeHtml(validationStatus || "MISSING")}</span>
    </div>
    ${blockers.map((item) => `<div class="state-line">${escapeHtml(translatePromotionBlocker(item))}</div>`).join("")}
    ${(Array.isArray(validation?.issues) ? validation.issues.slice(0, 3) : []).map((item) => `<div class="state-line">${escapeHtml(item)}</div>`).join("")}
    ${renderClosedQualityGuardTable(closedQualityRows)}
    ${helpDetails([
      { term: "데이터 축적 중", desc: "KEEP_SHADOW 상태입니다. 실제 매매 로직은 바꾸지 않고 결과만 비교합니다." },
      { term: "Guard 적용군", desc: "품질 가드가 위험하다고 판단해 Shadow 점수에 페널티를 준 후보입니다." },
      { term: "Top20 비교", desc: "기존 운영 랭킹과 Guard 적용 가정 랭킹의 상위 후보가 얼마나 달라지는지 보는 비교입니다." },
      { term: "snapshot fallback", desc: "매수 lot 원가가 부족해 계좌 스냅샷 평균단가로 보조 계산한 청산 거래입니다. 많으면 해석을 보수적으로 봅니다." },
    ])}
  `;
}

function renderClosedTradeReport(report) {
  const root = document.getElementById("closedTradePanel");
  if (!root) return;
  if (!report || !Object.keys(report).length) {
    root.innerHTML = `<div class="empty-state">live_closed_trade_report 산출물이 아직 없습니다.</div>`;
    return;
  }
  const overview = report.overview || {};
  const warnings = Array.isArray(report.warnings) ? report.warnings.slice(0, 3) : [];
  const recentRows = Array.isArray(report.recent_closed_trades) ? report.recent_closed_trades.slice(0, 6) : [];
  const recentHtml = recentRows.length
    ? `
      <div class="table-wrap" style="margin-top:12px;">
        <table class="status-table">
          <thead>
            <tr>
              <th>종목</th>
              <th>Intent</th>
              <th class="right">수량</th>
              <th class="right">매칭</th>
              <th class="right">미매칭</th>
              <th class="right">실현손익</th>
              <th class="right">수익률</th>
              <th>매칭</th>
            </tr>
          </thead>
          <tbody>
            ${recentRows.map((row) => `
              <tr>
                <td><span class="mono">${escapeHtml(row.code || "")}</span> ${escapeHtml(row.name || "")}</td>
                <td>${escapeHtml(row.intent_type || "-")}</td>
                <td class="right">${fmtNum(row.sell_qty)}</td>
                <td class="right">${fmtNum(row.matched_qty)}</td>
                <td class="right">${fmtNum(row.unmatched_qty)}</td>
                <td class="right ${signedClass(row.realized_net_pnl)}">${escapeHtml(fmtNum(row.realized_net_pnl))}</td>
                <td class="right ${signedClass(row.realized_return)}">${escapeHtml(fmtPct(row.realized_return, 2))}</td>
                <td><span class="chip ${row.match_status === "MATCHED" ? "good" : "warn"}">${escapeHtml(row.match_status || "-")}</span></td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    `
    : `<div class="empty-state" style="margin-top:12px;">닫힌 거래가 아직 없습니다.</div>`;
  root.innerHTML = `
    <p class="card-guide">매도까지 끝난 거래만 모아 실제로 돈을 벌었는지 확인하는 카드입니다. 아직 표본이 적으면 방향성만 참고합니다.</p>
    <div class="kv">
      <div class="kv-row"><span>${metricLabel("최근 청산일", "가장 최근 매도 체결이 반영된 날짜입니다.")}</span><strong>${escapeHtml(report.latest_closed_date || "-")}</strong></div>
      <div class="kv-row"><span>${metricLabel("표본 상태", "청산 거래 표본이 판단 가능한 수준인지 나타냅니다.")}</span><strong>${statusChip(report.sample_status)}</strong></div>
      <div class="kv-row"><span>${metricLabel("닫힌 거래 / 관찰", "매도까지 끝난 거래 수와 수익률 계산에 사용 가능한 거래 수입니다.")}</span><strong>${fmtNum(overview.closed_trade_count)} / ${fmtNum(overview.observed_count)}</strong></div>
      <div class="kv-row"><span>${metricLabel("매칭 방식", "매도 체결을 어떤 매수 원가와 연결해 손익을 계산했는지 나타냅니다.")}</span><strong>${escapeHtml(overview.match_method || "-")}</strong></div>
      <div class="kv-row"><span>${metricLabel("실현손익", "닫힌 거래의 수수료/세금 반영 후 손익 합계입니다.")}</span><strong class="${signedClass(overview.realized_net_pnl)}">${escapeHtml(fmtNum(overview.realized_net_pnl))}</strong></div>
      <div class="kv-row"><span>${metricLabel("평균 수익률 / 승률", "닫힌 거래당 평균 수익률과 플러스 거래 비율입니다.")}</span><strong class="${signedClass(overview.avg_realized_return)}">${escapeHtml(fmtPct(overview.avg_realized_return, 2))} / ${escapeHtml(fmtPct(overview.win_rate, 1))}</strong></div>
      <div class="kv-row"><span>${metricLabel("최대손실 / 매칭주의", "가장 큰 단일 거래 손실과 원가 매칭에 주의가 필요한 건수입니다.")}</span><strong class="${signedClass(overview.max_loss)}">${escapeHtml(fmtPct(overview.max_loss, 2))} / ${fmtNum(overview.unmatched_count)} (${fmtNum(overview.partial_basis_count)} partial)</strong></div>
      <div class="kv-row"><span>${metricLabel("스냅샷 원가 보조", "정확한 매수 lot이 부족해 보유 스냅샷 평균단가로 보조 계산한 건수입니다.")}</span><strong>${fmtNum(overview.snapshot_fallback_count)}</strong></div>
    </div>
    ${warnings.map((item) => `<div class="state-line">${escapeHtml(item)}</div>`).join("")}
    ${recentHtml}
    ${helpDetails([
      { term: "닫힌 거래", desc: "매수 후 매도까지 발생해 실현손익을 계산할 수 있는 거래입니다." },
      { term: "FIFO", desc: "먼저 산 수량을 먼저 판 것으로 보고 매도 원가를 연결하는 방식입니다." },
      { term: "매칭주의", desc: "매도 수량에 대응되는 매수 원가가 일부 부족하거나 보조 계산이 들어간 상태입니다." },
      { term: "표본 기준", desc: "현재 승격 판단에서는 청산 관찰 표본 30건 이상을 최소 기준으로 봅니다." },
    ])}
  `;
}

function summarizeLiveGateForBanner(summary, intents, preview, execution, consistency) {
  const intentRows = Array.isArray(intents?.intents) ? intents.intents : [];
  const previewRows = Array.isArray(preview?.items) ? preview.items : [];
  const executionRows = Array.isArray(execution?.items) ? execution.items : [];
  const consistencyCounts = consistency?.counts || {};
  const gateStatus = String(
    intents?.gate_status ||
    preview?.gate_status ||
    summary?.preview_gate_status ||
    ""
  ).toUpperCase();
  const previewDisplayStatus = String(
    preview?.gate_display_status ||
    summary?.preview_gate_display_status ||
    gateStatus ||
    ""
  ).toUpperCase();
  const buyIntentCount = intentRows.filter((row) => String(row.intent_type || "").toUpperCase() === "BUY").length;
  const executableBuyIntentCount = intentRows.filter((row) =>
    String(row.intent_type || "").toUpperCase() === "BUY" && row.executable
  ).length;
  const executablePreviewCount = previewRows.filter((row) => row.executable_now).length;
  const executableBuyPreviewCount = previewRows.filter((row) =>
    row.executable_now && String(row.side || "").toUpperCase() === "BUY"
  ).length;
  const executionSubmittedCount = Number(execution?.summary?.submitted_count || 0);
  const executionSubmittedBuyCount = executionRows.filter((row) =>
    String(row.submission_status || "").toLowerCase() === "submitted" &&
    String(row.side || "").toUpperCase() === "BUY"
  ).length;
  const submittedCount = Math.max(Number(consistencyCounts.submitted_count || 0), executionSubmittedCount);
  const submittedBuyCount = Math.max(Number(consistencyCounts.buy_intent_count || 0), executionSubmittedBuyCount);
  const filledCount = Number(consistencyCounts.filled_count || 0);
  const missingFillCount = Array.isArray(consistency?.submitted_without_fill)
    ? consistency.submitted_without_fill.length
    : null;
  const limitedEntryEvidence =
    gateStatus === "PILOT" ||
    previewDisplayStatus === "PILOT" ||
    /limited auto-buy|PILOT limited/i.test(JSON.stringify([intentRows, previewRows]));

  return {
    gateStatus,
    previewDisplayStatus,
    buyIntentCount,
    executableBuyIntentCount,
    executablePreviewCount,
    executableBuyPreviewCount,
    submittedCount,
    submittedBuyCount,
    filledCount,
    missingFillCount,
    blockedPreviewCount: previewRows.filter((row) => row.blocked_reason).length,
    limitedEntryEvidence,
    consistencyAsOfDate: consistency?.as_of_date || consistency?.asof_date || null,
  };
}

function renderDecisionBanner(summary, intents, preview, execution, runtime, consistency) {
  const root = document.getElementById("decisionBanner");
  const liveGate = summarizeLiveGateForBanner(summary, intents, preview, execution, consistency);
  const gate = liveGate.gateStatus;
  const skippedCount = Number(execution?.summary?.skipped_count || 0);
  const executeOn = !!runtime?.policy?.auto_trade_execute;
  const buyOn = !!runtime?.policy?.auto_trade_allow_buy;
  const accountSyncedAt = summary?.summary?.generated_at || runtime?.live_account_sync_scheduler?.last_success_at || "-";
  const executionFlow = summarizeExecutionFlow(preview, execution, runtime);
  let headline = "Status not resolved";
  let headlineTone = "warn";
  let headlineDetail = "Required live trading artifacts are not complete yet.";

  if (liveGate.submittedCount > 0) {
    headline = liveGate.submittedBuyCount > 0
      ? `BUY 제출 ${fmtNum(liveGate.submittedBuyCount)}건 / 전체 ${fmtNum(liveGate.submittedCount)}건`
      : `주문 제출 ${fmtNum(liveGate.submittedCount)}건`;
    headlineTone = "primary";
    headlineDetail = liveGate.missingFillCount > 0
      ? `브로커 주문 제출 기록은 있지만 ${fmtNum(liveGate.missingFillCount)}건은 아직 체결 행과 연결되지 않았습니다.`
      : "실제 주문 제출 기록이 있습니다. 제출과 체결은 다를 수 있으니 체결 결과를 별도로 확인하세요.";
  } else if (!executeOn) {
    headline = "실주문 스위치 OFF";
    headlineTone = "warn";
    headlineDetail = "전략 판단과 주문 초안은 만들 수 있지만 브로커 주문은 제출하지 않습니다.";
  } else if (gate === "BLOCK" && liveGate.limitedEntryEvidence) {
    headline = "공식 Gate는 차단, 제한 진입 흔적 있음";
    headlineTone = "warn";
    headlineDetail = "공식 포트폴리오 Gate는 BLOCK이지만 최근 산출물에 PILOT 또는 제한 진입 흐름이 있습니다. 주문 초안과 제출 결과를 함께 확인하세요.";
  } else if (gate === "BLOCK") {
    headline = "신규 진입 차단";
    headlineTone = "bad";
    headlineDetail = "공식 Gate가 BLOCK이라 신규 BUY는 차단되고 리스크 축소가 우선입니다.";
  } else if (gate === "WATCH") {
    headline = "WATCH 제한 진입 구간";
    headlineTone = "warn";
    headlineDetail = "제한된 신규 진입만 고려하는 구간입니다. 완전한 BUY_ALLOWED 운용은 아닙니다.";
  } else if (gate === "PILOT") {
    headline = "PILOT 제한 실운용";
    headlineTone = "primary";
    headlineDetail = "제한된 BUY 진입은 제출될 수 있지만, 풀 비중 배정과 점수 기반 교체매매는 아직 제한됩니다.";
  } else if (skippedCount > 0) {
    headline = `주문 보류 ${fmtNum(skippedCount)}건`;
    headlineTone = executionFlow.tone || "warn";
    headlineDetail = executionFlow.detail;
  } else if (liveGate.executablePreviewCount > 0) {
    headline = `제출 가능 주문초안 ${fmtNum(liveGate.executablePreviewCount)}건`;
    headlineTone = "primary";
    headlineDetail = "차단되지 않은 주문 초안이 있습니다. 제출 전 실행 스위치와 승인 정책을 확인하세요.";
  }

  root.innerHTML = `
    <article class="decision-card ${headlineTone}">
      <h2 class="decision-title">오늘 결론</h2>
      <div class="decision-value">${escapeHtml(headline)}</div>
      <div class="decision-detail">${escapeHtml(headlineDetail)}</div>
    </article>
    <article class="decision-card">
      <h2 class="decision-title">진입 모드</h2>
      <div class="decision-value">${escapeHtml(gateChipText(gate))}</div>
      <div class="decision-detail">BUY 판단 ${fmtNum(liveGate.buyIntentCount)}건 · BUY 주문초안 ${fmtNum(liveGate.executableBuyPreviewCount)}건 · 제한진입 ${liveGate.limitedEntryEvidence ? "예" : "아니오"} · 매수스위치 ${buyOn ? "ON" : "OFF"}</div>
    </article>
    <article class="decision-card">
      <h2 class="decision-title">계좌 동기화</h2>
      <div class="decision-value">${escapeHtml(String(accountSyncedAt).slice(0, 16) || "-")}</div>
      <div class="decision-detail">제출가능 ${fmtNum(liveGate.executablePreviewCount)}건 · 차단 ${fmtNum(liveGate.blockedPreviewCount)}건 · 제출 ${fmtNum(liveGate.submittedCount)}건 · 체결 ${fmtNum(liveGate.filledCount)}건${liveGate.consistencyAsOfDate ? ` · 기준 ${escapeHtml(liveGate.consistencyAsOfDate)}` : ""}</div>
    </article>
  `;
}

function ensurePreviewHeaderColumnV2() {
  const headerRow = document.querySelector("#previewWrap thead tr");
  if (!headerRow) return;
  if (headerRow.children.length >= 13) return;
  const policyHeaders = ["정책상태", "차단유형", "심각도"];
  const beforeQty = headerRow.children[6] || null;
  policyHeaders.forEach((label) => {
    const th = document.createElement("th");
    th.textContent = label;
    headerRow.insertBefore(th, beforeQty);
  });
  ["요약사유", "상세사유"].forEach((label) => {
    const th = document.createElement("th");
    th.textContent = label;
    headerRow.appendChild(th);
  });
}

function ensureExecutionHeaderColumnsV2() {
  const headerRow = document.querySelector("#executionWrap thead tr");
  if (!headerRow) return;
  if (headerRow.children.length >= 9) return;
  const failureType = document.createElement("th");
  failureType.textContent = "실패유형";
  headerRow.insertBefore(failureType, headerRow.children[4] || null);
  const errorCode = document.createElement("th");
  errorCode.textContent = "에러코드";
  headerRow.insertBefore(errorCode, headerRow.children[5] || null);
}

function renderDiagnosticSummary(diagnostics) {
  const root = document.getElementById("diagnosticSummaryPanel");
  if (!root) return;
  const summary = diagnostics?.summary || {};
  const runId = diagnostics?.run_id || "-";
  root.innerHTML = `
    <div class="kv">
      <div class="kv-row"><span>run_id</span><strong class="mono">${escapeHtml(runId)}</strong></div>
      <div class="kv-row"><span>?? AI ??</span><strong>${fmtNum(summary.recommendation_count)}</strong></div>
      <div class="kv-row"><span>?? ??</span><strong>${fmtNum(summary.order_candidate_count)}</strong></div>
      <div class="kv-row"><span>?? ?? ??</span><strong>${fmtNum(summary.submit_allowed_count)}</strong></div>
      <div class="kv-row"><span>?? ??</span><strong>${fmtNum(summary.policy_blocked_count)}</strong></div>
      <div class="kv-row"><span>??? ??</span><strong>${fmtNum(summary.broker_rejected_count)}</strong></div>
      <div class="kv-row"><span>?? ??</span><strong>${fmtNum(summary.sell_candidate_count)}</strong></div>
      <div class="kv-row"><span>?? ?? ??</span><strong>${fmtNum(summary.sell_submit_allowed_count)}</strong></div>
      <div class="kv-row"><span>?? BUY ??</span><strong>${summary.new_buy_allowed ? "YES" : "NO"}</strong></div>
      <div class="kv-row"><span>live grade</span><strong>${escapeHtml(summary.live_grade || "-")}</strong></div>
      <div class="kv-row"><span>?? ??</span><strong>${escapeHtml(summary.market_status_ko || summary.market_status || "-")}</strong></div>
      <div class="kv-row"><span>??? ??</span><strong>${escapeHtml(summary.last_run_at || "-")}</strong></div>
      <div class="kv-row"><span>??? ?? ??</span><strong>${escapeHtml(summary.last_order_attempt_at || summary.last_execution_at || "-")}</strong></div>
      <div class="kv-row"><span>Scheduler</span><strong>${escapeHtml(summary.scheduler_status || "-")}</strong></div>
      <div class="kv-row"><span>Scheduler ?? ????</span><strong>${escapeHtml(summary.scheduler_last_failure_at || summary.scheduler_last_success_at || "-")}</strong></div>
      <div class="kv-row"><span>Refresh</span><strong>${escapeHtml(summary.refresh_status || "-")}</strong></div>
      <div class="kv-row"><span>Refresh step</span><strong>${escapeHtml(summary.refresh_failing_step || "-")}</strong></div>
    </div>
  `;
}

function renderWhyNoTrade(diagnostics) {
  const root = document.getElementById("whyNoTradeBox");
  if (!root) return;
  const summary = diagnostics?.summary || {};
  const items = Array.isArray(diagnostics?.diagnostics) ? diagnostics.diagnostics : [];
  const primary = items[0] || null;
  const secondary = items[1] || null;
  root.innerHTML = `
    <h3 class="explain-title">? ??? ? ????</h3>
    <div class="explain-body">
      ${escapeHtml(summary.main_user_message_ko || "?? ?? ? ?? ??? ?? ?????.")}
      <br>
      ??: ${escapeHtml(summary.main_block_reason || "-")}
      ${secondary?.raw_reason ? `<br>?? ??: ${escapeHtml(secondary.raw_reason)}` : ""}
      ${primary?.recommended_action ? `<br>??: ${escapeHtml(primary.recommended_action)}` : ""}
      ${summary.scheduler_last_error ? `<br>Scheduler ??: ${escapeHtml(summary.scheduler_last_error)}` : ""}
      ${summary.refresh_failing_step ? `<br>Refresh step: ${escapeHtml(summary.refresh_failing_step)}` : ""}
      ${summary.refresh_failure_reason ? `<br>Refresh reason: ${escapeHtml(summary.refresh_failure_reason)}` : ""}
    </div>
  `;
}

function renderPreview(preview, runtime) {
  const tbody = document.getElementById("previewTbody");
  const rows = preview?.items || [];
  ensurePreviewHeaderColumnV2();
  if (!rows.length) {
    document.getElementById("previewWrap").innerHTML = `<div class="empty-state">order requests preview ?곗텧臾쇱씠 ?꾩쭅 ?놁뒿?덈떎.</div>`;
    return;
  }
  tbody.innerHTML = rows.map((row) => `
    <tr>
      <td>${orderStateChip(row)}</td>
      <td class="mono">${escapeHtml(row.request_id || "-")}</td>
      <td class="mono">${escapeHtml(row.code || "-")}</td>
      <td>${escapeHtml(row.name || "-")}</td>
      <td>${escapeHtml(row.side || "-")}</td>
      <td>${escapeHtml(row.intent_type || "-")}</td>
      <td>${escapeHtml(row.policy_status || "-")}</td>
      <td>${escapeHtml(row.block_type || "-")}</td>
      <td>${escapeHtml(row.severity || "-")}</td>
      <td class="right">${fmtNum(row.final_request_qty)}</td>
      <td class="right">${fmtNum(row.allowed_qty)}</td>
      <td>${escapeHtml(row.user_message_ko || describePreviewExecutionRisk(row, runtime))}</td>
      <td><details><summary>보기</summary>${escapeHtml(row.raw_reason || row.reason || row.blocked_reason || "-")}</details></td>
    </tr>
  `).join("");
}

function renderExecutionV2(execution, preview, runtime) {
  const wrap = document.getElementById("executionWrap");
  const tbody = document.getElementById("executionTbody");
  const rows = execution?.items || [];
  ensureExecutionHeaderColumnsV2();
  if (!rows.length) {
    const summary = summarizeExecutionFlow(preview, execution, runtime);
    wrap.innerHTML = `<div class="empty-state"><strong>${escapeHtml(summary.title)}</strong><br>${escapeHtml(summary.detail)}</div>`;
    return;
  }
  tbody.innerHTML = rows.map((row) => `
    <tr>
      <td>${executionStateChip(row)}</td>
      <td class="mono">${escapeHtml(row.request_id || "-")}</td>
      <td class="mono">${escapeHtml(row.code || "-")}</td>
      <td>${escapeHtml(row.side || "-")}</td>
      <td>${escapeHtml(row.broker_result || (String(row.submission_status || "").toLowerCase() === "failed" ? "BROKER_REJECT" : row.block_type || "-"))}</td>
      <td class="mono">${escapeHtml(row.broker_error_code || ((String(row.skip_reason || "").match(/msg_cd=([A-Z0-9_-]+)/i) || [])[1]) || "-")}</td>
      <td class="right">${fmtNum(row.final_request_qty)}</td>
      <td class="mono">${escapeHtml(row.broker_order_id || "-")}</td>
      <td>${escapeHtml(row.broker_error_message || ((String(row.skip_reason || "").match(/msg1=(.+)$/i) || [])[1]) || describeExecutionReason(row.skip_reason, row, runtime))}</td>
    </tr>
  `).join("");
}

async function main() {
  const state = document.getElementById("pageState");
  state.textContent = "실자동매매 데이터를 불러오는 중입니다.";
  try {
    const [summary, intents, preview, execution, runtime, holdings, consistency, tradeReview, tradeReviewSummary, liveKpiDaily, qualityGuardReview, closedTradeReport, qualityGuardOutputCheck, diagnostics] = await Promise.all([
      fetchJsonMaybe("/api/live-account/summary"),
      fetchJsonMaybe("/api/trade-intents"),
      fetchJsonMaybe("/api/order-requests-preview"),
      fetchJsonMaybe("/api/order-requests-execution"),
      fetchJsonMaybe("/api/auto-trading/runtime-status"),
      fetchJsonMaybe("/api/live-account/holdings"),
      fetchJsonMaybe("/api/live-trade-consistency"),
      fetchJsonMaybe("/api/live-trade-review-report"),
      fetchJsonMaybe("/api/live-trade-review-summary"),
      fetchJsonMaybe("/api/live-kpi-daily-report"),
      fetchJsonMaybe("/api/quality-risk-guard-live-review"),
      fetchJsonMaybe("/api/live-closed-trade-report"),
      fetchJsonMaybe("/api/live-quality-guard-output-check"),
      fetchJsonMaybe("/api/live-auto-trading-diagnostics"),
    ]);

    renderHero(summary, intents, preview, holdings, execution);
    renderDecisionBanner(summary, intents, preview, execution, runtime, consistency);
    renderStatusV3(summary, intents, preview, execution, runtime);
    renderDiagnosticSummary(diagnostics);
    renderWhyNoTrade(diagnostics);
    renderConsistency(consistency);
    renderTradeReview(tradeReview, tradeReviewSummary);
    renderLiveKpiDaily(liveKpiDaily);
    renderQualityGuardReview(qualityGuardReview, qualityGuardOutputCheck);
    renderClosedTradeReport(closedTradeReport);
    renderAccountDetailsV3(summary, runtime);
    renderRunSummary(intents, preview, holdings, runtime);
    renderFocus(intents, preview, holdings);
    renderOperationalExplain(intents, preview, runtime, holdings);
    renderIntents(intents);
    renderPreview(preview, runtime);
    renderExecutionV2(execution, preview, runtime);
    renderHoldings(holdings);

    const loaded = [
      summary ? "summary" : null,
      intents ? "intents" : null,
      preview ? "preview" : null,
      execution ? "execution" : null,
      runtime ? "runtime" : null,
      holdings ? "holdings" : null,
      consistency ? "consistency" : null,
      tradeReview ? "tradeReview" : null,
      tradeReviewSummary ? "tradeReviewSummary" : null,
      liveKpiDaily ? "liveKpiDaily" : null,
      qualityGuardReview ? "qualityGuardReview" : null,
      closedTradeReport ? "closedTradeReport" : null,
      qualityGuardOutputCheck ? "qualityGuardOutputCheck" : null,
      diagnostics ? "diagnostics" : null,
    ].filter(Boolean);
    state.textContent = loaded.length
      ? `불러온 데이터: ${loaded.join(", ")}`
      : "실자동매매 산출물이 아직 없습니다.";
  } catch (error) {
    console.error(error);
    state.textContent = `실자동매매 데이터를 불러오지 못했습니다: ${error.message}`;
  }
}

void main();
