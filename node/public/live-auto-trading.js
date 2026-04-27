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
  heroGrid.innerHTML = `
    <article class="hero-card">
      <div class="card-label">기준일</div>
      <div class="card-value">${escapeHtml(intents?.asof_date || preview?.asof_date || "-")}</div>
      <div class="card-detail">주문 판단 생성 ${escapeHtml(intents?.generated_at || "-")}</div>
    </article>
    <article class="hero-card">
      <div class="card-label">Gate</div>
      <div class="card-value">${escapeHtml(intents?.gate_status || preview?.gate_status || summary?.preview_gate_status || "-")}</div>
      <div class="card-detail">${escapeHtml(intents?.gate_guidance || "gate 설명 정보 없음")}</div>
    </article>
    <article class="hero-card">
      <div class="card-label">Intent / Preview / Execute</div>
      <div class="card-value">${fmtNum(intents?.intent_count)} / ${fmtNum(preview?.summary?.request_count)}</div>
      <div class="card-detail">실행 후보 ${fmtNum((intents?.intents || []).filter((item) => item.executable).length)}건 | 주문 초안 ${fmtNum(summary?.order_preview_count ?? preview?.summary?.request_count)} | 실제 제출 ${fmtNum(execution?.summary?.submitted_count)}</div>
    </article>
    <article class="hero-card">
      <div class="card-label">실계좌 보유</div>
      <div class="card-value">${fmtNum(summary?.holding_count ?? holdings?.count)}</div>
      <div class="card-detail">계좌 요약 파일 ${summaryInfo ? "연결" : "없음"} | holdings csv ${holdings?.count ? "존재" : "없음"}</div>
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
    body = "실운영에서도 소액 제한 진입만 허용됩니다. 다만 현재 표는 실제 산출물만 보여주므로, 가정 후보는 아래 WATCH 시뮬레이션에서 따로 봐야 합니다.";
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

function renderWatchSimulation(simulation) {
  const explainRoot = document.getElementById("simulationExplain");
  const grid = document.getElementById("watchSimulationGrid");
  if (!explainRoot || !grid) return;
  if (!simulation || !Array.isArray(simulation.scenarios) || !simulation.scenarios.length) {
    explainRoot.innerHTML = `
      <h3 class="explain-title">WATCH 시뮬레이션 산출물이 아직 없습니다</h3>
      <div class="explain-body">운영 refresh 이후에 가정용 진입 후보가 별도 산출됩니다. 이 섹션은 실운영 주문 결과와 분리되어야 하므로, 산출물이 없으면 비워두는 편이 맞습니다.</div>
    `;
    grid.innerHTML = `<div class="empty-state">watch_auto_buy_simulation payload not found.</div>`;
    return;
  }

  const emptyScenario = simulation.scenarios.find((item) => item?.scenario_key === "watch_empty_account");
  const emptyCodes = ((emptyScenario?.summary?.enter_codes) || []).filter(Boolean);
  const summaryLine = emptyCodes.length
    ? `빈 계좌 WATCH 가정에서는 ${emptyCodes.join(", ")} 같은 신규 진입 후보를 별도 계산합니다. 이 결과는 연구용 참고이며 실제 주문 제출과 직접 연결되지 않습니다.`
    : "WATCH 가정 결과는 실운영 표와 별도입니다. 가정상 신규 진입 후보가 없으면 이 섹션도 비어 보일 수 있습니다.";

  explainRoot.innerHTML = `
    <h3 class="explain-title">가정용 후보를 실운영과 분리해서 보여줍니다</h3>
    <div class="explain-body">${escapeHtml(summaryLine)}</div>
  `;

  grid.innerHTML = simulation.scenarios.map((scenario) => {
    const actions = Array.isArray(scenario.actions) ? scenario.actions : [];
    const enters = actions.filter((row) => String(row.action || "").toUpperCase() === "ENTER");
    const itemsHtml = enters.length
      ? enters.slice(0, 5).map((row) => `
        <li class="sim-item">
          <div class="sim-item-row">
            <strong>${escapeHtml(row.code || "-")} ${escapeHtml(row.name || "")}</strong>
            <span>${escapeHtml(row.action || "-")}</span>
          </div>
          <div class="sim-item-row">
            <span>${escapeHtml(row.reason || "-")}</span>
            <span>${fmtPct(row.target_weight, 1)}</span>
          </div>
        </li>
      `).join("")
      : `<li class="sim-item"><div class="sim-item-row"><strong>신규 진입 후보 없음</strong><span>-</span></div><div class="sim-item-row"><span>이 시나리오에서는 ENTER action이 생성되지 않았습니다.</span><span></span></div></li>`;

    return `
      <article class="sim-card">
        <div class="sim-card-header">
          <div>
            <h3 class="sim-card-title">${escapeHtml(scenario.scenario_label || scenario.scenario_key || "WATCH simulation")}</h3>
            <div class="sim-meta">보유기준 ${escapeHtml(scenario.holdings_source || "-")} | 보유 ${fmtNum(scenario.holding_count)}종목 | 신규 ${fmtNum(scenario.summary?.enter_count)}건</div>
          </div>
          <span class="chip watch">실운영 아님</span>
        </div>
        <div class="chip-row">
          <span class="chip ${scenario.summary?.enter_count ? "good" : "warn"}">신규 진입 ${fmtNum(scenario.summary?.enter_count)}건</span>
          <span class="chip watch">총 비중 ${fmtPct(scenario.summary?.total_target_weight, 1)}</span>
          <span class="chip watch">최대 단일 ${fmtPct(scenario.summary?.max_single_target_weight, 1)}</span>
        </div>
        <ul class="sim-list">${itemsHtml}</ul>
      </article>
    `;
  }).join("");
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
      ? `Submitted BUY ${fmtNum(liveGate.submittedBuyCount)} / total ${fmtNum(liveGate.submittedCount)}`
      : `Submitted orders ${fmtNum(liveGate.submittedCount)}`;
    headlineTone = "primary";
    headlineDetail = liveGate.missingFillCount > 0
      ? `Broker submissions exist, but ${fmtNum(liveGate.missingFillCount)} submitted rows still have no matching fill row.`
      : "Broker submission records exist. Check fills separately because submission and fill can differ.";
  } else if (!executeOn) {
    headline = "Execution switch OFF";
    headlineTone = "warn";
    headlineDetail = "The system may build intents and previews, but it will not submit broker orders.";
  } else if (gate === "BLOCK" && liveGate.limitedEntryEvidence) {
    headline = "Formal gate BLOCK, limited-entry evidence exists";
    headlineTone = "warn";
    headlineDetail = "The formal portfolio gate is BLOCK, but recent artifacts include PILOT or limited-entry buy flow. Check preview and execution rows before treating this as a pure no-buy day.";
  } else if (gate === "BLOCK") {
    headline = "New entries blocked";
    headlineTone = "bad";
    headlineDetail = "The formal gate is BLOCK, so new BUY intents are blocked and risk reduction has priority.";
  } else if (gate === "WATCH") {
    headline = "WATCH limited-entry zone";
    headlineTone = "warn";
    headlineDetail = "Only constrained entry should be considered. This is not full BUY_ALLOWED operation.";
  } else if (gate === "PILOT") {
    headline = "PILOT limited live entry";
    headlineTone = "primary";
    headlineDetail = "Limited BUY entries can be submitted, but full-size allocation and score-driven replacement are still restricted.";
  } else if (skippedCount > 0) {
    headline = `Skipped orders ${fmtNum(skippedCount)}`;
    headlineTone = executionFlow.tone || "warn";
    headlineDetail = executionFlow.detail;
  } else if (liveGate.executablePreviewCount > 0) {
    headline = `Executable preview ${fmtNum(liveGate.executablePreviewCount)}`;
    headlineTone = "primary";
    headlineDetail = "Preview rows are executable. Confirm runtime switches and approval policy before submission.";
  }

  root.innerHTML = `
    <article class="decision-card ${headlineTone}">
      <h2 class="decision-title">Live Conclusion</h2>
      <div class="decision-value">${escapeHtml(headline)}</div>
      <div class="decision-detail">${escapeHtml(headlineDetail)}</div>
    </article>
    <article class="decision-card">
      <h2 class="decision-title">Gate / Limited Entry</h2>
      <div class="decision-value">${escapeHtml(gate || "-")}</div>
      <div class="decision-detail">BUY intents ${fmtNum(liveGate.buyIntentCount)} / executable BUY preview ${fmtNum(liveGate.executableBuyPreviewCount)} / limited ${liveGate.limitedEntryEvidence ? "YES" : "NO"} / buy switch ${buyOn ? "ON" : "OFF"}</div>
    </article>
    <article class="decision-card">
      <h2 class="decision-title">Account Sync</h2>
      <div class="decision-value">${escapeHtml(String(accountSyncedAt).slice(0, 16) || "-")}</div>
      <div class="decision-detail">Executable ${fmtNum(liveGate.executablePreviewCount)} / blocked ${fmtNum(liveGate.blockedPreviewCount)} / submitted ${fmtNum(liveGate.submittedCount)} / filled ${fmtNum(liveGate.filledCount)}${liveGate.consistencyAsOfDate ? ` / basis ${escapeHtml(liveGate.consistencyAsOfDate)}` : ""}</div>
    </article>
  `;
}

async function main() {
  const state = document.getElementById("pageState");
  state.textContent = "실자동매매 데이터를 불러오는 중입니다.";
  try {
    const [summary, intents, preview, execution, runtime, holdings, watchSimulation, consistency, tradeReview, tradeReviewSummary] = await Promise.all([
      fetchJsonMaybe("/api/live-account/summary"),
      fetchJsonMaybe("/api/trade-intents"),
      fetchJsonMaybe("/api/order-requests-preview"),
      fetchJsonMaybe("/api/order-requests-execution"),
      fetchJsonMaybe("/api/auto-trading/runtime-status"),
      fetchJsonMaybe("/api/live-account/holdings"),
      fetchJsonMaybe("/api/watch-auto-buy-simulation"),
      fetchJsonMaybe("/api/live-trade-consistency"),
      fetchJsonMaybe("/api/live-trade-review-report"),
      fetchJsonMaybe("/api/live-trade-review-summary"),
    ]);

    renderHero(summary, intents, preview, holdings, execution);
    renderDecisionBanner(summary, intents, preview, execution, runtime, consistency);
    renderStatusV2(summary, intents, preview, execution, runtime);
    renderConsistency(consistency);
    renderTradeReview(tradeReview, tradeReviewSummary);
    renderAccountDetailsV3(summary, runtime);
    renderRunSummary(intents, preview, holdings, runtime);
    renderFocus(intents, preview, holdings);
    renderOperationalExplain(intents, preview, runtime, holdings);
    renderWatchSimulation(watchSimulation);
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
      watchSimulation ? "watchSimulation" : null,
      consistency ? "consistency" : null,
      tradeReview ? "tradeReview" : null,
      tradeReviewSummary ? "tradeReviewSummary" : null,
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
