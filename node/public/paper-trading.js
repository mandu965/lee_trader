const fmtNum = (value, digits = 0) => {
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  return n.toLocaleString("ko-KR", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
};

const fmtPct = (value, digits = 2) => {
  const n = Number(value);
  if (!Number.isFinite(n)) return "-";
  return `${(n * 100).toFixed(digits)}%`;
};

const getToneClass = (value) => {
  const n = Number(value);
  if (!Number.isFinite(n)) return "";
  if (n > 0) return "pos";
  if (n < 0) return "neg";
  return "";
};

function toDate(value) {
  if (!value) return null;
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? null : d;
}

function isBusinessDay(date) {
  const day = date.getDay();
  return day !== 0 && day !== 6;
}

function diffBusinessDays(fromValue, toValue) {
  const from = toDate(fromValue);
  const to = toDate(toValue);
  if (!from || !to) return null;
  if (to < from) return 0;
  const cursor = new Date(from);
  let count = 0;
  while (cursor < to) {
    cursor.setDate(cursor.getDate() + 1);
    if (isBusinessDay(cursor)) count += 1;
  }
  return count;
}

function addBusinessDays(value, days) {
  const start = toDate(value);
  if (!start || !Number.isFinite(days)) return null;
  const cursor = new Date(start);
  let added = 0;
  while (added < days) {
    cursor.setDate(cursor.getDate() + 1);
    if (isBusinessDay(cursor)) added += 1;
  }
  return cursor.toISOString().slice(0, 10);
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`HTTP ${res.status}`);
  }
  return res.json();
}

function buildExplainGrid() {
  document.getElementById("explainGrid").innerHTML = `
    <article class="explain-card">
      <div class="card-label">전략 구조</div>
      <div class="card-value">3개 계좌</div>
      <div class="card-detail">top5, top8, top10은 하나의 계좌가 아니라 서로 독립된 3개 가상 전략입니다.</div>
    </article>
    <article class="explain-card">
      <div class="card-label">매일 처리</div>
      <div class="card-value">신규만 추가</div>
      <div class="card-detail">매일 순위가 바뀌어도 기존 보유를 전량 교체하지 않고, 그날 새 후보만 코호트로 추가합니다.</div>
    </article>
    <article class="explain-card">
      <div class="card-label">보유 방식</div>
      <div class="card-value">20거래일 보유</div>
      <div class="card-detail">한번 진입한 종목은 기본적으로 20거래일 보유합니다. 그래서 top5 전략도 현재 보유 종목 수는 5개를 넘을 수 있습니다.</div>
    </article>
    <article class="explain-card">
      <div class="card-label">중복 처리</div>
      <div class="card-value">재매수 스킵</div>
      <div class="card-detail">같은 전략 안에서 이미 들고 있는 종목이 다시 상위에 와도 추가 매수하지 않고 중복 스킵으로 처리합니다.</div>
    </article>
  `;
}

function buildMetaStrip(run, strategies, selectedStrategy) {
  const metaStrip = document.getElementById("metaStrip");
  if (!run) {
    metaStrip.innerHTML = `<div class="empty-state">paper trading run 정보가 없습니다.</div>`;
    return;
  }
  const latestNav = strategies.reduce((acc, item) => {
    const nav = Number(item.nav);
    return Number.isFinite(nav) && nav > acc ? nav : acc;
  }, 0);
  const totalOpen = strategies.reduce((acc, item) => acc + (Number(item.open_position_count) || 0), 0);
  const strategyLabel = selectedStrategy || "전체 전략";

  metaStrip.innerHTML = `
    <article class="meta-card">
      <div class="card-label">Run Tag</div>
      <div class="card-value">${run.run_tag || "-"}</div>
      <div class="card-detail">기준일 ${run.asof_date || "-"} · source ${run.source_mode || "-"}</div>
    </article>
    <article class="meta-card">
      <div class="card-label">선택 전략</div>
      <div class="card-value">${strategyLabel}</div>
      <div class="card-detail">hold_days ${fmtNum(run.hold_days)} · 초기자금 ${fmtNum(run.initial_nav)}</div>
    </article>
    <article class="meta-card">
      <div class="card-label">현재 스냅샷</div>
      <div class="card-value">${fmtNum(latestNav, 2)}</div>
      <div class="card-detail">전체 open positions ${fmtNum(totalOpen)}</div>
    </article>
    <article class="meta-card">
      <div class="card-label">추적 행 수</div>
      <div class="card-value">${fmtNum(run.positions_row_count)} / ${fmtNum(run.nav_row_count)}</div>
      <div class="card-detail">positions / nav rows</div>
    </article>
  `;

  document.getElementById("runInfo").textContent = `run_id ${run.paper_run_id} · 기준일 ${run.asof_date || "-"}`;
}

function buildStrategyCards(strategies) {
  const grid = document.getElementById("strategyGrid");
  if (!strategies.length) {
    grid.innerHTML = `<div class="empty-state">전략 요약 데이터가 없습니다.</div>`;
    return;
  }
  grid.innerHTML = strategies
    .map((item) => {
      const retCls = getToneClass(item.cumulative_return);
      const ddCls = getToneClass(item.drawdown);
      return `
        <article class="strategy-card">
          <div class="strategy-head">
            <div>
              <h3 class="strategy-name">${item.strategy}</h3>
              <div class="strategy-date">latest ${item.latest_date || "-"}</div>
            </div>
            <div class="strategy-return ${retCls}">${fmtPct(item.cumulative_return)}</div>
          </div>
          <div class="strategy-metrics">
            <div class="metric-box">
              <div class="metric-box__label">NAV</div>
              <div class="metric-box__value">${fmtNum(item.nav, 2)}</div>
            </div>
            <div class="metric-box">
              <div class="metric-box__label">Drawdown</div>
              <div class="metric-box__value ${ddCls}">${fmtPct(item.drawdown)}</div>
            </div>
            <div class="metric-box">
              <div class="metric-box__label">현재 보유 수</div>
              <div class="metric-box__value">${fmtNum(item.open_position_count)}</div>
            </div>
            <div class="metric-box">
              <div class="metric-box__label">누적 청산 수</div>
              <div class="metric-box__value">${fmtNum(item.closed_trade_count)}</div>
            </div>
          </div>
        </article>
      `;
    })
    .join("");
}

function computeFocusSummary(run, navItems, openItems) {
  const asofDate = run?.asof_date || null;
  const holdDays = Number(run?.hold_days) || 20;
  const latestEntryDate = openItems.reduce((acc, item) => {
    return !acc || String(item.entry_date) > String(acc) ? item.entry_date : acc;
  }, null);

  const newItems = latestEntryDate ? openItems.filter((item) => item.entry_date === latestEntryDate) : [];
  const ongoingItems = latestEntryDate ? openItems.filter((item) => item.entry_date !== latestEntryDate) : openItems.slice();
  const nearExitItems = openItems.filter((item) => Number.isFinite(item.remaining_days) && item.remaining_days <= 5);
  const latestNav = navItems.length ? navItems[navItems.length - 1] : null;

  return {
    asofDate,
    holdDays,
    latestEntryDate,
    newItems,
    ongoingItems,
    nearExitItems,
    latestNav,
  };
}

function buildStatusStrip(summary, strategyLabel) {
  const wrap = document.getElementById("statusStrip");
  const latestEntryText = summary.latestEntryDate || "없음";
  const activeText = summary.latestNav ? fmtNum(summary.latestNav.active_position_count) : "-";
  const duplicateText = summary.latestNav ? fmtNum(summary.latestNav.duplicate_skip_count) : "-";

  const newList = summary.newItems.slice(0, 4).map((item) => `<li>${item.code} ${item.name}</li>`).join("");
  const ongoingList = summary.ongoingItems.slice(0, 4).map((item) => `<li>${item.code} ${item.name}</li>`).join("");
  const nearExitList = summary.nearExitItems.slice(0, 4).map((item) => `<li>${item.code} ${item.name} · 잔여 ${fmtNum(item.remaining_days)}일</li>`).join("");

  wrap.innerHTML = `
    <article class="status-card">
      <div class="card-label">오늘 신규 편입 해석</div>
      <div class="card-value">${fmtNum(summary.newItems.length)}개</div>
      <div class="card-detail">${strategyLabel} 기준 최근 신규 코호트 진입일은 ${latestEntryText} 입니다.</div>
      <ul class="status-card__list">${newList || "<li>이번 기준일에는 새로 들어온 종목이 없습니다.</li>"}</ul>
    </article>
    <article class="status-card">
      <div class="card-label">계속 보유 중</div>
      <div class="card-value">${fmtNum(summary.ongoingItems.length)}개</div>
      <div class="card-detail">기존 코호트가 계속 살아있기 때문에 현재 보유 종목 수가 top5보다 커질 수 있습니다. 활성 포지션 ${activeText}개입니다.</div>
      <ul class="status-card__list">${ongoingList || "<li>계속 보유 중인 종목이 없습니다.</li>"}</ul>
    </article>
    <article class="status-card">
      <div class="card-label">곧 청산 / 중복 스킵</div>
      <div class="card-value">${fmtNum(summary.nearExitItems.length)}개</div>
      <div class="card-detail">잔여일수 5일 이하 종목 기준입니다. 최근 NAV 기준 중복 스킵은 ${duplicateText}건입니다.</div>
      <ul class="status-card__list">${nearExitList || "<li>곧 청산할 종목은 아직 많지 않습니다.</li>"}</ul>
    </article>
  `;
}

function buildNavTable(items) {
  const tbody = document.getElementById("navTbody");
  if (!items.length) {
    tbody.innerHTML = `<tr><td colspan="8" class="center">NAV 데이터가 없습니다.</td></tr>`;
    return;
  }
  tbody.innerHTML = items.map((row) => `
    <tr>
      <td>${row.strategy || "-"}</td>
      <td>${row.date || "-"}</td>
      <td class="right">${fmtNum(row.nav, 2)}</td>
      <td class="right ${getToneClass(row.cumulative_return)}">${fmtPct(row.cumulative_return)}</td>
      <td class="right ${getToneClass(row.drawdown)}">${fmtPct(row.drawdown)}</td>
      <td class="right">${fmtNum(row.active_position_count)}</td>
      <td class="right">${fmtNum(row.opened_today)}</td>
      <td class="right">${fmtNum(row.duplicate_skip_count)}</td>
    </tr>
  `).join("");
}

function buildPositionsTable(items) {
  const tbody = document.getElementById("positionsTbody");
  if (!items.length) {
    tbody.innerHTML = `<tr><td colspan="12" class="center">포지션 데이터가 없습니다.</td></tr>`;
    return;
  }
  const statusChip = (row) => {
    if (row.remaining_days !== null && row.remaining_days <= 5) {
      return `<span class="chip is-near">곧 마감</span>`;
    }
    if (row.held_days !== null && row.held_days <= 1) {
      return `<span class="chip is-new">신규</span>`;
    }
    return `<span class="chip is-hold">보유중</span>`;
  };
  tbody.innerHTML = items.map((row) => `
    <tr>
      <td>${statusChip(row)}</td>
      <td>${row.strategy || "-"}</td>
      <td>${row.code || "-"}</td>
      <td>${row.name || "-"}</td>
      <td>
        <div class="date-stack">
          <div class="date-stack__main">${row.entry_date || "-"}</div>
          <div class="date-stack__sub">진입</div>
        </div>
      </td>
      <td>
        <div class="date-stack">
          <div class="date-stack__main">${row.expected_exit_date || "-"}</div>
          <div class="date-stack__sub">20거래일 기준</div>
        </div>
      </td>
      <td class="right">${fmtNum(row.held_days)}</td>
      <td class="right">${fmtNum(row.remaining_days)}</td>
      <td class="right">${fmtNum(row.entry_exec_price, 2)}</td>
      <td class="right">${fmtNum(row.shares, 2)}</td>
      <td class="right ${getToneClass(row.net_return)}">${fmtPct(row.net_return)}</td>
      <td class="right">${fmtNum(row.confidence_score, 2)}</td>
      <td class="right">${fmtNum(row.final_score, 2)}</td>
    </tr>
  `).join("");
}

function enrichPositions(items, run) {
  const asofDate = run?.asof_date || null;
  const holdDays = Number(run?.hold_days) || 20;
  return items.map((item) => {
    const heldDays = diffBusinessDays(item.entry_date, asofDate);
    const remainingDays = Number.isFinite(heldDays) ? Math.max(0, holdDays - heldDays) : null;
    const expectedExitDate = item.planned_exit_date || addBusinessDays(item.entry_date, holdDays);
    return {
      ...item,
      held_days: heldDays,
      remaining_days: remainingDays,
      expected_exit_date: expectedExitDate,
    };
  });
}

function buildPositionsTable(items) {
  const tbody = document.getElementById("positionsTbody");
  const theadRow = document.querySelector(".positions-table thead tr");
  if (theadRow) {
    theadRow.innerHTML = `
      <th>상태</th>
      <th>전략</th>
      <th>코드</th>
      <th>종목명</th>
      <th>진입일</th>
      <th>예상 마감일</th>
      <th class="right">보유일수</th>
      <th class="right">잔여일수</th>
      <th class="right">진입가</th>
      <th class="right">수량</th>
      <th class="right">현재가</th>
      <th class="right">평가손익</th>
      <th class="right">현재 평가수익률</th>
      <th class="right">Confidence</th>
      <th class="right">Final Score</th>
    `;
  }

  if (!items.length) {
    tbody.innerHTML = `<tr><td colspan="15" class="center">포지션 데이터가 없습니다.</td></tr>`;
    return;
  }

  const statusChip = (row) => {
    if (row.remaining_days !== null && row.remaining_days <= 5) {
      return `<span class="chip is-near">곧 마감</span>`;
    }
    if (row.held_days !== null && row.held_days <= 1) {
      return `<span class="chip is-new">신규</span>`;
    }
    return `<span class="chip is-hold">보유중</span>`;
  };

  tbody.innerHTML = items.map((row) => `
    <tr>
      <td>${statusChip(row)}</td>
      <td>${row.strategy || "-"}</td>
      <td>${row.code || "-"}</td>
      <td>${row.name || "-"}</td>
      <td>
        <div class="date-stack">
          <div class="date-stack__main">${row.entry_date || "-"}</div>
          <div class="date-stack__sub">진입</div>
        </div>
      </td>
      <td>
        <div class="date-stack">
          <div class="date-stack__main">${row.expected_exit_date || "-"}</div>
          <div class="date-stack__sub">20거래일 기준</div>
        </div>
      </td>
      <td class="right">${fmtNum(row.held_days)}</td>
      <td class="right">${fmtNum(row.remaining_days)}</td>
      <td class="right">${fmtNum(row.entry_exec_price, 2)}</td>
      <td class="right">${fmtNum(row.shares, 2)}</td>
      <td class="right">${fmtNum(row.current_price, 2)}</td>
      <td class="right ${getToneClass(row.current_pnl_amount)}">${fmtNum(row.current_pnl_amount, 2)}</td>
      <td class="right ${getToneClass(row.current_return)}">${fmtPct(row.current_return)}</td>
      <td class="right">${fmtNum(row.confidence_score, 2)}</td>
      <td class="right">${fmtNum(row.final_score, 2)}</td>
    </tr>
  `).join("");
}

async function loadPaperTrading() {
  const strategy = document.getElementById("strategyFilter").value.trim();
  const status = document.getElementById("statusFilter").value.trim();
  const state = document.getElementById("pageState");
  const strategyLabel = strategy || "전체 전략";
  state.textContent = "모의투자 데이터를 불러오는 중입니다.";

  try {
    const summary = await fetchJson("/api/paper-trading/summary");
    const navQs = new URLSearchParams();
    const posQs = new URLSearchParams();
    if (strategy) {
      navQs.set("strategy", strategy);
      posQs.set("strategy", strategy);
    }
    if (status) {
      posQs.set("status", status);
    }

    const [nav, positions] = await Promise.all([
      fetchJson(`/api/paper-trading/nav?${navQs.toString()}`),
      fetchJson(`/api/paper-trading/positions?${posQs.toString()}`),
    ]);

    const openItems = enrichPositions(
      (positions.items || []).filter((item) => (item.status || "OPEN") === "OPEN"),
      summary.run
    );

    buildExplainGrid();
    buildMetaStrip(summary.run, summary.strategies || [], strategyLabel);
    buildStrategyCards(summary.strategies || []);
    buildStatusStrip(computeFocusSummary(summary.run, nav.items || [], openItems), strategyLabel);
    buildNavTable(nav.items || []);
    buildPositionsTable(openItems);

    state.textContent = `run ${summary.run?.paper_run_id || "-"} 기준 ${strategyLabel} 해석 완료 · 현재 open positions ${openItems.length}건`;
  } catch (error) {
    console.error(error);
    document.getElementById("explainGrid").innerHTML = "";
    document.getElementById("metaStrip").innerHTML = `<div class="empty-state">모의투자 데이터를 불러오지 못했습니다.</div>`;
    document.getElementById("strategyGrid").innerHTML = "";
    document.getElementById("statusStrip").innerHTML = "";
    buildNavTable([]);
    buildPositionsTable([]);
    state.textContent = `조회 실패: ${error.message}`;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("refreshBtn").addEventListener("click", () => {
    loadPaperTrading().catch((error) => console.error(error));
  });
  document.getElementById("strategyFilter").addEventListener("change", () => {
    loadPaperTrading().catch((error) => console.error(error));
  });
  document.getElementById("statusFilter").addEventListener("change", () => {
    loadPaperTrading().catch((error) => console.error(error));
  });
  loadPaperTrading().catch((error) => console.error(error));
});
