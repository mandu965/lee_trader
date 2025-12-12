// ranking.js

async function fetchTodayActions(dateValue) {
  const params = new URLSearchParams();
  if (dateValue) params.set("date", dateValue);

  const res = await fetch("/api/dashboard/today-actions?" + params.toString());
  if (!res.ok) throw new Error("today-actions API error");
  return res.json();
}

async function fetchTop20(dateValue) {
  const params = new URLSearchParams();
  if (dateValue) params.set("date", dateValue);
  params.set("horizon", "60d");
  params.set("limit", "20");

  const res = await fetch("/api/signals/top20?" + params.toString());
  if (!res.ok) throw new Error("top20 API error");
  return res.json();
}

function formatPercent(x) {
  if (x === null || x === undefined) return "-";
  return (x * 100).toFixed(1) + "%";
}

function formatNumber(x) {
  if (x === null || x === undefined) return "-";
  return x.toLocaleString();
}

function createActionLi(item) {
  const li = document.createElement("li");
  li.className = "action-item";
  li.innerHTML = `
    <div class="action-main">
      <span class="action-name">${item.name}</span>
      <span class="action-code">${item.code}</span>
    </div>
    <div class="action-meta">
      <span class="pos">예측: ${formatPercent(item.pred_return_60d)}</span>
      <span class="neg">MDD: ${formatPercent(item.pred_mdd_60d)}</span>
      <span class="prob">Top20: ${formatPercent(item.prob_top20_60d)}</span>
    </div>
    <div class="action-reason">${item.reason}</div>
  `;
  li.onclick = () => {
    // 상세 페이지로 이동 (쿼리 파라미터 방식)
    window.location.href = `holdingsDetail.html?code=${item.code}`;
  };
  return li;
}

function renderTodayActions(data) {
  document.getElementById("todayDate").textContent = data.date;

  const buyList = document.getElementById("buyList");
  const addList = document.getElementById("addList");
  const trimList = document.getElementById("trimList");

  buyList.innerHTML = "";
  addList.innerHTML = "";
  trimList.innerHTML = "";

  if (data.buy_candidates.length === 0) {
    buyList.innerHTML = '<li class="empty">해당 없음</li>';
  } else {
    data.buy_candidates.forEach((item) => buyList.appendChild(createActionLi(item)));
  }

  if (data.add_candidates.length === 0) {
    addList.innerHTML = '<li class="empty">해당 없음</li>';
  } else {
    data.add_candidates.forEach((item) => addList.appendChild(createActionLi(item)));
  }

  if (data.trim_candidates.length === 0) {
    trimList.innerHTML = '<li class="empty">해당 없음</li>';
  } else {
    data.trim_candidates.forEach((item) => trimList.appendChild(createActionLi(item)));
  }
}

function renderTop20(data) {
  const tbody = document.getElementById("signalTbody");
  tbody.innerHTML = "";

  data.items.forEach((item) => {
    const tr = document.createElement("tr");
    tr.className = item.is_holding ? "row-holding" : "";
    tr.onclick = () => {
      window.location.href = `holdingsDetail.html?code=${item.code}`;
    };

    tr.innerHTML = `
      <td>${item.rank}</td>
      <td>
        <div class="cell-name">
          <span class="name">${item.name}</span>
          <span class="code">${item.code}</span>
        </div>
      </td>
      <td>
        <div class="cell-market">
          <span>${item.market}</span>
          <span class="sector">${item.sector}</span>
        </div>
      </td>
      <td class="num">${formatNumber(item.close)}</td>
      <td class="num">${formatPercent(item.pred_return)}</td>
      <td class="num">${formatPercent(item.pred_mdd)}</td>
      <td class="num">${formatPercent(item.prob_top20)}</td>
      <td class="num">
        ${item.final_score != null ? item.final_score.toFixed(1) : "-"}
      </td>
      <td>
        ${
          item.is_holding
            ? '<span class="badge badge-holding">보유중</span>'
            : ""
        }
      </td>
    `;
    tbody.appendChild(tr);
  });
}

async function loadAll() {
  const dateInput = document.getElementById("signalDate");
  const dateValue = dateInput.value || "";

  try {
    const [actions, top20] = await Promise.all([
      fetchTodayActions(dateValue),
      fetchTop20(dateValue),
    ]);

    renderTodayActions(actions);
    renderTop20(top20);

    // 사용자가 날짜를 지정하지 않았다면 응답 날짜를 입력에 반영
    if (!dateValue) {
      const inferredDate = actions?.date || top20?.date;
      if (inferredDate) {
        document.getElementById("signalDate").value = inferredDate;
      }
    }
  } catch (e) {
    console.error(e);
    alert("데이터를 불러오는데 실패했습니다.");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const dateInput = document.getElementById("signalDate");
  const reloadBtn = document.getElementById("reloadBtn");

  reloadBtn.addEventListener("click", loadAll);
  dateInput.addEventListener("change", loadAll);

  loadAll().catch(() => {});
});
