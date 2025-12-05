function formatNumber(x) {
  if (x === null || x === undefined) return "-";
  return x.toLocaleString();
}
function formatSigned(x) {
  if (x === null || x === undefined) return "-";
  const s = x.toLocaleString();
  return x > 0 ? "+" + s : s;
}

async function fetchTrades(params) {
  const qs = new URLSearchParams(params);
  const res = await fetch("/api/trades/history?" + qs.toString());
  if (!res.ok) throw new Error("trades API error");
  return res.json();
}

function renderSummary(items) {
  let sumBuy = 0;
  let sumSell = 0;
  let sumRealized = 0;

  for (const it of items) {
    if (it.side === "BUY") {
      sumBuy += it.amount;
    } else if (it.side === "SELL") {
      sumSell += -it.amount; // amount는 매도 시 음수
      sumRealized += it.realized || 0;
    }
  }

  const sumRealizedEl = document.getElementById("sumRealized");
  sumRealizedEl.textContent = formatSigned(Math.round(sumRealized));
  sumRealizedEl.className =
    "summary-value " + (sumRealized >= 0 ? "pos" : "neg");

  document.getElementById("sumBuyAmount").textContent = formatNumber(
    Math.round(sumBuy)
  );
  document.getElementById("sumSellAmount").textContent = formatNumber(
    Math.round(sumSell)
  );
  document.getElementById("sumTrades").textContent = items.length.toString();
}

function renderTable(items) {
  const tbody = document.getElementById("tradeTbody");
  tbody.innerHTML = "";

  if (!items.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 10;
    td.textContent = "거래 내역이 없습니다.";
    td.style.textAlign = "center";
    td.style.color = "#9ca3af";
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }

  for (const it of items) {
    const tr = document.createElement("tr");
    tr.className = it.side === "SELL" ? "row-sell" : "row-buy";

    const sideBadge =
      it.side === "SELL"
        ? '<span class="badge-sell">매도</span>'
        : '<span class="badge-buy">매수</span>';

    tr.innerHTML = `
      <td>${it.date}</td>
      <td>${sideBadge}</td>
      <td>${it.name} <span style="font-size:11px;color:#9ca3af;">${it.code}</span></td>
      <td class="num">${formatNumber(it.qty)}</td>
      <td class="num">${formatNumber(it.price)}</td>
      <td class="num">${formatNumber(it.amount)}</td>
      <td class="num">${formatSigned(Math.round(it.realized || 0))}</td>
      <td class="num">${formatSigned(
        Math.round(it.realized_acc_code || 0)
      )}</td>
      <td class="num">${formatNumber(it.remain_qty)}</td>
      <td class="num">${formatNumber(
        it.avg_price ? Math.round(it.avg_price) : 0
      )}</td>
    `;

    tbody.appendChild(tr);
  }
}

async function loadTrades() {
  const keyword = document.getElementById("codeFilter").value.trim();
  const from = document.getElementById("fromDate").value;
  const to = document.getElementById("toDate").value;

  const params = {};
  if (keyword) {
    if (/^[0-9]+$/.test(keyword)) {
      params.code = keyword; // 숫자만: 코드 정확검색
    } else {
      params.q = keyword; // 코드/이름 부분검색
    }
  }
  if (from) params.from = from;
  if (to) params.to = to;

  const data = await fetchTrades(params);
  const items = data.items || [];

  renderSummary(items);
  renderTable(items);
}

document.addEventListener("DOMContentLoaded", () => {
  const filterBtn = document.getElementById("filterBtn");
  filterBtn.addEventListener("click", () => {
    loadTrades().catch((e) => {
      console.error(e);
      alert("매매 이력을 불러오는 중 오류가 발생했습니다.");
    });
  });

  // 엔터 검색
  document
    .getElementById("codeFilter")
    .addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        loadTrades().catch((err) => {
          console.error(err);
          alert("매매 이력을 불러오는 중 오류가 발생했습니다.");
        });
      }
    });

  // 빠른 기간 버튼
  document.querySelectorAll(".quick-range button[data-range]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const today = new Date();
      const fmt = (d) =>
        [
          d.getFullYear(),
          String(d.getMonth() + 1).padStart(2, "0"),
          String(d.getDate()).padStart(2, "0"),
        ].join("-");

      const range = btn.dataset.range;
      let from = null;
      if (range === "ytd") {
        from = new Date(today.getFullYear(), 0, 1);
      } else if (range === "1m") {
        from = new Date(today);
        from.setMonth(from.getMonth() - 1);
      } else if (range === "1w") {
        from = new Date(today);
        from.setDate(from.getDate() - 7);
      }

      document.getElementById("toDate").value = fmt(today);
      if (from) document.getElementById("fromDate").value = fmt(from);
      loadTrades().catch((err) => {
        console.error(err);
        alert("매매 이력을 불러오는 중 오류가 발생했습니다.");
      });
    });
  });

  loadTrades().catch((err) => {
    console.error(err);
    alert("매매 이력을 불러오는 중 오류가 발생했습니다.");
  });
});
