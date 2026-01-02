(() => {
  const tokenInput = document.getElementById("admin-token");
  const saveTokenBtn = document.getElementById("save-token");
  const refreshBtn = document.getElementById("refresh-tables");
  const tableList = document.getElementById("table-list");
  const filterInput = document.getElementById("table-filter");
  const limitInput = document.getElementById("table-limit");
  const loadTableBtn = document.getElementById("load-table");
  const tableResult = document.getElementById("table-result");
  const sqlInput = document.getElementById("sql-input");
  const runQueryBtn = document.getElementById("run-query");
  const queryResult = document.getElementById("query-result");

  const getToken = () => localStorage.getItem("admin_token") || "";
  const setToken = (v) => localStorage.setItem("admin_token", v || "");

  tokenInput.value = getToken();

  function headers() {
    const h = { "Content-Type": "application/json" };
    const t = getToken();
    if (t) h["x-admin-token"] = t;
    return h;
  }

  function renderTable(container, data) {
    const { columns = [], rows = [] } = data || {};
    if (!columns.length) {
      container.innerHTML = "<p>no rows</p>";
      return;
    }
    const thead = `<thead><tr>${columns.map((c) => `<th>${c}</th>`).join("")}</tr></thead>`;
    const tbody = `<tbody>${rows
      .map(
        (r) =>
          "<tr>" +
          columns.map((c) => `<td>${r[c] !== undefined && r[c] !== null ? r[c] : ""}</td>`).join("") +
          "</tr>"
      )
      .join("")}</tbody>`;
    container.innerHTML = `<table>${thead}${tbody}</table>`;
  }

  async function loadTables() {
    try {
      const res = await fetch("/api/admin/db/tables", { headers: headers() });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      tableList.innerHTML = "";
      (data.tables || []).forEach((name) => {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        tableList.appendChild(opt);
      });
    } catch (e) {
      alert("테이블 목록을 불러오지 못했습니다: " + e.message);
    }
  }

  async function loadTablePreview() {
    const name = tableList.value;
    if (!name) return alert("테이블을 선택하세요.");
    const params = new URLSearchParams();
    params.set("name", name);
    const limit = Number(limitInput.value) || 100;
    params.set("limit", limit);
    const filter = filterInput.value.trim();
    if (filter) params.set("filter", filter);
    try {
      const res = await fetch(`/api/admin/db/table?${params.toString()}`, { headers: headers() });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      renderTable(tableResult, data);
    } catch (e) {
      alert("테이블 미리보기 실패: " + e.message);
    }
  }

  async function runQuery() {
    const sql = sqlInput.value.trim();
    if (!sql) return alert("SELECT 쿼리를 입력하세요.");
    try {
      const res = await fetch("/api/admin/db/query", {
        method: "POST",
        headers: headers(),
        body: JSON.stringify({ sql }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const rows = data.rows || [];
      const cols = rows.length ? Object.keys(rows[0]) : [];
      renderTable(queryResult, { columns: cols, rows });
    } catch (e) {
      alert("쿼리 실행 실패: " + e.message);
    }
  }

  saveTokenBtn.addEventListener("click", () => {
    setToken(tokenInput.value.trim());
    alert("토큰을 저장했습니다.");
  });
  refreshBtn.addEventListener("click", loadTables);
  loadTableBtn.addEventListener("click", loadTablePreview);
  runQueryBtn.addEventListener("click", runQuery);

  loadTables();
})();
