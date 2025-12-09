const express = require("express");
const cors = require("cors");
const fs = require("fs");
const path = require("path");
require("dotenv").config();
const { Pool } = require("pg");

const app = express();
const PORT = process.env.PORT || 3000;

// ---------------------
// Env / Postgres Pool
// ---------------------
function resolveDataDir() {
  const candidates = [
    "/app/data",
    path.join(__dirname, "data"),
    path.join(__dirname, "..", "data"),
    path.join(process.cwd(), "data"),
  ];
  for (const p of candidates) {
    try {
      if (fs.existsSync(p)) return p;
    } catch {}
  }
  return path.join(__dirname, "data");
}
const DATA_DIR = resolveDataDir();
console.log("[DATA_DIR]", DATA_DIR);

const { DATABASE_URL } = process.env;
if (!DATABASE_URL) {
  console.error("DATABASE_URL not set. API will fail to reach Postgres.");
}

const pool = new Pool({
  connectionString: DATABASE_URL,
  max: 10,
  idleTimeoutMillis: 0,
  connectionTimeoutMillis: 5000,
});

// ---------------------
// Helpers
// ---------------------
function readCsv(filePath) {
  try {
    if (!fs.existsSync(filePath)) return null;
    const content = fs.readFileSync(filePath, "utf-8");
    const lines = content.trim().split(/\r?\n/);
    if (!lines.length) return [];
    const headers = lines.shift().split(",");
    return lines
      .filter((l) => l)
      .map((line) => {
        const row = {};
        line.split(",").forEach((v, idx) => {
          row[headers[idx]] = v;
        });
        return row;
      });
  } catch (e) {
    console.warn("readCsv error", e.message);
    return null;
  }
}

function toNum(v) {
  if (v === null || v === undefined) return null;
  if (typeof v === "number" && Number.isFinite(v)) return v;
  const s = String(v).replace(/,/g, "").trim();
  if (!s) return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
}

function toIsoDate(v) {
  if (v === null || v === undefined) return "";
  if (v instanceof Date) {
    if (Number.isNaN(v.getTime())) return "";
    return v.toISOString().slice(0, 10);
  }
  const s = String(v).trim();
  if (!s) return "";
  // yyyy-mm-dd string
  if (/^\d{4}-\d{2}-\d{2}$/.test(s)) return s;
  const d = new Date(s);
  if (!Number.isNaN(d.getTime())) return d.toISOString().slice(0, 10);
  // fallback: first 10 chars
  return s.slice(0, 10);
}

function boolify(v) {
  const s = String(v).toLowerCase();
  if (["true", "1", "t", "yes"].includes(s)) return true;
  if (["false", "0", "f", "no"].includes(s)) return false;
  return null;
}

async function queryRows(sql, params = []) {
  const { rows } = await pool.query(sql, params);
  return rows;
}

async function getLatestDate(table) {
  const rows = await queryRows(`SELECT MAX(date) AS d FROM ${table}`);
  return rows[0]?.d || null;
}

// ---------------------
// Universe loader
// ---------------------
const UNIVERSE_CSV = path.join(DATA_DIR, "universe.csv");
let universeMap = new Map();
function getName(code) {
  const v = universeMap.get(code);
  return (v && v.name) || code;
}
function getMarket(code) {
  const v = universeMap.get(code);
  return (v && v.market) || null;
}
function getSector(code) {
  const v = universeMap.get(code);
  return (v && v.sector) || null;
}
function getShares(code) {
  const v = universeMap.get(code);
  return (v && v.shares) || null;
}
function getMktcap(code) {
  const v = universeMap.get(code);
  return (v && v.mktcap) || null;
}

function loadUniverse() {
  try {
    const rows = readCsv(UNIVERSE_CSV);
    const map = new Map();
    if (rows && rows.length) {
      rows.forEach((r) => {
        const code = (r.code || "").trim();
        if (!code) return;
        const shares = toNum(r.shares || r.shares_outstanding);
        const mktcap = toNum(r.mktcap || r.marketcap);
        map.set(code, {
          name: (r.name || code).trim(),
          market: (r.market || "").trim(),
          sector: (r.sector || "").trim(),
          shares: Number.isFinite(shares) ? shares : null,
          mktcap: Number.isFinite(mktcap) ? mktcap : null,
        });
      });
    }
    universeMap = map;
    console.log(`[universe] loaded ${universeMap.size} tickers`);
  } catch (e) {
    console.warn("Failed to load universe.csv", e.message);
  }
}

loadUniverse();
fs.watchFile(UNIVERSE_CSV, { interval: 5000 }, () => loadUniverse());

// ---------------------
// DB loaders
// ---------------------
async function loadMarketStatusLatest() {
  try {
    const rows = await queryRows(
      "SELECT * FROM market_status ORDER BY date DESC LIMIT 1"
    );
    if (rows.length) return rows[0];
  } catch (e) {
    console.warn("[market_status] DB load fail:", e.message);
  }

  try {
    const rows = readCsv(path.join(DATA_DIR, "market_status.csv")) || [];
    if (rows.length) {
      return rows
        .sort((a, b) => String(a.date || "").localeCompare(String(b.date || "")))
        .pop();
    }
  } catch (e) {
    console.warn("[market_status] CSV load fail:", e.message);
  }
  return null;
}

async function getPredictions() {
  try {
    return await queryRows("SELECT * FROM predictions");
  } catch (e) {
    console.warn("[predictions] DB load fail:", e.message);
    return readCsv(path.join(DATA_DIR, "predictions.csv")) || [];
  }
}

async function getFeatures(whereClause = "", params = []) {
  try {
    return await queryRows(
      `SELECT code, date, close, ma_5, ma_20, ma_60, rsi_14, vol_20, volume FROM features ${whereClause}`,
      params
    );
  } catch (e) {
    console.warn("[features] DB load fail:", e.message);
    return readCsv(path.join(DATA_DIR, "features.csv")) || [];
  }
}

async function getRanking(targetDate) {
  try {
    const date = targetDate || (await getLatestDate("daily_ranking"));
    if (!date) return null;
    const rows = await queryRows(
      "SELECT * FROM daily_ranking WHERE date = $1 ORDER BY final_score DESC NULLS LAST",
      [date]
    );
    return { date, rows };
  } catch (e) {
    console.warn("[daily_ranking] DB load fail:", e.message);
    const csvRows = readCsv(path.join(DATA_DIR, "ranking_final.csv"));
    if (!csvRows || !csvRows.length) return null;
    const date = targetDate || csvRows.map((r) => r.date).filter(Boolean).sort().pop();
    const rows = csvRows.filter((r) => String(r.date || "") === String(date));
    return { date, rows };
  }
}

async function getRankingLatestByCode() {
  const res = await getRanking();
  const map = new Map();
  if (res && res.rows) {
    res.rows.forEach((r) => map.set(r.code, r));
  }
  return map;
}

async function getFeatureStatsForCodes(codes) {
  if (!codes || !codes.length) return { latestClose: new Map(), ret3m: new Map() };
  const feats = await getFeatures("WHERE code = ANY($1) ORDER BY code, date", [codes]);
  const latestClose = new Map();
  const ret3m = new Map();

  let current = null;
  let buffer = [];
  const flush = () => {
    if (!buffer.length || !current) return;
    const last = buffer[buffer.length - 1];
    latestClose.set(current, toNum(last.close));
    if (buffer.length >= 60) {
      const prev = buffer[buffer.length - 60];
      const prevClose = toNum(prev.close);
      const lastClose = toNum(last.close);
      if (Number.isFinite(prevClose) && Number.isFinite(lastClose) && prevClose !== 0) {
        ret3m.set(current, lastClose / prevClose - 1);
      } else {
        ret3m.set(current, null);
      }
    }
    buffer = [];
  };

  for (const row of feats) {
    if (current !== row.code) {
      flush();
      current = row.code;
    }
    buffer.push(row);
  }
  flush();
  return { latestClose, ret3m };
}

// ---------------------
// Trades
// ---------------------
async function ensureTradesTable() {
  await pool.query(
    `
    CREATE TABLE IF NOT EXISTS trades (
      trade_id    BIGSERIAL PRIMARY KEY,
      date        DATE NOT NULL,
      side        TEXT NOT NULL,
      code        TEXT NOT NULL,
      name        TEXT,
      market      TEXT,
      sector      TEXT,
      qty         NUMERIC,
      price       NUMERIC,
      amount      NUMERIC,
      fee         NUMERIC,
      memo        TEXT,
      created_at  TIMESTAMPTZ DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS idx_trades_code_date ON trades(code, date);
    CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(date);
    `
  );
}

async function listTrades() {
  await ensureTradesTable();
  const rows = await queryRows(
    "SELECT * FROM trades ORDER BY date ASC, trade_id ASC"
  );
  return rows.map((r) => ({
    trade_id: r.trade_id,
    date: r.date ? String(r.date) : null,
    side: (r.side || "").toUpperCase(),
    code: r.code,
    name: r.name || null,
    market: r.market || null,
    sector: r.sector || null,
    qty: toNum(r.qty) || 0,
    price: toNum(r.price) || 0,
    amount: toNum(r.amount) || 0,
    fee: toNum(r.fee) || 0,
    memo: r.memo || "",
    created_at: r.created_at || null,
  }));
}

async function insertTrade(payload) {
  await ensureTradesTable();
  const {
    date,
    side,
    code,
    name = null,
    market = null,
    sector = null,
    qty,
    price,
    amount = null,
    fee = null,
    memo = null,
  } = payload;

  const { rows } = await pool.query(
    `
    INSERT INTO trades (date, side, code, name, market, sector, qty, price, amount, fee, memo)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
    RETURNING trade_id;
    `,
    [date, side, code, name, market, sector, qty, price, amount, fee, memo]
  );
  return rows[0];
}

// ---------------------
// Holdings helper
// ---------------------
function computePositions(trades) {
  const stateByCode = new Map();
  const sorted = trades
    .slice()
    .sort((a, b) => {
      const da = toIsoDate(a.date || "");
      const db = toIsoDate(b.date || "");
      if (da !== db) return da < db ? -1 : 1;
      return (Number(a.trade_id) || 0) - (Number(b.trade_id) || 0);
    });

  for (const t of sorted) {
    const code = (t.code || "").trim();
    if (!code) continue;
    const side = (t.side || "").toUpperCase().trim();
    const qRaw = toNum(t.qty);
    const p = toNum(t.price);
    if (!Number.isFinite(qRaw) || !Number.isFinite(p) || p <= 0) continue;
    const q = Math.abs(qRaw);
    if (q <= 0) continue;

    if (!stateByCode.has(code)) {
      stateByCode.set(code, { qty: 0, avgPrice: 0, realizedAcc: 0, totalBuy: 0 });
    }
    const st = stateByCode.get(code);

    if (side === "BUY") {
      const newQty = st.qty + q;
      st.avgPrice = (st.avgPrice * st.qty + p * q) / newQty;
      st.qty = newQty;
      st.totalBuy += p * q;
    } else if (side === "SELL") {
      const sellQty = Math.min(q, st.qty);
      st.realizedAcc += (p - st.avgPrice) * sellQty;
      st.qty -= sellQty;
      if (st.qty <= 0) {
        st.qty = 0;
        st.avgPrice = 0;
      }
    }
  }
  return stateByCode;
}

function buildHoldings(trades, latestRankByCode) {
  const stateByCode = computePositions(trades);
  const holdings = [];

  for (const [code, st] of stateByCode.entries()) {
    const qty = st.qty;
    if (qty <= 0) continue;

    const rankRow = latestRankByCode.get(code) || {};
    const name = rankRow.name || getName(code) || code;
    const market = rankRow.market || getMarket(code) || null;
    const sector = rankRow.sector || getSector(code) || null;
    const currentPrice = toNum(rankRow.close);
    const currentValue = Number.isFinite(currentPrice) ? currentPrice * qty : null;

    const avgBuyPrice = st.avgPrice > 0 ? st.avgPrice : null;
    const costBasis = avgBuyPrice && qty ? avgBuyPrice * qty : null;
    const unrealized =
      currentValue != null && costBasis != null ? currentValue - costBasis : null;
    const unrealizedPct =
      currentValue != null && costBasis
        ? (currentValue / costBasis - 1) * 100
        : null;

    const realizedPct = st.totalBuy > 0 ? (st.realizedAcc / st.totalBuy) * 100 : null;

    holdings.push({
      code,
      name,
      market,
      sector,
      current_qty: qty,
      avg_buy_price: avgBuyPrice,
      current_price: currentPrice,
      current_value: currentValue,
      cost_basis: costBasis,
      realized_pnl: st.realizedAcc,
      realized_pnl_pct: realizedPct,
      unrealized_pnl: unrealized,
      unrealized_pnl_pct: unrealizedPct,
      final_score: rankRow.final_score || rankRow.score || null,
    });
  }

  holdings.sort((a, b) => a.code.localeCompare(b.code));

  return holdings;
}

// ---------------------
// Express setup
// ---------------------
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

// Health
app.get("/api/health", (req, res) => {
  const demo = fs.existsSync(path.join(DATA_DIR, ".demo"));
  res.json({ status: "ok", message: "API running", demo });
});

// Market status
app.get("/api/market/status", async (req, res) => {
  try {
    const row = await loadMarketStatusLatest();
    if (!row) return res.status(404).json({ error: "market_status not found" });
    const status_date =
      row.date || row.status_date || row.market_status_date || null;
    res.json({
      status_date,
      market_up: (() => {
        const b = boolify(row.market_up);
        return b === null ? null : b;
      })(),
      kospi_close: toNum(row.kospi_close ?? row.close ?? row.kospi),
      kospi_ma20: toNum(row.kospi_ma20 ?? row.ma20),
      vol_5d: toNum(row.vol_5d ?? row.volatility_5d),
      foreign_5d: toNum(row.foreign_5d ?? row.foreign_trading_5d ?? row.foreign_net_5d),
    });
  } catch (e) {
    console.error("GET /api/market/status error", e);
    res.status(500).json({ error: "internal error" });
  }
});

// Sectors list
app.get("/api/sectors", (req, res) => {
  try {
    const sectors = Array.from(universeMap.values())
      .map((v) => v.sector || "")
      .filter((s) => s);
    res.json(Array.from(new Set(sectors)).sort((a, b) => a.localeCompare(b, "ko")));
  } catch {
    res.json([]);
  }
});

// Stocks list (predictions + latest feature snapshot)
app.get("/api/stocks", async (req, res) => {
  try {
    const preds = await getPredictions();
    if (!preds || !preds.length) {
      return res.status(404).json({ error: "predictions not found" });
    }
    const feats = await getFeatures();

    const byCode = new Map();
    feats.forEach((r) => {
      const code = r.code;
      if (!code) return;
      if (!byCode.has(code)) byCode.set(code, []);
      byCode.get(code).push(r);
    });

    const latestCloseMap = new Map();
    const ret3mMap = new Map();
    for (const [code, arr] of byCode.entries()) {
      arr.sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
      if (!arr.length) continue;
      const last = arr[arr.length - 1];
      latestCloseMap.set(code, toNum(last.close));
      if (arr.length >= 60) {
        const prev = arr[arr.length - 60];
        const prevClose = toNum(prev.close);
        const lastClose = toNum(last.close);
        if (Number.isFinite(prevClose) && Number.isFinite(lastClose) && prevClose !== 0) {
          ret3mMap.set(code, lastClose / prevClose - 1);
        }
      }
    }

    const marketFilter = (req.query.market || "").toUpperCase();
    const sectorFilter = (req.query.sector || "");

    let data = preds.map((r) => {
      const code = r.code;
      const info = universeMap.get(code) || {};
      const close = latestCloseMap.get(code);
      const shares = info.shares;
      const mktcap = Number.isFinite(info.mktcap)
        ? info.mktcap
        : Number.isFinite(shares) && Number.isFinite(close)
        ? shares * close
        : null;
      return {
        date: r.date,
        code,
        name: getName(code),
        market: (info.market || "").toUpperCase(),
        sector: info.sector || null,
        close,
        mktcap,
        ret_3m: ret3mMap.get(code) ?? null,
        pred_return_60d: toNum(r.pred_return_60d),
        pred_return_90d: r.pred_return_90d !== undefined && r.pred_return_90d !== "" ? toNum(r.pred_return_90d) : null,
        score: r.score !== undefined && r.score !== "" ? toNum(r.score) : null,
      };
    });

    if (marketFilter && marketFilter !== "ALL") {
      data = data.filter((d) => d.market === marketFilter);
    }
    if (sectorFilter && sectorFilter !== "ALL") {
      data = data.filter((d) => (d.sector || "") === sectorFilter);
    }

    res.json(data);
  } catch (e) {
    console.error("api/stocks error", e);
    res.status(500).json({ error: "internal error" });
  }
});

// Stock detail
app.get("/api/stocks/:code", async (req, res) => {
  try {
    const code = req.params.code;
    const limit = parseInt(req.query.limit || "180", 10);

    const feats = await getFeatures("WHERE code = $1 ORDER BY date", [code]);
    if (!feats || !feats.length) {
      return res.status(404).json({ error: `no data for code ${code}` });
    }
    const sliced = limit > 0 ? feats.slice(-limit) : feats;
    const rows = sliced.map((r) => ({
      date: r.date,
      close: toNum(r.close),
      ma_5: toNum(r.ma_5),
      ma_20: toNum(r.ma_20),
      ma_60: toNum(r.ma_60),
      rsi_14: toNum(r.rsi_14),
      vol_20: toNum(r.vol_20),
      volume: toNum(r.volume),
    }));
    const latest = rows[rows.length - 1];

    const pred = (
      await queryRows(
        "SELECT * FROM predictions WHERE code = $1 ORDER BY date DESC LIMIT 1",
        [code]
      )
    )[0];
    const rank = (
      await queryRows(
        "SELECT * FROM daily_ranking WHERE code = $1 ORDER BY date DESC LIMIT 1",
        [code]
      )
    )[0];

    res.json({
      code,
      name: getName(code),
      count: rows.length,
      latest,
      pred_return_60d: pred ? toNum(pred.pred_return_60d) : null,
      pred_return_90d: pred ? toNum(pred.pred_return_90d) : null,
      ret_score: rank ? toNum(rank.ret_score) : null,
      prob_score: rank ? toNum(rank.prob_score) : null,
      qual_score: rank ? toNum(rank.qual_score) : null,
      tech_score: rank ? toNum(rank.tech_score) : null,
      pred_score: rank ? toNum(rank.pred_score) : null,
      risk_penalty: rank ? toNum(rank.risk_penalty) : null,
      pred_mdd_60d: rank ? toNum(rank.pred_mdd_60d) : null,
      prob_top20_60d: rank ? toNum(rank.prob_top20_60d) : null,
      prob_top20_90d: rank ? toNum(rank.prob_top20_90d) : null,
      final_score_rank: rank ? toNum(rank.final_score) : null,
      rows,
    });
  } catch (e) {
    console.error("api/stocks/:code error", e);
    res.status(500).json({ error: "internal error" });
  }
});

// Ranking list
app.get("/api/ranking", async (req, res) => {
  try {
    const targetDate = (req.query.date || "").trim() || null;
    const rankingRes = await getRanking(targetDate);
    if (!rankingRes) return res.status(404).json({ error: "ranking data not found" });
    const { date, rows } = rankingRes;

    const codes = rows.map((r) => r.code);
    const { latestClose, ret3m } = await getFeatureStatsForCodes(codes);

    let marketUp = true;
    if (rows.length && rows[0].market_up !== undefined) {
      const b = boolify(rows[0].market_up);
      marketUp = b === null ? true : b;
    }

    const marketFilter = (req.query.market || "").toUpperCase();
    const sectorFilter = (req.query.sector || "");

    let data = rows.map((r) => {
      const code = r.code;
      return {
        date: r.date,
        code,
        name: (r.name && r.name.trim()) || getName(code),
        market: ((r.market && r.market.trim()) || getMarket(code) || "").toUpperCase(),
        sector: (r.sector && r.sector.trim()) || getSector(code) || null,
        close: latestClose.get(code) ?? toNum(r.close),
        ret_3m: ret3m.get(code) ?? null,
        pred_return_60d: toNum(r.pred_return_60d),
        pred_return_90d: toNum(r.pred_return_90d),
        pred_mdd_60d: toNum(r.pred_mdd_60d),
        pred_mdd_90d: toNum(r.pred_mdd_90d),
        risk_penalty: toNum(r.risk_penalty),
        prob_top20_60d: toNum(r.prob_top20_60d),
        prob_top20_90d: toNum(r.prob_top20_90d),
        score: toNum(r.final_score ?? r.score),
        final_score: toNum(r.final_score),
        tech_score: toNum(r.tech_score),
        qual_score: toNum(r.qual_score),
        ret_score: toNum(r.ret_score),
        prob_score: toNum(r.prob_score),
        pred_score: toNum(r.pred_score),
        market_up: marketUp,
        market_status_date: r.market_status_date || null,
        market_kospi_close: toNum(r.market_kospi_close),
        market_kospi_ma20: toNum(r.market_kospi_ma20),
        market_vol_5d: toNum(r.market_vol_5d),
        market_foreign_5d: toNum(r.market_foreign_5d),
      };
    });

    if (marketFilter && marketFilter !== "ALL") {
      data = data.filter((d) => d.market === marketFilter);
    }
    if (sectorFilter && sectorFilter !== "ALL") {
      data = data.filter((d) => (d.sector || "") === sectorFilter);
    }

    data.sort((a, b) => {
      const af = Number.isFinite(+a.final_score) ? +a.final_score : -Infinity;
      const bf = Number.isFinite(+b.final_score) ? +b.final_score : -Infinity;
      return bf - af;
    });

    // 기존 프런트 호환: 배열 형태를 그대로 반환
    res.setHeader("X-Ranking-Date", date);
    res.json(data);
  } catch (e) {
    console.error("api/ranking error", e);
    res.status(500).json({ error: "failed to read ranking", detail: String(e) });
  }
});

// Top20 summary
app.get("/api/top20", async (req, res) => {
  try {
    const rankingRes = await getRanking(req.query.date || null);
    if (!rankingRes) return res.status(404).json({ error: "ranking data is empty" });

    const { date, rows } = rankingRes;
    const sorted = rows
      .slice()
      .sort(
        (a, b) =>
          (Number.isFinite(+b.final_score) ? +b.final_score : -Infinity) -
          (Number.isFinite(+a.final_score) ? +a.final_score : -Infinity)
      )
      .slice(0, 20);

    if (!sorted.length) return res.status(404).json({ error: "no ranking rows" });

    const first = sorted[0];
    const marketMeta = {
      market_up: (() => {
        const b = boolify(first.market_up);
        return b === null ? true : b;
      })(),
      status_date: first.market_status_date || null,
      kospi_close: toNum(first.market_kospi_close),
      kospi_ma20: toNum(first.market_kospi_ma20),
      vol_5d: toNum(first.market_vol_5d),
      foreign_5d: toNum(first.market_foreign_5d),
    };

    const fmtPct = (v, d = 1) => {
      const n = toNum(v);
      if (!Number.isFinite(n)) return "-";
      return (n * 100).toFixed(d) + "%";
    };

    const items = sorted.map((r, idx) => {
      const code = r.code;
      const name = (r.name && r.name.trim()) || getName(code);
      const sector = r.sector || getSector(code) || null;
      const market = (r.market || getMarket(code) || "").toUpperCase();
      const close = toNum(r.close);
      const pred60 = toNum(r.pred_return_60d);
      const pred90 = toNum(r.pred_return_90d);
      const mdd60 = toNum(r.pred_mdd_60d);
      const mdd90 = toNum(r.pred_mdd_90d);
      const prob60 = toNum(r.prob_top20_60d);
      const prob90 = toNum(r.prob_top20_90d);
      const retScore = toNum(r.ret_score);
      const probScore = toNum(r.prob_score);
      const riskPenalty = toNum(r.risk_penalty);
      const finalScore = toNum(r.final_score ?? r.score);

      const summary_ko = [
        `(${idx + 1}) ${name} (${code})${sector ? ` · 섹터: ${sector}` : ""}${market ? ` · 시장: ${market}` : ""}`,
        `- 예상 수익률 60d ${fmtPct(pred60)}, 90d ${fmtPct(pred90)}`,
        `- 상위20% 확률: 60d ${fmtPct(prob60)}, 90d ${fmtPct(prob90)}`,
        `- 예상 MDD: 60d ${fmtPct(Math.abs(mdd60))}, 90d ${fmtPct(Math.abs(mdd90))}`,
        `- 점수: 수익 ${retScore?.toFixed?.(1) ?? "-"}, 확률 ${probScore?.toFixed?.(1) ?? "-"}, 리스크 ${riskPenalty?.toFixed?.(1) ?? "-"}`,
        `- 최종 점수(final_score): ${Number.isFinite(finalScore) ? finalScore.toFixed(2) : "-"}`,
      ].join("\n");

      return {
        rank: idx + 1,
        date: r.date,
        code,
        name,
        sector,
        market,
        close,
        pred_return_60d: pred60,
        pred_return_90d: pred90,
        pred_mdd_60d: mdd60,
        pred_mdd_90d: mdd90,
        prob_top20_60d: prob60,
        prob_top20_90d: prob90,
        ret_score: retScore,
        prob_score: probScore,
        risk_penalty: riskPenalty,
        final_score: finalScore,
        score: finalScore,
        summary_ko,
      };
    });

    res.json({ date, market: marketMeta, count: items.length, items });
  } catch (e) {
    console.error("api/top20 error", e);
    res.status(500).json({ error: "failed to build top20 summary", detail: String(e) });
  }
});

// Signals top20
app.get("/api/signals/top20", async (req, res) => {
  try {
    const horizon = req.query.horizon === "90d" ? "90d" : "60d";
    const limit = Math.max(1, Math.min(100, Number(req.query.limit) || 20));
    const onlyNew = req.query.only_new === "1" || req.query.only_new === "true";
    const rankingRes = await getRanking(req.query.date || null);
    if (!rankingRes) return res.status(404).json({ error: "ranking data is empty" });
    const { date, rows } = rankingRes;

    const sortKey = horizon === "90d" ? "pred_return_90d" : "pred_return_60d";
    const mddKey = horizon === "90d" ? "pred_mdd_90d" : "pred_mdd_60d";
    const probKey = horizon === "90d" ? "prob_top20_90d" : "prob_top20_60d";

    const filtered = rows
      .slice()
      .sort((a, b) => (toNum(b[sortKey]) || 0) - (toNum(a[sortKey]) || 0));

    const latestByCode = new Map();
    rows.forEach((r) => {
      const code = (r.code || "").trim();
      if (!code) return;
      const prev = latestByCode.get(code);
      if (!prev || String(prev.date || "") < String(r.date || "")) {
        latestByCode.set(code, r);
      }
    });

    const trades = await listTrades();
    const holdings = buildHoldings(trades, latestByCode);
    const holdingCodes = new Set(holdings.map((h) => h.code));

    const items = [];
    for (const r of filtered) {
      const code = (r.code || "").trim();
      if (!code) continue;
      const isHolding = holdingCodes.has(code);
      if (onlyNew && isHolding) continue;
      items.push({
        rank: items.length + 1,
        code,
        name: r.name || getName(code) || code,
        market: r.market || getMarket(code) || "",
        sector: r.sector || getSector(code) || "",
        close: toNum(r.close),
        pred_return_60d: toNum(r.pred_return_60d),
        pred_return_90d: toNum(r.pred_return_90d),
        pred_mdd_60d: toNum(r.pred_mdd_60d),
        pred_mdd_90d: toNum(r.pred_mdd_90d),
        pred_return: toNum(r[sortKey]),
        pred_mdd: toNum(r[mddKey]),
        prob_top20: toNum(r[probKey]),
        final_score: toNum(r.final_score),
        ret_score: toNum(r.ret_score),
        prob_score: toNum(r.prob_score),
        qual_score: toNum(r.qual_score),
        tech_score: toNum(r.tech_score),
        pred_score: toNum(r.pred_score),
        risk_penalty: toNum(r.risk_penalty),
        is_holding: isHolding,
      });
      if (items.length >= limit) break;
    }

    res.json({ date, horizon, items });
  } catch (e) {
    console.error("GET /api/signals/top20 error", e);
    res.status(500).json({ error: "internal error" });
  }
});

// Today actions
app.get("/api/dashboard/today-actions", async (req, res) => {
  try {
    const rankingRes = await getRanking(req.query.date || null);
    if (!rankingRes) return res.status(404).json({ error: "ranking_final is empty" });
    const { date: targetDate, rows } = rankingRes;

    const latestByCode = new Map();
    rows.forEach((r) => {
      const code = (r.code || "").trim();
      if (!code) return;
      const prev = latestByCode.get(code);
      if (!prev || String(prev.date || "") < String(r.date || "")) {
        latestByCode.set(code, r);
      }
    });

    const trades = await listTrades();
    const holdings = buildHoldings(trades, latestByCode);
    const holdingCodes = new Set(holdings.map((h) => h.code));

    const BUY_MIN_RET = 0.30;
    const BUY_MIN_PROB = 0.40;
    const BUY_MIN_MDD = -0.35;
    const ADD_MIN_RET = 0.25;
    const ADD_MIN_PROB = 0.30;
    const ADD_MIN_MDD = -0.40;
    const TRIM_MAX_RET = 0.05;
    const TRIM_MIN_MDD = -0.45;

    const sorted = rows
      .slice()
      .sort((a, b) => (toNum(b.pred_return_60d) || 0) - (toNum(a.pred_return_60d) || 0))
      .slice(0, 50);

    const buyCandidates = [];
    const addCandidates = [];
    const trimCandidates = [];

    for (let idx = 0; idx < sorted.length; idx++) {
      const r = sorted[idx];
      const code = (r.code || "").trim();
      if (!code) continue;

      const predRet = toNum(r.pred_return_60d);
      const predMdd = toNum(r.pred_mdd_60d);
      const prob = toNum(r.prob_top20_60d);
      const finalScore = toNum(r.final_score);
      const isHolding = holdingCodes.has(code);

      if (
        !isHolding &&
        idx < 10 &&
        predRet != null &&
        predRet >= BUY_MIN_RET &&
        prob != null &&
        prob >= BUY_MIN_PROB &&
        predMdd != null &&
        predMdd >= BUY_MIN_MDD
      ) {
        buyCandidates.push({
          code,
          name: r.name || getName(code) || code,
          market: r.market || getMarket(code) || "",
          sector: r.sector || getSector(code) || "",
          close: toNum(r.close),
          pred_return_60d: predRet,
          pred_mdd_60d: predMdd,
          prob_top20_60d: prob,
          final_score: finalScore,
          reason: "예상 수익/확률 양호 + 리스크 완화 범위",
        });
      }

      if (
        isHolding &&
        predRet != null &&
        predRet >= ADD_MIN_RET &&
        prob != null &&
        prob >= ADD_MIN_PROB &&
        predMdd != null &&
        predMdd >= ADD_MIN_MDD
      ) {
        addCandidates.push({
          code,
          name: r.name || getName(code) || code,
          market: r.market || getMarket(code) || "",
          sector: r.sector || getSector(code) || "",
          close: toNum(r.close),
          pred_return_60d: predRet,
          pred_mdd_60d: predMdd,
          prob_top20_60d: prob,
          final_score: finalScore,
          reason: "보유 중이며 모멘텀/확률 충분",
        });
      }

      if (
        isHolding &&
        ((predRet != null && predRet < TRIM_MAX_RET) ||
          (predMdd != null && predMdd < TRIM_MIN_MDD))
      ) {
        trimCandidates.push({
          code,
          name: r.name || getName(code) || code,
          market: r.market || getMarket(code) || "",
          sector: r.sector || getSector(code) || "",
          close: toNum(r.close),
          pred_return_60d: predRet,
          pred_mdd_60d: predMdd,
          prob_top20_60d: prob,
          final_score: finalScore,
          reason: "수익 기대치 낮거나 MDD 위험 확대",
        });
      }
    }

    res.json({
      date: targetDate,
      horizon: "60d",
      buy_candidates: buyCandidates.slice(0, 5),
      add_candidates: addCandidates.slice(0, 5),
      trim_candidates: trimCandidates.slice(0, 5),
    });
  } catch (e) {
    console.error("GET /api/dashboard/today-actions error", e);
    res.status(500).json({ error: "internal error" });
  }
});

// Trades list
app.get("/api/trades", async (req, res) => {
  try {
    const trades = await listTrades();
    res.json({ count: trades.length, items: trades });
  } catch (e) {
    console.error("api/trades GET error", e);
    res.status(500).json({ error: "failed to load trades", detail: String(e) });
  }
});

// Create trade
app.post("/api/trades", async (req, res) => {
  try {
    const { side, code, date, qty, price, fee, memo } = req.body || {};
    const s = (side || "").toUpperCase();
    if (!["BUY", "SELL"].includes(s)) {
      return res.status(400).json({ error: "side must be BUY or SELL" });
    }
    if (!code) return res.status(400).json({ error: "code required" });
    if (!date) return res.status(400).json({ error: "date required" });

    const q = Number(qty);
    const p = Number(price);
    if (!Number.isFinite(q) || q <= 0) return res.status(400).json({ error: "qty > 0" });
    if (!Number.isFinite(p) || p <= 0) return res.status(400).json({ error: "price > 0" });

    const amount = q * p;
    const info = universeMap.get(code) || {};
    const inserted = await insertTrade({
      date,
      side: s,
      code,
      name: info.name || null,
      market: info.market || null,
      sector: info.sector || null,
      qty: q,
      price: p,
      amount,
      fee: fee ?? null,
      memo: memo ?? null,
    });

    res.json({ success: true, trade_id: inserted.trade_id });
  } catch (e) {
    console.error("[POST /api/trades] error:", e);
    res.status(500).json({ error: "failed to save", detail: String(e) });
  }
});

// Holdings summary
app.get("/api/holdings", async (req, res) => {
  try {
    const trades = await listTrades();
    if (!trades.length) return res.json({ count: 0, items: [] });

    const latestRankByCode = await getRankingLatestByCode();
    const holdings = buildHoldings(trades, latestRankByCode);

    let totalValue = 0;
    let totalCost = 0;
    let totalRealized = 0;
    holdings.forEach((h) => {
      if (Number.isFinite(h.current_value)) totalValue += h.current_value;
      if (Number.isFinite(h.cost_basis)) totalCost += h.cost_basis;
      if (Number.isFinite(h.realized_pnl)) totalRealized += h.realized_pnl;
    });

    const totalUnrealized = totalValue - totalCost;
    const totalUnrealizedPct = totalCost > 0 ? (totalUnrealized / totalCost) * 100 : null;
    const totalPnl = totalRealized + totalUnrealized;
    const totalPnlPct = totalCost > 0 ? (totalPnl / totalCost) * 100 : null;

    res.json({
      count: holdings.length,
      total_cost: totalCost,
      total_value: totalValue,
      total_realized_pnl: totalRealized,
      total_unrealized_pnl: totalUnrealized,
      total_unrealized_pnl_pct: totalUnrealizedPct,
      total_pnl: totalPnl,
      total_pnl_pct: totalPnlPct,
      items: holdings,
    });
  } catch (e) {
    console.error("api/holdings error", e);
    res.status(500).json({ error: "failed to build holdings", detail: String(e) });
  }
});

// Trades history with running stats
app.get("/api/trades/history", async (req, res) => {
  try {
    const rows = await listTrades();
    if (!rows || !rows.length) return res.json({ items: [] });

    const codeFilter = (req.query.code || "").trim();
    const q = (req.query.q || "").trim().toLowerCase();
    const from = (req.query.from || "").trim();
    const to = (req.query.to || "").trim();

    const stateByCode = new Map();
    const items = [];
    let totalRealizedAcc = 0;

    // 1) 계산용: 오름차순(날짜 asc, trade_id asc)
    rows.sort((a, b) => {
      const da = toIsoDate(a.date || "");
      const db = toIsoDate(b.date || "");
      if (da !== db) return da < db ? -1 : 1; // asc
      return (Number(a.trade_id) || 0) - (Number(b.trade_id) || 0);
    });

    for (const r of rows) {
      const date = toIsoDate(r.date || "");
      const code = String(r.code || "").trim();
      if (!code) continue;

      const name = r.name || getName(code) || code;
      const inDate = (!from || date >= from) && (!to || date <= to);
      const inCode = !codeFilter || code === codeFilter;
      let include = inDate && inCode;
      if (q) {
        include = include && (code.toLowerCase().includes(q) || name.toLowerCase().includes(q));
      }

      const side = (r.side || "").toUpperCase().trim();
      const qty = toNum(r.qty) || 0;
      const price = toNum(r.price) || 0;
      const amount = qty * price * (side === "SELL" ? -1 : 1);

      if (!stateByCode.has(code)) {
        stateByCode.set(code, { qty: 0, avgPrice: 0, realizedAcc: 0 });
      }
      const st = stateByCode.get(code);

      let realized = 0;
      if (side === "BUY") {
        const newQty = st.qty + qty;
        const newCost = st.qty * st.avgPrice + qty * price;
        st.qty = newQty;
        st.avgPrice = newQty > 0 ? newCost / newQty : 0;
      } else if (side === "SELL") {
        const sellQty = Math.min(qty, st.qty > 0 ? st.qty : qty);
        realized = (price - st.avgPrice) * sellQty;
        st.qty -= sellQty;
        if (st.qty < 0) st.qty = 0;
        if (st.qty === 0) st.avgPrice = 0;
        st.realizedAcc += realized;
        totalRealizedAcc += realized;
      }

      if (include) {
        items.push({
          trade_id: r.trade_id,
          date: String(date).slice(0, 10), // YYYY-MM-DD
          code,
          name,
          side,
          qty: side === "SELL" ? -qty : qty,
          price,
          amount,
          realized,
          realized_acc_code: st.realizedAcc,
          realized_acc_total: totalRealizedAcc,
          remain_qty: st.qty,
          avg_price: st.avgPrice,
        });
      }
    }

    // 2) 응답용: 시간순(날짜 asc, trade_id asc) + 날짜 포맷 고정
    items.sort((a, b) => {
      const da = toIsoDate(a.date || "");
      const db = toIsoDate(b.date || "");
      if (da !== db) return da < db ? -1 : 1;
      return (Number(a.trade_id) || 0) - (Number(b.trade_id) || 0);
    });

    res.json({ items });
  } catch (e) {
    console.error("GET /api/trades/history error", e);
    res.status(500).json({ error: "internal error" });
  }
});

// Holding detail
app.get("/api/holding/:code", async (req, res) => {
  try {
    const rawCode = (req.params.code || "").trim();
    if (!rawCode) return res.status(400).json({ error: "code required" });
    const code = rawCode;

    const tradesAll = await listTrades();
    const trades = tradesAll
      .filter((t) => (t.code || "").trim() === code)
      .sort((a, b) => {
        const da = String(a.date || "");
        const db = String(b.date || "");
        if (da !== db) return da < db ? -1 : 1;
        return (Number(a.trade_id) || 0) - (Number(b.trade_id) || 0);
      });

    const rank = (
      await queryRows(
        "SELECT * FROM daily_ranking WHERE code = $1 ORDER BY date DESC LIMIT 1",
        [code]
      )
    )[0];

    const name = (rank && rank.name) || (trades[0] && trades[0].name) || getName(code) || code;
    const market = (rank && rank.market) || (trades[0] && trades[0].market) || getMarket(code) || null;
    const sector = (rank && rank.sector) || (trades[0] && trades[0].sector) || getSector(code) || null;

    if (!trades.length) {
      return res.json({ code, name, market, sector, holding: null, latest: rank || null, trades: [] });
    }

    let positionQty = 0;
    let avgCost = 0;
    let realizedPnl = 0;
    let totalBuyAmount = 0;
    let firstBuyDate = null;
    let lastTradeDate = null;

    const tradesWithRun = trades.map((t) => {
      const side = (t.side || "").toUpperCase();
      const q = toNum(t.qty);
      const p = toNum(t.price);
      const dateStr = String(t.date || "");

      if (!firstBuyDate && side === "BUY") firstBuyDate = dateStr;
      if (side === "BUY" && dateStr && firstBuyDate && dateStr < firstBuyDate) firstBuyDate = dateStr;
      if (!lastTradeDate || (dateStr && dateStr > lastTradeDate)) lastTradeDate = dateStr;

      if (Number.isFinite(q) && Number.isFinite(p) && q > 0 && p > 0) {
        if (side === "BUY") {
          const newQty = positionQty + q;
          avgCost = (avgCost * positionQty + p * q) / newQty;
          positionQty = newQty;
          totalBuyAmount += p * q;
        } else if (side === "SELL" && positionQty > 0) {
          const sellQty = Math.min(q, positionQty);
          realizedPnl += (p - avgCost) * sellQty;
          positionQty -= sellQty;
        }
      }

      return {
        trade_id: t.trade_id,
        date: t.date,
        side,
        qty: t.qty,
        price: t.price,
        amount: t.amount,
        fee: t.fee,
        memo: t.memo,
        running_qty: positionQty,
        running_avg_price: avgCost,
        running_realized_pnl: realizedPnl,
      };
    });

    let holding = null;
    if (positionQty > 0) {
      const currentPrice =
        rank && rank.close !== undefined && rank.close !== "" ? Number(rank.close) : null;
      const avgBuyPrice = avgCost > 0 ? avgCost : null;
      const costBasis =
        Number.isFinite(avgBuyPrice) && Number.isFinite(positionQty)
          ? avgBuyPrice * positionQty
          : null;
      const currentValue =
        Number.isFinite(currentPrice) && Number.isFinite(positionQty)
          ? currentPrice * positionQty
          : null;

      let unrealizedPnl = null;
      let unrealizedPnlPct = null;
      if (Number.isFinite(currentValue) && Number.isFinite(costBasis) && costBasis !== 0) {
        unrealizedPnl = currentValue - costBasis;
        unrealizedPnlPct = (currentValue / costBasis - 1) * 100;
      }

      let realizedPnlPct = null;
      if (Number.isFinite(realizedPnl) && totalBuyAmount > 0) {
        realizedPnlPct = (realizedPnl / totalBuyAmount) * 100;
      }

      holding = {
        code,
        name,
        market,
        sector,
        current_qty: positionQty,
        avg_buy_price: avgBuyPrice,
        current_price: currentPrice,
        current_value: currentValue,
        cost_basis: costBasis,
        unrealized_pnl: unrealizedPnl,
        unrealized_pnl_pct: unrealizedPnlPct,
        realized_pnl: realizedPnl,
        realized_pnl_pct: realizedPnlPct,
        final_score: rank ? Number(rank.final_score) : null,
        first_buy_date: firstBuyDate,
        last_trade_date: lastTradeDate,
      };
    }

    res.json({ code, name, market, sector, latest: rank || null, holding, trades: tradesWithRun });
  } catch (e) {
    console.error("api/holding error", e);
    res.status(500).json({ error: "failed to build holding detail", detail: String(e) });
  }
});

// Debug data dir
app.get("/api/debug/data-dir", (req, res) => {
  res.json({ DATA_DIR });
});

app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
