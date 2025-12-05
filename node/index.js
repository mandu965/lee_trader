const express = require("express");
const cors = require("cors");
const fs = require("fs");
const path = require("path");
const Database = require("better-sqlite3");

const app = express();

const PORT = process.env.PORT || 3000;
console.log('---------------------------------------------------')

function resolveDataDir() {
  const candidates = [
    "/app/data", // container volume mount 우선
    path.join(__dirname, "data"), // local dev: node/data
    path.join(__dirname, "..", "data"), // repo root data
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
console.log("DATA_DIR fixed to:", DATA_DIR);
const DB_PATH = path.join(DATA_DIR, "lee_trader.db");
let dbHandle = null;

function getDb() {
  if (dbHandle) return dbHandle;
  try {
    dbHandle = new Database(DB_PATH, { fileMustExist: false });
    dbHandle.pragma("foreign_keys = ON");
    // ensure trades table exists (others are created by Python migrate)
    dbHandle.prepare(
      `
      CREATE TABLE IF NOT EXISTS trades (
        trade_id    INTEGER PRIMARY KEY AUTOINCREMENT,
        date        TEXT NOT NULL,
        side        TEXT NOT NULL,
        code        TEXT NOT NULL,
        name        TEXT,
        market      TEXT,
        sector      TEXT,
        qty         REAL,
        price       REAL,
        amount      REAL,
        fee         REAL,
        memo        TEXT,
        created_at  TEXT
      );
      `
    ).run();
    return dbHandle;
  } catch (e) {
    console.warn("[DB] open failed:", e.message);
    return null;
  }
}

// -------- DB loaders -------- //
function dbLatestRanking() {
  const db = getDb();
  if (!db) return null;
  try {
    const hit = db.prepare("SELECT MAX(date) AS d FROM daily_ranking").get();
    if (!hit || !hit.d) return null;
    const rows = db.prepare("SELECT * FROM daily_ranking WHERE date = ?").all(hit.d);
    return { date: hit.d, rows };
  } catch (e) {
    console.warn("[DB] ranking load fail:", e.message);
    return null;
  }
}

function dbPredictions() {
  const db = getDb();
  if (!db) return null;
  try {
    return db.prepare("SELECT * FROM predictions").all();
  } catch (e) {
    console.warn("[DB] predictions load fail:", e.message);
    return null;
  }
}

function dbFeatures() {
  const db = getDb();
  if (!db) return null;
  try {
    return db.prepare("SELECT * FROM features").all();
  } catch (e) {
    console.warn("[DB] features load fail:", e.message);
    return null;
  }
}

function dbRankingAll() {
  const db = getDb();
  if (!db) return null;
  try {
    return db.prepare("SELECT * FROM daily_ranking").all();
  } catch (e) {
    console.warn("[DB] daily_ranking load fail:", e.message);
    return null;
  }
}

function dbRankingByDate(date) {
  const db = getDb();
  if (!db) return null;
  try {
    if (date) {
      const rows = db.prepare("SELECT * FROM daily_ranking WHERE date = ?").all(date);
      if (rows && rows.length) return { date, rows };
    }
    const hit = db.prepare("SELECT MAX(date) AS d FROM daily_ranking").get();
    if (!hit || !hit.d) return null;
    const rows = db.prepare("SELECT * FROM daily_ranking WHERE date = ?").all(hit.d);
    return { date: hit.d, rows };
  } catch (e) {
    console.warn("[DB] ranking load fail:", e.message);
    return null;
  }
}

function dbMarketStatusLatest() {
  const db = getDb();
  if (!db) return null;
  try {
    const hit = db.prepare("SELECT * FROM market_status ORDER BY date DESC LIMIT 1").get();
    return hit || null;
  } catch (e) {
    console.warn("[DB] market_status load fail:", e.message);
    return null;
  }
}

function loadMarketStatusLatest() {
  // 1) DB
  const dbHit = dbMarketStatusLatest();
  if (dbHit) return dbHit;

  // 2) CSV fallback
  try {
    const rows = readCsv(MARKET_STATUS_CSV) || [];
    if (rows && rows.length) {
      const sorted = rows
        .map((r) => r)
        .sort((a, b) => {
          const da = String(a.date || a.status_date || a.market_status_date || "");
          const db = String(b.date || b.status_date || b.market_status_date || "");
          if (da < db) return -1;
          if (da > db) return 1;
          return 0;
        });
      return sorted[sorted.length - 1];
    }
  } catch (e) {
    console.warn("[market_status] CSV load fail:", e.message);
  }
  return null;
}

const RANKING_PATH = path.join(DATA_DIR, "ranking_final.csv");
const RANKING_HISTORY_DIR = path.join(DATA_DIR, "history");
const TRADES_CSV_PATH = path.join(DATA_DIR, "trades.csv");
const MARKET_STATUS_CSV = path.join(DATA_DIR, "market_status.csv");

app.use(cors());
app.use(express.json());
const PUBLIC_DIR = path.join(__dirname, "public");
app.use(express.static(PUBLIC_DIR));
const UNIVERSE_CSV = path.join(DATA_DIR, "universe.csv");
let universeMap = new Map();
/** universeMap: code -> { name, market, sector, shares?, mktcap? } */
function getName(code) {
  const v = universeMap.get(code);
  return (v && v.name) || v || code;
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

function boolify(v) {
  if (v === true || v === "true" || v === "TRUE" || v === "True") return true;
  if (v === false || v === "false" || v === "FALSE" || v === "False") return false;
  if (v === 1 || v === "1") return true;
  if (v === 0 || v === "0") return false;
  return null;
}

// Helpers
function readCsv(filePath) {
  // DB 우선 읽기: ranking/predictions/features/market_status
  try {
    const db = getDb();
    const base = path.basename(filePath).toLowerCase();
    if (db) {
      if (base === "ranking_final.csv") {
        const maxDate = db.prepare("SELECT MAX(date) as d FROM daily_ranking").get();
        if (maxDate && maxDate.d) {
          const rows = db.prepare("SELECT * FROM daily_ranking WHERE date = ?").all(maxDate.d);
          return rows;
        }
      } else if (base === "predictions.csv") {
        const rows = db.prepare("SELECT * FROM predictions").all();
        if (rows && rows.length) return rows;
      } else if (base === "features.csv") {
        const rows = db.prepare("SELECT * FROM features").all();
        if (rows && rows.length) return rows;
      } else if (base === "market_status.csv") {
        const rows = db.prepare("SELECT * FROM market_status").all();
        if (rows && rows.length) return rows;
      } else if (base === "universe.csv") {
        const rows = db.prepare("SELECT code, name, market, sector FROM stocks").all();
        if (rows && rows.length) return rows;
      }
    }
  } catch (e) {
    console.warn("[readCsv->DB] fallback to file:", e.message);
  }

  try {
    if (!fs.existsSync(filePath)) return null;
    const content = fs.readFileSync(filePath, "utf-8");
    const lines = content.trim().split(/\r?\n/);
    if (lines.length === 0) return [];
    const headers = lines.shift().split(",");
    const rows = [];
    for (const line of lines) {
      if (!line) continue;
      const values = line.split(",");
      const row = {};
      headers.forEach((h, i) => {
        row[h] = values[i] !== undefined ? values[i] : "";
      });
      rows.push(row);
    }
    return rows;
  } catch (e) {
    console.error("readCsv error", e);
    return null;
  }
}

// trades.csv 없으면 헤더 만들어두기
// trades load/save (DB first, fallback CSV)
function ensureTradesTable() {
  const db = getDb();
  if (!db) return null;
  db.prepare(
    `
    CREATE TABLE IF NOT EXISTS trades (
      trade_id    INTEGER PRIMARY KEY AUTOINCREMENT,
      date        TEXT NOT NULL,
      side        TEXT NOT NULL,
      code        TEXT NOT NULL,
      name        TEXT,
      market      TEXT,
      sector      TEXT,
      qty         REAL,
      price       REAL,
      amount      REAL,
      fee         REAL,
      memo        TEXT,
      created_at  TEXT
    );
    `
  ).run();
  return db;
}

function loadTrades() {
  const db = ensureTradesTable();
  if (!db) {
    throw new Error("DB unavailable for trades");
  }
  const rows = db.prepare("SELECT * FROM trades ORDER BY date ASC, trade_id ASC").all();
  return rows
    .map((r) => ({
      trade_id: r.trade_id,
      date: r.date,
      side: (r.side || "").toUpperCase(),
      code: r.code,
      name: r.name || null,
      market: r.market || null,
      sector: r.sector || null,
      qty: Number(r.qty) || 0,
      price: Number(r.price) || 0,
      amount: Number(r.amount) || 0,
      fee: Number(r.fee) || 0,
      memo: r.memo || "",
      created_at: r.created_at || null,
    }))
    .filter((t) => t.code && t.qty !== 0 && t.side);
}

// Append trade to CSV for redundancy/logging
function appendTradeCsv(trade) {
  try {
    const headers = [
      "trade_id",
      "date",
      "side",
      "code",
      "name",
      "market",
      "sector",
      "qty",
      "price",
      "amount",
      "fee",
      "memo",
      "created_at",
    ];
    const row = headers
      .map((h) => (trade[h] !== undefined && trade[h] !== null ? String(trade[h]) : ""))
      .join(",");
    const exists = fs.existsSync(TRADES_CSV_PATH);
    if (!exists) {
      fs.writeFileSync(TRADES_CSV_PATH, headers.join(",") + "\n" + row + "\n", "utf-8");
    } else {
      fs.appendFileSync(TRADES_CSV_PATH, row + "\n", "utf-8");
    }
  } catch (e) {
    console.warn("[CSV] failed to append trade:", e.message);
  }
}


/**
 * trades 배열을 기반으로 보유 종목(포지션) 계산
 * @param {Array} trades
 * @param {Map<string, any>} latestRankByCode  code -> ranking_final 최신행
 */


/**
 * trades 배열을 기반으로 보유 종목(포지션) + 실현손익 계산
 * - 평균단가 방식
 * - 각 종목별로 모든 거래를 날짜 순으로 정렬해서 처리
 * @param {Array} trades
 * @param {Map<string, any>} latestRankByCode  code -> ranking_final 최신행
 */

/**
 * trades 배열을 기반으로 보유 종목(포지션) + 실현손익 계산
 * 평균단가 방식
 */
function buildHoldings(trades, latestRankByCode) {
  // 코드별로 트레이드 묶기
  const byCode = new Map();
  for (const t of trades) {
    const code = (t.code || "").trim();
    if (!code) continue;
    if (!byCode.has(code)) byCode.set(code, []);
    byCode.get(code).push(t);
  }

  const holdings = [];

  // 종목별 포지션 계산
  for (const [code, list] of byCode.entries()) {
    // 날짜 순 정렬
    list.sort((a, b) => {
      const da = String(a.date || "");
      const db = String(b.date || "");
      if (da < db) return -1;
      if (da > db) return 1;

      const ia = Number(a.trade_id);
      const ib = Number(b.trade_id);
      return ia - ib;
    });

    let qty = 0;             // 현재 포지션
    let avg = 0;             // 평균단가
    let realized = 0;        // 실현손익(누적)
    let totalBuyAmount = 0;  // 매수금액 합 (실현손익 % 계산용)

    for (const t of list) {
      const side = (t.side || "").toUpperCase();
      const q = Number(t.qty);
      const p = Number(t.price);

      if (!Number.isFinite(q) || !Number.isFinite(p) || q <= 0 || p <= 0) continue;

      if (side === "BUY") {
        // 평균단가 갱신
        const newQty = qty + q;
        avg = (avg * qty + p * q) / newQty;
        qty = newQty;
        totalBuyAmount += p * q;

      } else if (side === "SELL") {
        if (qty <= 0) continue;

        const sellQty = Math.min(q, qty);
        realized += (p - avg) * sellQty; // 실현손익 추가
        qty -= sellQty;
        // avg는 그대로 유지
      }
    }

    if (qty <= 0) continue; // 보유하지 않음

    // 최신 종가 가져오기
    const r = latestRankByCode.get(code) || {};
    const name = r.name || getName(code) || code;
    const market = r.market || getMarket(code) || null;
    const sector = r.sector || getSector(code) || null;

    const currentPrice = Number(r.close) || null;
    const currentValue = currentPrice ? currentPrice * qty : null;

    const avgBuyPrice = avg > 0 ? avg : null;
    const costBasis = avgBuyPrice && qty ? avgBuyPrice * qty : null;
    const unrealized = currentValue != null && costBasis != null ? currentValue - costBasis : null;
    const unrealizedPct = currentValue != null && costBasis ? (currentValue / costBasis - 1) * 100 : null;

    const realizedPct = totalBuyAmount > 0 ? (realized / totalBuyAmount) * 100 : null;

    // 20% 목표가
    let target = null;
    let progress = null;
    if (avgBuyPrice) {
      target = avgBuyPrice * 1.2;
      if (currentPrice) {
        progress = Math.max(0, Math.min(100, ((currentPrice - avgBuyPrice) / (target - avgBuyPrice)) * 100));
      }
    }

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

      // 🔥 실현손익 추가
      realized_pnl: realized,
      realized_pnl_pct: realizedPct,

      // 평가손익
      unrealized_pnl: unrealized,
      unrealized_pnl_pct: unrealizedPct,

      target_price: target,
      progress_to_target: progress,
      final_score: r.final_score || r.score || null
    });
  }

  // 기본 정렬: 평가손익률 DESC
  holdings.sort((a, b) => {
    const pa = Number(a.unrealized_pnl_pct) || -Infinity;
    const pb = Number(b.unrealized_pnl_pct) || -Infinity;
    return pb - pa;
  });

  return holdings;
}



function toNum(v) {
  if (v === null || v === undefined) return null;
  if (typeof v === "number" && Number.isFinite(v)) return v;
  let s = String(v).trim();
  if (!s) return null;

  // Normalize: remove spaces and commas
  s = s.replace(/,/g, "").replace(/\s+/g, "");

  // Handle Korean units: 조(1e12), 억(1e8), 만(1e4)
  const unitMap = { "조": 1e12, "억": 1e8, "만": 1e4 };
  let total = 0;
  let matched = false;
  const re = /([+-]?\d+(?:\.\d+)?)(조|억|만)/g;
  let m;
  while ((m = re.exec(s)) !== null) {
    const num = parseFloat(m[1]);
    const mul = unitMap[m[2]];
    if (Number.isFinite(num) && mul) {
      total += num * mul;
      matched = true;
    }
  }
  if (matched) return Number.isFinite(total) ? total : null;

  // Fallback: keep only number-like chars
  const cleaned = s.replace(/[^0-9.+\-eE]/g, "");
  const n = parseFloat(cleaned);
  return Number.isFinite(n) ? n : null;
}

// Load universe (code->name) from CSV and watch for changes
function loadUniverse() {
  try {
    const rows = readCsv(UNIVERSE_CSV);
    const map = new Map();
    if (rows && rows.length) {
      rows.forEach((r) => {
        const code = (r.code || "").trim();
        const name = (r.name || "").trim();
        const market = (r.market || "").trim();
        const sector = (r.sector || "").trim();
        const sharesRaw = r.shares !== undefined ? r.shares : (r.shares_outstanding !== undefined ? r.shares_outstanding : "");
        const mktcapRaw = r.mktcap !== undefined ? r.mktcap : (r.marketcap !== undefined ? r.marketcap : "");
        const shares = toNum(sharesRaw);
        const mktcap = toNum(mktcapRaw);
        if (code) {
          map.set(code, {
            name: name || code,
            market: market || null,
            sector: sector || null,
            shares: Number.isFinite(shares) ? shares : null,
            mktcap: Number.isFinite(mktcap) ? mktcap : null,
          });
        }
      });
    }
    universeMap = map;
    console.log(`Universe loaded: ${universeMap.size} entries`);
  } catch (e) {
    console.warn("Failed to load universe.csv", e);
  }
}

function ensureRankingHistorySnapshot() {
  try {
    if (!fs.existsSync(RANKING_PATH)) {
      console.warn("[ranking_history] ranking_final.csv not found");
      return;
    }

    if (!fs.existsSync(RANKING_HISTORY_DIR)) {
      fs.mkdirSync(RANKING_HISTORY_DIR, { recursive: true });
    }

    const rows = readCsv(RANKING_PATH);
    if (!rows || rows.length === 0) {
      console.warn("[ranking_history] ranking_final.csv is empty");
      return;
    }

    // date 컬럼에서 최신 날짜 추출
    const dates = rows
      .map((r) => String(r.date || "").trim())
      .filter((d) => d.length > 0)
      .sort();
    if (!dates.length) {
      console.warn("[ranking_history] no date column in ranking_final.csv");
      return;
    }
    const latestDate = dates[dates.length - 1]; // 예: "2025-11-26"

    const snapshotName = `ranking_final_${latestDate}.csv`;
    const snapshotPath = path.join(RANKING_HISTORY_DIR, snapshotName);

    if (fs.existsSync(snapshotPath)) {
      console.log(
        `[ranking_history] snapshot already exists for ${latestDate}: ${snapshotName}`
      );
      return;
    }

    fs.copyFileSync(RANKING_PATH, snapshotPath);
    console.log(
      `[ranking_history] snapshot created for ${latestDate}: ${snapshotName}`
    );
  } catch (e) {
    console.error("[ranking_history] error while creating snapshot", e);
  }
}

ensureRankingHistorySnapshot();  // ✅ 서버 뜰 때 스냅샷 한 번 생성

loadUniverse();
fs.watchFile(UNIVERSE_CSV, { interval: 5000 }, () => {
  try {
    loadUniverse();
  } catch {}
});

// Health
app.get("/api/health", (req, res) => {
  const demo = fs.existsSync(path.join(DATA_DIR, ".demo"));
  res.json({ status: "ok", message: "API running", demo });
});

// 최신 시장 지표
app.get("/api/market/status", (req, res) => {
  try {
    const row = loadMarketStatusLatest();
    if (!row) {
      return res.status(404).json({ error: "market_status not found" });
    }
    const status_date = row.date || row.status_date || row.market_status_date || null;
    const payload = {
      status_date,
      market_up: (() => {
        const b = boolify(row.market_up);
        return b === null ? null : b;
      })(),
      kospi_close: toNum(row.kospi_close ?? row.close ?? row.kospi),
      kospi_ma20: toNum(row.kospi_ma20 ?? row.ma20),
      vol_5d: toNum(row.vol_5d ?? row.volatility_5d),
      foreign_5d: toNum(row.foreign_5d ?? row.foreign_trading_5d ?? row.foreign_net_5d),
    };
    return res.json(payload);
  } catch (e) {
    console.error("GET /api/market/status error", e);
    return res.status(500).json({ error: "internal error" });
  }
});

app.get("/api/sectors", (req, res) => {
  try {
    const sectors = Array.from(universeMap.values())
      .map(v => (v && v.sector) || "")
      .filter(s => s && s.trim().length > 0);
    const uniq = Array.from(new Set(sectors)).sort((a, b) => a.localeCompare(b, "ko"));
    res.json(uniq);
  } catch (e) {
    res.json([]);
  }
});


// GET /api/stocks
// predictions.csv + features.csv 기반 리스트 반환 (code, date, close, ret_3m, pred_return_60d, score)
app.get("/api/stocks", (req, res) => {
  const preds = dbPredictions() || readCsv(path.join(DATA_DIR, "predictions.csv"));
  if (preds === null) {
    return res.status(404).json({ error: "predictions not found" });
  }
  const feats = dbFeatures() || readCsv(path.join(DATA_DIR, "features.csv")) || [];

  // code별 최신 종가와 3개월(약 60영업일) 수익률 계산
  const latestCloseMap = new Map();
  const ret3mMap = new Map();
  const byCode = new Map();
  for (const r of feats) {
    const code = r.code;
    if (!code) continue;
    if (!byCode.has(code)) byCode.set(code, []);
    byCode.get(code).push(r);
  }
  for (const [code, arr] of byCode.entries()) {
    // 날짜 오름차순 정렬 가정(features.csv는 이미 정렬되어 있지만 안전장치)
    arr.sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
    const n = arr.length;
    if (n === 0) continue;
    const last = arr[n - 1];
    const lastClose = toNum(last.close);
    latestCloseMap.set(code, lastClose);
    if (n >= 60) {
      const prev = arr[n - 60];
      const prevClose = toNum(prev.close);
      if (Number.isFinite(lastClose) && Number.isFinite(prevClose) && prevClose !== 0) {
        ret3mMap.set(code, (lastClose / prevClose) - 1);
      } else {
        ret3mMap.set(code, null);
      }
    } else {
      ret3mMap.set(code, null);
    }
  }

  const marketFilter = (req.query.market || "").toUpperCase();
  const sectorFilter = (req.query.sector || "");
  let data = preds.map((r) => ({
    date: r.date,
    code: r.code,
    name: getName(r.code),
    market: (getMarket(r.code) || "").toUpperCase(),
    sector: getSector(r.code) || null,
    close: latestCloseMap.get(r.code) ?? null,
    // mktcap: prefer value from universe.csv, else shares * close if both available
    mktcap: (() => {
      const info = universeMap.get(r.code);
      const lc = latestCloseMap.get(r.code);
      if (info && Number.isFinite(+info.mktcap)) return +info.mktcap;
      if (info && Number.isFinite(+info.shares) && Number.isFinite(+lc)) return (+info.shares) * (+lc);
      return null;
    })(),
    ret_3m: ret3mMap.get(r.code) ?? null,
    pred_return_60d: toNum(r.pred_return_60d),
    pred_return_90d: r.pred_return_90d !== undefined && r.pred_return_90d !== "" ? toNum(r.pred_return_90d) : null,
    score: r.score !== undefined && r.score !== "" ? toNum(r.score) : null,
  }));
  if (marketFilter && marketFilter !== "ALL") {
    data = data.filter((d) => d.market === marketFilter);
  }
  if (sectorFilter && sectorFilter !== "ALL") {
    data = data.filter((d) => (d.sector || "") === sectorFilter);
  }
  res.json(data);
});

// GET /api/stocks/:code
// ?? ?? ???? (features + scores_final + daily_ranking.final_score)
app.get("/api/stocks/:code", (req, res) => {
  const code = req.params.code;
  const limit = parseInt(req.query.limit || "180", 10); // ?? 180? ??
  const featPath = path.join(DATA_DIR, "features.csv");
  const scorePath = path.join(DATA_DIR, "scores_final.csv");
  const rankingPath = path.join(DATA_DIR, "ranking_final.csv");

  const feats = readCsv(featPath);
  if (feats === null) {
    return res.status(404).json({ error: "features.csv not found" });
  }
  const scoresRows = readCsv(scorePath) || [];
  const scoreMap = new Map(
    scoresRows
      .filter((r) => r.code === code)
      .map((r) => [r.date, toNum(r.score)])
  );
  let finalScoreMap = new Map();
  try {
    const dbRank = dbRankingByDate(); // latest if no date param
    const rows =
      (dbRank && dbRank.rows && dbRank.rows.filter((r) => r.code === code)) ||
      ((readCsv(rankingPath) || []).filter((r) => r.code === code));
    finalScoreMap = new Map(rows.map((r) => [r.date, toNum(r.final_score)]));
  } catch {}

  const filtered = feats.filter((r) => r.code === code);
  if (filtered.length === 0) {
    return res.status(404).json({ error: `no data for code ${code}` });
  }
  const sliced = limit > 0 ? filtered.slice(-limit) : filtered;

  const rows = sliced.map((r) => ({
    date: r.date,
    close: toNum(r.close),
    ma_5: toNum(r.ma_5),
    ma_20: toNum(r.ma_20),
    ma_60: toNum(r.ma_60),
    rsi_14: toNum(r.rsi_14),
    vol_20: toNum(r.vol_20),
    volume: toNum(r.volume),
    score: scoreMap.get(r.date) ?? null,
    final_score: finalScoreMap.get(r.date) ?? null,
  }));
  const latest = rows[rows.length - 1];

  // latest ?? ???(60d/90d)? predictions.csv?? ??
  let pred60 = null;
  let pred90 = null;
  let latestRankRow = null;
  try {
    // predictions (for pred_return)
    const predRows = readCsv(path.join(DATA_DIR, "predictions.csv")) || [];
    const hit = predRows.find((p) => p.code === code);
    if (hit) {
      pred60 = toNum(hit.pred_return_60d);
      if (hit.pred_return_90d !== undefined && hit.pred_return_90d !== "") {
        pred90 = toNum(hit.pred_return_90d);
      }
    }
    // ranking_final or daily_ranking for score breakdown
    const rankRows =
      (dbRankingByDate() && dbRankingByDate().rows && dbRankingByDate().rows.filter((r) => r.code === code)) ||
      ((readCsv(rankingPath) || []).filter((r) => r.code === code));
    if (rankRows && rankRows.length) {
      // latest by date
      latestRankRow = rankRows.reduce((acc, cur) => {
        return (!acc || String(cur.date || "") > String(acc.date || "")) ? cur : acc;
      }, null);
    }
  } catch {}

  res.json({
    code,
    name: getName(code),
    count: rows.length,
    latest,
    pred_return_60d: pred60,
    pred_return_90d: pred90,
    // score breakdown (from latest ranking row if available)
    ret_score: latestRankRow ? toNum(latestRankRow.ret_score) : null,
    prob_score: latestRankRow ? toNum(latestRankRow.prob_score) : null,
    qual_score: latestRankRow ? toNum(latestRankRow.qual_score) : null,
    tech_score: latestRankRow ? toNum(latestRankRow.tech_score) : null,
    pred_score: latestRankRow ? toNum(latestRankRow.pred_score) : null,
    risk_penalty: latestRankRow ? toNum(latestRankRow.risk_penalty) : null,
    pred_mdd_60d: latestRankRow ? toNum(latestRankRow.pred_mdd_60d) : null,
    prob_top20_60d: latestRankRow ? toNum(latestRankRow.prob_top20_60d) : null,
    prob_top20_90d: latestRankRow ? toNum(latestRankRow.prob_top20_90d) : null,
    final_score_rank: latestRankRow ? toNum(latestRankRow.final_score) : null,
    rows,
  });
});

app.get("/api/backtest", (req, res) => {
  // 백테스트 결과는 메인 화면에서 숨김: 호출 시 빈 리스트 반환
  res.json([]);
});

app.get("/api/ranking", (req, res) => {
  try {
    const targetDate = (req.query.date || "").trim();
    const dbRank = dbRankingByDate(targetDate);
    let ranking = (dbRank && dbRank.rows) || readCsv(path.join(DATA_DIR, "ranking_final.csv"));
    if (ranking && targetDate) {
      ranking = ranking.filter((r) => String(r.date) === targetDate);
    }
    if (ranking === null) {
      return res.status(404).json({ error: "ranking data not found" });
    }

    // 🔥 market_up 값 읽어오기
    let marketUp = true;
    if (ranking.length > 0 && "market_up" in ranking[0]) {
      const raw = ranking[0].market_up;
      marketUp =
        raw === true ||
        raw === "True" ||
        raw === "true" ||
        raw === "1" ||
        raw === 1;
    }

    const feats = dbFeatures() || readCsv(path.join(DATA_DIR, "features.csv")) || [];

    // Build latest close and 3M return maps from features
    const latestCloseMap = new Map();
    const ret3mMap = new Map();
    const byCode = new Map();
    for (const r of feats) {
      const code = r.code;
      if (!code) continue;
      if (!byCode.has(code)) byCode.set(code, []);
      byCode.get(code).push(r);
    }
    for (const [code, arr] of byCode.entries()) {
      arr.sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
      const n = arr.length;
      if (n === 0) continue;
      const last = arr[n - 1];
      const lastClose = toNum(last.close);
      latestCloseMap.set(code, lastClose);
      if (n >= 60) {
        const prev = arr[n - 60];
        const prevClose = toNum(prev.close);
        if (Number.isFinite(lastClose) && Number.isFinite(prevClose) && prevClose !== 0) {
          ret3mMap.set(code, (lastClose / prevClose) - 1);
        } else {
          ret3mMap.set(code, null);
        }
      } else {
        ret3mMap.set(code, null);
      }
    }

    const marketFilter = (req.query.market || "").toUpperCase();
    const sectorFilter = (req.query.sector || "");

    let data = ranking.map((r) => {
      // prefer name/market/sector from ranking file if present, else universe map
      const code = r.code;
      return {
        date: r.date,
        code,
        name: (r.name && r.name.trim().length ? r.name : getName(code)),
        market: ((r.market && r.market.trim().length ? r.market : (getMarket(code) || ""))).toUpperCase(),
        sector: (r.sector && r.sector.trim().length ? r.sector : (getSector(code) || null)),
        close: latestCloseMap.get(code) ?? toNum(r.close),
        ret_3m: ret3mMap.get(code) ?? null,
        pred_return_60d: r.pred_return_60d !== undefined && r.pred_return_60d !== "" ? toNum(r.pred_return_60d) : null,
        pred_return_90d: r.pred_return_90d !== undefined && r.pred_return_90d !== "" ? toNum(r.pred_return_90d) : null,

        // 🔥 예측 MDD + 리스크 감점 추가
        pred_mdd_60d: r.pred_mdd_60d !== undefined && r.pred_mdd_60d !== "" ? toNum(r.pred_mdd_60d) : null,
        pred_mdd_90d: r.pred_mdd_90d !== undefined && r.pred_mdd_90d !== "" ? toNum(r.pred_mdd_90d) : null,
        risk_penalty: r.risk_penalty !== undefined && r.risk_penalty !== "" ? toNum(r.risk_penalty) : null,

        prob_top20_60d: r.prob_top20_60d !== undefined && r.prob_top20_60d !== "" ? toNum(r.prob_top20_60d) : null,
        prob_top20_90d: r.prob_top20_90d !== undefined && r.prob_top20_90d !== "" ? toNum(r.prob_top20_90d) : null,
        // expose final_score under 'score' so existing UI column shows the composite score
        score: r.final_score !== undefined && r.final_score !== "" ? toNum(r.final_score) : null,
        final_score: r.final_score !== undefined && r.final_score !== "" ? toNum(r.final_score) : null,
        tech_score: r.tech_score !== undefined && r.tech_score !== "" ? toNum(r.tech_score) : null,
        qual_score: r.qual_score !== undefined && r.qual_score !== "" ? toNum(r.qual_score) : null,
        ret_score: r.ret_score !== undefined && r.ret_score !== "" ? toNum(r.ret_score) : null,
        prob_score: r.prob_score !== undefined && r.prob_score !== "" ? toNum(r.prob_score) : null,
        pred_score: r.pred_score !== undefined && r.pred_score !== "" ? toNum(r.pred_score) : null,

        // 🔥 시장 상태 정보 추가
        market_up: r.market_up !== undefined ? (String(r.market_up).toLowerCase() === "true") : true,
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
    // Default sort: final_score desc if available, else score desc
    data.sort((a, b) => {
      const af = Number.isFinite(+a.final_score) ? +a.final_score : (Number.isFinite(+a.score) ? +a.score : -Infinity);
      const bf = Number.isFinite(+b.final_score) ? +b.final_score : (Number.isFinite(+b.score) ? +b.score : -Infinity);
      return bf - af;
    });
    // 🔥 market_up 값을 각 row에 붙여주기
    data.forEach((row) => {
      row.market_up = marketUp;
    });

    // 🔥 응답은 예전처럼 "배열" 그대로
    return res.json(data);
  } catch (e) {
    res.status(500).json({ error: "failed to read ranking", detail: String(e) });
  }
});

app.get("/api/top20", (req, res) => {
  try {
    const rankPath = path.join(DATA_DIR, "ranking_final.csv");
    const rows = readCsv(rankPath);

    if (!rows || rows.length === 0) {
      return res.status(404).json({ error: "ranking_final.csv is empty" });
    }

    // 1) 최신 날짜 찾기
    const dates = rows
      .map((r) => r.date)
      .filter((d) => d && d.trim().length > 0);
    if (dates.length === 0) {
      return res
        .status(400)
        .json({ error: "No valid date column in ranking_final.csv" });
    }
    const latestDate = dates.sort()[dates.length - 1];

    // 2) 최신 날짜만 필터링
    let latestRows = rows.filter((r) => r.date === latestDate);
    if (latestRows.length === 0) {
      latestRows = rows; // 혹시라도 date가 안 맞으면 전체에서라도
    }

    // 3) 점수 순으로 정렬 (final_score 우선, 없으면 score)
    latestRows.sort((a, b) => {
      const af = toNum(a.final_score ?? a.score);
      const bf = toNum(b.final_score ?? b.score);
      const aa = Number.isFinite(af) ? af : -Infinity;
      const bb = Number.isFinite(bf) ? bf : -Infinity;
      return bb - aa;
    });

    // 4) 상위 N개만 추출
    const N = 20;
    const top = latestRows.slice(0, N);

    if (top.length === 0) {
      return res.status(404).json({ error: "No ranking rows for latest date" });
    }

    // 5) 시장 메타 정보 (첫 행 기준)
    const first = top[0];
    const marketMeta = {
      market_up:
        first.market_up === true ||
        String(first.market_up).toLowerCase() === "true" ||
        String(first.market_up) === "1",
      status_date: first.market_status_date || null,
      kospi_close: toNum(first.market_kospi_close),
      kospi_ma20: toNum(first.market_kospi_ma20),
      vol_5d: toNum(first.market_vol_5d),
      foreign_5d: toNum(first.market_foreign_5d),
    };

    // 6) 각 종목별 요약 텍스트 만들기
    function fmtPct(v, digits = 1) {
      const n = toNum(v);
      if (!Number.isFinite(n)) return "-";
      return (n * 100).toFixed(digits) + "%";
    }

    function fmtAbsPct(v, digits = 1) {
      const n = toNum(v);
      if (!Number.isFinite(n)) return "-";
      return (Math.abs(n) * 100).toFixed(digits) + "%";
    }

    function fmtProb(v, digits = 1) {
      const n = toNum(v);
      if (!Number.isFinite(n)) return "-";
      return (n * 100).toFixed(digits) + "%";
    }

    const items = top.map((r, idx) => {
      const code = r.code;
      const name = (r.name && r.name.trim().length ? r.name : getName(code));
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

      // 한글 요약 문장
      const summaryLines = [];

      summaryLines.push(
        `(${idx + 1}위) ${name} (${code})` +
          (sector ? ` · 섹터: ${sector}` : "") +
          (market ? ` · 시장: ${market}` : "")
      );

      if (Number.isFinite(pred60) || Number.isFinite(pred90)) {
        summaryLines.push(
          `- 예측 수익률: 60일 ${fmtPct(pred60)}, 90일 ${fmtPct(pred90)}`
        );
      }

      if (Number.isFinite(prob60) || Number.isFinite(prob90)) {
        summaryLines.push(
          `- 상위 20% 진입 확률: 60일 ${fmtProb(prob60)}, 90일 ${fmtProb(
            prob90
          )}`
        );
      }

      if (Number.isFinite(mdd60) || Number.isFinite(mdd90)) {
        summaryLines.push(
          `- 예상 최대 낙폭(MDD): 60일 약 ${fmtAbsPct(mdd60)}, 90일 약 ${fmtAbsPct(mdd90)} 하락`
        );
      }

      

      summaryLines.push(
        `- 점수 구성: 수익률 점수 ${retScore?.toFixed?.(1) ?? "-"}, 확률 점수 ${
          probScore?.toFixed?.(1) ?? "-"
        }, 리스크 감점 ${riskPenalty?.toFixed?.(1) ?? "-"}`
      );
      summaryLines.push(
        `- 최종 점수(final_score): ${
          Number.isFinite(finalScore) ? finalScore.toFixed(2) : "-"
        }`
      );

      const summary_ko = summaryLines.join("\n");

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

    return res.json({
      date: latestDate,
      market: marketMeta,
      count: items.length,
      items,
    });
  } catch (e) {
    console.error("api/top20 error", e);
    return res.status(500).json({
      error: "failed to build top20 summary",
      detail: String(e),
    });
  }
});


// === Step2: Top20 시그널 / 오늘의 액션 API ===

// GET /api/signals/top20?horizon=60d&limit=20&date=YYYY-MM-DD&only_new=1
app.get("/api/signals/top20", (req, res) => {
  try {
    const horizon = req.query.horizon === "90d" ? "90d" : "60d";
    const limit = Math.max(1, Math.min(100, Number(req.query.limit) || 20));
    const onlyNew = req.query.only_new === "1" || req.query.only_new === "true";

    const dbRows = dbRankingAll();
    const rows = dbRows || readCsv(path.join(DATA_DIR, "ranking_final.csv"));

    if (!rows || rows.length === 0) {
      return res.status(404).json({ error: "ranking data is empty" });
    }

    // 기준 날짜 (없으면 최신 날짜)
    let targetDate = (req.query.date || "").trim();
    if (!targetDate) {
      const dates = rows
        .map((r) => r.date)
        .filter((d) => d && d.trim().length > 0)
        .sort();
      targetDate = dates[dates.length - 1];
    }

    let filtered = rows.filter((r) => String(r.date || "") === targetDate);
    if (!filtered.length) {
      filtered = rows.slice(); // fallback
    }

    const sortKey = horizon === "90d" ? "pred_return_90d" : "pred_return_60d";
    const mddKey  = horizon === "90d" ? "pred_mdd_90d"   : "pred_mdd_60d";
    const probKey = horizon === "90d" ? "prob_top20_90d" : "prob_top20_60d";

    filtered.sort((a, b) => {
      const av = toNum(a[sortKey]) || 0;
      const bv = toNum(b[sortKey]) || 0;
      return bv - av;
    });

    // holdings 계산용 최신 ranking 행
    const latestByCode = new Map();
    for (const r of rows) {
      const code = (r.code || "").trim();
      if (!code) continue;
      const prev = latestByCode.get(code);
      if (!prev || String(prev.date || "") < String(r.date || "")) {
        latestByCode.set(code, r);
      }
    }

    const trades = loadTrades();
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

    return res.json({
      date: targetDate,
      horizon,
      items,
    });
  } catch (e) {
    console.error("GET /api/signals/top20 error", e);
    return res.status(500).json({ error: "internal error" });
  }
});

// GET /api/dashboard/today-actions?date=YYYY-MM-DD
app.get("/api/dashboard/today-actions", (req, res) => {
  try {
    const rankPath = path.join(DATA_DIR, "ranking_final.csv");
    const rows = readCsv(rankPath);

    if (!rows || rows.length === 0) {
      return res.status(404).json({ error: "ranking_final.csv is empty" });
    }

    // 기준 날짜 (없으면 최신)
    let targetDate = (req.query.date || "").trim();
    if (!targetDate) {
      const dates = rows
        .map((r) => r.date)
        .filter((d) => d && d.trim().length > 0)
        .sort();
      targetDate = dates[dates.length - 1];
    }

    const dayRows = rows.filter((r) => String(r.date || "") === targetDate);
    const baseRows = dayRows.length ? dayRows : rows;

    // holdings 계산용 최신 ranking 행
    const latestByCode = new Map();
    for (const r of rows) {
      const code = (r.code || "").trim();
      if (!code) continue;
      const prev = latestByCode.get(code);
      if (!prev || String(prev.date || "") < String(r.date || "")) {
        latestByCode.set(code, r);
      }
    }
    const trades = loadTrades();
    const holdings = buildHoldings(trades, latestByCode);
    const holdingCodes = new Set(holdings.map((h) => h.code));

    // 기준값
    const BUY_MIN_RET  = 0.30;
    const BUY_MIN_PROB = 0.40;
    const BUY_MIN_MDD  = -0.35;

    const ADD_MIN_RET  = 0.25;
    const ADD_MIN_PROB = 0.30;
    const ADD_MIN_MDD  = -0.40;

    const TRIM_MAX_RET = 0.05;
    const TRIM_MIN_MDD = -0.45;

    const buyCandidates  = [];
    const addCandidates  = [];
    const trimCandidates = [];

    const sorted = baseRows
      .slice()
      .sort((a, b) => (toNum(b.pred_return_60d) || 0) - (toNum(a.pred_return_60d) || 0))
      .slice(0, 50);  // 상위 50개 안에서만 검사

    for (let idx = 0; idx < sorted.length; idx++) {
      const r = sorted[idx];
      const code = (r.code || "").trim();
      if (!code) continue;

      const predRet   = toNum(r.pred_return_60d);
      const predMdd   = toNum(r.pred_mdd_60d);
      const prob      = toNum(r.prob_top20_60d);
      const finalScore = toNum(r.final_score);
      const isHolding = holdingCodes.has(code);

      // 신규 매수 후보
      if (
        !isHolding &&
        idx < 10 &&
        predRet != null && predRet >= BUY_MIN_RET &&
        prob    != null && prob    >= BUY_MIN_PROB &&
        predMdd != null && predMdd >= BUY_MIN_MDD
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
          reason: "예측 수익률 상위 + 위험도 허용 범위",
        });
      }

      // 증액 후보
      if (
        isHolding &&
        predRet != null && predRet >= ADD_MIN_RET &&
        prob    != null && prob    >= ADD_MIN_PROB &&
        predMdd != null && predMdd >= ADD_MIN_MDD
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
          reason: "보유 중이지만 모델이 여전히 강하게 추천",
        });
      }

      // 감축 / 청산 후보
      if (
        isHolding &&
        (
          (predRet != null && predRet < TRIM_MAX_RET) ||
          (predMdd != null && predMdd < TRIM_MIN_MDD)
        )
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
          reason: "예측 수익률 저조 또는 예상 MDD 과도",
        });
      }
    }

    return res.json({
      date: targetDate,
      horizon: "60d",
      buy_candidates:  buyCandidates.slice(0, 5),
      add_candidates:  addCandidates.slice(0, 5),
      trim_candidates: trimCandidates.slice(0, 5),
    });
  } catch (e) {
    console.error("GET /api/dashboard/today-actions error", e);
    return res.status(500).json({ error: "internal error" });
  }
});



// ------------------------------
//  매수/매도 입력 및 조회
// ------------------------------

// ------------------------------
//  보유 종목 단일 상세 조회
// ------------------------------

// GET /api/holding/:code
// - 해당 종목의 보유 현황 + 모델 정보 + 매매 이력 반환
app.get("/api/holding/:code", (req, res) => {
  try {
    const rawCode = (req.params.code || "").trim();
    if (!rawCode) {
      return res.status(400).json({ error: "code required" });
    }

    const code = rawCode;
    const tradesAll = loadTrades();
    const trades = tradesAll
      .filter((t) => (t.code || "").trim() === code)
      .sort((a, b) => {
        const da = String(a.date || "");
        const db = String(b.date || "");
        if (da < db) return -1;
        if (da > db) return 1;

        const ca = String(a.created_at || "");
        const cb = String(b.created_at || "");
        if (ca < cb) return -1;
        if (ca > cb) return 1;

        const ia = Number(a.trade_id);
        const ib = Number(b.trade_id);
        if (Number.isFinite(ia) && Number.isFinite(ib)) return ia - ib;
        return 0;
      });

    // 랭킹 파일에서 최신 정보 가져오기
    const ranking = dbRankingAll() || readCsv(path.join(DATA_DIR, "ranking_final.csv")) || [];
    let latestRank = null;
    for (const r of ranking) {
      const c = (r.code || "").trim();
      if (c !== code) continue;
      if (!latestRank || (latestRank.date || "") < (r.date || "")) {
        latestRank = r;
      }
    }

    // 기본 메타 정보
    const name =
      (latestRank && latestRank.name) ||
      (trades[0] && trades[0].name) ||
      getName(code) ||
      code;
    const market =
      (latestRank && latestRank.market) ||
      (trades[0] && trades[0].market) ||
      getMarket(code) ||
      null;
    const sector =
      (latestRank && latestRank.sector) ||
      (trades[0] && trades[0].sector) ||
      getSector(code) ||
      null;

    // 매매 이력이 없으면 바로 리턴
    if (!trades.length) {
      return res.json({
        code,
        name,
        market,
        sector,
        holding: null,
        latest: latestRank || null,
        trades: [],
      });
    }

    // 포지션/실현손익 계산 (평균단가 방식)
    let positionQty = 0;
    let avgCost = 0;
    let realizedPnl = 0;
    let totalBuyAmount = 0;
    let firstBuyDate = null;
    let lastTradeDate = null;

    const tradesWithRun = trades.map((t) => {
      const side = (t.side || "").toUpperCase();
      const q = Number(t.qty);
      const p = Number(t.price);
      const dateStr = String(t.date || "");

      if (!firstBuyDate && side === "BUY") firstBuyDate = dateStr;
      if (side === "BUY" && dateStr && firstBuyDate && dateStr < firstBuyDate) {
        firstBuyDate = dateStr;
      }
      if (!lastTradeDate || (dateStr && dateStr > lastTradeDate)) {
        lastTradeDate = dateStr;
      }

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
        side: side,
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

    // 현재 보유 포지션 계산
    let holding = null;
    if (positionQty > 0) {
      const currentPrice =
        latestRank && latestRank.close !== undefined && latestRank.close !== ""
          ? Number(latestRank.close)
          : null;

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
      if (
        Number.isFinite(currentValue) &&
        Number.isFinite(costBasis) &&
        costBasis !== 0
      ) {
        unrealizedPnl = currentValue - costBasis;
        unrealizedPnlPct = (currentValue / costBasis - 1) * 100;
      }

      let realizedPnlPct = null;
      if (Number.isFinite(realizedPnl) && totalBuyAmount > 0) {
        realizedPnlPct = (realizedPnl / totalBuyAmount) * 100;
      }

      // 목표가(이미 buildHoldings 로직과 동일 비율 사용)
      let targetPrice = null;
      let progressToTarget = null;
      if (Number.isFinite(avgBuyPrice) && avgBuyPrice > 0) {
        // 여기서 비율은 buildHoldings와 맞춰줘야 함 (예: 20%면 1.2)
        const TARGET_PROFIT_PCT = 0.20; // 현재 20% 기준
        targetPrice = avgBuyPrice * (1 + TARGET_PROFIT_PCT);
        if (Number.isFinite(currentPrice)) {
          const denom = targetPrice - avgBuyPrice;
          if (denom !== 0) {
            const raw = ((currentPrice - avgBuyPrice) / denom) * 100;
            progressToTarget = Math.max(0, Math.min(100, raw));
          }
        }
      }

      const finalScore =
        latestRank && latestRank.final_score !== undefined
          ? Number(latestRank.final_score)
          : latestRank && latestRank.score !== undefined
          ? Number(latestRank.score)
          : null;

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
        target_price: targetPrice,
        progress_to_target: progressToTarget,
        final_score: finalScore,
        first_buy_date: firstBuyDate,
        last_trade_date: lastTradeDate,
      };
    }

    return res.json({
      code,
      name,
      market,
      sector,
      latest: latestRank || null,
      holding,
      trades: tradesWithRun,
    });
  } catch (e) {
    console.error("api/holding error", e);
    res.status(500).json({ error: "failed to build holding detail", detail: String(e) });
  }
});


// 거래 히스토리 조회 (옵션)
app.get("/api/trades", (req, res) => {
  try {
    const trades = loadTrades();
    res.json({ count: trades.length, items: trades });
  } catch (e) {
    console.error("api/trades GET error", e);
    res.status(500).json({ error: "failed to load trades", detail: String(e) });
  }
});

app.post("/api/trades", (req, res) => {
  try {
    console.log("[POST /api/trades] body =", req.body);

    const { side, code, date, qty, price } = req.body || {};
    const s = (side || "").toUpperCase();

    if (!["BUY", "SELL"].includes(s)) {
      console.warn("  -> invalid side:", side);
      return res.status(400).json({ error: "side must be BUY or SELL" });
    }
    if (!code) {
      console.warn("  -> missing code");
      return res.status(400).json({ error: "code required" });
    }
    if (!date) {
      console.warn("  -> missing date");
      return res.status(400).json({ error: "date required" });
    }

    const q = Number(qty);
    const p = Number(price);
    if (!Number.isFinite(q) || q <= 0) {
      console.warn("  -> invalid qty:", qty);
      return res.status(400).json({ error: "qty > 0" });
    }
    if (!Number.isFinite(p) || p <= 0) {
      console.warn("  -> invalid price:", price);
      return res.status(400).json({ error: "price > 0" });
    }

    const db = ensureTradesTable();
    if (!db) {
      return res.status(500).json({ error: "DB unavailable for trades" });
    }

    const created_at = new Date().toISOString();
    const stmt = db.prepare(
      `INSERT INTO trades (date, side, code, name, market, sector, qty, price, amount, fee, memo, created_at)
       VALUES (?, ?, ?, '', '', '', ?, ?, ?, 0, '', ?)`
    );
    const info = stmt.run(date, s, code, q, p, q * p, created_at);
    const trade_id = info.lastInsertRowid;
    console.log("  -> trade saved id=" + trade_id);

    // append to CSV for redundancy
    appendTradeCsv({
      trade_id,
      date,
      side: s,
      code,
      name: "",
      market: "",
      sector: "",
      qty: q,
      price: p,
      amount: q * p,
      fee: 0,
      memo: "",
      created_at,
    });

    res.json({ success: true, trade_id });
  } catch (e) {
    console.error("[POST /api/trades] error:", e);
    res.status(500).json({ error: "failed to save", detail: String(e) });
  }
});


// ------------------------------
//  보유 종목 조회 API
// ------------------------------
app.get("/api/holdings", (req, res) => {
  try {
    const trades = loadTrades();
    if (!trades.length) {
      return res.json({ count: 0, items: [] });
    }

    const ranking = dbRankingAll() || readCsv(path.join(DATA_DIR, "ranking_final.csv")) || [];
    const latestByCode = new Map();

    for (const r of ranking) {
      const code = (r.code || "").trim();
      if (!code) continue;
      const prev = latestByCode.get(code);
      if (!prev || (prev.date || "") < (r.date || "")) {
        latestByCode.set(code, r);
      }
    }

    const holdings = buildHoldings(trades, latestByCode);

let totalValue = 0;
let totalCost = 0;
let totalRealized = 0;

for (const h of holdings) {
  if (Number.isFinite(h.current_value)) totalValue += h.current_value;
  if (Number.isFinite(h.cost_basis)) totalCost += h.cost_basis;
  if (Number.isFinite(h.realized_pnl)) totalRealized += h.realized_pnl;
}

let totalUnrealized = totalValue - totalCost;
let totalUnrealizedPct = totalCost > 0 ? (totalUnrealized / totalCost) * 100 : null;

let totalPnl = totalRealized + totalUnrealized;
let totalPnlPct = totalCost > 0 ? (totalPnl / totalCost) * 100 : null;

res.json({
  count: holdings.length,
  total_cost: totalCost,
  total_value: totalValue,

  total_realized_pnl: totalRealized,
  total_unrealized_pnl: totalUnrealized,
  total_unrealized_pnl_pct: totalUnrealizedPct,

  total_pnl: totalPnl,
  total_pnl_pct: totalPnlPct,

  items: holdings
});

  } catch (e) {
    console.error("api/holdings error", e);
    res.status(500).json({ error: "failed to build holdings", detail: String(e) });
  }
});


// 매매 이력 조회 API
app.get("/api/trades/history", (req, res) => {
  try {
    const rows = loadTrades();
    if (!rows || rows.length === 0) {
      return res.json({ items: [] });
    }

    const codeFilter = (req.query.code || "").trim();
    const q = (req.query.q || "").trim().toLowerCase(); // 코드/이름 부분검색
    const from = (req.query.from || "").trim();
    const to = (req.query.to || "").trim();

    const stateByCode = new Map();
    const items = [];
    let totalRealizedAcc = 0;

    // 안정 정렬: 날짜 -> trade_id
    rows.sort((a, b) => {
      const da = String(a.date || "");
      const db = String(b.date || "");
      if (da < db) return -1;
      if (da > db) return 1;
      const ia = Number(a.trade_id) || 0;
      const ib = Number(b.trade_id) || 0;
      return ia - ib;
    });

    for (const r of rows) {
      const date = String(r.date || "").slice(0, 10);
      const code = String(r.code || "").trim();
      if (!code) continue;

      const name = r.name || getName(code) || code;
      const inDate = (!from || date >= from) && (!to || date <= to);
      const inCode = !codeFilter || code === codeFilter;
      let include = inDate && inCode;
      if (q) {
        const lowerName = (name || "").toLowerCase();
        include = include && (code.toLowerCase().includes(q) || lowerName.includes(q));
      }
      const side = String(r.side || "").toUpperCase(); // BUY/SELL
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
          date,
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

    return res.json({ items });
  } catch (e) {
    console.error("GET /api/trades/history error", e);
    return res.status(500).json({ error: "internal error" });
  }
});




app.get("/api/debug/data-dir", (req, res) => {
  res.json({ DATA_DIR });
});


app.listen(PORT, () => console.log("Server running on port " + PORT));


