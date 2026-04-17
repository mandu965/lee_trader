const { Pool } = require("pg");
const path = require("path");
let Database; // lazy load better-sqlite3 to avoid requirement when not used

const DATA_DIR = path.join(__dirname, "data");
const DATABASE_URL = process.env.DATABASE_URL;
const DB_DIALECT = (process.env.DB_DIALECT || "").toLowerCase();
const SQLITE_PATH = process.env.SQLITE_PATH || path.join(DATA_DIR, "lee_trader.db");

const useSqlite = DB_DIALECT === "sqlite" || (!DATABASE_URL && process.env.DB_DIALECT !== "postgres");

let pgPool = null;
let sqlite = null;

function ensurePg() {
  if (pgPool) return pgPool;
  if (!DATABASE_URL) {
    throw new Error("DATABASE_URL not set and DB_DIALECT is not sqlite");
  }
  pgPool = new Pool({
    connectionString: DATABASE_URL,
    max: 10,
    idleTimeoutMillis: 0,
    connectionTimeoutMillis: 5000,
  });
  return pgPool;
}

function ensureSqlite() {
  if (sqlite) return sqlite;
  try {
    Database = require("better-sqlite3");
  } catch (e) {
    throw new Error("better-sqlite3 not installed; set DB_DIALECT=postgres or add dependency");
  }
  sqlite = new Database(SQLITE_PATH, { fileMustExist: true });
  return sqlite;
}

function quoteIdentPg(name) {
  if (!/^[a-zA-Z0-9_]+$/.test(name)) throw new Error("Invalid identifier");
  return `"${name}"`;
}
function quoteIdentSqlite(name) {
  if (!/^[a-zA-Z0-9_]+$/.test(name)) throw new Error("Invalid identifier");
  return `"${name}"`;
}

async function listTables() {
  if (useSqlite) {
    const db = ensureSqlite();
    const stmt = db.prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name");
    return stmt.all().map((r) => r.name);
  }
  const pool = ensurePg();
  const { rows } = await pool.query(
    `
    SELECT table_schema || '.' || table_name AS name
    FROM information_schema.tables
    WHERE table_schema NOT IN ('pg_catalog','information_schema')
    ORDER BY table_schema, table_name
    `
  );
  return rows.map((r) => r.name);
}

async function getColumns(table) {
  if (useSqlite) {
    const db = ensureSqlite();
    const stmt = db.prepare(`PRAGMA table_info(${quoteIdentSqlite(table)})`);
    return stmt.all().map((r) => r.name);
  }
  const pool = ensurePg();
  const [schema, name] = table.includes(".") ? table.split(".") : ["public", table];
  const { rows } = await pool.query(
    `
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = $1 AND table_name = $2
    ORDER BY ordinal_position
    `,
    [schema, name]
  );
  return rows.map((r) => r.column_name);
}

async function selectTable(table, { limit = 100, offset = 0, filter } = {}) {
  const cols = await getColumns(table);
  if (!cols.length) {
    return { columns: [], rows: [] };
  }

  if (useSqlite) {
    const db = ensureSqlite();
    const whereParts = [];
    const params = [];
    if (filter) {
      const like = `%${filter}%`;
      whereParts.push(
        cols.map((c) => `${quoteIdentSqlite(c)} LIKE ?`).join(" OR ")
      );
      params.push(like);
    }
    params.push(limit);
    params.push(offset);
    const where = whereParts.length ? `WHERE ${whereParts.join(" AND ")}` : "";
    const sql = `SELECT * FROM ${quoteIdentSqlite(table)} ${where} LIMIT ? OFFSET ?`;
    const rows = db.prepare(sql).all(...params);
    return { columns: cols, rows };
  }

  // Postgres
  const pool = ensurePg();
  const identifiers = table.includes(".")
    ? table.split(".").map(quoteIdentPg).join(".")
    : quoteIdentPg(table);
  const params = [];
  let where = "";
  if (filter) {
    params.push(`%${filter}%`);
    const likeExpr = cols.map((c) => `${quoteIdentPg(c)}::text ILIKE $1`).join(" OR ");
    where = `WHERE ${likeExpr}`;
  }
  params.push(limit);
  params.push(offset);
  const limitIdx = params.length - 1;
  const offsetIdx = params.length;
  const sql = `SELECT * FROM ${identifiers} ${where} LIMIT $${limitIdx} OFFSET $${offsetIdx}`;
  const { rows } = await pool.query(sql, params);
  return { columns: cols, rows };
}

async function runSelect(sql, params = []) {
  const trimmed = (sql || "").trim().toLowerCase();
  if (!trimmed.startsWith("select")) {
    throw new Error("Only SELECT queries are allowed");
  }
  if (useSqlite) {
    const db = ensureSqlite();
    const stmt = db.prepare(sql);
    return stmt.all(...params);
  }
  const pool = ensurePg();
  const { rows } = await pool.query(sql, params);
  return rows;
}

async function runAny(sql, params = []) {
  const trimmed = (sql || "").trim().toLowerCase();
  if (!trimmed) throw new Error("SQL is empty");

  if (useSqlite) {
    const db = ensureSqlite();
    const stmt = db.prepare(sql);
    if (trimmed.startsWith("select")) {
      const rows = stmt.all(...params);
      return { rows, rowCount: rows.length };
    }
    const result = stmt.run(...params);
    return { rows: [], rowCount: result.changes ?? 0 };
  }

  const pool = ensurePg();
  const result = await pool.query(sql, params);
  return { rows: result.rows || [], rowCount: result.rowCount ?? 0 };
}

module.exports = {
  useSqlite,
  listTables,
  getColumns,
  selectTable,
  runSelect,
  runAny,
};
