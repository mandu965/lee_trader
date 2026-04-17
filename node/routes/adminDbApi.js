const express = require("express");
const adminAuth = require("../adminAuth");
const db = require("../dbAdapter");

const router = express.Router();
router.use(adminAuth);

router.get("/db/tables", async (req, res) => {
  try {
    const tables = await db.listTables();
    res.json({ tables });
  } catch (e) {
    res.status(500).json({ error: "failed to list tables", detail: String(e) });
  }
});

router.get("/db/table", async (req, res) => {
  const name = req.query.name;
  if (!name) return res.status(400).json({ error: "name is required" });
  const limit = Math.min(parseInt(req.query.limit || "100", 10), 1000);
  const offset = Math.max(parseInt(req.query.offset || "0", 10), 0);
  const filter = req.query.filter || null;
  try {
    const result = await db.selectTable(name, { limit, offset, filter });
    res.json(result);
  } catch (e) {
    res.status(500).json({ error: "failed to fetch table", detail: String(e) });
  }
});

router.post("/db/query", express.json({ limit: "1mb" }), async (req, res) => {
  const sql = (req.body && req.body.sql) || "";
  const params = (req.body && req.body.params) || [];
  try {
    const result = await db.runAny(sql, params);
    res.json(result);
  } catch (e) {
    res.status(500).json({ error: "query failed", detail: String(e) });
  }
});

module.exports = router;
