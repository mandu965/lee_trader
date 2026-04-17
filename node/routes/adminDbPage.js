const express = require("express");
const path = require("path");
const router = express.Router();

// Admin 페이지(토큰 입력 화면)는 누구나 열 수 있게 열어둔다.
// 실제 데이터 조회 API는 adminDbApi에서 adminAuth로 보호된다.
router.get("/admin/db", (req, res) => {
  res.sendFile(path.join(__dirname, "..", "public", "admin_db.html"));
});

module.exports = router;
