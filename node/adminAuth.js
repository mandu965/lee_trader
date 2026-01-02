// adminAuth middleware: protects admin endpoints.
// Checks x-admin-token header or ?token= query. If ADMIN_TOKEN is unset, allow all.
module.exports = function adminAuth(req, res, next) {
  const expected = process.env.ADMIN_TOKEN;
  if (!expected) return next();

  const token = req.headers["x-admin-token"] || req.query.token || req.headers["authorization"];
  if (token && token === expected) {
    return next();
  }
  return res.status(401).json({ error: "unauthorized" });
};
