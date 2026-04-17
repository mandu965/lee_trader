const crypto = require("crypto");

const COOKIE_NAME = "lt_operator_auth";
const ONE_WEEK_SECONDS = 60 * 60 * 24 * 7;

function getPassword() {
  return String(process.env.OPERATOR_PASSWORD || "").trim();
}

function isEnabled() {
  return Boolean(getPassword());
}

function cookieSecret() {
  return String(process.env.OPERATOR_AUTH_SECRET || `lt-operator:${getPassword()}`);
}

function expectedToken() {
  const password = getPassword();
  if (!password) return "";
  return crypto.createHmac("sha256", cookieSecret()).update(password).digest("hex");
}

function parseCookies(req) {
  const raw = String(req.headers.cookie || "");
  const out = {};
  raw.split(";").forEach((part) => {
    const idx = part.indexOf("=");
    if (idx < 0) return;
    const key = part.slice(0, idx).trim();
    const value = part.slice(idx + 1).trim();
    if (!key) return;
    out[key] = decodeURIComponent(value);
  });
  return out;
}

function requestToken(req) {
  const cookies = parseCookies(req);
  const authHeader = String(req.headers.authorization || "").trim();
  const bearer = authHeader.toLowerCase().startsWith("bearer ") ? authHeader.slice(7).trim() : "";
  return (
    cookies[COOKIE_NAME] ||
    String(req.headers["x-operator-token"] || "").trim() ||
    String(req.headers["x-admin-token"] || "").trim() ||
    String(req.query.operator_token || "").trim() ||
    bearer
  );
}

function isAuthorized(req) {
  if (!isEnabled()) return true;
  const token = requestToken(req);
  return Boolean(token && token === expectedToken());
}

function setAuthCookie(res) {
  const secure = String(process.env.NODE_ENV || "").toLowerCase() === "production";
  const pieces = [
    `${COOKIE_NAME}=${encodeURIComponent(expectedToken())}`,
    "Path=/",
    `Max-Age=${ONE_WEEK_SECONDS}`,
    "HttpOnly",
    "SameSite=Lax",
  ];
  if (secure) pieces.push("Secure");
  res.setHeader("Set-Cookie", pieces.join("; "));
}

function clearAuthCookie(res) {
  const secure = String(process.env.NODE_ENV || "").toLowerCase() === "production";
  const pieces = [
    `${COOKIE_NAME}=`,
    "Path=/",
    "Max-Age=0",
    "HttpOnly",
    "SameSite=Lax",
  ];
  if (secure) pieces.push("Secure");
  res.setHeader("Set-Cookie", pieces.join("; "));
}

function apiGuard(req, res, next) {
  if (isAuthorized(req)) return next();
  return res.status(401).json({ error: "operator_auth_required" });
}

function pageGuard(req, res, next) {
  if (isAuthorized(req)) return next();
  const nextPath = encodeURIComponent(req.originalUrl || req.url || "/");
  return res.redirect(`/operator-login?next=${nextPath}`);
}

function login(req, res) {
  if (!isEnabled()) return res.json({ success: true, enabled: false, authorized: true });
  const provided = typeof req.body?.password === "string" ? req.body.password : "";
  if (!provided || provided !== getPassword()) {
    clearAuthCookie(res);
    return res.status(401).json({ error: "invalid_password" });
  }
  setAuthCookie(res);
  return res.json({ success: true, enabled: true, authorized: true });
}

function logout(req, res) {
  clearAuthCookie(res);
  return res.json({ success: true, enabled: isEnabled(), authorized: false });
}

function status(req, res) {
  return res.json({
    enabled: isEnabled(),
    authorized: isAuthorized(req),
  });
}

module.exports = {
  COOKIE_NAME,
  isEnabled,
  isAuthorized,
  apiGuard,
  pageGuard,
  login,
  logout,
  status,
};
