(function () {
  function toNum(v) {
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }

  function resolveGrade(score, grade) {
    if (grade) {
      const g = String(grade).trim().toLowerCase();
      if (g === "high" || g === "medium" || g === "low") return g;
      if (g === "a" || g === "b") return "high";
      if (g === "c") return "medium";
      if (g === "d" || g === "e") return "low";
    }
    const n = toNum(score);
    if (n === null) return "muted";
    if (n >= 75) return "high";
    if (n >= 50) return "medium";
    return "low";
  }

  function gradeLabel(score, grade) {
    if (grade) {
      const raw = String(grade).trim().toUpperCase();
      if (["A", "B", "C", "D", "E"].includes(raw)) return raw;
      if (["HIGH", "MEDIUM", "LOW"].includes(raw)) return raw;
    }
    return resolveGrade(score, grade).toUpperCase();
  }

  function defaultCaption(score, grade) {
    const n = toNum(score);
    if (n === null) return "신뢰도 데이터가 아직 없습니다.";
    if (grade === "high") return "데이터 품질과 신호 안정성이 비교적 양호합니다.";
    if (grade === "medium") return "해석 가능하지만 보조 지표 확인이 필요합니다.";
    if (n <= 0) return "초기 축적 단계이거나 설명 데이터가 부족할 수 있습니다.";
    return "보수적 해석이 필요하며 추가 확인을 권장합니다.";
  }

  function statusText(score, grade, explain) {
    const n = toNum(score);
    const text = String(explain || "");
    if (text) {
      if (text.includes("부족")) return "설명 데이터 부족";
      if (text.includes("백테스트")) return "백테스트 축적 필요";
      if (text.includes("초기")) return "초기 해석 단계";
      if (text.includes("일관성")) return "신호 일관성 낮음";
    }
    if (n === null) return "설명 데이터 부족";
    if (n <= 0) return "초기 해석 단계";
    if (grade === "high") return "해석 안정성 양호";
    if (grade === "medium") return "보조 확인 필요";
    return "백테스트 축적 필요";
  }

  function headlineText(score, grade) {
    const n = toNum(score);
    if (n === null) return "신뢰도 정보 없음";
    if (grade === "high") return "신뢰도 양호";
    if (grade === "medium") return "중간 신뢰도";
    if (n <= 0) return "초기 해석 단계";
    return "보수적 해석 필요";
  }

  function fallbackReasons(score, grade, explain) {
    const reasons = [];
    const n = toNum(score);
    const text = String(explain || "").toLowerCase();

    if (!explain) reasons.push("설명 데이터 없음");
    if (n !== null && n <= 0) {
      reasons.push("데이터 부족");
      reasons.push("초기 백테스트 축적 단계");
    } else if (grade === "low") {
      reasons.push("신호 일관성 낮음");
    }
    if (text.includes("data") || text.includes("없") || text.includes("부족")) reasons.push("데이터 부족");
    if (text.includes("backtest") || text.includes("백테스트")) reasons.push("초기 백테스트 축적 단계");
    if (text.includes("일관성")) reasons.push("신호 일관성 낮음");

    return Array.from(new Set(reasons)).slice(0, 3);
  }

  function renderConfidence(score, grade, explain, options) {
    const opts = options || {};
    const resolvedGrade = resolveGrade(score, grade);
    const value = toNum(score);
    const mode = opts.mode || "card";
    const label = opts.label || "Confidence";
    const caption = explain || opts.caption || defaultCaption(value, resolvedGrade);
    const headline = opts.headline || headlineText(value, resolvedGrade);
    const state = opts.state || statusText(value, resolvedGrade, explain);
    const reasons = fallbackReasons(value, resolvedGrade, explain);
    const gradeText = resolvedGrade === "muted" ? "N/A" : gradeLabel(value, grade);
    const width = value === null ? 0 : Math.max(6, Math.min(value, 100));
    const scoreText = value === null ? "-" : value.toFixed(1);

    return `
      <section class="confidence-widget confidence-widget--${mode} is-${resolvedGrade}">
        <div class="confidence-widget__header">
          <span class="confidence-widget__label">${label}</span>
          <span class="confidence-widget__grade">${gradeText}</span>
        </div>
        <div class="confidence-widget__headline">${headline}</div>
        <div class="confidence-widget__scoreline">
          <div class="confidence-widget__score">${scoreText}</div>
          <div class="confidence-widget__state">${state}</div>
        </div>
        <div class="confidence-widget__meter">
          <div class="confidence-widget__meter-fill" style="width:${width}%"></div>
        </div>
        <div class="confidence-widget__caption">${caption}</div>
        ${reasons.length ? `<div class="confidence-widget__reasons">${reasons.map((item) => `<span class="confidence-widget__reason">${item}</span>`).join("")}</div>` : ""}
      </section>
    `;
  }

  const SCORE_LABELS = {
    qual_score: "품질",
    tech_score: "기술",
    ret_score: "예측수익",
    prob_score: "확률",
    pred_score: "연구용 모델",
    safety_score: "안정성",
    liquidity_score: "유동성",
    risk_penalty: "리스크"
  };

  function scoreTone(key, value) {
    const n = toNum(value);
    if (n === null) return "neutral";
    if (key === "risk_penalty") {
      if (n >= 8) return "negative";
      if (n >= 4) return "neutral";
      return "positive";
    }
    if (n >= 70) return "positive";
    if (n >= 45) return "neutral";
    return "negative";
  }

  function collectScoreItems(data) {
    return Object.keys(SCORE_LABELS).map((key) => ({
      key,
      label: SCORE_LABELS[key],
      score: toNum(data[key]),
      contribution: toNum(data[`contrib_${key.replace("_score", "")}`])
    })).filter((item) => item.score !== null || item.contribution !== null);
  }

  function buildExplainSummary(items) {
    const positives = items.filter((item) => scoreTone(item.key, item.score) === "positive").sort((a, b) => (b.score ?? -Infinity) - (a.score ?? -Infinity));
    const negatives = items.filter((item) => scoreTone(item.key, item.score) === "negative").sort((a, b) => (a.score ?? Infinity) - (b.score ?? Infinity));
    if (!positives.length && !negatives.length) return "하위 점수 설명 정보가 충분하지 않습니다.";
    const parts = [];
    if (positives.length) parts.push(`강점은 ${positives.slice(0, 2).map((item) => item.label).join(", ")}입니다.`);
    if (negatives.length) parts.push(`약점은 ${negatives.slice(0, 2).map((item) => item.label).join(", ")}입니다.`);
    return parts.join(" ");
  }

  function renderScoreExplain(data, options) {
    const opts = options || {};
    const title = opts.title || "Final Score Breakdown";
    const items = collectScoreItems(data || {});
    if (!items.length) {
      return `<section class="score-explain-widget"><div class="score-explain-widget__title">${title}</div><div class="score-explain-widget__empty">표시할 하위 점수 정보가 없습니다.</div></section>`;
    }
    const scoreMax = Math.max(...items.map((item) => Math.abs(item.score ?? 0)), 1);
    const contribMax = Math.max(...items.map((item) => Math.abs(item.contribution ?? 0)), 1);
    return `
      <section class="score-explain-widget">
        <div class="score-explain-widget__header">
          <div class="score-explain-widget__title">${title}</div>
          <div class="score-explain-widget__summary">${buildExplainSummary(items)}</div>
        </div>
        <div class="score-explain-widget__list">
          ${items.map((item) => {
            const tone = scoreTone(item.key, item.score);
            const scoreWidth = item.score === null ? 0 : Math.max(8, (Math.abs(item.score) / scoreMax) * 100);
            const contribWidth = item.contribution === null ? 0 : Math.max(8, (Math.abs(item.contribution) / contribMax) * 100);
            const contribText = item.contribution === null ? "-" : `${item.contribution >= 0 ? "+" : ""}${item.contribution.toFixed(1)}`;
            return `
              <article class="score-explain-row is-${tone}">
                <div class="score-explain-row__head">
                  <div class="score-explain-row__label">${item.label}</div>
                  <div class="score-explain-row__meta">
                    <span class="score-explain-row__score">${item.score === null ? "-" : item.score.toFixed(1)}</span>
                    <span class="score-explain-row__contrib">기여 ${contribText}</span>
                  </div>
                </div>
                <div class="score-explain-row__bars">
                  <div class="score-explain-bar"><div class="score-explain-bar__fill is-score is-${tone}" style="width:${scoreWidth}%"></div></div>
                  <div class="score-explain-bar score-explain-bar--contrib"><div class="score-explain-bar__fill is-contrib is-${tone}" style="width:${contribWidth}%"></div></div>
                </div>
              </article>
            `;
          }).join("")}
        </div>
      </section>
    `;
  }

  window.ConfidenceUI = { toNum, resolveGrade, render: renderConfidence };
  window.ScoreExplainUI = { render: renderScoreExplain, collectItems: collectScoreItems };
})();
