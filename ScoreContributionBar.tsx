import React from "react";

type ContributionItem = {
  key: string;
  label: string;
  percent: number; // 예: 45.3, -12.7
};

interface ScoreContributionBarProps {
  items: ContributionItem[];
  className?: string;
}

/**
 * final_score를 구성하는 항목별 기여도(%)를
 * 가로 스택 바 + 작은 레전드로 보여주는 컴포넌트.
 *
 * - percent > 0 : 양수 기여 (수익률/확률/퀄리티/기술 등)
 * - percent < 0 : 음수 기여 (리스크/시장 페널티 등)
 *
 * 사용 예시:
 *   <ScoreContributionBar
 *     items={[
 *       { key: "return",  label: "수익률", percent: stock.contrib_return },
 *       { key: "prob",    label: "확률",   percent: stock.contrib_prob },
 *       { key: "qual",    label: "퀄리티", percent: stock.contrib_qual },
 *       { key: "tech",    label: "기술",   percent: stock.contrib_tech },
 *       { key: "risk",    label: "리스크", percent: stock.contrib_risk },
 *       { key: "market",  label: "시장",   percent: stock.contrib_market },
 *     ]}
 *   />
 */
const ScoreContributionBar: React.FC<ScoreContributionBarProps> = ({
  items,
  className = "",
}) => {
  if (!items || items.length === 0) {
    return null;
  }

  // 전체 절대값 합 기준으로 width 계산 (양수/음수 모두 비중만 보기 위함)
  const totalAbs = items.reduce(
    (sum, item) => sum + Math.abs(item.percent || 0),
    0
  );

  if (totalAbs === 0) {
    return null;
  }

  return (
    <div className={`flex flex-col gap-1 ${className}`}>
      {/* 가로 스택 바 */}
      <div className="flex h-2 w-full overflow-hidden rounded-full bg-slate-800">
        {items.map((item) => {
          const widthPercent = (Math.abs(item.percent) / totalAbs) * 100;
          if (widthPercent <= 0) return null;

          const isNegative = item.percent < 0;

          const barClass = isNegative
            ? "bg-red-500/80 hover:bg-red-400"
            : "bg-emerald-400/80 hover:bg-emerald-300";

          const title = `${item.label}: ${item.percent.toFixed(1)}%`;

          return (
            <div
              key={item.key}
              className={`h-full ${barClass} transition-colors`}
              style={{ width: `${widthPercent}%` }}
              title={title}
            />
          );
        })}
      </div>

      {/* 레전드 + 숫자 */}
      <div className="flex flex-wrap gap-2 text-[10px] text-slate-300">
        {items.map((item) => {
          if (!Number.isFinite(item.percent)) return null;

          const isNegative = item.percent < 0;
          const dotClass = isNegative ? "bg-red-500" : "bg-emerald-400";
          const textClass = isNegative ? "text-red-300" : "text-emerald-200";

          return (
            <div key={item.key} className="flex items-center gap-1">
              <span
                className={`inline-block h-2 w-2 rounded-full ${dotClass}`}
              />
              <span>{item.label}</span>
              <span className={textClass}>
                {item.percent >= 0 ? "+" : ""}
                {item.percent.toFixed(1)}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ScoreContributionBar;
