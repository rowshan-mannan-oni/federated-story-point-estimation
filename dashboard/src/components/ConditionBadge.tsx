import type { ConditionKey } from "../types/results";
import { CONDITION_META } from "../lib/conditions";

export function ConditionBadge({ condition, short }: { condition: ConditionKey; short?: boolean }) {
  const meta = CONDITION_META[condition];
  return (
    <span
      className="badge"
      style={{ background: `color-mix(in srgb, ${meta.color} 16%, transparent)`, color: meta.color }}
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: meta.color }} />
      {short ? meta.shortLabel : meta.label}
    </span>
  );
}
