import { NavLink } from "react-router-dom";
import { LayoutDashboard, LineChart, Boxes, Grid3x3, Radio, GitCompareArrows, Activity, Network } from "lucide-react";

const NAV_ITEMS = [
  { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/rounds", label: "Round History", icon: LineChart },
  { to: "/projects", label: "Per-Project", icon: Boxes },
  { to: "/conditions", label: "Condition Compare", icon: GitCompareArrows },
  { to: "/confusion", label: "Confusion Matrix", icon: Grid3x3 },
  { to: "/communication", label: "Communication Cost", icon: Radio },
  { to: "/architecture", label: "Architecture", icon: Network },
];

export function Sidebar() {
  return (
    <aside
      className="hidden md:flex w-64 flex-col gap-1 border-r p-4 shrink-0"
      style={{ borderColor: "var(--color-border-soft)", background: "var(--color-surface)" }}
    >
      <div className="flex items-center gap-2.5 px-2 py-3 mb-2">
        <div
          className="flex h-9 w-9 items-center justify-center rounded-xl"
          style={{
            background: "linear-gradient(135deg, var(--color-fedprox), var(--color-fedavg))",
          }}
        >
          <Activity className="h-5 w-5 text-white" strokeWidth={2.5} />
        </div>
        <div>
          <p className="text-sm font-bold leading-tight" style={{ color: "var(--color-text-primary)" }}>
            FedScope
          </p>
          <p className="text-xs leading-tight" style={{ color: "var(--color-text-muted)" }}>
            FL Results Explorer
          </p>
        </div>
      </div>

      <nav className="flex flex-col gap-1">
        {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
          <NavLink key={to} to={to} end={end} className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
            <Icon className="h-4.5 w-4.5" strokeWidth={2} />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="mt-auto pt-4 px-2">
        <p className="text-xs leading-relaxed" style={{ color: "var(--color-text-muted)" }}>
          Reads result artifacts from{" "}
          <code className="text-[11px]" style={{ color: "var(--color-text-secondary)" }}>
            public/data/runs/
          </code>
          . Sync a real run with{" "}
          <code className="text-[11px]" style={{ color: "var(--color-text-secondary)" }}>
            npm run sync-data
          </code>
          .
        </p>
      </div>
    </aside>
  );
}
