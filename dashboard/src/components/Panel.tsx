import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

interface PanelProps {
  title: string;
  icon?: LucideIcon;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}

export function Panel({ title, icon: Icon, actions, children, className, bodyClassName }: PanelProps) {
  return (
    <div className={`panel flex flex-col ${className ?? ""}`}>
      <div className="panel-header">
        <div className="flex items-center gap-2">
          {Icon && <Icon className="h-4 w-4" style={{ color: "var(--color-text-secondary)" }} strokeWidth={2} />}
          <h3 className="text-sm font-semibold" style={{ color: "var(--color-text-primary)" }}>
            {title}
          </h3>
        </div>
        {actions}
      </div>
      <div className={`p-5 ${bodyClassName ?? ""}`}>{children}</div>
    </div>
  );
}
