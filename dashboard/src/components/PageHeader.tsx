import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

interface PageHeaderProps {
  icon: LucideIcon;
  title: string;
  description?: string;
  actions?: ReactNode;
}

export function PageHeader({ icon: Icon, title, description, actions }: PageHeaderProps) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-4 px-4 sm:px-6 pt-6">
      <div className="flex items-start gap-3">
        <div
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl"
          style={{ background: "var(--color-surface-2)", color: "var(--color-fedavg)" }}
        >
          <Icon className="h-5 w-5" strokeWidth={2} />
        </div>
        <div>
          <h2 className="text-lg font-semibold" style={{ color: "var(--color-text-primary)" }}>
            {title}
          </h2>
          {description && (
            <p className="mt-0.5 text-sm" style={{ color: "var(--color-text-secondary)" }}>
              {description}
            </p>
          )}
        </div>
      </div>
      {actions}
    </div>
  );
}
