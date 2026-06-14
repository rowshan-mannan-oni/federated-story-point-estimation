import { AlertTriangle, Loader2, type LucideIcon } from "lucide-react";

interface StatusScreenProps {
  icon?: LucideIcon;
  title: string;
  description?: string;
  spinning?: boolean;
}

/** Full-bleed centered state for loading / error / empty-data conditions. */
export function StatusScreen({ icon: Icon = AlertTriangle, title, description, spinning }: StatusScreenProps) {
  return (
    <div className="flex flex-1 items-center justify-center p-10">
      <div className="flex max-w-md flex-col items-center gap-3 text-center">
        <div
          className="flex h-12 w-12 items-center justify-center rounded-2xl"
          style={{ background: "var(--color-surface-2)" }}
        >
          {spinning ? (
            <Loader2 className="h-6 w-6 animate-spin" style={{ color: "var(--color-fedavg)" }} />
          ) : (
            <Icon className="h-6 w-6" style={{ color: "var(--color-text-muted)" }} />
          )}
        </div>
        <h2 className="text-base font-semibold" style={{ color: "var(--color-text-primary)" }}>
          {title}
        </h2>
        {description && (
          <p className="text-sm leading-relaxed" style={{ color: "var(--color-text-secondary)" }}>
            {description}
          </p>
        )}
      </div>
    </div>
  );
}
