export function formatPercent(value: number | undefined, decimals = 1): string {
  if (value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(decimals)}%`;
}

export function formatNumber(value: number | undefined, decimals = 3): string {
  if (value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(decimals);
}

export function formatInt(value: number | undefined): string {
  if (value === undefined || Number.isNaN(value)) return "—";
  return value.toLocaleString();
}

export function formatCompactNumber(value: number | undefined): string {
  if (value === undefined || Number.isNaN(value)) return "—";
  return new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

export function formatBytes(bytes: number | undefined): string {
  if (bytes === undefined || Number.isNaN(bytes)) return "—";
  if (bytes === 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** exponent;
  return `${value.toFixed(value >= 100 || exponent === 0 ? 0 : 1)} ${units[exponent]}`;
}

/** Friendly label for a project/client id, e.g. "The_MongoDB_Engineering" -> "The MongoDB Engineering". */
export function formatProjectName(id: string): string {
  if (id === "global") return "Global (all clients)";
  return id.replace(/_/g, " ");
}
