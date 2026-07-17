import { Outlet } from "react-router-dom";
import { DatabaseZap } from "lucide-react";
import { Sidebar } from "./Sidebar";
import { Topbar } from "./Topbar";
import { StatusScreen } from "../StatusScreen";
import { useRun } from "../../context/RunContext";

export function AppLayout() {
  const { runData, loading, error, runs } = useRun();

  return (
    <div className="flex min-h-screen" style={{ background: "var(--color-canvas)" }}>
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <Topbar />
        <main className="flex flex-1 flex-col overflow-y-auto">
          {error && runs.length === 0 ? (
            <StatusScreen
              icon={DatabaseZap}
              title="No result data found"
              description={error}
            />
          ) : loading && !runData ? (
            <StatusScreen title="Loading results…" spinning />
          ) : runData ? (
            <Outlet />
          ) : (
            <StatusScreen
              icon={DatabaseZap}
              title="No results loaded"
              description='Click "Load folder" in the top bar and pick a results directory (one containing config.json and *_per_project.json) to explore it here.'
            />
          )}
        </main>
      </div>
    </div>
  );
}
