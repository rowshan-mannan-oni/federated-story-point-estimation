import { BrowserRouter, Routes, Route } from "react-router-dom";
import { RunProvider } from "./context/RunContext";
import { AppLayout } from "./components/layout/AppLayout";
import { Overview } from "./pages/Overview";
import { RoundHistory } from "./pages/RoundHistory";
import { ProjectBreakdown } from "./pages/ProjectBreakdown";
import { ConditionCompare } from "./pages/ConditionCompare";
import { ConfusionMatrixPage } from "./pages/ConfusionMatrixPage";
import { CommunicationCostPage } from "./pages/CommunicationCostPage";
import { ArchitecturePage } from "./pages/ArchitecturePage";

export default function App() {
  return (
    <BrowserRouter>
      <RunProvider>
        <Routes>
          <Route element={<AppLayout />}>
            <Route index element={<Overview />} />
            <Route path="rounds" element={<RoundHistory />} />
            <Route path="projects" element={<ProjectBreakdown />} />
            <Route path="conditions" element={<ConditionCompare />} />
            <Route path="confusion" element={<ConfusionMatrixPage />} />
            <Route path="communication" element={<CommunicationCostPage />} />
            <Route path="architecture" element={<ArchitecturePage />} />
          </Route>
        </Routes>
      </RunProvider>
    </BrowserRouter>
  );
}
