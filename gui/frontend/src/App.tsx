import { useState } from "react";
import { SimulationProvider, useSimulation } from "./context/SimulationContext";
import { Toolbar } from "./components/Toolbar";
import { GridMapPixi } from "./components/GridMapPixi";
import { EventFeed } from "./components/EventFeed";
import { AgentInspector } from "./components/AgentInspector";
import { MacroDashboard } from "./components/MacroDashboard";
import { SocialNetwork } from "./components/SocialNetwork";
import { UploadScreen } from "./components/UploadScreen";

type Tab = "map" | "network" | "dashboard";

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`px-4 py-2 text-xs font-semibold rounded-lg transition-all duration-200 ${
        active
          ? "bg-gradient-to-r from-indigo-500/30 to-cyan-500/20 text-indigo-200 border border-indigo-500/30 shadow-lg shadow-indigo-900/20"
          : "text-slate-500 hover:text-slate-300 hover:bg-white/5"
      }`}
    >
      {children}
    </button>
  );
}

function SimulationView() {
  const [tab, setTab] = useState<Tab>("map");

  return (
    <>
      {/* Toolbar */}
      <div className="flex-none">
        <Toolbar />
      </div>

      {/* Body */}
      <div className="flex flex-1 min-h-0 gap-2 p-2 overflow-hidden">
        {/* Left: tabs + main view */}
        <div className="flex flex-col flex-1 min-w-0 min-h-0 gap-2">
          <div className="flex gap-1.5 px-1 flex-none">
            <TabButton active={tab === "map"} onClick={() => setTab("map")}>🗺 Grid Map</TabButton>
            <TabButton active={tab === "network"} onClick={() => setTab("network")}>🕸 Social Network</TabButton>
            <TabButton active={tab === "dashboard"} onClick={() => setTab("dashboard")}>📊 Macro Dashboard</TabButton>
          </div>
          <div className="flex-1 min-h-0 rounded-xl glass overflow-hidden">
            {tab === "map" && <GridMapPixi />}
            {tab === "network" && <SocialNetwork />}
            {tab === "dashboard" && <MacroDashboard />}
          </div>
        </div>

        {/* Right sidebar */}
        <div className="flex flex-col w-72 flex-none min-h-0 gap-2">
          <div className="flex-1 min-h-0 rounded-xl glass overflow-hidden">
            <AgentInspector />
          </div>
          <div className="flex-1 min-h-0 rounded-xl glass overflow-hidden">
            <EventFeed />
          </div>
        </div>
      </div>
    </>
  );
}

function AppInner() {
  const { status } = useSimulation();
  const showSim = status === "connected" || status === "done";

  return (
    <div className="flex flex-col h-screen overflow-hidden font-sans text-slate-100">
      {showSim ? (
        <SimulationView />
      ) : (
        <UploadScreen />
      )}
    </div>
  );
}

export default function App() {
  return (
    <SimulationProvider>
      <AppInner />
    </SimulationProvider>
  );
}
