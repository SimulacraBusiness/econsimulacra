import React from "react";
import { X } from "lucide-react";
import { useSimulation } from "../context/SimulationContext";
import type { StressEntry } from "../types";

// ── Stress helpers ──────────────────────────────────────────────────────────

const STRESS_LABEL: Record<string, string> = {
  consumption:      "Consumption",
  move:             "Move",
  state_evaluation: "State Eval",
  sleep:            "Sleep",
  obs:              "Observation",
};

function stressGradient(value: number): string {
  if (value >= 70) return "linear-gradient(90deg, #ef4444, #dc2626)";
  if (value >= 40) return "linear-gradient(90deg, #f97316, #ea580c)";
  if (value >= 20) return "linear-gradient(90deg, #eab308, #ca8a04)";
  return "linear-gradient(90deg, #10b981, #059669)";
}

function stressTextColor(value: number): string {
  if (value >= 70) return "text-red-400";
  if (value >= 40) return "text-orange-400";
  if (value >= 20) return "text-yellow-400";
  return "text-emerald-400";
}

function StressBar({ label, entry }: { label: string; entry: StressEntry }) {
  const [expanded, setExpanded] = React.useState(false);
  return (
    <div className="rounded-lg overflow-hidden bg-white/3 border border-white/6">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center gap-2 px-2.5 py-1.5 hover:bg-white/4 transition-colors text-left"
      >
        <span className="w-20 text-slate-400 truncate text-xs flex-none">{label}</span>
        <div className="flex-1 h-1.5 bg-white/8 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-700"
            style={{ width: `${entry.value}%`, background: stressGradient(entry.value) }}
          />
        </div>
        <span className={`w-8 text-right text-xs font-mono font-bold flex-none ${stressTextColor(entry.value)}`}>
          {entry.value}
        </span>
        <span className="text-slate-600 text-xs flex-none">{expanded ? "▲" : "▼"}</span>
      </button>
      {expanded && entry.reason && (
        <div className="px-3 pb-2 pt-0.5">
          <p className="text-xs text-slate-500 leading-relaxed italic border-l-2 border-indigo-500/30 pl-2">
            {entry.reason}
          </p>
        </div>
      )}
    </div>
  );
}

// ── Inventory bar ────────────────────────────────────────────────────────────

function InventoryBar({ name, amount, max }: { name: string; amount: number; max: number }) {
  const pct = max > 0 ? Math.min((amount / max) * 100, 100) : 0;
  return (
    <div className="flex items-center gap-2">
      <span className="w-20 text-slate-400 truncate text-xs flex-none">{name}</span>
      <div className="flex-1 h-1.5 bg-white/8 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full"
          style={{
            width: `${pct}%`,
            background: "linear-gradient(90deg, #6366f1, #06b6d4)",
          }}
        />
      </div>
      <span className="w-10 text-right text-xs text-slate-300 flex-none font-mono">
        {typeof amount === "number" ? amount.toFixed(amount % 1 === 0 ? 0 : 1) : amount}
      </span>
    </div>
  );
}

// ── Panel section header ─────────────────────────────────────────────────────

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2 text-xs text-slate-500 font-semibold uppercase tracking-wider mb-1.5">
      <span
        className="w-1 h-3 rounded-full flex-none"
        style={{ background: "linear-gradient(180deg, #818cf8, #06b6d4)" }}
      />
      {children}
    </div>
  );
}

// ── Main component ───────────────────────────────────────────────────────────

export function AgentInspector() {
  const { agents, selectedAgentId, setSelectedAgentId, events } = useSimulation();

  if (selectedAgentId === null) {
    return (
      <div className="flex flex-col h-full">
        <div className="text-xs text-slate-500 px-3 py-2 border-b border-white/8 font-semibold uppercase tracking-wider flex items-center gap-2">
          <span
            className="w-1.5 h-1.5 rounded-full bg-slate-600"
          />
          Agent Inspector
        </div>
        <div className="flex-1 flex flex-col items-center justify-center gap-2 text-slate-600">
          <span className="text-3xl opacity-30">👤</span>
          <span className="text-xs">Click an agent on the map</span>
        </div>
      </div>
    );
  }

  const agent = agents[selectedAgentId];
  if (!agent) return null;

  const agentEvents = events.filter((e) => e.agentId === selectedAgentId).slice(0, 18);
  const inventoryEntries = Object.entries(agent.inventory).filter(([name]) => name !== "Yen");
  const maxInv = Math.max(...inventoryEntries.map(([, v]) => v), 1);
  const wealth = agent.wealth;
  const wealthFmt = wealth >= 1_000_000
    ? `¥${(wealth / 1_000_000).toFixed(2)}M`
    : `¥${Math.round(wealth).toLocaleString()}`;

  const typeLabel =
    agent.agent_type === "LLMAgent"
      ? "Household"
      : agent.agent_type === "AutoReactLLMAgent"
      ? "Retailer"
      : "Restaurant";

  const stressEntries = Object.entries(agent.stress ?? {});
  const avgStress = stressEntries.length
    ? Math.round(stressEntries.reduce((s, [, e]) => s + e.value, 0) / stressEntries.length)
    : null;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-white/8">
        <span className="text-xs text-slate-500 font-semibold uppercase tracking-wider flex items-center gap-2">
          <span
            className="w-1.5 h-1.5 rounded-full"
            style={{ background: "linear-gradient(135deg,#818cf8,#06b6d4)" }}
          />
          Agent Inspector
        </span>
        <button
          onClick={() => setSelectedAgentId(null)}
          className="text-slate-600 hover:text-slate-300 transition-colors p-0.5 rounded hover:bg-white/8"
        >
          <X size={14} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto min-h-0 space-y-4 p-3">
        {/* Identity */}
        <div>
          <div className="flex items-center gap-2 mb-1 flex-wrap">
            <span className="text-base font-bold text-slate-100">{agent.name}</span>
            <span
              className={`text-xs px-2 py-0.5 rounded-full font-medium border ${
                agent.is_household
                  ? "bg-indigo-500/15 text-indigo-300 border-indigo-500/25"
                  : "bg-orange-500/15 text-orange-300 border-orange-500/25"
              }`}
            >
              {typeLabel}
            </span>
            {agent.isSleeping && (
              <span className="text-xs px-2 py-0.5 rounded-full bg-slate-700/40 text-slate-400 border border-slate-600/30">
                😴 sleeping
              </span>
            )}
          </div>
          <p className="text-xs text-slate-600 font-mono">
            #{agent.id} · ({agent.pos[0]},{agent.pos[1]}) · follows {agent.numFollows} / {agent.numFollowers} flw
          </p>
        </div>

        {/* Wealth */}
        <div
          className="rounded-xl p-3 border border-white/8"
          style={{ background: "rgba(16,185,129,0.06)" }}
        >
          <div className="text-xs text-slate-500 mb-0.5">Wealth</div>
          <div
            className="text-2xl font-bold font-mono"
            style={{
              background: "linear-gradient(90deg, #34d399, #06b6d4)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}
          >
            {wealthFmt}
          </div>
          <div className="mt-2 h-1.5 bg-white/8 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full"
              style={{
                width: `${Math.min((wealth / 400_000) * 100, 100)}%`,
                background: "linear-gradient(90deg, #34d399, #06b6d4)",
              }}
            />
          </div>
        </div>

        {/* Stress */}
        {stressEntries.length > 0 && (
          <div>
            <SectionLabel>
              Stress
              {avgStress !== null && (
                <span className={`ml-1 font-mono ${stressTextColor(avgStress)}`}>
                  avg {avgStress}/100
                </span>
              )}
            </SectionLabel>
            <div className="space-y-1.5">
              {stressEntries.map(([type, entry]) => (
                <StressBar key={type} label={STRESS_LABEL[type] ?? type} entry={entry} />
              ))}
            </div>
          </div>
        )}

        {/* Inventory */}
        {inventoryEntries.length > 0 && (
          <div>
            <SectionLabel>Inventory</SectionLabel>
            <div className="space-y-1.5">
              {inventoryEntries.map(([name, amount]) => (
                <InventoryBar key={name} name={name} amount={amount} max={maxInv} />
              ))}
              {agent.inventory["Yen"] !== undefined && (
                <div className="flex items-center gap-2 pt-1 border-t border-white/6 mt-1">
                  <span className="w-20 text-slate-400 text-xs flex-none">Yen</span>
                  <span className="text-xs text-yellow-300 font-mono">
                    ¥{Math.round(agent.inventory["Yen"]).toLocaleString()}
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Inner Thought */}
        {agent.lastInnerThought && (
          <div>
            <SectionLabel>Inner Thought</SectionLabel>
            <div
              className="rounded-xl p-3 text-xs text-violet-200 leading-relaxed italic border border-violet-500/15"
              style={{ background: "rgba(139,92,246,0.08)" }}
            >
              💭 {agent.lastInnerThought}
            </div>
          </div>
        )}

        {/* Last Tweet */}
        {agent.lastTweet && (
          <div>
            <SectionLabel>Last Tweet</SectionLabel>
            <div
              className="rounded-xl p-3 text-xs text-sky-200 leading-relaxed border border-sky-500/15"
              style={{ background: "rgba(14,165,233,0.07)" }}
            >
              🐦 {agent.lastTweet}
            </div>
          </div>
        )}

        {/* Recent actions */}
        {agentEvents.length > 0 && (
          <div>
            <SectionLabel>Recent Actions</SectionLabel>
            <div className="space-y-1">
              {agentEvents.map((ev) => (
                <div
                  key={ev.id}
                  className="text-xs text-slate-400 flex gap-2 leading-snug py-0.5"
                >
                  <span className="text-slate-600 font-mono whitespace-nowrap flex-none">
                    {ev.time.slice(5, 16)}
                  </span>
                  <span className="min-w-0">{ev.message}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
