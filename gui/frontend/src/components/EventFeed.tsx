import { useEffect, useRef } from "react";
import { useSimulation } from "../context/SimulationContext";

const TYPE_META: Record<string, { icon: string; color: string; bg: string }> = {
  tweet:           { icon: "🐦", color: "text-sky-300",    bg: "bg-sky-500/10" },
  inner_thought:   { icon: "💭", color: "text-violet-300", bg: "bg-violet-500/10" },
  move:            { icon: "🚶", color: "text-slate-400",  bg: "" },
  order:           { icon: "🛒", color: "text-yellow-300", bg: "bg-yellow-500/8" },
  order_reaction:  { icon: "✅", color: "text-emerald-300",bg: "bg-emerald-500/8" },
  consumption:     { icon: "🍽", color: "text-orange-300", bg: "bg-orange-500/8" },
  sleep_start:     { icon: "😴", color: "text-indigo-300", bg: "" },
  sleep_end:       { icon: "☀️", color: "text-amber-300",  bg: "" },
  follow:          { icon: "➕", color: "text-teal-300",   bg: "" },
  unfollow:        { icon: "➖", color: "text-rose-400",   bg: "" },
  change_price:    { icon: "💴", color: "text-emerald-300",bg: "bg-emerald-500/8" },
  proposal:        { icon: "🤝", color: "text-pink-300",   bg: "bg-pink-500/8" },
  proposal_reaction:{ icon: "📋",color: "text-rose-300",   bg: "" },
};

export function EventFeed({ hideHeader }: { hideHeader?: boolean }) {
  const { events, setSelectedAgentId, selectedAgentId } = useSimulation();
  const bottomRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const atBottomRef = useRef(true);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const onScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = container;
      atBottomRef.current = scrollHeight - scrollTop - clientHeight < 40;
    };
    container.addEventListener("scroll", onScroll);
    return () => container.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    if (atBottomRef.current) {
      bottomRef.current?.scrollIntoView({ behavior: "instant" });
    }
  }, [events]);

  return (
    <div className="flex flex-col h-full">
      {!hideHeader && (
        <div className="text-xs text-slate-500 px-3 py-2 border-b border-white/8 font-semibold uppercase tracking-wider flex items-center justify-between">
          <span className="flex items-center gap-2">
            <span
              className="w-1.5 h-1.5 rounded-full animate-pulse"
              style={{ background: "linear-gradient(135deg,#818cf8,#06b6d4)" }}
            />
            Event Feed
          </span>
          <span className="text-slate-600 font-normal normal-case tracking-normal">
            {events.length} recent
          </span>
        </div>
      )}
      <div
        ref={containerRef}
        className="flex-1 overflow-y-auto text-xs min-h-0"
      >
        {[...events].reverse().map((ev) => {
          const meta = TYPE_META[ev.type] ?? { icon: "•", color: "text-slate-400", bg: "" };
          const isHighlight = ev.agentId === selectedAgentId;
          return (
            <div
              key={ev.id}
              onClick={() => ev.agentId !== undefined && setSelectedAgentId(ev.agentId)}
              className={`px-3 py-1.5 flex gap-2 cursor-pointer transition-all duration-150 border-b border-white/4 ${
                isHighlight
                  ? "bg-indigo-500/12 border-l-2 border-l-indigo-400/60"
                  : `hover:bg-white/4 ${meta.bg}`
              }`}
            >
              <span className="flex-none mt-0.5 text-sm">{meta.icon}</span>
              <div className="min-w-0 flex-1">
                <p className={`${meta.color} break-words leading-snug`}>
                  {ev.message}
                </p>
                <p className="text-slate-600 mt-0.5 font-mono">
                  step {ev.timeStep} · {ev.time.slice(5, 16)}
                </p>
              </div>
            </div>
          );
        })}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
