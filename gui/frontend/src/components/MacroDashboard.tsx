import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
} from "recharts";
import { useSimulation } from "../context/SimulationContext";

const ITEM_COLORS: Record<string, string> = {
  Rice:  "#86efac",
  Pizza: "#fb923c",
  Soda:  "#38bdf8",
  Soup:  "#c4b5fd",
  Bread: "#fde68a",
  Salad: "#6ee7b7",
};

const CHART_COMMON = {
  style: { fontSize: 10 },
  stroke: "#1e293b",
};

const TOOLTIP_STYLE = {
  contentStyle: {
    background: "rgba(10,15,30,0.95)",
    border: "1px solid rgba(255,255,255,0.1)",
    borderRadius: 8,
    fontSize: 11,
    backdropFilter: "blur(12px)",
  },
  labelStyle: { color: "#94a3b8" },
};

function KpiCard({
  label,
  value,
  sub,
  gradient,
}: {
  label: string;
  value: string | number;
  sub: string;
  gradient: string;
}) {
  return (
    <div
      className="rounded-xl p-3 border border-white/8 flex flex-col gap-1"
      style={{ background: "rgba(255,255,255,0.03)" }}
    >
      <div className="text-xs text-slate-500">{label}</div>
      <div
        className="text-xl font-bold font-mono"
        style={{ background: gradient, WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}
      >
        {value}
      </div>
      <div className="text-xs text-slate-600">{sub}</div>
    </div>
  );
}

function ChartSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div
      className="rounded-xl p-3 border border-white/8"
      style={{ background: "rgba(255,255,255,0.025)" }}
    >
      <div className="text-xs text-slate-500 font-semibold uppercase tracking-wider mb-3 flex items-center gap-2">
        <span
          className="w-1 h-3 rounded-full flex-none"
          style={{ background: "linear-gradient(180deg,#818cf8,#06b6d4)" }}
        />
        {title}
      </div>
      {children}
    </div>
  );
}

const tickFmt = (v: number) =>
  v >= 1_000_000 ? `${(v / 1_000_000).toFixed(1)}M`
  : v >= 1000 ? `${(v / 1000).toFixed(0)}k`
  : String(v);

export function MacroDashboard() {
  const { macroHistory, prices, agents, metadata } = useSimulation();

  const items = metadata ? Object.keys(metadata.items).filter((n) => n !== "Yen") : [];
  const households = Object.values(agents).filter((a) => a.is_household);
  const numSleeping = households.filter((a) => a.isSleeping).length;
  const numActive = households.length - numSleeping;

  const priceHistory = macroHistory.map((d) => ({
    step: d.step,
    ...items.reduce((acc, item) => {
      acc[item] = d.prices[item] ?? null;
      return acc;
    }, {} as Record<string, number | null>),
  }));

  const lastWealth = macroHistory.length > 0
    ? tickFmt(macroHistory[macroHistory.length - 1].avgWealth)
    : "—";

  return (
    <div className="flex flex-col h-full overflow-y-auto p-3 gap-3">
      <div className="text-xs text-slate-500 font-semibold uppercase tracking-wider flex items-center gap-2">
        <span
          className="w-1.5 h-1.5 rounded-full"
          style={{ background: "linear-gradient(135deg,#818cf8,#06b6d4)" }}
        />
        Macro Dashboard
      </div>

      {/* KPI row */}
      <div className="grid grid-cols-3 gap-2">
        <KpiCard
          label="Active Agents"
          value={`${numActive} / ${households.length}`}
          sub={`${numSleeping} sleeping`}
          gradient="linear-gradient(90deg,#818cf8,#06b6d4)"
        />
        <KpiCard
          label="Avg Wealth"
          value={lastWealth}
          sub="households"
          gradient="linear-gradient(90deg,#34d399,#06b6d4)"
        />
        <KpiCard
          label="Item Types"
          value={items.length}
          sub="tracked"
          gradient="linear-gradient(90deg,#c4b5fd,#818cf8)"
        />
      </div>

      {/* Current prices */}
      <ChartSection title="Current Prices">
        <div className="grid grid-cols-3 gap-x-4 gap-y-1.5">
          {items.map((item) => (
            <div key={item} className="flex justify-between items-center text-xs">
              <span className="text-slate-400">{item}</span>
              <span
                className="font-mono font-semibold"
                style={{ color: ITEM_COLORS[item] ?? "#cbd5e1" }}
              >
                ¥{(prices[item] ?? 0).toLocaleString()}
              </span>
            </div>
          ))}
        </div>
      </ChartSection>

      {/* Average Wealth */}
      {macroHistory.length > 1 && (
        <ChartSection title="Average Household Wealth">
          <ResponsiveContainer width="100%" height={120}>
            <LineChart data={macroHistory} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="step" tick={{ fill: "#475569", fontSize: 9 }} tickLine={false} axisLine={false} />
              <YAxis tickFormatter={tickFmt} tick={{ fill: "#475569", fontSize: 9 }} tickLine={false} axisLine={false} width={38} />
              <Tooltip {...TOOLTIP_STYLE} formatter={(v: number) => [`¥${v.toLocaleString()}`, "Avg Wealth"]} />
              <Line
                type="monotone"
                dataKey="avgWealth"
                stroke="url(#wealthGrad)"
                strokeWidth={2}
                dot={false}
              />
              <defs>
                <linearGradient id="wealthGrad" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#34d399" />
                  <stop offset="100%" stopColor="#06b6d4" />
                </linearGradient>
              </defs>
            </LineChart>
          </ResponsiveContainer>
        </ChartSection>
      )}

      {/* Price history */}
      {priceHistory.length > 1 && (
        <ChartSection title="Item Prices Over Time">
          <ResponsiveContainer width="100%" height={150}>
            <LineChart data={priceHistory} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="step" tick={{ fill: "#475569", fontSize: 9 }} tickLine={false} axisLine={false} />
              <YAxis tickFormatter={tickFmt} tick={{ fill: "#475569", fontSize: 9 }} tickLine={false} axisLine={false} width={38} />
              <Tooltip {...TOOLTIP_STYLE} formatter={(v: number, name: string) => [`¥${v?.toLocaleString() ?? "—"}`, name]} />
              <Legend wrapperStyle={{ fontSize: 9, color: "#64748b" }} />
              {items.map((item) => (
                <Line
                  key={item}
                  type="monotone"
                  dataKey={item}
                  stroke={ITEM_COLORS[item] ?? "#94a3b8"}
                  strokeWidth={1.5}
                  dot={false}
                  connectNulls
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </ChartSection>
      )}

      {/* Activity per step */}
      {macroHistory.length > 1 && (
        <ChartSection title="Activity per Step (last 40)">
          <ResponsiveContainer width="100%" height={100}>
            <BarChart data={macroHistory.slice(-40)} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
              <XAxis dataKey="step" tick={{ fill: "#475569", fontSize: 9 }} tickLine={false} axisLine={false} />
              <YAxis tick={{ fill: "#475569", fontSize: 9 }} tickLine={false} axisLine={false} width={24} />
              <Tooltip {...TOOLTIP_STYLE} />
              <defs>
                <linearGradient id="orderGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#f59e0b" stopOpacity={0.9} />
                  <stop offset="100%" stopColor="#f59e0b" stopOpacity={0.3} />
                </linearGradient>
                <linearGradient id="tweetGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#38bdf8" stopOpacity={0.9} />
                  <stop offset="100%" stopColor="#38bdf8" stopOpacity={0.3} />
                </linearGradient>
              </defs>
              <Bar dataKey="orderCount" name="Orders" fill="url(#orderGrad)" maxBarSize={10} radius={[2,2,0,0]} />
              <Bar dataKey="tweetCount" name="Tweets" fill="url(#tweetGrad)" maxBarSize={10} radius={[2,2,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartSection>
      )}
    </div>
  );
}
