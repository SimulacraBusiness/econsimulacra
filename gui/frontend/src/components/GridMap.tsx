import { useCallback, useEffect, useRef } from "react";
import type { AgentState } from "../types";
import { useSimulation } from "../context/SimulationContext";

const GRID = 10;
const CELL = 52;
const PAD = 16;
const CANVAS_SIZE = GRID * CELL + PAD * 2;

const AGENT_R = 9;
const STORE_R = 13;

function hueForId(id: number) {
  return (id * 67) % 360;
}

function drawAgent(
  ctx: CanvasRenderingContext2D,
  a: AgentState,
  isSelected: boolean
) {
  const cx = PAD + a.pos[0] * CELL + CELL / 2;
  const cy = PAD + (GRID - 1 - a.pos[1]) * CELL + CELL / 2;

  const r = a.is_household ? AGENT_R : STORE_R;
  const alpha = a.isSleeping ? 0.3 : 1.0;

  // Selection ring (outer glow)
  if (isSelected) {
    ctx.save();
    ctx.shadowColor = "#fbbf24";
    ctx.shadowBlur = 14;
    ctx.beginPath();
    ctx.arc(cx, cy, r + 7, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(251,191,36,0.7)";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.restore();
  }

  ctx.save();
  ctx.globalAlpha = alpha;

  if (a.is_household) {
    const hue = hueForId(a.id);
    // Glow
    ctx.shadowColor = `hsla(${hue},80%,60%,0.5)`;
    ctx.shadowBlur = 8;
    // Circle
    const grad = ctx.createRadialGradient(cx - 2, cy - 2, 1, cx, cy, r);
    grad.addColorStop(0, `hsla(${hue},85%,78%,1)`);
    grad.addColorStop(1, `hsla(${hue},70%,48%,1)`);
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();
    if (a.isSleeping) {
      ctx.strokeStyle = "rgba(148,163,184,0.4)";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  } else {
    // Store: rounded square with glow
    const isRest = a.agent_type === "DiscountRestaurant";
    const color = isRest ? "#f97316" : "#22c55e";
    ctx.shadowColor = color;
    ctx.shadowBlur = 12;
    const s = STORE_R * 1.9;
    ctx.beginPath();
    ctx.roundRect(cx - s / 2, cy - s / 2, s, s, 6);
    const grad = ctx.createLinearGradient(cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2);
    if (isRest) {
      grad.addColorStop(0, "#fb923c");
      grad.addColorStop(1, "#dc2626");
    } else {
      grad.addColorStop(0, "#4ade80");
      grad.addColorStop(1, "#16a34a");
    }
    ctx.fillStyle = grad;
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.3)";
    ctx.lineWidth = 1;
    ctx.stroke();

    // Icon
    ctx.shadowBlur = 0;
    ctx.globalAlpha = 1;
    ctx.font = "bold 11px sans-serif";
    ctx.fillStyle = "#fff";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(isRest ? "🍕" : "🏪", cx, cy);
  }

  ctx.restore();
}

export function GridMap() {
  const { agents, selectedAgentId, setSelectedAgentId, metadata } = useSimulation();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    // Navy background
    const bg = ctx.createLinearGradient(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    bg.addColorStop(0, "#07101f");
    bg.addColorStop(1, "#050d1a");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    // Grid lines — subtle indigo tint
    ctx.strokeStyle = "rgba(99,102,241,0.12)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= GRID; i++) {
      const x = PAD + i * CELL;
      const y = PAD + i * CELL;
      ctx.beginPath(); ctx.moveTo(x, PAD); ctx.lineTo(x, PAD + GRID * CELL); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(PAD, y); ctx.lineTo(PAD + GRID * CELL, y); ctx.stroke();
    }

    // Coord labels
    ctx.fillStyle = "rgba(99,102,241,0.25)";
    ctx.font = "8px monospace";
    ctx.textAlign = "center";
    for (let i = 0; i < GRID; i++) {
      ctx.fillText(String(i), PAD + i * CELL + CELL / 2, PAD - 4);
      ctx.textAlign = "right";
      ctx.fillText(String(GRID - 1 - i), PAD - 4, PAD + i * CELL + CELL / 2 + 3);
      ctx.textAlign = "center";
    }

    // Draw stores first, then households
    const agentList = Object.values(agents);
    const stores = agentList.filter((a) => !a.is_household);
    const households = agentList.filter((a) => a.is_household);
    for (const a of [...stores, ...households]) {
      drawAgent(ctx, a, a.id === selectedAgentId);
    }

    // Selected name tooltip
    if (selectedAgentId !== null && agents[selectedAgentId]) {
      const a = agents[selectedAgentId];
      const cx = PAD + a.pos[0] * CELL + CELL / 2;
      const cy = PAD + (GRID - 1 - a.pos[1]) * CELL + CELL / 2;
      const label = a.name;
      ctx.font = "bold 10px sans-serif";
      const tw = ctx.measureText(label).width;
      const bx = cx - tw / 2 - 6;
      const by = cy - AGENT_R - 22;
      ctx.fillStyle = "rgba(6,11,22,0.88)";
      ctx.roundRect(bx, by, tw + 12, 16, 4);
      ctx.fill();
      ctx.strokeStyle = "rgba(251,191,36,0.5)";
      ctx.lineWidth = 1;
      ctx.roundRect(bx, by, tw + 12, 16, 4);
      ctx.stroke();
      ctx.fillStyle = "#fbbf24";
      ctx.textAlign = "center";
      ctx.fillText(label, cx, by + 12);
    }
  }, [agents, selectedAgentId]);

  useEffect(() => {
    draw();
  }, [draw]);

  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
      const scaleX = CANVAS_SIZE / rect.width;
      const scaleY = CANVAS_SIZE / rect.height;
      const mx = (e.clientX - rect.left) * scaleX;
      const my = (e.clientY - rect.top) * scaleY;

      let closest: AgentState | null = null;
      let minDist = 999;
      for (const a of Object.values(agents)) {
        const cx = PAD + a.pos[0] * CELL + CELL / 2;
        const cy = PAD + (GRID - 1 - a.pos[1]) * CELL + CELL / 2;
        const d = Math.hypot(mx - cx, my - cy);
        const r = a.is_household ? AGENT_R : STORE_R;
        if (d < r + 7 && d < minDist) {
          minDist = d;
          closest = a;
        }
      }
      setSelectedAgentId(closest ? closest.id : null);
    },
    [agents, setSelectedAgentId]
  );

  const gridW = metadata?.grid_size[0] ?? GRID;
  const gridH = metadata?.grid_size[1] ?? GRID;

  return (
    <div className="flex flex-col h-full">
      <div className="text-xs text-slate-500 px-3 py-2 border-b border-white/8 font-semibold uppercase tracking-wider flex items-center gap-2">
        <span
          className="w-1.5 h-1.5 rounded-full"
          style={{ background: "linear-gradient(135deg,#818cf8,#06b6d4)" }}
        />
        Grid Map
        <span className="text-slate-600 font-normal normal-case tracking-normal">
          {gridW}×{gridH}
        </span>
      </div>
      <div className="flex-1 flex items-center justify-center p-3 min-h-0 overflow-hidden">
        <canvas
          ref={canvasRef}
          width={CANVAS_SIZE}
          height={CANVAS_SIZE}
          onClick={handleClick}
          className="cursor-pointer rounded-lg"
          style={{ maxWidth: "100%", maxHeight: "100%", imageRendering: "pixelated" }}
        />
      </div>
      <div className="flex gap-4 px-3 py-1.5 text-xs text-slate-500 border-t border-white/6">
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-2.5 h-2.5 rounded-full bg-indigo-400/70" />
          Household
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-2.5 h-2.5 rounded bg-green-500/70" />
          Retailer
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block w-2.5 h-2.5 rounded bg-orange-500/70" />
          Restaurant
        </span>
      </div>
    </div>
  );
}
