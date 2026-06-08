import { useEffect, useRef, useCallback } from "react";
import { useSimulation } from "../context/SimulationContext";

const W = 320;
const H = 320;
const NODE_R = 6;
const STORE_R = 10;

interface Vec { x: number; y: number }

function applyForces(
  positions: Map<number, Vec>,
  edges: Array<{ source: number; target: number }>,
  agentIds: number[],
  alpha: number
) {
  const REPEL = 2400;
  const ATTRACT = 0.11;
  const CENTER = 0.025;
  const cx = W / 2, cy = H / 2;

  for (let i = 0; i < agentIds.length; i++) {
    for (let j = i + 1; j < agentIds.length; j++) {
      const a = positions.get(agentIds[i])!;
      const b = positions.get(agentIds[j])!;
      const dx = a.x - b.x, dy = a.y - b.y;
      const d2 = dx * dx + dy * dy || 1;
      const f = (REPEL / d2) * alpha;
      a.x += dx * f; a.y += dy * f;
      b.x -= dx * f; b.y -= dy * f;
    }
  }

  for (const e of edges) {
    const a = positions.get(e.source), b = positions.get(e.target);
    if (!a || !b) continue;
    const dx = b.x - a.x, dy = b.y - a.y;
    a.x += dx * ATTRACT * alpha;
    a.y += dy * ATTRACT * alpha;
    b.x -= dx * ATTRACT * alpha;
    b.y -= dy * ATTRACT * alpha;
  }

  for (const id of agentIds) {
    const p = positions.get(id)!;
    p.x += (cx - p.x) * CENTER * alpha;
    p.y += (cy - p.y) * CENTER * alpha;
    p.x = Math.max(NODE_R + 4, Math.min(W - NODE_R - 4, p.x));
    p.y = Math.max(NODE_R + 4, Math.min(H - NODE_R - 4, p.y));
  }
}

export function SocialNetwork() {
  const { agents, socialEdges, selectedAgentId, setSelectedAgentId } = useSimulation();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const posRef = useRef<Map<number, Vec>>(new Map());
  const rafRef = useRef<number>(0);
  const alphaRef = useRef(1.0);

  const agentIds = Object.keys(agents).map(Number);

  useEffect(() => {
    for (const id of agentIds) {
      if (!posRef.current.has(id)) {
        const angle = (id / agentIds.length) * Math.PI * 2;
        const r = Math.min(W, H) * 0.33;
        posRef.current.set(id, {
          x: W / 2 + r * Math.cos(angle),
          y: H / 2 + r * Math.sin(angle),
        });
      }
    }
    alphaRef.current = 1.0;
  }, [agentIds.length]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, W, H);

    // Background gradient matching overall theme
    const bg = ctx.createLinearGradient(0, 0, W, H);
    bg.addColorStop(0, "#07101f");
    bg.addColorStop(1, "#050d1a");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, W, H);

    // Tick physics
    if (alphaRef.current > 0.004) {
      for (let i = 0; i < 3; i++) {
        applyForces(posRef.current, socialEdges, agentIds, alphaRef.current);
      }
      alphaRef.current *= 0.992;
    }

    // Edges (directional, with gradient)
    for (const e of socialEdges) {
      const a = posRef.current.get(e.source);
      const b = posRef.current.get(e.target);
      if (!a || !b) continue;
      const dx = b.x - a.x, dy = b.y - a.y;
      const d = Math.hypot(dx, dy) || 1;
      const tx = b.x - (dx / d) * (NODE_R + 4);
      const ty = b.y - (dy / d) * (NODE_R + 4);

      const grad = ctx.createLinearGradient(a.x, a.y, tx, ty);
      grad.addColorStop(0, "rgba(99,102,241,0.15)");
      grad.addColorStop(1, "rgba(6,182,212,0.3)");
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(tx, ty);
      ctx.strokeStyle = grad;
      ctx.lineWidth = 1;
      ctx.stroke();

      // Arrowhead
      const angle = Math.atan2(dy, dx);
      ctx.beginPath();
      ctx.moveTo(tx, ty);
      ctx.lineTo(tx - 5 * Math.cos(angle - 0.4), ty - 5 * Math.sin(angle - 0.4));
      ctx.lineTo(tx - 5 * Math.cos(angle + 0.4), ty - 5 * Math.sin(angle + 0.4));
      ctx.closePath();
      ctx.fillStyle = "rgba(6,182,212,0.4)";
      ctx.fill();
    }

    // Nodes
    for (const id of agentIds) {
      const p = posRef.current.get(id);
      if (!p) continue;
      const a = agents[id];
      if (!a) continue;
      const r = a.is_household ? NODE_R : STORE_R;
      const isSelected = id === selectedAgentId;

      // Selection halo
      if (isSelected) {
        ctx.save();
        ctx.shadowColor = "#fbbf24";
        ctx.shadowBlur = 14;
        ctx.beginPath();
        ctx.arc(p.x, p.y, r + 5, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(251,191,36,0.6)";
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.restore();
      }

      // Node body
      ctx.save();
      if (a.is_household) {
        const hue = (id * 67) % 360;
        ctx.shadowColor = `hsla(${hue},80%,60%,0.6)`;
        ctx.shadowBlur = 8;
        const grad = ctx.createRadialGradient(p.x - 1, p.y - 1, 0, p.x, p.y, r);
        grad.addColorStop(0, `hsla(${hue},80%,78%,1)`);
        grad.addColorStop(1, `hsla(${hue},65%,50%,1)`);
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx.fillStyle = grad;
        ctx.fill();
      } else {
        const isRest = a.agent_type === "DiscountRestaurant";
        ctx.shadowColor = isRest ? "#f97316" : "#22c55e";
        ctx.shadowBlur = 10;
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        const grad = ctx.createRadialGradient(p.x - 2, p.y - 2, 0, p.x, p.y, r);
        if (isRest) {
          grad.addColorStop(0, "#fb923c"); grad.addColorStop(1, "#dc2626");
        } else {
          grad.addColorStop(0, "#4ade80"); grad.addColorStop(1, "#16a34a");
        }
        ctx.fillStyle = grad;
        ctx.fill();
      }
      ctx.restore();
    }

    rafRef.current = requestAnimationFrame(draw);
  }, [agents, socialEdges, agentIds, selectedAgentId]);

  useEffect(() => {
    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [draw]);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
    const scaleX = W / rect.width;
    const scaleY = H / rect.height;
    const mx = (e.clientX - rect.left) * scaleX;
    const my = (e.clientY - rect.top) * scaleY;

    let closest: number | null = null;
    let minDist = 999;
    for (const id of agentIds) {
      const p = posRef.current.get(id);
      if (!p) continue;
      const d = Math.hypot(mx - p.x, my - p.y);
      const a = agents[id];
      const r = (a?.is_household ? NODE_R : STORE_R) + 5;
      if (d < r && d < minDist) { minDist = d; closest = id; }
    }
    setSelectedAgentId(closest);
  };

  return (
    <div className="flex flex-col h-full">
      <div className="text-xs text-slate-500 px-3 py-2 border-b border-white/8 font-semibold uppercase tracking-wider flex items-center justify-between">
        <span className="flex items-center gap-2">
          <span
            className="w-1.5 h-1.5 rounded-full"
            style={{ background: "linear-gradient(135deg,#818cf8,#06b6d4)" }}
          />
          Social Network
        </span>
        <span className="text-slate-600 font-normal normal-case tracking-normal">
          {socialEdges.length} follow edges
        </span>
      </div>
      <div className="flex-1 flex items-center justify-center min-h-0 p-2">
        <canvas
          ref={canvasRef}
          width={W}
          height={H}
          onClick={handleClick}
          className="cursor-pointer rounded-lg"
          style={{ maxWidth: "100%", maxHeight: "100%" }}
        />
      </div>
    </div>
  );
}
