import { useEffect, useRef, useCallback } from "react";
import type { AgentState } from "../types";
import { useSimulation } from "../context/SimulationContext";

const W = 520;
const H = 520;
const NODE_R = 10;
const STORE_R = 14;
const MARGIN = 18;
const EMOJI_FONT = "Apple Color Emoji,Segoe UI Emoji,Noto Color Emoji,sans-serif";

function avgStress(agent: AgentState): number {
  const vals = Object.values(agent.stress).map((s) => s.value);
  return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
}

function nodeEmoji(agent: AgentState): string {
  if (!agent.is_household)
    return agent.agent_type === "DiscountRestaurant" ? "🍕" : "🏪";
  if (agent.isSleeping) return "😴";
  const s = avgStress(agent);
  if (s <= 25) return "😊";
  if (s <= 50) return "😐";
  if (s <= 75) return "😟";
  return "😫";
}

interface Vec { x: number; y: number }

function applyForces(
  positions: Map<number, Vec>,
  edges: Array<{ source: number; target: number }>,
  agentIds: number[],
  alpha: number
) {
  const REPEL = 6500;
  const ATTRACT = 0.09;
  const CENTER = 0.02;
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
    p.x = Math.max(MARGIN, Math.min(W - MARGIN, p.x));
    p.y = Math.max(MARGIN, Math.min(H - MARGIN, p.y));
  }
}

interface Bubble { text: string; alpha: number }

function drawTweetBubble(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  text: string,
  alpha: number
) {
  const label = text.length > 28 ? text.slice(0, 27) + "…" : text;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.font = "10px sans-serif";
  const tw = ctx.measureText(label).width;
  const bw = tw + 16;
  const bh = 20;
  const bx = x - bw / 2;
  const by = y - bh - 16;

  // Rounded rect background
  ctx.fillStyle = "rgba(255,255,255,0.95)";
  ctx.strokeStyle = "rgba(99,102,241,0.7)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(bx, by, bw, bh, 5);
  ctx.fill();
  ctx.stroke();

  // Tail
  ctx.beginPath();
  ctx.moveTo(x - 5, by + bh);
  ctx.lineTo(x + 5, by + bh);
  ctx.lineTo(x, by + bh + 7);
  ctx.closePath();
  ctx.fillStyle = "rgba(255,255,255,0.95)";
  ctx.fill();
  ctx.strokeStyle = "rgba(99,102,241,0.7)";
  ctx.lineWidth = 1;
  ctx.stroke();

  // Text
  ctx.fillStyle = "#1e293b";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(label, x, by + bh / 2);
  ctx.restore();
}

export function SocialNetwork() {
  const { agents, socialEdges, selectedAgentId, setSelectedAgentId, tweetEvents } = useSimulation();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const posRef = useRef<Map<number, Vec>>(new Map());
  const rafRef = useRef<number>(0);
  const alphaRef = useRef(1.0);
  const bubblesRef = useRef<Map<number, Bubble>>(new Map());
  const seenTweetIdsRef = useRef<Set<number>>(new Set());

  const agentIds = Object.keys(agents).map(Number);

  // Register tweet bubbles
  useEffect(() => {
    for (const ev of tweetEvents) {
      if (seenTweetIdsRef.current.has(ev.id)) continue;
      seenTweetIdsRef.current.add(ev.id);
      if (ev.agentId !== undefined) {
        const text = String(ev.raw.message ?? "");
        bubblesRef.current.set(ev.agentId, { text, alpha: 1.0 });
      }
    }
  }, [tweetEvents]);

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

    // Background
    const bg = ctx.createLinearGradient(0, 0, W, H);
    bg.addColorStop(0, "#111e33");
    bg.addColorStop(1, "#0c1829");
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
      grad.addColorStop(0, "rgba(129,140,248,0.5)");
      grad.addColorStop(1, "rgba(6,182,212,0.75)");
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(tx, ty);
      ctx.strokeStyle = grad;
      ctx.lineWidth = 1.3;
      ctx.stroke();

      // Arrowhead
      const angle = Math.atan2(dy, dx);
      ctx.beginPath();
      ctx.moveTo(tx, ty);
      ctx.lineTo(tx - 7 * Math.cos(angle - 0.4), ty - 7 * Math.sin(angle - 0.4));
      ctx.lineTo(tx - 7 * Math.cos(angle + 0.4), ty - 7 * Math.sin(angle + 0.4));
      ctx.closePath();
      ctx.fillStyle = "rgba(6,182,212,0.85)";
      ctx.fill();
    }

    // Nodes — emoji icons
    for (const id of agentIds) {
      const p = posRef.current.get(id);
      if (!p) continue;
      const a = agents[id];
      if (!a) continue;
      const isSelected = id === selectedAgentId;
      const fontSize = a.is_household ? 20 : 26;

      // Selection halo
      if (isSelected) {
        ctx.save();
        ctx.shadowColor = "#fbbf24";
        ctx.shadowBlur = 20;
        ctx.beginPath();
        ctx.arc(p.x, p.y, fontSize / 2 + 7, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(251,191,36,0.85)";
        ctx.lineWidth = 2.5;
        ctx.stroke();
        ctx.restore();
      }

      // Emoji
      ctx.save();
      ctx.font = `${fontSize}px ${EMOJI_FONT}`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.globalAlpha = a.isSleeping ? 0.4 : 1.0;
      ctx.fillText(nodeEmoji(a), p.x, p.y);
      ctx.restore();
    }

    // Tweet bubbles — fade over time
    for (const [id, bubble] of bubblesRef.current) {
      const p = posRef.current.get(id);
      if (!p) continue;
      if (bubble.alpha <= 0) { bubblesRef.current.delete(id); continue; }
      drawTweetBubble(ctx, p.x, p.y, bubble.text, bubble.alpha);
      bubble.alpha -= 0.004;
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
      const r = a?.is_household ? 15 : 18;
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
      <div className="flex flex-wrap gap-x-3 gap-y-1 px-3 py-1.5 text-xs text-slate-500 border-t border-white/6">
        <span>😊 calm</span>
        <span>😐 normal</span>
        <span>😟 stressed</span>
        <span>😫 max</span>
        <span>😴 sleeping</span>
        <span className="ml-auto flex gap-3">
          <span>🏪 Retailer</span>
          <span>🍕 Restaurant</span>
        </span>
      </div>
    </div>
  );
}
