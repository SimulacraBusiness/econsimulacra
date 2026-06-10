import { useCallback, useEffect, useRef } from "react";
import * as PIXI from "pixi.js";
import type { AgentState } from "../types";
import { useSimulation } from "../context/SimulationContext";

const GRID = 10;
const CELL = 52;
const PAD = 16;
const CANVAS_SIZE = GRID * CELL + PAD * 2;
const LERP = 0.18;

function avgStress(agent: AgentState): number {
  const vals = Object.values(agent.stress).map((s) => s.value);
  return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
}

function agentEmoji(agent: AgentState): string {
  if (!agent.is_household)
    return agent.agent_type === "DiscountRestaurant" ? "🍕" : "🏪";
  if (agent.isSleeping) return "😴";
  const s = avgStress(agent);
  if (s <= 25) return "😊";
  if (s <= 50) return "😐";
  if (s <= 75) return "😟";
  return "😫";
}

function cellXY(pos: [number, number]) {
  return {
    x: PAD + pos[0] * CELL + CELL / 2,
    y: PAD + (GRID - 1 - pos[1]) * CELL + CELL / 2,
  };
}

const EMOJI_FONT =
  "Apple Color Emoji,Segoe UI Emoji,Noto Color Emoji,sans-serif";

export function GridMapPixi() {
  const { agents, selectedAgentId, setSelectedAgentId, metadata, orderReactions } =
    useSimulation();

  const mountRef = useRef<HTMLDivElement>(null);
  const appRef = useRef<PIXI.Application | null>(null);
  const agentLayerRef = useRef<PIXI.Container | null>(null);
  const ringLayerRef = useRef<PIXI.Graphics | null>(null);
  const labelLayerRef = useRef<PIXI.Container | null>(null);
  const popupLayerRef = useRef<PIXI.Container | null>(null);
  const spritesRef = useRef<Map<number, PIXI.Text>>(new Map());
  const tweensRef = useRef<Map<number, { tx: number; ty: number }>>(new Map());
  const seenEventsRef = useRef<Set<number>>(new Set());

  // Initialize PixiJS once on mount
  useEffect(() => {
    if (!mountRef.current) return;

    const app = new PIXI.Application({
      width: CANVAS_SIZE,
      height: CANVAS_SIZE,
      backgroundAlpha: 0,
      antialias: true,
      resolution: Math.min(window.devicePixelRatio || 1, 2),
      autoDensity: true,
    });
    mountRef.current.appendChild(app.view as HTMLCanvasElement);
    (app.view as HTMLCanvasElement).style.cssText =
      "border-radius:8px;max-width:100%;max-height:100%;display:block;";
    appRef.current = app;

    // Background
    const bg = new PIXI.Graphics();
    bg.beginFill(0x07101f);
    bg.drawRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    bg.endFill();
    app.stage.addChild(bg);

    // Grid lines
    const gridGfx = new PIXI.Graphics();
    gridGfx.lineStyle(1, 0x6366f1, 0.15);
    for (let i = 0; i <= GRID; i++) {
      const gx = PAD + i * CELL;
      const gy = PAD + i * CELL;
      gridGfx.moveTo(gx, PAD).lineTo(gx, PAD + GRID * CELL);
      gridGfx.moveTo(PAD, gy).lineTo(PAD + GRID * CELL, gy);
    }
    app.stage.addChild(gridGfx);

    // Coordinate labels
    const coordStyle: Partial<PIXI.ITextStyle> = {
      fontSize: 8,
      fill: 0x6366f1,
      fontFamily: "monospace",
    };
    for (let i = 0; i < GRID; i++) {
      const xLabel = new PIXI.Text(String(i), coordStyle);
      xLabel.anchor.set(0.5, 1);
      xLabel.x = PAD + i * CELL + CELL / 2;
      xLabel.y = PAD - 2;
      app.stage.addChild(xLabel);

      const yLabel = new PIXI.Text(String(GRID - 1 - i), coordStyle);
      yLabel.anchor.set(1, 0.5);
      yLabel.x = PAD - 3;
      yLabel.y = PAD + i * CELL + CELL / 2;
      app.stage.addChild(yLabel);
    }

    // Ordered layers (z-depth)
    const ring = new PIXI.Graphics();
    app.stage.addChild(ring);
    ringLayerRef.current = ring;

    const agentLayer = new PIXI.Container();
    app.stage.addChild(agentLayer);
    agentLayerRef.current = agentLayer;

    const labelLayer = new PIXI.Container();
    app.stage.addChild(labelLayer);
    labelLayerRef.current = labelLayer;

    const popupLayer = new PIXI.Container();
    app.stage.addChild(popupLayer);
    popupLayerRef.current = popupLayer;

    // Ticker: lerp movements + float/fade popups
    app.ticker.add(() => {
      for (const [id, tween] of tweensRef.current) {
        const sprite = spritesRef.current.get(id);
        if (!sprite) { tweensRef.current.delete(id); continue; }
        sprite.x += (tween.tx - sprite.x) * LERP;
        sprite.y += (tween.ty - sprite.y) * LERP;
        if (Math.abs(tween.tx - sprite.x) < 0.3 && Math.abs(tween.ty - sprite.y) < 0.3) {
          sprite.x = tween.tx;
          sprite.y = tween.ty;
          tweensRef.current.delete(id);
        }
      }

      if (popupLayerRef.current) {
        for (let i = popupLayerRef.current.children.length - 1; i >= 0; i--) {
          const p = popupLayerRef.current.children[i] as PIXI.Container;
          p.y -= 1.2;
          p.alpha -= 0.022;
          if (p.alpha <= 0) popupLayerRef.current.removeChildAt(i);
        }
      }
    });

    return () => {
      app.destroy(true, { children: true, texture: true });
      appRef.current = null;
      spritesRef.current.clear();
      tweensRef.current.clear();
    };
  }, []);

  // Sync agent sprites whenever agents state changes
  useEffect(() => {
    const layer = agentLayerRef.current;
    if (!layer) return;
    const sprites = spritesRef.current;
    const tweens = tweensRef.current;

    for (const agent of Object.values(agents)) {
      const { x, y } = cellXY(agent.pos);
      const emoji = agentEmoji(agent);
      const alpha = agent.isSleeping ? 0.4 : 1.0;
      const fontSize = agent.is_household ? 20 : 24;

      if (!sprites.has(agent.id)) {
        const t = new PIXI.Text(emoji, { fontSize, fontFamily: EMOJI_FONT });
        t.anchor.set(0.5);
        t.x = x;
        t.y = y;
        t.alpha = alpha;
        sprites.set(agent.id, t);
        layer.addChild(t);
      } else {
        const t = sprites.get(agent.id)!;
        if (t.text !== emoji) t.text = emoji;
        t.alpha = alpha;
        const cur = tweens.get(agent.id);
        if (!cur || cur.tx !== x || cur.ty !== y) {
          tweens.set(agent.id, { tx: x, ty: y });
        }
      }
    }

    // Remove sprites for agents no longer present
    for (const id of [...sprites.keys()]) {
      if (!agents[id]) {
        const s = sprites.get(id)!;
        layer.removeChild(s);
        s.destroy();
        sprites.delete(id);
        tweens.delete(id);
      }
    }
  }, [agents]);

  // Selection ring + name label
  useEffect(() => {
    const ring = ringLayerRef.current;
    const labelLayer = labelLayerRef.current;
    if (!ring || !labelLayer) return;

    ring.clear();
    labelLayer.removeChildren();

    if (selectedAgentId !== null && agents[selectedAgentId]) {
      const sprite = spritesRef.current.get(selectedAgentId);
      const px = sprite?.x ?? cellXY(agents[selectedAgentId].pos).x;
      const py = sprite?.y ?? cellXY(agents[selectedAgentId].pos).y;

      ring.lineStyle(2.5, 0xfbbf24, 0.9);
      ring.drawCircle(px, py, 20);
      ring.lineStyle(7, 0xfbbf24, 0.12);
      ring.drawCircle(px, py, 25);

      // Agent name tooltip above sprite
      const name = agents[selectedAgentId].name;
      const nameText = new PIXI.Text(name, {
        fontSize: 10,
        fill: 0xfbbf24,
        fontFamily: "sans-serif",
        fontWeight: "bold",
      });
      nameText.anchor.set(0.5, 1);
      nameText.x = px;
      nameText.y = py - 28;

      const tw = nameText.width + 14;
      const th = nameText.height + 8;
      const labelBg = new PIXI.Graphics();
      labelBg.beginFill(0x060b16, 0.9);
      labelBg.lineStyle(1, 0xfbbf24, 0.5);
      labelBg.drawRoundedRect(px - tw / 2, py - 28 - th + 2, tw, th, 4);
      labelBg.endFill();
      labelLayer.addChild(labelBg);
      labelLayer.addChild(nameText);
    }
  }, [agents, selectedAgentId]);

  // order_reaction popup: aggregate by (agentId, item_name) per batch
  useEffect(() => {
    const popupLayer = popupLayerRef.current;
    if (!popupLayer) return;

    // Collect newly arrived events
    const fresh: typeof orderReactions = [];
    for (const ev of orderReactions) {
      if (seenEventsRef.current.has(ev.id)) continue;
      seenEventsRef.current.add(ev.id);
      if (ev.agentId !== undefined && agents[ev.agentId]) fresh.push(ev);
    }
    if (fresh.length === 0) return;

    // Group by (agentId, item_name) and sum accept_amount
    const groups = new Map<string, { ev: (typeof fresh)[0]; total: number }>();
    for (const ev of fresh) {
      const item = String(ev.raw.item_name ?? "?");
      const key = `${ev.agentId}:${item}`;
      if (groups.has(key)) {
        groups.get(key)!.total += (ev.raw.accept_amount as number) || 0;
      } else {
        groups.set(key, { ev, total: (ev.raw.accept_amount as number) || 0 });
      }
    }

    // One popup per group
    for (const { ev, total } of groups.values()) {
      const item = String(ev.raw.item_name ?? "?");
      const label = `${item} x${total}`;

      const sprite = spritesRef.current.get(ev.agentId!);
      const base = cellXY(agents[ev.agentId!].pos);
      const px = sprite?.x ?? base.x;
      const py = (sprite?.y ?? base.y) - 26;

      const bubble = new PIXI.Container();
      const labelText = new PIXI.Text(label, {
        fontSize: 11,
        fill: 0xffffff,
        fontFamily: "sans-serif",
      });
      labelText.anchor.set(0.5, 0.5);

      const bw = labelText.width + 16;
      const bh = labelText.height + 8;
      const bgRect = new PIXI.Graphics();
      bgRect.beginFill(0x0a1628, 0.93);
      bgRect.lineStyle(1, 0x94a3b8, 0.3);
      bgRect.drawRoundedRect(-bw / 2, -bh / 2, bw, bh, 8);
      bgRect.endFill();

      bubble.addChild(bgRect);
      bubble.addChild(labelText);
      bubble.x = px;
      bubble.y = py;
      popupLayer.addChild(bubble);
    }
  }, [orderReactions, agents]);

  // Click → select nearest agent
  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      const canvas = appRef.current?.view as HTMLCanvasElement | null;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const sx = CANVAS_SIZE / rect.width;
      const sy = CANVAS_SIZE / rect.height;
      const mx = (e.clientX - rect.left) * sx;
      const my = (e.clientY - rect.top) * sy;

      let closest: AgentState | null = null;
      let minD = 30;
      for (const a of Object.values(agents)) {
        const sp = spritesRef.current.get(a.id);
        const px = sp?.x ?? cellXY(a.pos).x;
        const py = sp?.y ?? cellXY(a.pos).y;
        const d = Math.hypot(mx - px, my - py);
        if (d < minD) { minD = d; closest = a; }
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
      <div
        className="flex-1 flex items-center justify-center p-3 min-h-0 overflow-hidden"
        onClick={handleClick}
        style={{ cursor: "pointer" }}
      >
        <div ref={mountRef} style={{ display: "contents" }} />
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-1 px-3 py-1.5 text-xs text-slate-500 border-t border-white/6">
        <span>😊 calm</span>
        <span>😐 normal</span>
        <span>😟 stressed</span>
        <span>😫 max stress</span>
        <span>😴 sleeping</span>
        <span className="ml-auto flex gap-3">
          <span>🏪 Retailer</span>
          <span>🍕 Restaurant</span>
        </span>
      </div>
    </div>
  );
}
