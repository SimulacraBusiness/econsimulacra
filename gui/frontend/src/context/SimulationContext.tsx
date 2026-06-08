import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";
import type {
  AgentState,
  ConnectionStatus,
  EventEntry,
  MacroDataPoint,
  Metadata,
  SocialEdge,
} from "../types";

const MAX_EVENTS = 80;
let eventSeq = 0;

interface SimCtx {
  metadata: Metadata | null;
  agents: Record<number, AgentState>;
  prices: Record<string, number>;
  events: EventEntry[];
  socialEdges: SocialEdge[];
  macroHistory: MacroDataPoint[];
  currentStep: number;
  totalSteps: number;
  isPlaying: boolean;
  speed: number;
  status: ConnectionStatus;
  selectedAgentId: number | null;
  hasDefault: boolean;
  setSelectedAgentId: (id: number | null) => void;
  play: () => void;
  pause: () => void;
  setSpeed: (s: number) => void;
  seek: (step: number) => void;
  upload: (file: File) => Promise<string>;
  connect: (sessionId: string | null) => Promise<void>;
  reset: () => void;
}

const Ctx = createContext<SimCtx | null>(null);

export function useSimulation() {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useSimulation must be used inside SimulationProvider");
  return ctx;
}

function agentLabel(agents: Record<number, AgentState>, id: number) {
  return agents[id]?.name ?? `Agent#${id}`;
}

function buildEventMessage(
  r: Record<string, unknown>,
  agents: Record<number, AgentState>
): string {
  const t = r.type as string;
  const aid = r.agent_id as number | undefined;
  const name = aid !== undefined ? agentLabel(agents, aid) : "";
  switch (t) {
    case "tweet":
      return `${name}: "${r.message}"`;
    case "inner_thought":
      return `${name} 💭 ${r.inner_thought}`;
    case "move":
      return `${name} moved (${(r.old_pos as number[]).join(",")}) → (${(r.new_pos as number[]).join(",")})`;
    case "order":
      return `${name} ordered ${r.item_amount}× ${r.item_name} from ${agentLabel(agents, r.counterparty_id as number)}`;
    case "order_reaction":
      return `${agentLabel(agents, r.agent_id as number)} accepted ${r.accept_amount}× ${r.item_name} @ ¥${r.price}`;
    case "consumption":
      return `${name} consumed ${r.item_amount}× ${r.item_name}`;
    case "sleep_start":
      return `${name} went to sleep until ${r.until}`;
    case "sleep_end":
      return `${name} woke up`;
    case "follow":
      return `${name} followed ${agentLabel(agents, r.target_agent_id as number)}`;
    case "unfollow":
      return `${name} unfollowed ${agentLabel(agents, r.target_agent_id as number)}`;
    case "change_price":
      return `${name} changed ${r.item_name}: ¥${r.old_price} → ¥${r.new_price}`;
    case "proposal":
      return `${agentLabel(agents, r.proposer_agent_id as number)} proposed ${r.give_item_amount}× ${r.give_item_name} ↔ ${r.get_item_amount}× ${r.get_item_name}`;
    default:
      return t;
  }
}

const VISIBLE_TYPES = new Set([
  "tweet", "inner_thought", "move", "order", "order_reaction",
  "consumption", "sleep_start", "sleep_end", "follow", "unfollow",
  "change_price", "proposal", "proposal_reaction",
]);

export function SimulationProvider({ children }: { children: React.ReactNode }) {
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [agents, setAgents] = useState<Record<number, AgentState>>({});
  const [prices, setPrices] = useState<Record<string, number>>({});
  const [events, setEvents] = useState<EventEntry[]>([]);
  const [socialEdges, setSocialEdges] = useState<SocialEdge[]>([]);
  const [macroHistory, setMacroHistory] = useState<MacroDataPoint[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [totalSteps, setTotalSteps] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeedState] = useState(3);
  const [status, setStatus] = useState<ConnectionStatus>("idle");
  const [selectedAgentId, setSelectedAgentId] = useState<number | null>(null);
  const [hasDefault, setHasDefault] = useState(false);

  // Check if server has a default log
  useEffect(() => {
    fetch("/api/has_default")
      .then((r) => r.json())
      .then((d) => setHasDefault(d.has_default))
      .catch(() => setHasDefault(false));
  }, []);

  const wsRef = useRef<WebSocket | null>(null);
  // snapshot refs for use inside WS message handler
  const agentsRef = useRef(agents);
  agentsRef.current = agents;
  const pricesRef = useRef(prices);
  pricesRef.current = prices;

  const send = useCallback((msg: object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  const play = useCallback(() => {
    setIsPlaying(true);
    send({ action: "resume" });
  }, [send]);

  const pause = useCallback(() => {
    setIsPlaying(false);
    send({ action: "pause" });
  }, [send]);

  const setSpeed = useCallback(
    (s: number) => {
      setSpeedState(s);
      send({ action: "set_speed", speed: s });
    },
    [send]
  );

  const seek = useCallback(
    (step: number) => {
      setCurrentStep(step);
      send({ action: "seek", step });
    },
    [send]
  );

  // Upload a log file → returns session_id
  const upload = useCallback(async (file: File): Promise<string> => {
    const form = new FormData();
    form.append("file", file);
    const resp = await fetch("/api/upload", { method: "POST", body: form });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: "Upload failed" }));
      throw new Error(err.detail ?? "Upload failed");
    }
    const data = await resp.json();
    return data.session_id as string;
  }, []);

  // Reset to upload screen
  const reset = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    setStatus("idle");
    setMetadata(null);
    setAgents({});
    setPrices({});
    setEvents([]);
    setSocialEdges([]);
    setMacroHistory([]);
    setCurrentStep(0);
    setTotalSteps(0);
    setIsPlaying(false);
    setSelectedAgentId(null);
  }, []);

  const connect = useCallback(async (sessionId: string | null) => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    setStatus("connecting");

    // Fetch metadata
    const qs = sessionId ? `?session_id=${sessionId}` : "";
    const resp = await fetch(`/api/metadata${qs}`);
    if (!resp.ok) throw new Error("Failed to fetch metadata");
    const meta: Metadata = await resp.json();
    setMetadata(meta);
    setTotalSteps(meta.total_steps);
    setPrices(meta.items);

    // Initialize agent states
    const initAgents: Record<number, AgentState> = {};
    for (const a of meta.agents) {
      initAgents[a.id] = {
        ...a,
        isSleeping: false,
        numFollows: 0,
        numFollowers: 0,
        lastInnerThought: null,
        lastTweet: null,
        stress: {},
      };
    }
    setAgents(initAgents);
    agentsRef.current = initAgents;

    const wsProto = window.location.protocol === "https:" ? "wss" : "ws";
    const wsUrl = `${wsProto}://${window.location.host}/ws/replay${qs}`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("connected");
      setIsPlaying(true);
      ws.send(JSON.stringify({ action: "set_speed", speed: 3 }));
    };

    ws.onerror = () => setStatus("error");

    ws.onmessage = (ev) => {
      const payload = JSON.parse(ev.data) as {
        time_step: number;
        records: Record<string, unknown>[];
        done?: boolean;
      };

      if (payload.done) {
        setStatus("done");
        setIsPlaying(false);
        return;
      }

      const { time_step: step, records } = payload;
      if (step < 0) return; // init records handled via metadata

      setCurrentStep(step);

      // Accumulators for this batch
      const agentPatch: Record<number, Partial<AgentState>> = {};
      const newEvents: EventEntry[] = [];
      const pricePatch: Record<string, number> = {};
      const edgePatch = { add: [] as SocialEdge[], remove: [] as SocialEdge[] };
      let orderCount = 0;
      let tweetCount = 0;
      let timeLabel = "";

      const cur = agentsRef.current;

      for (const r of records) {
        const rtype = r.type as string;
        if (!timeLabel && r.time) timeLabel = r.time as string;

        if (rtype === "move") {
          const aid = r.agent_id as number;
          agentPatch[aid] = {
            ...(agentPatch[aid] ?? {}),
            pos: r.new_pos as [number, number],
          };
        } else if (rtype === "state_evaluation") {
          const aid = r.agent_id as number;
          const inv: Record<string, number> = {};
          for (const [k, v] of Object.entries(r)) {
            if (k.startsWith("inventory_")) inv[k.slice(10)] = v as number;
          }
          agentPatch[aid] = {
            ...(agentPatch[aid] ?? {}),
            wealth: r.wealth as number,
            inventory: Object.keys(inv).length ? inv : (cur[aid]?.inventory ?? {}),
          };
        } else if (rtype === "sleep_start") {
          const aid = r.agent_id as number;
          agentPatch[aid] = { ...(agentPatch[aid] ?? {}), isSleeping: true };
        } else if (rtype === "sleep_end") {
          const aid = r.agent_id as number;
          agentPatch[aid] = { ...(agentPatch[aid] ?? {}), isSleeping: false };
        } else if (rtype === "follow") {
          const src = r.agent_id as number;
          const tgt = r.target_agent_id as number;
          agentPatch[src] = {
            ...(agentPatch[src] ?? {}),
            numFollows: r.num_follows as number,
          };
          agentPatch[tgt] = {
            ...(agentPatch[tgt] ?? {}),
            numFollowers: (agentPatch[tgt]?.numFollowers ?? cur[tgt]?.numFollowers ?? 0) + 1,
          };
          edgePatch.add.push({ source: src, target: tgt });
        } else if (rtype === "unfollow") {
          const src = r.agent_id as number;
          const tgt = r.target_agent_id as number;
          agentPatch[src] = {
            ...(agentPatch[src] ?? {}),
            numFollows: r.num_follows as number,
          };
          edgePatch.remove.push({ source: src, target: tgt });
        } else if (rtype === "tweet") {
          const aid = r.agent_id as number;
          agentPatch[aid] = {
            ...(agentPatch[aid] ?? {}),
            lastTweet: r.message as string,
          };
          tweetCount++;
        } else if (rtype === "inner_thought") {
          const aid = r.agent_id as number;
          agentPatch[aid] = {
            ...(agentPatch[aid] ?? {}),
            lastInnerThought: r.inner_thought as string,
          };
        } else if (rtype === "change_price") {
          pricePatch[r.item_name as string] = r.new_price as number;
        } else if (rtype === "order_reaction") {
          orderCount++;
        } else if (rtype === "observation" && r.obs_type === "memory") {
          const aid = r.agent_id as number;
          const obs = r.obs as Record<string, unknown>;
          const stressPatch: Record<string, { value: number; reason: string }> = {};
          for (const [k, v] of Object.entries(obs)) {
            if (k.endsWith("_history_stress") && !k.endsWith("_stress_reason")) {
              const stressType = k.replace("_history_stress", "");
              const reasonKey = `${k}_reason`;
              stressPatch[stressType] = {
                value: Number(v),
                reason: (obs[reasonKey] as string) ?? "",
              };
            }
          }
          if (Object.keys(stressPatch).length > 0) {
            agentPatch[aid] = {
              ...(agentPatch[aid] ?? {}),
              stress: {
                ...(cur[aid]?.stress ?? {}),
                ...stressPatch,
              },
            };
          }
        }

        if (VISIBLE_TYPES.has(rtype)) {
          const aid = r.agent_id as number | undefined;
          newEvents.push({
            id: ++eventSeq,
            timeStep: step,
            time: (r.time as string) ?? "",
            type: rtype,
            agentId: aid,
            agentName: aid !== undefined ? (cur[aid]?.name ?? `Agent#${aid}`) : undefined,
            message: buildEventMessage(r, cur),
            raw: r,
          });
        }
      }

      // Apply patches
      setAgents((prev) => {
        const next = { ...prev };
        for (const [aid, patch] of Object.entries(agentPatch)) {
          const id = Number(aid);
          next[id] = { ...next[id], ...patch };
        }
        return next;
      });

      if (Object.keys(pricePatch).length) {
        setPrices((prev) => ({ ...prev, ...pricePatch }));
      }

      if (newEvents.length) {
        setEvents((prev) => {
          const combined = [...newEvents, ...prev];
          return combined.slice(0, MAX_EVENTS);
        });
      }

      if (edgePatch.add.length || edgePatch.remove.length) {
        setSocialEdges((prev) => {
          let edges = [...prev];
          for (const e of edgePatch.remove) {
            edges = edges.filter(
              (x) => !(x.source === e.source && x.target === e.target)
            );
          }
          for (const e of edgePatch.add) {
            if (!edges.some((x) => x.source === e.source && x.target === e.target)) {
              edges.push(e);
            }
          }
          return edges;
        });
      }

      // Macro snapshot
      if (step % 3 === 0) {
        setMacroHistory((prev) => {
          const curAgents = agentsRef.current;
          const curPrices = pricesRef.current;
          const household = Object.values(curAgents).filter((a) => a.is_household);
          const avgWealth =
            household.length > 0
              ? household.reduce((s, a) => s + a.wealth, 0) / household.length
              : 0;
          const point: MacroDataPoint = {
            step,
            time: timeLabel,
            avgWealth: Math.round(avgWealth),
            orderCount,
            tweetCount,
            prices: { ...curPrices, ...pricePatch },
          };
          return [...prev.slice(-120), point];
        });
      }
    };

    ws.onclose = () => {
      // don't reset to idle automatically — keep "done" or current state
    };
  }, []);

  // Cleanup
  useEffect(() => {
    return () => {
      wsRef.current?.close();
    };
  }, []);

  return (
    <Ctx.Provider
      value={{
        metadata,
        agents,
        prices,
        events,
        socialEdges,
        macroHistory,
        currentStep,
        totalSteps,
        isPlaying,
        speed,
        status,
        selectedAgentId,
        hasDefault,
        setSelectedAgentId,
        play,
        pause,
        setSpeed,
        seek,
        upload,
        connect,
        reset,
      }}
    >
      {children}
    </Ctx.Provider>
  );
}
