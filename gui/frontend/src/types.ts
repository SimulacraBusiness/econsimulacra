export interface AgentMeta {
  id: number;
  name: string;
  agent_type: string;
  is_household: boolean;
  wealth: number;
  inventory: Record<string, number>;
  pos: [number, number];
}

export interface StressEntry {
  value: number;   // 0–100
  reason: string;
}

export interface AgentState {
  id: number;
  name: string;
  agent_type: string;
  is_household: boolean;
  pos: [number, number];
  wealth: number;
  inventory: Record<string, number>;
  isSleeping: boolean;
  numFollows: number;
  numFollowers: number;
  lastInnerThought: string | null;
  lastTweet: string | null;
  stress: Record<string, StressEntry>;  // e.g. "consumption" → {value, reason}
}

export interface EventEntry {
  id: number;
  timeStep: number;
  time: string;
  type: string;
  agentId?: number;
  agentName?: string;
  message: string;
  raw: Record<string, unknown>;
}

export interface MacroDataPoint {
  step: number;
  time: string;
  avgWealth: number;
  orderCount: number;
  tweetCount: number;
  prices: Record<string, number>;
}

export interface SocialEdge {
  source: number;
  target: number;
}

export interface Metadata {
  agents: AgentMeta[];
  items: Record<string, number>;
  grid_size: [number, number];
  total_steps: number;
}

export type ConnectionStatus = "idle" | "connecting" | "connected" | "done" | "error";
