import { FolderOpen, Pause, Play, SkipForward } from "lucide-react";
import { useSimulation } from "../context/SimulationContext";

const STATUS_STYLES: Record<string, string> = {
  connected: "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30",
  done:      "bg-violet-500/20 text-violet-300 border border-violet-500/30",
  connecting:"bg-amber-500/20 text-amber-300 border border-amber-500/30",
  error:     "bg-red-500/20 text-red-300 border border-red-500/30",
  idle:      "bg-slate-700/40 text-slate-400 border border-slate-600/30",
};

export function Toolbar() {
  const {
    isPlaying, play, pause, speed, setSpeed,
    currentStep, totalSteps, seek, status, reset,
  } = useSimulation();

  const progress = totalSteps > 0 ? (currentStep / totalSteps) * 100 : 0;

  return (
    <div
      className="flex items-center gap-3 px-4 py-2.5 border-b border-white/8"
      style={{
        background: "linear-gradient(135deg, rgba(99,102,241,0.15) 0%, rgba(6,182,212,0.08) 100%)",
        backdropFilter: "blur(20px)",
        WebkitBackdropFilter: "blur(20px)",
      }}
    >
      {/* Logo */}
      <div className="flex items-center gap-2 flex-none">
        <span
          className="font-bold text-sm tracking-wide"
          style={{
            background: "linear-gradient(90deg, #818cf8, #a5f3fc)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}
        >
          EconSimulacra
        </span>
        <span className="text-slate-600 text-xs hidden sm:inline">
          Interactive Demo
        </span>
      </div>

      <div className="w-px h-5 bg-white/10 flex-none" />

      {/* Play / Pause */}
      <div className="flex items-center gap-1 flex-none">
        <button
          onClick={isPlaying ? pause : play}
          className="p-1.5 rounded-lg bg-white/8 hover:bg-white/14 border border-white/10 text-slate-200 transition-all"
        >
          {isPlaying ? <Pause size={15} /> : <Play size={15} />}
        </button>
        <button
          onClick={() => seek(Math.min(currentStep + 10, totalSteps - 1))}
          className="p-1.5 rounded-lg bg-white/8 hover:bg-white/14 border border-white/10 text-slate-200 transition-all"
        >
          <SkipForward size={15} />
        </button>
      </div>

      {/* Speed */}
      <div className="flex items-center gap-1.5 flex-none">
        <span className="text-xs text-slate-500">Speed</span>
        <select
          value={speed}
          onChange={(e) => setSpeed(Number(e.target.value))}
          className="bg-white/8 border border-white/10 rounded-md px-1.5 py-0.5 text-slate-200 text-xs focus:outline-none focus:border-indigo-500/50"
        >
          {[0.5, 1, 2, 3, 5, 10, 20].map((s) => (
            <option key={s} value={s} className="bg-[#0d1428]">
              {s}×
            </option>
          ))}
        </select>
      </div>

      {/* Seek slider */}
      <div className="flex-1 flex items-center gap-2 min-w-0">
        <input
          type="range"
          min={0}
          max={Math.max(totalSteps - 1, 1)}
          value={currentStep}
          onChange={(e) => seek(Number(e.target.value))}
          className="flex-1 h-1 rounded-full appearance-none cursor-pointer"
          style={{ accentColor: "#818cf8" }}
        />
        <span className="text-xs text-slate-500 whitespace-nowrap font-mono">
          {String(currentStep).padStart(3, " ")} / {totalSteps}
        </span>
      </div>

      {/* Progress bar (thin bottom line) */}
      <div className="absolute bottom-0 left-0 right-0 h-[2px] bg-white/5">
        <div
          className="h-full transition-all duration-500 rounded-full"
          style={{
            width: `${progress}%`,
            background: "linear-gradient(90deg, #6366f1, #06b6d4)",
          }}
        />
      </div>

      {/* Open new file */}
      <button
        onClick={reset}
        title="Open another file"
        className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium text-slate-400 border border-white/10 bg-white/4 hover:bg-white/8 hover:text-slate-200 transition-all flex-none"
      >
        <FolderOpen size={13} />
        <span className="hidden sm:inline">New file</span>
      </button>

      {/* Status badge */}
      <span className={`text-xs px-2.5 py-0.5 rounded-full font-medium flex-none ${STATUS_STYLES[status] ?? STATUS_STYLES.idle}`}>
        {status}
      </span>
    </div>
  );
}
