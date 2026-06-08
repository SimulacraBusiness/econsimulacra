import React, { useCallback, useState } from "react";
import { Upload, FileText, Zap, AlertCircle } from "lucide-react";
import { useSimulation } from "../context/SimulationContext";

function GradientText({ children }: { children: React.ReactNode }) {
  return (
    <span
      style={{
        background: "linear-gradient(90deg, #818cf8, #a5f3fc)",
        WebkitBackgroundClip: "text",
        WebkitTextFillColor: "transparent",
      }}
    >
      {children}
    </span>
  );
}

export function UploadScreen() {
  const { upload, connect, hasDefault } = useSimulation();
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFile = useCallback(
    async (file: File) => {
      if (!file.name.endsWith(".txt")) {
        setError("Please select a .txt log file.");
        return;
      }
      setError(null);
      setUploading(true);
      setProgress("Uploading…");

      try {
        const sessionId = await upload(file);
        setProgress("Connecting…");
        await connect(sessionId);
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : "Upload failed.";
        setError(msg);
        setUploading(false);
        setProgress(null);
      }
    },
    [upload, connect]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    e.target.value = "";
  };

  const useDemo = useCallback(async () => {
    setUploading(true);
    setProgress("Loading demo…");
    try {
      await connect(null);
    } catch {
      setError("Failed to load demo.");
      setUploading(false);
      setProgress(null);
    }
  }, [connect]);

  return (
    <div className="flex-1 flex items-center justify-center p-8">
      <div className="w-full max-w-lg flex flex-col gap-6">
        {/* Hero */}
        <div className="text-center">
          <h1 className="text-3xl font-bold mb-2">
            <GradientText>EconSimulacra</GradientText>
          </h1>
          <p className="text-slate-400 text-sm leading-relaxed">
            LLM-powered socio-economic multi-agent simulation.<br />
            Upload a simulation log to explore agents, markets, and social dynamics interactively.
          </p>
        </div>

        {/* Drop zone */}
        <label
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={`relative flex flex-col items-center justify-center gap-4 rounded-2xl border-2 border-dashed p-10 cursor-pointer transition-all duration-200 ${
            dragging
              ? "border-indigo-400/80 bg-indigo-500/10 scale-[1.01]"
              : "border-white/15 bg-white/3 hover:border-indigo-500/50 hover:bg-white/5"
          } ${uploading ? "pointer-events-none opacity-60" : ""}`}
        >
          <input
            type="file"
            accept=".txt"
            className="hidden"
            onChange={onInputChange}
            disabled={uploading}
          />

          {uploading ? (
            <>
              <div
                className="w-12 h-12 rounded-full border-2 border-t-transparent animate-spin"
                style={{ borderColor: "rgba(129,140,248,0.6)", borderTopColor: "transparent" }}
              />
              <p className="text-slate-300 font-medium">{progress}</p>
            </>
          ) : (
            <>
              <div
                className="w-16 h-16 rounded-2xl flex items-center justify-center"
                style={{ background: "linear-gradient(135deg, rgba(99,102,241,0.2), rgba(6,182,212,0.15))" }}
              >
                <Upload size={28} className="text-indigo-300" />
              </div>
              <div className="text-center">
                <p className="text-slate-200 font-semibold text-base">
                  Drop your log file here
                </p>
                <p className="text-slate-500 text-xs mt-1">
                  or click to browse · <span className="text-indigo-400">.txt</span> files only · max 200 MB
                </p>
              </div>
            </>
          )}
        </label>

        {/* Error */}
        {error && (
          <div className="flex items-center gap-2 rounded-xl px-4 py-3 text-sm text-red-300 bg-red-500/10 border border-red-500/20">
            <AlertCircle size={16} className="flex-none" />
            {error}
          </div>
        )}

        {/* Demo button */}
        {hasDefault && !uploading && (
          <div className="flex items-center gap-3">
            <div className="flex-1 h-px bg-white/8" />
            <span className="text-xs text-slate-600">or</span>
            <div className="flex-1 h-px bg-white/8" />
          </div>
        )}
        {hasDefault && !uploading && (
          <button
            onClick={useDemo}
            className="flex items-center justify-center gap-2 w-full py-3 rounded-xl font-semibold text-sm text-indigo-200 border border-indigo-500/30 bg-indigo-500/10 hover:bg-indigo-500/18 transition-all duration-200"
            style={{ boxShadow: "0 0 20px rgba(99,102,241,0.12)" }}
          >
            <Zap size={15} />
            Launch Demo (built-in log)
          </button>
        )}

        {/* Feature pills */}
        <div className="flex flex-wrap justify-center gap-2 pt-2">
          {[
            "🗺 Grid Map",
            "🕸 Social Network",
            "💭 Inner Thoughts",
            "📊 Macro Dashboard",
            "😰 Stress Tracking",
          ].map((f) => (
            <span
              key={f}
              className="text-xs px-3 py-1 rounded-full text-slate-400 bg-white/4 border border-white/8"
            >
              {f}
            </span>
          ))}
        </div>

        {/* File format hint */}
        <div
          className="rounded-xl p-3 border border-white/6 text-xs text-slate-500"
          style={{ background: "rgba(255,255,255,0.02)" }}
        >
          <div className="flex items-center gap-1.5 mb-1.5 text-slate-400 font-medium">
            <FileText size={12} />
            Expected format
          </div>
          <p className="font-mono leading-relaxed">
            {"{"}"type":"move","time_step":0,"agent_id":1,...{"}"}<br />
            {"{"}"type":"tweet","time_step":0,"message":"..."{"}"}<br />
            <span className="text-slate-600">…one JSON object per line (JSON Lines)</span>
          </p>
        </div>
      </div>
    </div>
  );
}
