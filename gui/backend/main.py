"""FastAPI backend for EconSimulacra GUI demo — with file upload & session management."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="EconSimulacra GUI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session storage ───────────────────────────────────────────────────────────
# session_id → {steps: dict[int, list], created_at: float}
_sessions: dict[str, dict[str, Any]] = {}
SESSION_TTL = 3600  # 1 hour


def _cleanup_sessions() -> None:
    now = time.time()
    expired = [
        sid for sid, s in _sessions.items() if now - s["created_at"] > SESSION_TTL
    ]
    for sid in expired:
        del _sessions[sid]


def _get_steps(session_id: str | None) -> tuple[dict[int, list], int]:
    """Return (steps, max_step) for a session or the default log."""
    if session_id and session_id in _sessions:
        steps = _sessions[session_id]["steps"]
    elif _DEFAULT_STEPS:
        steps = _DEFAULT_STEPS
    else:
        raise HTTPException(404, "No session found and no default log available.")
    max_step = max((k for k in steps if k >= 0), default=0)
    return steps, max_step


# ── Default log (optional) ───────────────────────────────────────────────────
_DEFAULT_LOG_PATH = Path(__file__).parent.parent / "demo_log.txt"


def _parse_log(text: str) -> dict[int, list[dict[str, Any]]]:
    steps: dict[int, list[dict[str, Any]]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts: int = record.get("time_step", -1)
        if ts not in steps:
            steps[ts] = []
        steps[ts].append(record)
    return steps


_DEFAULT_STEPS: dict[int, list] = {}
if _DEFAULT_LOG_PATH.exists():
    print("Loading default log file…")
    with open(_DEFAULT_LOG_PATH, encoding="utf-8") as _f:
        _DEFAULT_STEPS = _parse_log(_f.read())
    _default_max = max((k for k in _DEFAULT_STEPS if k >= 0), default=0)
    print(
        f"Default log: {_default_max + 1} steps, "
        f"{sum(len(v) for v in _DEFAULT_STEPS.values())} records"
    )
else:
    print("No default log found — upload-only mode.")


# ── Metadata extraction ───────────────────────────────────────────────────────
def _build_metadata(steps: dict[int, list], max_step: int) -> dict[str, Any]:
    init_records = steps.get(-1, [])
    agents: dict[int, dict[str, Any]] = {}
    items: dict[str, float] = {}

    for r in init_records:
        if r["type"] == "agent_generation":
            aid = r["agent_id"]
            agents[aid] = {
                "id": aid,
                "name": r["agent_name"],
                "agent_type": r["agent_type"],
                "is_household": r["agent_type"] == "LLMAgent",
                "wealth": r.get("wealth", 0),
                "inventory": {
                    k[len("inventory_") :]: v
                    for k, v in r.items()
                    if k.startswith("inventory_")
                },
                "pos": [0, 0],
            }
        elif r["type"] == "item_generation":
            items[r["item_name"]] = float(r["price"])

    for r in init_records:
        if r["type"] == "space_assign" and r["agent_id"] in agents:
            agents[r["agent_id"]]["pos"] = r["pos"]

    for r in steps.get(0, []):
        if r["type"] == "change_price":
            items[r["item_name"]] = float(r["new_price"])

    return {
        "agents": list(agents.values()),
        "items": items,
        "grid_size": [10, 10],
        "total_steps": max_step + 1,
    }


# ── REST endpoints ────────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_log(file: UploadFile = File(...)) -> dict[str, Any]:
    """Accept a .txt log file, parse it, and return a session_id."""
    _cleanup_sessions()

    if not file.filename or not file.filename.endswith(".txt"):
        raise HTTPException(400, "Only .txt log files are accepted.")

    MAX_BYTES = 200 * 1024 * 1024  # 200 MB
    content = await file.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise HTTPException(413, "File too large (max 200 MB).")

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, "File must be UTF-8 encoded.")

    steps = _parse_log(text)
    if not steps:
        raise HTTPException(400, "No valid log records found in file.")

    max_step = max((k for k in steps if k >= 0), default=0)
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {"steps": steps, "created_at": time.time()}

    return {
        "session_id": session_id,
        "total_steps": max_step + 1,
        "num_records": sum(len(v) for v in steps.values()),
    }


@app.get("/api/metadata")
async def get_metadata(session_id: str | None = None) -> dict[str, Any]:
    steps, max_step = _get_steps(session_id)
    return _build_metadata(steps, max_step)


@app.get("/api/has_default")
async def has_default() -> dict[str, bool]:
    return {"has_default": bool(_DEFAULT_STEPS)}


# ── WebSocket replay ──────────────────────────────────────────────────────────
@app.websocket("/ws/replay")
async def replay(websocket: WebSocket, session_id: str | None = None) -> None:
    await websocket.accept()

    try:
        steps, max_step = _get_steps(session_id)
    except HTTPException as e:
        await websocket.send_json({"error": e.detail})
        await websocket.close()
        return

    speed: float = 3.0
    paused: bool = False
    current_step: list[int] = [0]
    stop_event = asyncio.Event()

    async def receive_controls() -> None:
        nonlocal speed, paused
        while not stop_event.is_set():
            try:
                msg = await websocket.receive_json()
                action = msg.get("action", "")
                if action == "pause":
                    paused = True
                elif action == "resume":
                    paused = False
                elif action == "seek":
                    current_step[0] = int(msg.get("step", current_step[0]))
                elif action == "set_speed":
                    speed = max(0.1, float(msg.get("speed", speed)))
            except Exception:
                stop_event.set()
                break

    recv_task = asyncio.create_task(receive_controls())

    try:
        await websocket.send_json({"time_step": -1, "records": steps.get(-1, [])})

        while not stop_event.is_set():
            if paused:
                await asyncio.sleep(0.05)
                continue

            step = current_step[0]
            if step > max_step:
                await websocket.send_json(
                    {"done": True, "time_step": step, "records": []}
                )
                break

            records = steps.get(step, [])
            await websocket.send_json({"time_step": step, "records": records})
            current_step[0] += 1
            await asyncio.sleep(1.0 / speed)

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        stop_event.set()
        recv_task.cancel()


# ── Serve built frontend ──────────────────────────────────────────────────────
_frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="static")
