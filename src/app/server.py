"""League Timer web server.

Single Python process that:
- serves the static frontend (``src/app/static``),
- exposes a ``/ws`` WebSocket for add/remove/trigger/reset + state broadcast, and
- starts a :class:`~src.app.detector.LiveDetector` for any added champion that
  has a model, routing detected casts into the :class:`CooldownEngine`.

Run::

    python -m src.app.server          # then open http://127.0.0.1:8000

Screen-recording permission is only needed once you add a modeled champion
(detection starts then, same as ``src.infer.live``).
"""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from src.app.cooldowns import CooldownEngine
from src.common.config import Config

ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = ROOT / "configs"
MODELS_DIR = ROOT / "models"
STATIC_DIR = Path(__file__).resolve().parent / "static"

BROADCAST_HZ = 4.0
DEVICE = os.environ.get("LEAGUE_TIMER_DEVICE", "auto")


def champion_assets(name: str):
    """Resolve (slug, config_path|None, model_path|None) for a champion name."""
    slug = name.strip().lower().replace(" ", "")
    config = CONFIGS_DIR / f"{slug}.yaml"
    model = MODELS_DIR / slug / "best.pt"
    return slug, (config if config.exists() else None), (model if model.exists() else None)


def timer_spec_from_config(
    config_path: Path,
) -> tuple[Dict[str, Union[float, List[float]]], List[str], Dict[str, str]]:
    """Read tracked abilities, summoners, and class mapping from a champion yaml."""
    cfg = Config.load(str(config_path))
    infer = cfg.section("infer")
    timers = cfg.section("timers")

    track = {str(k) for k in (infer.get("track") or [])}
    summoners = [str(s) for s in (timers.get("summoners") or [])]
    summoner_set = set(summoners)

    ability_cds: Dict[str, Union[float, List[float]]] = {}
    for key, val in (timers.get("abilities") or {}).items():
        key = str(key)
        if key in summoner_set:
            continue
        if track and key not in track:
            continue
        ability_cds[key] = val

    class_to_key = {
        str(k): str(v) for k, v in (timers.get("class_to_key") or {}).items()
    }
    return ability_cds, summoners, class_to_key


class DetectionManager:
    """Owns the (single) active LiveDetector and routes its events to the engine.

    Scope: one auto-detector at a time (single screen, ~230 ms/inference). A
    second modeled champion added while one runs stays manual-only.
    """

    def __init__(self, engine: CooldownEngine) -> None:
        self.engine = engine
        self._detector = None
        self._champion_id: Optional[str] = None

    @property
    def busy(self) -> bool:
        return self._detector is not None

    def _build_and_start(self, champion_id: str, config_path: str, model_path: str) -> None:
        # Imported lazily so the server can start without torch loaded.
        from src.app.detector import LiveDetector

        engine = self.engine

        def on_event(ability: str, score: float, t: float) -> None:
            engine.on_detection(champion_id, ability)

        def on_status(status: Dict) -> None:
            engine.set_detector_status(champion_id, status)

        det = LiveDetector(
            config_path, model_path, device_str=DEVICE,
            on_event=on_event, on_status=on_status,
        )
        det.start()
        self._detector = det
        self._champion_id = champion_id

    async def maybe_start(self, champion_id: str, config_path: Path,
                          model_path: Path) -> bool:
        """Start detection for a modeled champion if none is running. Returns started."""
        if self.busy:
            return False
        self.engine.set_detector_status(champion_id, {"loading": True})
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None, self._build_and_start, champion_id,
                str(config_path), str(model_path))
            return True
        except Exception as exc:  # model load / capture setup failure
            self.engine.set_detector_status(
                champion_id, {"error": f"{type(exc).__name__}: {exc}"})
            return False

    def stop_for(self, champion_id: str) -> None:
        if self._champion_id == champion_id and self._detector is not None:
            self._detector.stop()
            self._detector = None
            self._champion_id = None

    def stop_all(self) -> None:
        if self._detector is not None:
            self._detector.stop()
            self._detector = None
            self._champion_id = None


class Hub:
    """Tracks connected WebSocket clients and broadcasts JSON to all."""

    def __init__(self) -> None:
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self.active.discard(ws)

    async def broadcast(self, message: Dict) -> None:
        for ws in list(self.active):
            try:
                await ws.send_json(message)
            except Exception:
                self.disconnect(ws)


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = CooldownEngine()
    manager = DetectionManager(engine)
    hub = Hub()
    app.state.engine = engine
    app.state.manager = manager
    app.state.hub = hub

    async def broadcaster():
        interval = 1.0 / BROADCAST_HZ
        while True:
            await asyncio.sleep(interval)
            if hub.active:
                await hub.broadcast({"type": "state", "champions": engine.snapshot()})

    task = asyncio.create_task(broadcaster())
    try:
        yield
    finally:
        task.cancel()
        manager.stop_all()


app = FastAPI(title="League Timer", lifespan=lifespan)


async def _handle_message(app: FastAPI, data: Dict) -> Optional[Dict]:
    """Apply a client command. Returns an optional error message dict."""
    engine: CooldownEngine = app.state.engine
    manager: DetectionManager = app.state.manager
    mtype = data.get("type")

    if mtype == "add_champion":
        name = str(data.get("name", "")).strip()
        if not name:
            return {"type": "error", "message": "Champion name required"}
        slug, config_path, model_path = champion_assets(name)
        ability_cds: Dict[str, Union[float, List[float]]] = {}
        summoner_keys: List[str] = []
        class_to_key: Dict[str, str] = {}
        if config_path is not None:
            ability_cds, summoner_keys, class_to_key = timer_spec_from_config(
                config_path)
        will_auto = model_path is not None and config_path is not None and not manager.busy
        champ = engine.add_champion(
            name,
            ability_cooldowns=ability_cds,
            summoner_keys=summoner_keys,
            auto=will_auto,
            class_to_key=class_to_key,
        )
        if will_auto:
            started = await manager.maybe_start(champ["id"], config_path, model_path)
            if not started:
                engine.set_auto(champ["id"], False)  # downgrade to manual
        return None

    if mtype == "remove_champion":
        cid = str(data.get("id", ""))
        manager.stop_for(cid)
        engine.remove_champion(cid)
        return None

    if mtype == "trigger":
        cid = str(data.get("id", ""))
        key = str(data.get("key", ""))
        if not engine.trigger(cid, key):
            return {"type": "error", "message": f"Cannot trigger {key}"}
        return None

    if mtype == "reset":
        cid = str(data.get("id", ""))
        key = str(data.get("key", ""))
        engine.reset(cid, key)
        return None

    if mtype == "set_level":
        cid = str(data.get("id", ""))
        if not engine.set_level(cid, int(data.get("level", 1))):
            return {"type": "error", "message": "Champion not found"}
        return None

    if mtype == "set_ability_haste":
        cid = str(data.get("id", ""))
        if not engine.set_ability_haste(cid, int(data.get("haste", 0))):
            return {"type": "error", "message": "Champion not found"}
        return None

    if mtype == "set_summoner_haste":
        cid = str(data.get("id", ""))
        if not engine.set_summoner_haste(cid, int(data.get("haste", 0))):
            return {"type": "error", "message": "Champion not found"}
        return None

    if mtype == "set_ability_rank":
        cid = str(data.get("id", ""))
        key = str(data.get("key", ""))
        if not engine.set_ability_rank(cid, key, int(data.get("rank", 1))):
            return {"type": "error", "message": f"Cannot set rank for {key}"}
        return None

    return {"type": "error", "message": f"Unknown command: {mtype}"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    hub: Hub = app.state.hub
    engine: CooldownEngine = app.state.engine
    await hub.connect(ws)
    try:
        await ws.send_json({"type": "state", "champions": engine.snapshot()})
        while True:
            data = await ws.receive_json()
            err = await _handle_message(app, data)
            if err is not None:
                await ws.send_json(err)
            # Immediate state echo for snappy UI (broadcaster also runs periodically).
            await ws.send_json({"type": "state", "champions": engine.snapshot()})
    except WebSocketDisconnect:
        hub.disconnect(ws)
    except Exception:
        hub.disconnect(ws)


# Static frontend (mounted last so /ws and any future routes take priority).
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


def main() -> None:
    import uvicorn

    host = os.environ.get("LEAGUE_TIMER_HOST", "127.0.0.1")
    port = int(os.environ.get("LEAGUE_TIMER_PORT", "8000"))
    print(f"League Timer -> http://{host}:{port}  (device={DEVICE})")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
