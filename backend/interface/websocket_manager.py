# -*- coding: utf-8 -*-
"""
WebSocket Manager - Real-time Status Push

Manages WebSocket connections and broadcasts real-time updates:
- GPU status
- System info
- Model status
- Training progress (step, loss, EMA loss, LR)
- Download progress
- Cache progress
- Generation progress
"""

import asyncio
import json
import logging
from typing import Set, Dict, Any, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter(tags=["WebSocket"])


class ConnectionManager:
    """WebSocket connection manager with broadcast loop."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._broadcast_task: Optional[asyncio.Task] = None
        self._running = False

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

        # Send init message with full state
        try:
            init_data = await self._collect_full_status()
            init_data["type"] = "init"
            await websocket.send_text(json.dumps(init_data))
        except Exception as e:
            logger.error(f"Failed to send init message: {e}")

        if not self._running:
            self._running = True
            self._broadcast_task = asyncio.create_task(self._broadcast_loop())

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

        if not self.active_connections and self._running:
            self._running = False
            if self._broadcast_task:
                self._broadcast_task.cancel()

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast to all connected clients."""
        if not self.active_connections:
            return

        data = json.dumps(message)
        dead = set()
        for ws in self.active_connections:
            try:
                await ws.send_text(data)
            except Exception:
                dead.add(ws)

        self.active_connections -= dead

    async def _broadcast_loop(self):
        """Periodic status update broadcast."""
        last_log_idx = 0  # Track how many log lines we've already sent

        while self._running:
            try:
                # --- Drain training subprocess output FIRST ---
                new_lines: list = []
                try:
                    from ..interface.training_router import (
                        _current_process_id, _recent_logs,
                        _parse_training_info, _MAX_LOG_LINES,
                    )
                    from ..infrastructure.container import container
                    runner = container.training_runner()
                    if _current_process_id is not None:
                        while True:
                            line = runner.get_output(_current_process_id)
                            if line is None:
                                break
                            _recent_logs.append(line)
                            if len(_recent_logs) > _MAX_LOG_LINES:
                                _recent_logs[:] = _recent_logs[-_MAX_LOG_LINES:]

                    # Detect new log lines since last broadcast
                    current_len = len(_recent_logs)
                    if current_len > last_log_idx:
                        new_lines = _recent_logs[last_log_idx:current_len]
                        last_log_idx = current_len
                    elif current_len < last_log_idx:
                        # Logs were trimmed, reset
                        last_log_idx = current_len
                except Exception:
                    pass

                # --- Send training_log messages for each new line ---
                for raw_line in new_lines:
                    msg: dict = {"type": "training_log", "message": raw_line}

                    # If this is a [STEP] line, parse progress data
                    if "[STEP]" in raw_line:
                        step_data = {}
                        idx = raw_line.index("[STEP]") + len("[STEP]")
                        for part in raw_line[idx:].strip().split():
                            if "=" in part:
                                k, v = part.split("=", 1)
                                step_data[k.strip()] = v.strip()
                        if step_data:
                            progress: dict = {}
                            if "step" in step_data:
                                parts = step_data["step"].split("/")
                                progress["step"] = {
                                    "current": int(parts[0]),
                                    "total": int(parts[1]) if len(parts) > 1 else 0,
                                }
                            if "epoch" in step_data:
                                progress["epoch"] = {
                                    "current": int(step_data["epoch"]),
                                    "total": 0,
                                }
                            if "loss" in step_data:
                                progress["loss"] = float(step_data["loss"])
                            if "lr" in step_data:
                                progress["lr"] = float(step_data["lr"])
                            msg["progress"] = progress

                    await self.broadcast(msg)

                # --- Regular status update ---
                status = await self._collect_full_status()
                status["type"] = "status_update"
                await self.broadcast(status)
                await asyncio.sleep(1.0)  # 1 Hz update rate
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Broadcast error: {e}")
                await asyncio.sleep(2.0)

    async def _collect_full_status(self) -> Dict[str, Any]:
        """Collect full status from all domains."""
        status: Dict[str, Any] = {}

        try:
            from ..infrastructure.container import container

            # GPU info
            try:
                monitor = container.gpu_monitor()
                gpus = monitor.get_gpu_info()
                if gpus:
                    g = gpus[0]  # Primary GPU
                    status["gpu"] = {
                        "name": g.name,
                        "memoryUsed": float(g.memory_used_gb),
                        "memoryTotal": float(g.memory_total_gb),
                        "memoryPercent": round(g.memory_percent, 1),
                        "temperature": g.temperature,
                        "utilization": g.utilization,
                    }
            except Exception as e:
                logger.debug(f"GPU info error: {e}")

            # System info
            try:
                sys_provider = container.system_info_provider()
                sys_info = sys_provider.get_system_info()
                status["system_info"] = {
                    "platform": sys_info.os,
                    "python": sys_info.python_version,
                    "pytorch": sys_info.torch_version,
                    "cuda": sys_info.cuda_version,
                    "cudnn": "",
                    "diffusers": "",
                    "transformers": "",
                    "accelerate": "",
                    "xformers": "",
                    "bitsandbytes": "",
                }
                # Fill optional library versions
                try:
                    import diffusers
                    status["system_info"]["diffusers"] = diffusers.__version__
                except ImportError:
                    pass
                try:
                    import transformers
                    status["system_info"]["transformers"] = transformers.__version__
                except ImportError:
                    pass
                try:
                    import accelerate
                    status["system_info"]["accelerate"] = accelerate.__version__
                except ImportError:
                    pass
            except Exception as e:
                logger.debug(f"System info error: {e}")

            # Model status
            try:
                from ..domain.system.entities import ModelStatus as MS
                model_mgr = container.model_manager()
                model_info = model_mgr.get_model_status("zimage")
                status["model_status"] = {
                    "downloaded": model_info.status == MS.VALID,
                    "status": model_info.status.value,
                    "path": model_info.path,
                    "size_gb": 0,
                    "missing_files": model_info.missing_files,
                    "components": model_info.components,
                }
                # Download progress
                dl_progress = model_mgr.get_download_progress()
                if dl_progress:
                    from ..domain.system.entities import DownloadStatus as DS
                    status["download"] = {
                        "status": dl_progress.status.value,
                        "model_type": dl_progress.model_type,
                        "progress": round(dl_progress.progress_percent, 1),
                        "downloaded_size_gb": round(dl_progress.downloaded_mb / 1024, 2),
                        "speed_mbps": round(dl_progress.speed_mbps, 2),
                        "current_file": dl_progress.current_file,
                        "error": dl_progress.error_message,
                    }
            except Exception as e:
                logger.debug(f"Model status error: {e}")

            # Training state (with step data for frontend)
            try:
                from .training_router import (
                    _current_process_id, _recent_logs,
                    _parse_training_info, _MAX_LOG_LINES,
                )
                runner = container.training_runner()
                is_running = (
                    _current_process_id is not None
                    and runner.is_running(_current_process_id)
                )

                # Drain subprocess output so logs stay fresh
                if _current_process_id is not None:
                    while True:
                        line = runner.get_output(_current_process_id)
                        if line is None:
                            break
                        _recent_logs.append(line)
                        if len(_recent_logs) > _MAX_LOG_LINES:
                            _recent_logs[:] = _recent_logs[-_MAX_LOG_LINES:]

                training_data: dict = {"running": is_running, "loading": False}
                if is_running:
                    info = _parse_training_info(_recent_logs)
                    training_data.update(info)
                status["training"] = training_data
            except Exception:
                status["training"] = {"running": False, "loading": False}

            # Cache status
            try:
                from .cache_router import get_cache_status
                status["cache"] = get_cache_status()
            except Exception:
                status["cache"] = {
                    "latent": {"status": "idle", "progress": 0},
                    "text": {"status": "idle", "progress": 0},
                }

            # Generation status
            status["generation"] = {
                "running": False,
                "current_step": 0,
                "total_steps": 0,
                "progress": 0,
                "stage": "idle",
                "message": "",
                "error": None,
            }

            # Download status
            status["download"] = {
                "status": "idle",
                "progress": 0,
                "downloaded_size_gb": 0,
            }

        except Exception as e:
            logger.error(f"Status collection error: {e}")

        return status


# Global manager instance
manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Handle incoming messages (heartbeat, commands)
            data = await websocket.receive_text()

            # Handle plain text ping
            if data == "ping":
                await websocket.send_text("pong")
                continue

            # Handle JSON messages
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                elif msg.get("action") == "get_status":
                    full = await manager._collect_full_status()
                    full["type"] = "full_status"
                    await websocket.send_text(json.dumps(full))
                elif msg.get("action") == "clear_logs":
                    await websocket.send_text(json.dumps({"type": "logs_cleared"}))
            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)
