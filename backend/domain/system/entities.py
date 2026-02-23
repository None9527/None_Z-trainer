# -*- coding: utf-8 -*-
"""
System Domain - Entities

Core entities for system status, GPU monitoring, and model management.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

@dataclass
class GPUInfo:
    """GPU device information."""
    index: int = 0
    name: str = ""
    memory_total_mb: int = 0
    memory_used_mb: int = 0
    memory_free_mb: int = 0
    temperature: int = 0
    utilization: int = 0

    @property
    def memory_total_gb(self) -> str:
        return f"{self.memory_total_mb / 1024:.1f}"

    @property
    def memory_used_gb(self) -> str:
        return f"{self.memory_used_mb / 1024:.1f}"

    @property
    def memory_percent(self) -> float:
        if self.memory_total_mb == 0:
            return 0.0
        return self.memory_used_mb / self.memory_total_mb * 100


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

@dataclass
class SystemInfo:
    """Overall system information."""
    os: str = ""
    python_version: str = ""
    torch_version: str = ""
    cuda_version: str = ""
    gpus: List[GPUInfo] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Model - Spec & Registry
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """Model specification for registry."""
    name: str
    model_id: str                    # ModelScope model ID, e.g. "Tongyi-MAI/Z-Image"
    description: str
    default_path: str                # Suggested local directory name
    size_gb: float
    source_url: str = ""             # ModelScope page URL
    aliases: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Model - Status
# ---------------------------------------------------------------------------

class ModelStatus(Enum):
    """Local model status."""
    NOT_FOUND = "not_found"
    INCOMPLETE = "incomplete"
    VALID = "valid"
    DOWNLOADING = "downloading"


@dataclass
class ComponentDetail:
    """Validation result for a single model component."""
    path: str
    exists: bool = False
    valid: bool = False
    required: bool = True
    message: str = ""


@dataclass
class ModelInfo:
    """Local model status and details."""
    model_type: str = "zimage"
    status: ModelStatus = ModelStatus.NOT_FOUND
    path: str = ""
    missing_files: List[str] = field(default_factory=list)
    components: Dict[str, Any] = field(default_factory=dict)
    download_progress: float = 0.0


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

class DownloadStatus(Enum):
    """Download status."""
    IDLE = "idle"
    PENDING = "pending"
    DOWNLOADING = "downloading"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DownloadProgress:
    """Download progress information."""
    status: DownloadStatus = DownloadStatus.IDLE
    model_type: str = ""
    downloaded_bytes: int = 0
    total_bytes: int = 0
    speed_bps: float = 0.0
    eta_seconds: Optional[float] = None
    current_file: Optional[str] = None
    error_message: Optional[str] = None

    @property
    def progress_percent(self) -> float:
        if self.total_bytes <= 0:
            return 0.0
        return min(100.0, (self.downloaded_bytes / self.total_bytes) * 100)

    @property
    def downloaded_mb(self) -> float:
        return self.downloaded_bytes / (1024 * 1024)

    @property
    def speed_mbps(self) -> float:
        return self.speed_bps / (1024 * 1024)
