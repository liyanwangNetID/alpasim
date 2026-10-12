# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
"""Synchronous startup-barrier transport for AlpaSim static occupancy.

The exporter extracts and sends one complete scene snapshot over TCP, then waits
for an application-level READY acknowledgement before the simulation event loop
is allowed to start.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import struct
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from alpasim_utils.static_environment import (
    SCHEMA_VERSION,
    StaticEnvironmentConfig,
    extract_static_environment,
)

logger = logging.getLogger(__name__)

_HEADER_LENGTH = struct.Struct("!I")
_PAYLOAD_LENGTH = struct.Struct("!Q")


@dataclass(frozen=True)
class StaticOccupancyTcpConfig:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 15006
    startup_timeout_s: float = 30.0
    acknowledgement_timeout_s: float = 10.0
    retry_interval_s: float = 0.5

    @classmethod
    def from_environment(cls) -> "StaticOccupancyTcpConfig":
        enabled = os.environ.get("ALPASIM_STATIC_OCCUPANCY_ENABLED", "0").lower()
        return cls(
            enabled=enabled in {"1", "true", "yes", "on"},
            host=os.environ.get(
                "ALPASIM_STATIC_OCCUPANCY_HOST",
                os.environ.get("ALPASIM_ROS_BRIDGE_HOST", "127.0.0.1"),
            ),
            port=int(os.environ.get("ALPASIM_STATIC_OCCUPANCY_PORT", "15006")),
            startup_timeout_s=float(
                os.environ.get("ALPASIM_STATIC_OCCUPANCY_STARTUP_TIMEOUT_S", "30.0")
            ),
            acknowledgement_timeout_s=float(
                os.environ.get("ALPASIM_STATIC_OCCUPANCY_ACK_TIMEOUT_S", "10.0")
            ),
            retry_interval_s=float(
                os.environ.get("ALPASIM_STATIC_OCCUPANCY_RETRY_INTERVAL_S", "0.5")
            ),
        )

    def validate(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ValueError(f"Invalid static occupancy TCP port: {self.port}")
        for name, value in (
            ("startup_timeout_s", self.startup_timeout_s),
            ("acknowledgement_timeout_s", self.acknowledgement_timeout_s),
            ("retry_interval_s", self.retry_interval_s),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")


@dataclass(frozen=True)
class StaticOccupancyWireSnapshot:
    cache_key: str
    header: dict[str, Any]
    header_bytes: bytes
    payload: bytes

    @property
    def checksum(self) -> str:
        return str(self.header["payload_sha256"])


class StaticOccupancyTcpExporter:
    """Build, cache, send, and acknowledge static occupancy before simulation."""

    def __init__(
        self,
        transport_config: StaticOccupancyTcpConfig | None = None,
        extraction_config: StaticEnvironmentConfig | None = None,
    ) -> None:
        self.transport_config = (
            transport_config or StaticOccupancyTcpConfig.from_environment()
        )
        self.transport_config.validate()
        self.extraction_config = extraction_config or StaticEnvironmentConfig()
        self.extraction_config.validate()
        self._cache: dict[str, StaticOccupancyWireSnapshot] = {}
        self._cache_lock = threading.Lock()

    def prepare_and_send(
        self,
        *,
        usdz_path: str | Path,
        scene_id: str,
    ) -> StaticOccupancyWireSnapshot | None:
        """Block until the receiver confirms the complete scene snapshot is ready."""
        if not self.transport_config.enabled:
            logger.debug("Static occupancy startup export is disabled")
            return None
        snapshot = self._get_or_build_snapshot(Path(usdz_path), scene_id)
        deadline = time.monotonic() + self.transport_config.startup_timeout_s
        last_error: BaseException | None = None
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            try:
                self._send_once(snapshot)
                logger.info(
                    "Static occupancy READY: scene=%s columns=%d bytes=%d sha256=%s",
                    scene_id,
                    snapshot.header["column_count"],
                    len(snapshot.payload),
                    snapshot.checksum,
                )
                return snapshot
            except (OSError, ConnectionError, TimeoutError, ValueError) as exc:
                last_error = exc
                remaining = deadline - time.monotonic()
                logger.warning(
                    "Static occupancy startup handshake failed (attempt=%d, remaining=%.1fs): %s",
                    attempt,
                    max(0.0, remaining),
                    exc,
                )
                if remaining > 0:
                    time.sleep(min(self.transport_config.retry_interval_s, remaining))
        raise TimeoutError(
            "Static occupancy receiver did not become ready before startup timeout: "
            f"scene={scene_id!r}, destination="
            f"{self.transport_config.host}:{self.transport_config.port}, "
            f"last_error={last_error!r}"
        )

    def _snapshot_cache_key(self, usdz_path: Path, scene_id: str) -> str:
        stat = usdz_path.stat()
        config_json = json.dumps(
            asdict(self.extraction_config), sort_keys=True, separators=(",", ":")
        )
        config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()
        return "|".join(
            (
                scene_id,
                str(usdz_path.resolve()),
                str(stat.st_size),
                str(stat.st_mtime_ns),
                SCHEMA_VERSION,
                config_hash,
            )
        )

    def _get_or_build_snapshot(
        self, usdz_path: Path, scene_id: str
    ) -> StaticOccupancyWireSnapshot:
        cache_key = self._snapshot_cache_key(usdz_path, scene_id)
        with self._cache_lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            logger.info("Reusing cached static occupancy snapshot for scene %s", scene_id)
            return cached

        logger.info("Extracting static occupancy before simulation: scene=%s", scene_id)
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="alpasim-static-occupancy-") as temp:
            result = extract_static_environment(
                usdz_path,
                Path(temp),
                self.extraction_config,
            )
            metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            payload = result.columns_path.read_bytes()

        checksum = hashlib.sha256(payload).hexdigest()
        header: dict[str, Any] = {
            "message_type": "static_occupancy",
            "schema_version": metadata["schema_version"],
            "scene_id": metadata["scene_id"],
            "scene_uuid": metadata["scene_uuid"],
            "frame_id": metadata["frame_id"],
            "representation": metadata["representation"],
            "semantic_labels": metadata["semantic_labels"],
            "grid_size_xy_m": metadata["config"]["grid_size_xy_m"],
            "column_count": summary["column_count"],
            "payload_encoding": "npz_compressed",
            "payload_size": len(payload),
            "payload_sha256": checksum,
        }
        header_bytes = json.dumps(
            header, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        if len(header_bytes) > 0xFFFFFFFF:
            raise ValueError("Static occupancy header exceeds uint32 framing")
        snapshot = StaticOccupancyWireSnapshot(
            cache_key=cache_key,
            header=header,
            header_bytes=header_bytes,
            payload=payload,
        )
        with self._cache_lock:
            self._cache.clear()
            self._cache[cache_key] = snapshot
        logger.info(
            "Built static occupancy snapshot: scene=%s columns=%d bytes=%d elapsed=%.3fs",
            scene_id,
            header["column_count"],
            len(payload),
            time.perf_counter() - started,
        )
        return snapshot

    def _send_once(self, snapshot: StaticOccupancyWireSnapshot) -> None:
        config = self.transport_config
        connect_timeout = max(
            0.1,
            min(config.retry_interval_s, config.startup_timeout_s),
        )
        with socket.create_connection(
            (config.host, config.port), timeout=connect_timeout
        ) as sock:
            sock.settimeout(config.acknowledgement_timeout_s)
            sock.sendall(_HEADER_LENGTH.pack(len(snapshot.header_bytes)))
            sock.sendall(_PAYLOAD_LENGTH.pack(len(snapshot.payload)))
            sock.sendall(snapshot.header_bytes)
            sock.sendall(snapshot.payload)
            ack_length = _HEADER_LENGTH.unpack(self._recv_exact(sock, 4))[0]
            if ack_length <= 0 or ack_length > 1_000_000:
                raise ValueError(f"Invalid static occupancy ACK length: {ack_length}")
            ack = json.loads(self._recv_exact(sock, ack_length).decode("utf-8"))
            self._validate_ack(snapshot, ack)

    @staticmethod
    def _recv_exact(sock: socket.socket, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = sock.recv(remaining)
            if not chunk:
                raise ConnectionError("Connection closed before complete ACK was received")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    @staticmethod
    def _validate_ack(
        snapshot: StaticOccupancyWireSnapshot,
        ack: dict[str, Any],
    ) -> None:
        expected = {
            "message_type": "static_occupancy_ack",
            "status": "ready",
            "scene_id": snapshot.header["scene_id"],
            "scene_uuid": snapshot.header["scene_uuid"],
            "schema_version": snapshot.header["schema_version"],
            "payload_sha256": snapshot.header["payload_sha256"],
        }
        mismatches = {
            key: {"expected": value, "received": ack.get(key)}
            for key, value in expected.items()
            if ack.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Invalid static occupancy ACK: {mismatches}")


static_occupancy_tcp_exporter = StaticOccupancyTcpExporter()
