# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import socket
import struct
import threading
from pathlib import Path

import pytest

from alpasim_runtime.static_occupancy_tcp import (
    StaticOccupancyTcpConfig,
    StaticOccupancyTcpExporter,
    StaticOccupancyWireSnapshot,
)


def _snapshot() -> StaticOccupancyWireSnapshot:
    payload = b"test-npz-payload"
    header = {
        "message_type": "static_occupancy",
        "schema_version": "0.1",
        "scene_id": "clipgt-test",
        "scene_uuid": "uuid-test",
        "frame_id": "map",
        "representation": "single_interval_2p5d_static_occupancy",
        "semantic_labels": False,
        "grid_size_xy_m": 0.4,
        "column_count": 3,
        "payload_encoding": "npz_compressed",
        "payload_size": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    return StaticOccupancyWireSnapshot("key", header, header_bytes, payload)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise RuntimeError("short receive")
        data += chunk
    return data


def _server(listener: socket.socket, *, corrupt_ack: bool = False) -> None:
    conn, _ = listener.accept()
    with conn:
        header_size = struct.unpack("!I", _recv_exact(conn, 4))[0]
        payload_size = struct.unpack("!Q", _recv_exact(conn, 8))[0]
        header = json.loads(_recv_exact(conn, header_size))
        payload = _recv_exact(conn, payload_size)
        assert hashlib.sha256(payload).hexdigest() == header["payload_sha256"]
        ack = {
            "message_type": "static_occupancy_ack",
            "status": "ready",
            "scene_id": header["scene_id"],
            "scene_uuid": header["scene_uuid"],
            "schema_version": header["schema_version"],
            "payload_sha256": (
                "bad" if corrupt_ack else header["payload_sha256"]
            ),
        }
        encoded = json.dumps(ack, separators=(",", ":")).encode()
        conn.sendall(struct.pack("!I", len(encoded)) + encoded)


def _listener() -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    return sock


def test_send_once_accepts_valid_ready_ack() -> None:
    listener = _listener()
    thread = threading.Thread(target=_server, args=(listener,), daemon=True)
    thread.start()
    exporter = StaticOccupancyTcpExporter(
        StaticOccupancyTcpConfig(
            enabled=True,
            host="127.0.0.1",
            port=listener.getsockname()[1],
            startup_timeout_s=1.0,
            acknowledgement_timeout_s=1.0,
            retry_interval_s=0.1,
        )
    )
    exporter._send_once(_snapshot())
    thread.join(timeout=1.0)
    listener.close()


def test_send_once_rejects_wrong_checksum_ack() -> None:
    listener = _listener()
    thread = threading.Thread(
        target=_server, args=(listener,), kwargs={"corrupt_ack": True}, daemon=True
    )
    thread.start()
    exporter = StaticOccupancyTcpExporter(
        StaticOccupancyTcpConfig(
            enabled=True,
            host="127.0.0.1",
            port=listener.getsockname()[1],
            startup_timeout_s=1.0,
            acknowledgement_timeout_s=1.0,
            retry_interval_s=0.1,
        )
    )
    with pytest.raises(ValueError, match="Invalid static occupancy ACK"):
        exporter._send_once(_snapshot())
    thread.join(timeout=1.0)
    listener.close()


def test_disabled_exporter_does_not_touch_usdz(tmp_path: Path) -> None:
    exporter = StaticOccupancyTcpExporter(
        StaticOccupancyTcpConfig(enabled=False)
    )
    result = exporter.prepare_and_send(
        usdz_path=tmp_path / "missing.usdz",
        scene_id="clipgt-test",
    )
    assert result is None
