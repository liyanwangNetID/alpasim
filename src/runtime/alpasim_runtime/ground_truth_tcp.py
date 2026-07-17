"""Reliable TCP exporter for the complete recording GT ego trajectory."""

from __future__ import annotations

import json
import logging
import math
import queue
import socket
import struct
import threading
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _finite_vector(
    values: Any,
    expected_length: int,
    field_name: str,
) -> list:
    """Convert an array-like vector and reject non-finite values."""
    array = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    if array.size != expected_length:
        raise ValueError(
            f"{field_name} must contain {expected_length} values, "
            f"got shape {array.shape}"
        )

    result = [
        float(value)
        for value in array
    ]

    if not all(math.isfinite(value) for value in result):
        raise ValueError(
            f"{field_name} contains non-finite values: {result}"
        )

    return result


def serialize_ground_truth_trajectory(
    trajectory: Any,
    scene_id: str,
    frame_id: str = "map",
) -> dict[str, Any]:
    """Serialize a complete recording GT ego trajectory."""

    if trajectory is None or len(trajectory) == 0:
        raise ValueError(
            f"Scene {scene_id!r} has no recording GT trajectory"
        )

    timestamps = np.asarray(
        trajectory.timestamps_us,
        dtype=np.int64,
    )
    positions = np.asarray(
        trajectory.positions,
        dtype=np.float64,
    )
    quaternions = np.asarray(
        trajectory.quaternions,
        dtype=np.float64,
    )

    velocities = np.asarray(
        trajectory.velocities(),
        dtype=np.float64,
    )
    accelerations = np.asarray(
        trajectory.accelerations(),
        dtype=np.float64,
    )
    yaws = np.asarray(
        trajectory.yaws,
        dtype=np.float64,
    )
    yaw_rates = np.asarray(
        trajectory.yaw_rates(),
        dtype=np.float64,
    )
    yaw_accelerations = np.asarray(
        trajectory.yaw_accelerations(),
        dtype=np.float64,
    )

    count = len(timestamps)

    expected_shapes = {
        "positions": (count, 3),
        "quaternions": (count, 4),
        "velocities": (count, 3),
        "accelerations": (count, 3),
        "yaws": (count,),
        "yaw_rates": (count,),
        "yaw_accelerations": (count,),
    }

    actual_arrays = {
        "positions": positions,
        "quaternions": quaternions,
        "velocities": velocities,
        "accelerations": accelerations,
        "yaws": yaws,
        "yaw_rates": yaw_rates,
        "yaw_accelerations": yaw_accelerations,
    }

    for name, expected_shape in expected_shapes.items():
        actual_shape = actual_arrays[name].shape

        if actual_shape != expected_shape:
            raise ValueError(
                f"{name} has shape {actual_shape}; "
                f"expected {expected_shape}"
            )

    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(
            "Recording GT timestamps must be strictly increasing"
        )

    points: list[dict[str, Any]] = []

    for index in range(count):
        position = _finite_vector(
            positions[index],
            3,
            "position",
        )
        quaternion = _finite_vector(
            quaternions[index],
            4,
            "quaternion",
        )
        velocity = _finite_vector(
            velocities[index],
            3,
            "velocity",
        )
        acceleration = _finite_vector(
            accelerations[index],
            3,
            "acceleration",
        )

        yaw = float(yaws[index])
        yaw_rate = float(yaw_rates[index])
        yaw_acceleration = float(
            yaw_accelerations[index]
        )

        if not all(
            math.isfinite(value)
            for value in (
                yaw,
                yaw_rate,
                yaw_acceleration,
            )
        ):
            raise ValueError(
                f"Non-finite angular state at index {index}"
            )

        speed = float(
            np.linalg.norm(velocities[index])
        )

        points.append(
            {
                "timestamp_us": int(timestamps[index]),
                "position": {
                    "x": position[0],
                    "y": position[1],
                    "z": position[2],
                },
                "orientation": {
                    "x": quaternion[0],
                    "y": quaternion[1],
                    "z": quaternion[2],
                    "w": quaternion[3],
                },
                "linear_velocity": {
                    "x": velocity[0],
                    "y": velocity[1],
                    "z": velocity[2],
                },
                "linear_acceleration": {
                    "x": acceleration[0],
                    "y": acceleration[1],
                    "z": acceleration[2],
                },
                "yaw": yaw,
                "yaw_rate": yaw_rate,
                "yaw_acceleration": yaw_acceleration,
                "speed": speed,
            }
        )

    return {
        "message_type": "ground_truth_ego_trajectory",
        "scene_id": str(scene_id),
        "source_revision": 1,
        "pose_frame_id": frame_id,
        "dynamics_frame_id": frame_id,
        "start_timestamp_us": int(timestamps[0]),
        "end_timestamp_us": int(timestamps[-1]),
        "points": points,
    }


class GroundTruthTcpExporter:
    """Reliably export the complete recording trajectory once."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 15004,
    ) -> None:
        self.destination = (host, port)

        self.queue: queue.Queue[
            dict[str, Any]
        ] = queue.Queue(maxsize=1)

        self.socket: socket.socket | None = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        self.scheduled_keys: set[str] = set()

        self.thread = threading.Thread(
            target=self._worker,
            name="alpasim-ground-truth-tcp-exporter",
            daemon=True,
        )
        self.thread.start()

    def publish(
        self,
        trajectory: Any,
        scene_id: str,
        frame_id: str = "map",
    ) -> bool:
        """Queue one complete trajectory per scene."""

        key = str(scene_id)

        with self.lock:
            if key in self.scheduled_keys:
                return False

            self.scheduled_keys.add(key)

        try:
            message = serialize_ground_truth_trajectory(
                trajectory=trajectory,
                scene_id=scene_id,
                frame_id=frame_id,
            )
        except Exception:
            with self.lock:
                self.scheduled_keys.discard(key)

            logger.exception(
                "Failed to serialize recording GT trajectory "
                "for scene %s",
                scene_id,
            )
            raise

        try:
            self.queue.put_nowait(message)
        except queue.Full:
            try:
                replaced = self.queue.get_nowait()

                with self.lock:
                    self.scheduled_keys.discard(
                        str(replaced.get("scene_id", ""))
                    )
            except queue.Empty:
                pass

            self.queue.put_nowait(message)

        logger.info(
            "Queued recording GT ego trajectory: "
            "scene=%s, points=%d, duration=%.3fs",
            scene_id,
            len(message["points"]),
            (
                message["end_timestamp_us"]
                - message["start_timestamp_us"]
            )
            / 1e6,
        )

        return True

    def _connect(self) -> bool:
        self._close_socket()

        sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )
        sock.settimeout(1.0)

        try:
            sock.connect(self.destination)
        except OSError:
            sock.close()
            return False

        sock.settimeout(None)
        self.socket = sock

        logger.info(
            "Connected to GT trajectory server at "
            "tcp://%s:%d",
            self.destination[0],
            self.destination[1],
        )

        return True

    def _send(
        self,
        message: dict[str, Any],
    ) -> None:
        if self.socket is None:
            raise ConnectionError(
                "GT trajectory server is not connected"
            )

        payload = json.dumps(
            message,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

        if len(payload) > 0xFFFFFFFF:
            raise ValueError(
                "GT trajectory payload exceeds uint32 framing"
            )

        packet = (
            struct.pack("!I", len(payload))
            + payload
        )

        self.socket.sendall(packet)

        logger.info(
            "Sent recording GT trajectory: "
            "scene=%s, points=%d, bytes=%d",
            message["scene_id"],
            len(message["points"]),
            len(payload),
        )

    def _worker(self) -> None:
        pending: dict[str, Any] | None = None

        while not self.stop_event.is_set():
            if pending is None:
                try:
                    pending = self.queue.get(
                        timeout=0.1
                    )
                except queue.Empty:
                    continue

            if self.socket is None and not self._connect():
                time.sleep(0.5)
                continue

            try:
                self._send(pending)
                pending = None

            except (
                OSError,
                ConnectionError,
            ) as exc:
                logger.warning(
                    "GT trajectory send failed; "
                    "retaining data for retry: %s",
                    exc,
                )
                self._close_socket()
                time.sleep(0.5)

            except Exception:
                logger.exception(
                    "Unexpected GT trajectory send error; "
                    "retaining data for retry"
                )
                self._close_socket()
                time.sleep(0.5)

    def _close_socket(self) -> None:
        if self.socket is not None:
            try:
                self.socket.close()
            except OSError:
                pass

            self.socket = None


ground_truth_tcp_exporter = GroundTruthTcpExporter()
