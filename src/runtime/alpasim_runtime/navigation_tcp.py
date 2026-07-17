"""Asynchronous TCP exporter for dynamic navigation state."""

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


def _vector3(
    values: Any,
    field_name: str,
) -> dict[str, float]:
    array = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    if array.size < 3:
        raise ValueError(
            f"{field_name} must contain at least three values, "
            f"got shape {array.shape}"
        )

    result = {
        "x": float(array[0]),
        "y": float(array[1]),
        "z": float(array[2]),
    }

    if not all(
        math.isfinite(value)
        for value in result.values()
    ):
        raise ValueError(
            f"{field_name} contains non-finite values: {result}"
        )

    return result


def _quaternion(
    values: Any,
) -> dict[str, float]:
    array = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    if array.size != 4:
        raise ValueError(
            "Quaternion must contain four values, "
            f"got shape {array.shape}"
        )

    result = {
        "x": float(array[0]),
        "y": float(array[1]),
        "z": float(array[2]),
        "w": float(array[3]),
    }

    if not all(
        math.isfinite(value)
        for value in result.values()
    ):
        raise ValueError(
            f"Quaternion contains non-finite values: {result}"
        )

    return result


def serialize_route(
    polyline: Any,
    *,
    frame_id: str,
    source_frame_id: str,
    lookahead_distance: float,
    expected_point_count: int,
) -> dict[str, Any]:
    """Serialize one current Route while preserving invalid padded points."""
    if polyline is None:
        return {
            "frame_id": frame_id,
            "source_frame_id": source_frame_id,
            "lookahead_distance": float(
                lookahead_distance
            ),
            "expected_point_count": int(
                expected_point_count
            ),
            "points": [],
        }

    points = np.asarray(
        polyline.points,
        dtype=np.float64,
    )

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(
            "Route points must have shape (N, >=3), "
            f"got {points.shape}"
        )

    serialized_points: list[
        dict[str, Any]
    ] = []

    cumulative_distance = 0.0
    previous_valid_point: np.ndarray | None = None

    for point in points:
        valid = bool(
            np.all(np.isfinite(point[:3]))
        )

        if valid:
            xyz = point[:3]

            if previous_valid_point is not None:
                cumulative_distance += float(
                    np.linalg.norm(
                        xyz - previous_valid_point
                    )
                )

            previous_valid_point = xyz.copy()

            position = {
                "x": float(xyz[0]),
                "y": float(xyz[1]),
                "z": float(xyz[2]),
            }

            longitudinal_distance = (
                cumulative_distance
            )

        else:
            # Never place NaN in JSON or ROS geometry fields.
            position = {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
            }
            longitudinal_distance = 0.0

        serialized_points.append(
            {
                "valid": valid,
                "position": position,
                "longitudinal_distance": float(
                    longitudinal_distance
                ),
            }
        )

    return {
        "frame_id": frame_id,
        "source_frame_id": source_frame_id,
        "lookahead_distance": float(
            lookahead_distance
        ),
        "expected_point_count": int(
            expected_point_count
        ),
        "points": serialized_points,
    }


def serialize_trajectory(
    trajectory: Any,
) -> dict[str, Any]:
    """Serialize a map-frame model or controller trajectory."""
    if trajectory is None or len(trajectory) == 0:
        return {
            "start_timestamp_us": 0,
            "end_timestamp_us": 0,
            "points": [],
        }

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

    arrays = {
        "positions": positions,
        "quaternions": quaternions,
        "velocities": velocities,
        "accelerations": accelerations,
        "yaws": yaws,
        "yaw_rates": yaw_rates,
        "yaw_accelerations": yaw_accelerations,
    }

    for name, expected_shape in expected_shapes.items():
        if arrays[name].shape != expected_shape:
            raise ValueError(
                f"{name} has shape {arrays[name].shape}, "
                f"expected {expected_shape}"
            )

    if np.any(np.diff(timestamps) <= 0):
        raise ValueError(
            "Planned trajectory timestamps must be "
            "strictly increasing"
        )

    serialized_points: list[
        dict[str, Any]
    ] = []

    for index in range(count):
        velocity = velocities[index]
        speed = float(
            np.linalg.norm(velocity)
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
                speed,
            )
        ):
            raise ValueError(
                f"Non-finite trajectory state at index {index}"
            )

        serialized_points.append(
            {
                "timestamp_us": int(
                    timestamps[index]
                ),
                "position": _vector3(
                    positions[index],
                    "position",
                ),
                "orientation": _quaternion(
                    quaternions[index]
                ),
                "linear_velocity": _vector3(
                    velocity,
                    "linear_velocity",
                ),
                "linear_acceleration": _vector3(
                    accelerations[index],
                    "linear_acceleration",
                ),
                "yaw": yaw,
                "yaw_rate": yaw_rate,
                "yaw_acceleration": yaw_acceleration,
                "speed": speed,
            }
        )

    return {
        "start_timestamp_us": int(timestamps[0]),
        "end_timestamp_us": int(timestamps[-1]),
        "points": serialized_points,
    }


class NavigationTcpExporter:
    """Send the newest navigation update without blocking Runtime."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 15005,
        queue_size: int = 4,
    ) -> None:
        self.destination = (host, port)

        self.queue: queue.Queue[
            dict[str, Any]
        ] = queue.Queue(maxsize=queue_size)

        self.socket: socket.socket | None = None
        self.stop_event = threading.Event()

        self.sequence_lock = threading.Lock()
        self.next_sequence = 1

        self.thread = threading.Thread(
            target=self._worker,
            name="alpasim-navigation-tcp-exporter",
            daemon=True,
        )
        self.thread.start()

    def publish_update(
        self,
        *,
        reference_timestamp_us: int,
        route_generator_type: str,
        force_gt_active: bool,
        route_map: Any,
        route_model_input: Any,
        planned_trajectory: Any,
        plan_source: str,
        plan_producer: str,
        is_model_generated: bool,
    ) -> None:
        """Queue one complete current navigation update."""
        with self.sequence_lock:
            sequence = self.next_sequence
            self.next_sequence += 1

        route_map_message = None
        route_model_input_message = None

        if route_map is not None:
            route_map_message = serialize_route(
                route_map,
                frame_id="map",
                source_frame_id="base_link",
                lookahead_distance=80.0,
                expected_point_count=20,
            )

        if route_model_input is not None:
            route_model_input_message = serialize_route(
                route_model_input,
                frame_id="base_link",
                source_frame_id="base_link",
                lookahead_distance=80.0,
                expected_point_count=20,
            )

        trajectory_message = serialize_trajectory(
            planned_trajectory
        )

        message = {
            "message_type": "navigation_update",
            "sequence": sequence,
            "reference_timestamp_us": int(
                reference_timestamp_us
            ),
            "route_generator_type": str(
                route_generator_type
            ),
            "force_gt_active": bool(
                force_gt_active
            ),
            "route_map": route_map_message,
            "route_model_input": (
                route_model_input_message
            ),
            "planned_trajectory": {
                "pose_frame_id": "map",
                "dynamics_frame_id": "map",
                "source": str(plan_source),
                "producer": str(plan_producer),
                "is_model_generated": bool(
                    is_model_generated
                ),
                **trajectory_message,
            },
        }

        try:
            self.queue.put_nowait(message)

        except queue.Full:
            # Dynamic policy state: discard stale updates.
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass

            try:
                self.queue.put_nowait(message)
            except queue.Full:
                pass

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
            "Connected to ROS navigation bridge at "
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
                "ROS navigation bridge is not connected"
            )

        payload = json.dumps(
            message,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

        if len(payload) > 0xFFFFFFFF:
            raise ValueError(
                "Navigation payload exceeds uint32 framing"
            )

        packet = (
            struct.pack("!I", len(payload))
            + payload
        )

        self.socket.sendall(packet)

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
                # Dynamic state may be stale by reconnect time.
                pending = None
                time.sleep(0.2)
                continue

            try:
                self._send(pending)
                pending = None

            except (
                OSError,
                ConnectionError,
            ):
                self._close_socket()
                pending = None
                time.sleep(0.1)

    def _close_socket(self) -> None:
        if self.socket is not None:
            try:
                self.socket.close()
            except OSError:
                pass

            self.socket = None


navigation_tcp_exporter = NavigationTcpExporter()
