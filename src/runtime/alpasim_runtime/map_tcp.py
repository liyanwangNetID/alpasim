"""Reliable asynchronous TCP exporter for AlpaSim vector maps."""

from __future__ import annotations

import json
import logging
import math
import queue
import socket
import struct
import threading
import time
from typing import Any, Iterable

import numpy as np
from trajdata.maps.vec_map_elements import MapElementType

logger = logging.getLogger(__name__)


def _point_to_dict(point: Iterable[Any]) -> dict[str, float]:
    """Convert the first three point coordinates to a JSON dictionary."""
    values = list(point)

    if len(values) < 3:
        raise ValueError(
            f"Map point must have at least three coordinates, got {values}"
        )

    x = float(values[0])
    y = float(values[1])
    z = float(values[2])

    if not all(math.isfinite(value) for value in (x, y, z)):
        raise ValueError(
            f"Map point contains non-finite coordinates: {(x, y, z)}"
        )

    return {
        "x": x,
        "y": y,
        "z": z,
    }


def _valid_ids(values: Iterable[Any]) -> list:
    """Convert map references to sorted strings and remove invalid -1 IDs."""
    ids = {
        str(value)
        for value in values
        if value is not None and str(value) != "-1"
    }

    return sorted(ids)


def _association_ids(values: Any) -> list:
    """Normalize list, set, tuple, or dict map associations to string IDs."""
    if values is None:
        return []

    if isinstance(values, dict):
        # Current lane associations are normally list[str], but retaining both
        # keys and string values makes the exporter tolerant to map variants.
        candidates: list[Any] = list(values.keys())

        for value in values.values():
            if isinstance(value, (str, int)):
                candidates.append(value)

        return _valid_ids(candidates)

    if isinstance(values, (str, int)):
        return _valid_ids([values])

    try:
        return _valid_ids(values)
    except TypeError:
        return []


def serialize_polyline(polyline: Any) -> dict[str, Any]:
    """Serialize a trajdata Polyline.

    Supported layouts:
      [x, y, z]
      [x, y, z, heading]

    Headings are empty when the fourth coordinate is unavailable.
    """
    if polyline is None:
        return {
            "points": [],
            "headings": [],
        }

    points = np.asarray(polyline.points)

    if points.ndim != 2:
        raise ValueError(
            f"Polyline points must be a 2D array, got shape {points.shape}"
        )

    if points.shape[1] < 3:
        raise ValueError(
            "Polyline points must contain at least x, y, z; "
            f"got shape {points.shape}"
        )

    serialized_points = [
        _point_to_dict(row)
        for row in points
    ]

    headings: list[float] = []

    if points.shape[1] >= 4:
        headings = [
            float(value)
            for value in points[:, 3]
        ]

        if not all(math.isfinite(value) for value in headings):
            raise ValueError("Polyline contains non-finite headings")

    return {
        "points": serialized_points,
        "headings": headings,
    }


def serialize_lane(lane: Any) -> dict[str, Any]:
    """Serialize one trajdata RoadLane."""
    return {
        "id": str(lane.id),

        "centerline": serialize_polyline(
            lane.center
        ),
        "left_boundary": serialize_polyline(
            lane.left_edge
        ),
        "right_boundary": serialize_polyline(
            lane.right_edge
        ),

        "successor_ids": _valid_ids(
            lane.next_lanes
        ),
        "predecessor_ids": _valid_ids(
            lane.prev_lanes
        ),
        "left_adjacent_ids": _valid_ids(
            lane.adj_lanes_left
        ),
        "right_adjacent_ids": _valid_ids(
            lane.adj_lanes_right
        ),

        "traffic_sign_ids": _association_ids(
            lane.traffic_sign_ids
        ),
        "wait_line_ids": _association_ids(
            lane.wait_line_ids
        ),
        "road_area_ids": _association_ids(
            lane.road_area_ids
        ),
    }


def serialize_road_edge(road_edge: Any) -> dict[str, Any]:
    """Serialize one trajdata RoadEdge."""
    return {
        "id": str(road_edge.id),
        "polyline": serialize_polyline(
            road_edge.polyline
        ),
    }


def serialize_traffic_sign(traffic_sign: Any) -> dict[str, Any]:
    """Serialize one trajdata TrafficSign."""
    return {
        "id": str(traffic_sign.id),
        "sign_type": str(traffic_sign.sign_type),
        "position": _point_to_dict(
            traffic_sign.position
        ),
    }


def serialize_wait_line(wait_line: Any) -> dict[str, Any]:
    """Serialize one trajdata WaitLine."""
    return {
        "id": str(wait_line.id),
        "wait_line_type": str(
            wait_line.wait_line_type
        ),
        "is_implicit": bool(
            wait_line.is_implicit
        ),
        "polyline": serialize_polyline(
            wait_line.polyline
        ),
    }


def serialize_vector_map(
    vector_map: Any,
    scene_id: str,
    frame_id: str = "map",
) -> dict[str, Any]:
    """Serialize the complete Runtime VectorMap to a JSON-compatible dict."""
    if vector_map is None:
        raise ValueError(
            f"Scene {scene_id!r} does not have a VectorMap"
        )

    extent = np.asarray(
        vector_map.extent,
        dtype=np.float64,
    ).reshape(-1)

    if extent.size != 6:
        raise ValueError(
            "VectorMap extent must contain "
            "[min_x, min_y, min_z, max_x, max_y, max_z], "
            f"got shape {extent.shape} and values {extent}"
        )

    if not np.all(np.isfinite(extent)):
        raise ValueError(
            f"VectorMap extent contains non-finite values: {extent}"
        )

    minimum = {
        "x": float(extent[0]),
        "y": float(extent[1]),
        "z": float(extent[2]),
    }

    maximum = {
        "x": float(extent[3]),
        "y": float(extent[4]),
        "z": float(extent[5]),
    }

    if any(
        minimum[axis] > maximum[axis]
        for axis in ("x", "y", "z")
    ):
        raise ValueError(
            f"Invalid VectorMap extent: min={minimum}, max={maximum}"
        )

    lanes = sorted(
        vector_map.lanes,
        key=lambda lane: str(lane.id),
    )

    road_edges = sorted(
        vector_map.road_edges,
        key=lambda edge: str(edge.id),
    )

    traffic_sign_collection = vector_map.elements.get(
        MapElementType.TRAFFIC_SIGN,
        {},
    )

    wait_line_collection = vector_map.elements.get(
        MapElementType.WAIT_LINE,
        {},
    )

    traffic_signs = sorted(
        traffic_sign_collection.values(),
        key=lambda sign: str(sign.id),
    )

    wait_lines = sorted(
        wait_line_collection.values(),
        key=lambda wait_line: str(wait_line.id),
    )

    lane_messages = [
        serialize_lane(lane)
        for lane in lanes
    ]

    road_edge_messages = [
        serialize_road_edge(edge)
        for edge in road_edges
    ]

    traffic_sign_messages = [
        serialize_traffic_sign(sign)
        for sign in traffic_signs
    ]

    wait_line_messages = [
        serialize_wait_line(wait_line)
        for wait_line in wait_lines
    ]

    lane_ids = {
        lane["id"]
        for lane in lane_messages
    }

    traffic_sign_ids = {
        sign["id"]
        for sign in traffic_sign_messages
    }

    wait_line_ids = {
        wait_line["id"]
        for wait_line in wait_line_messages
    }

    # Non-fatal integrity checks. Preserve source references but report
    # dangling IDs, because upstream map data can contain partial associations.
    dangling_lane_references: set[str] = set()
    dangling_sign_references: set[str] = set()
    dangling_wait_line_references: set[str] = set()

    for lane in lane_messages:
        for key in (
            "successor_ids",
            "predecessor_ids",
            "left_adjacent_ids",
            "right_adjacent_ids",
        ):
            dangling_lane_references.update(
                reference
                for reference in lane[key]
                if reference not in lane_ids
            )

        dangling_sign_references.update(
            reference
            for reference in lane["traffic_sign_ids"]
            if reference not in traffic_sign_ids
        )

        dangling_wait_line_references.update(
            reference
            for reference in lane["wait_line_ids"]
            if reference not in wait_line_ids
        )

    if dangling_lane_references:
        logger.warning(
            "VectorMap contains %d dangling lane references. "
            "Examples: %s",
            len(dangling_lane_references),
            sorted(dangling_lane_references)[:10],
        )

    if dangling_sign_references:
        logger.warning(
            "VectorMap contains %d dangling traffic-sign references. "
            "Examples: %s",
            len(dangling_sign_references),
            sorted(dangling_sign_references)[:10],
        )

    if dangling_wait_line_references:
        logger.warning(
            "VectorMap contains %d dangling wait-line references. "
            "Examples: %s",
            len(dangling_wait_line_references),
            sorted(dangling_wait_line_references)[:10],
        )

    return {
        "message_type": "vector_map",
        "frame_id": frame_id,
        "scene_id": str(scene_id),
        "map_id": str(vector_map.map_id),

        # Runtime revision. The ROS map server will maintain its own cached
        # revision and may increment it when the received map changes.
        "source_revision": 1,

        "extent": {
            "minimum": minimum,
            "maximum": maximum,
        },

        "lanes": lane_messages,
        "road_edges": road_edge_messages,
        "traffic_signs": traffic_sign_messages,
        "wait_lines": wait_line_messages,
    }


class MapTcpExporter:
    """Reliably send the latest static vector map to the ROS map server.

    Unlike actor state, a map packet is not discarded when the receiver is
    unavailable. The worker retains the pending map and retries until sendall()
    succeeds.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 15003,
        queue_size: int = 1,
    ) -> None:
        self.destination = (host, port)

        self.queue: queue.Queue[
            dict[str, Any]
        ] = queue.Queue(maxsize=queue_size)

        self.socket: socket.socket | None = None
        self.stop_event = threading.Event()

        self.state_lock = threading.Lock()

        # Maps already scheduled during this Runtime process.
        self.scheduled_keys: set[str] = set()

        self.thread = threading.Thread(
            target=self._worker,
            name="alpasim-map-tcp-exporter",
            daemon=True,
        )
        self.thread.start()

    def has_map_for_scene(
        self,
        scene_id: str,
        map_id: str | None = None,
    ) -> bool:
        """Return whether this map was already scheduled for transmission."""
        key = self._map_key(
            scene_id=scene_id,
            map_id=map_id,
        )

        with self.state_lock:
            return key in self.scheduled_keys

    def publish_vector_map(
        self,
        scene_id: str,
        vector_map: Any,
        frame_id: str = "map",
    ) -> bool:
        """Serialize and queue one map unless it is already scheduled.

        Returns:
            True when a new map was queued.
            False when the same scene/map was already scheduled.
        """
        if vector_map is None:
            logger.warning(
                "Cannot export map for scene %s: vector_map is None",
                scene_id,
            )
            return False

        map_id = str(vector_map.map_id)
        key = self._map_key(
            scene_id=scene_id,
            map_id=map_id,
        )

        with self.state_lock:
            if key in self.scheduled_keys:
                return False

            # Reserve before serialization so repeated StepEvents cannot
            # serialize the same large map concurrently.
            self.scheduled_keys.add(key)

        try:
            message = serialize_vector_map(
                vector_map=vector_map,
                scene_id=scene_id,
                frame_id=frame_id,
            )
        except Exception:
            # Allow a later retry if serialization failed.
            with self.state_lock:
                self.scheduled_keys.discard(key)

            logger.exception(
                "Failed to serialize VectorMap for scene %s",
                scene_id,
            )
            raise

        try:
            self.queue.put_nowait(message)
        except queue.Full:
            # Keep only the newest complete map. This is mainly relevant when
            # multiple rollouts are started before the ROS map server connects.
            try:
                replaced = self.queue.get_nowait()

                replaced_key = self._map_key(
                    scene_id=str(
                        replaced.get("scene_id", "")
                    ),
                    map_id=str(
                        replaced.get("map_id", "")
                    ),
                )

                with self.state_lock:
                    self.scheduled_keys.discard(
                        replaced_key
                    )
            except queue.Empty:
                pass

            self.queue.put_nowait(message)

        logger.info(
            "Queued VectorMap for ROS export: "
            "scene=%s, map_id=%s, lanes=%d, road_edges=%d, "
            "traffic_signs=%d, wait_lines=%d",
            scene_id,
            map_id,
            len(message["lanes"]),
            len(message["road_edges"]),
            len(message["traffic_signs"]),
            len(message["wait_lines"]),
        )

        return True

    @staticmethod
    def _map_key(
        scene_id: str,
        map_id: str | None,
    ) -> str:
        return f"{scene_id}|{map_id or ''}"

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
            "Connected to ROS map server at tcp://%s:%d",
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
                "ROS map server is not connected"
            )

        payload = json.dumps(
            message,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

        if len(payload) > 0xFFFFFFFF:
            raise ValueError(
                f"Serialized map is too large for uint32 framing: "
                f"{len(payload)} bytes"
            )

        # Wire format:
        #   uint32 big-endian payload length
        #   UTF-8 JSON payload
        packet = struct.pack(
            "!I",
            len(payload),
        ) + payload

        self.socket.sendall(packet)

        logger.info(
            "Sent VectorMap to ROS map server: "
            "scene=%s, bytes=%d, lanes=%d, road_edges=%d, "
            "traffic_signs=%d, wait_lines=%d",
            message["scene_id"],
            len(payload),
            len(message["lanes"]),
            len(message["road_edges"]),
            len(message["traffic_signs"]),
            len(message["wait_lines"]),
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

            # Important difference from the actor exporter:
            # do not discard pending when the receiver is unavailable.
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
                    "Map TCP send failed; retaining map and retrying: %s",
                    exc,
                )
                self._close_socket()
                time.sleep(0.5)

            except Exception:
                # Serialization has already completed, so unexpected send
                # errors should be visible. Keep the map pending for retry.
                logger.exception(
                    "Unexpected error while sending VectorMap; "
                    "retaining map for retry"
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


map_tcp_exporter = MapTcpExporter()
