"""Asynchronous TCP exporter for AlpaSim rendered camera images."""

from __future__ import annotations

import json
import queue
import socket
import struct
import threading
import time
from typing import Any, Iterable

from google.protobuf.json_format import MessageToDict


class CameraTcpExporter:
    """Send rendered camera frames and calibration metadata."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 15001,
        queue_size: int = 16,
    ) -> None:
        self.destination = (host, port)

        self.queue: queue.Queue[
            tuple[dict[str, Any], bytes]
        ] = queue.Queue(maxsize=queue_size)

        self.calibrations: dict[str, dict[str, Any]] = {}

        self.socket: socket.socket | None = None
        self.stop_event = threading.Event()

        self.thread = threading.Thread(
            target=self._worker,
            name="alpasim-camera-tcp-exporter",
            daemon=True,
        )
        self.thread.start()

    def set_calibrations(
        self,
        available_cameras: Iterable[Any],
    ) -> None:
        """Store final CameraCatalog definitions indexed by logical ID."""

        calibrations: dict[str, dict[str, Any]] = {}

        for camera in available_cameras:
            camera_dict = MessageToDict(
                camera,
                preserving_proto_field_name=True,
                use_integers_for_enums=False,
            )

            logical_id = str(camera.logical_id)

            calibrations[logical_id] = {
                "logical_id": logical_id,
                "available_camera": camera_dict,
            }

        self.calibrations = calibrations

        print(
            "Camera TCP exporter received calibrations:",
            sorted(self.calibrations),
        )

    def publish(self, image: Any) -> None:
        """Queue one ImageWithMetadata for transmission."""

        logical_id = str(image.camera_logical_id)

        header = {
            "message_type": "camera_frame",
            "camera_logical_id": logical_id,
            "start_timestamp_us": int(image.start_timestamp_us),
            "end_timestamp_us": int(image.end_timestamp_us),
            "camera_metadata": self.calibrations.get(logical_id),
        }

        payload = bytes(image.image_bytes)
        item = (header, payload)

        try:
            self.queue.put_nowait(item)

        except queue.Full:
            # Never block the AlpaSim event loop.
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass

            try:
                self.queue.put_nowait(item)
            except queue.Full:
                pass

    def _connect(self) -> bool:
        self._close_socket()

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)

        try:
            sock.connect(self.destination)
        except OSError:
            sock.close()
            return False

        sock.settimeout(None)
        self.socket = sock

        return True

    def _send(
        self,
        header: dict[str, Any],
        payload: bytes,
    ) -> None:
        if self.socket is None:
            raise ConnectionError("Camera bridge is not connected")

        header_bytes = json.dumps(
            header,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

        packet = (
            struct.pack("!I", len(header_bytes))
            + header_bytes
            + struct.pack("!Q", len(payload))
            + payload
        )

        self.socket.sendall(packet)

    def _worker(self) -> None:
        pending: tuple[dict[str, Any], bytes] | None = None

        while not self.stop_event.is_set():
            if pending is None:
                try:
                    pending = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue

            if self.socket is None and not self._connect():
                pending = None
                time.sleep(0.2)
                continue

            try:
                self._send(*pending)
                pending = None

            except OSError:
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


camera_tcp_exporter = CameraTcpExporter()