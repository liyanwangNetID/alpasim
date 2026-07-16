"""Asynchronous actor-state TCP exporter."""

from __future__ import annotations

import json
import queue
import socket
import struct
import threading
import time
from typing import Any


class ActorTcpExporter:
    """Send actor snapshots without blocking the simulation loop."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 15002,
        queue_size: int = 4,
    ) -> None:
        self.destination = (host, port)

        self.queue: queue.Queue[dict[str, Any]] = queue.Queue(
            maxsize=queue_size
        )

        self.socket: socket.socket | None = None
        self.stop_event = threading.Event()

        self.thread = threading.Thread(
            target=self._worker,
            name="alpasim-actor-tcp-exporter",
            daemon=True,
        )
        self.thread.start()

    def publish(self, message: dict[str, Any]) -> None:
        """Queue the newest actor state without blocking Runtime."""

        try:
            self.queue.put_nowait(message)

        except queue.Full:
            # Actor state is time-sensitive. Drop the oldest packet.
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

        return True

    def _send(self, message: dict[str, Any]) -> None:
        if self.socket is None:
            raise ConnectionError("Actor bridge is not connected")

        payload = json.dumps(
            message,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

        # uint32 packet length followed by JSON payload
        packet = struct.pack("!I", len(payload)) + payload
        self.socket.sendall(packet)

    def _worker(self) -> None:
        pending: dict[str, Any] | None = None

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
                self._send(pending)
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


actor_tcp_exporter = ActorTcpExporter()
