#!/usr/bin/env python3

from __future__ import annotations

import json
import socket
import struct
from pathlib import Path


def read_exact(
    connection: socket.socket,
    size: int,
) -> bytes:
    chunks = []
    remaining = size

    while remaining > 0:
        chunk = connection.recv(remaining)

        if not chunk:
            raise ConnectionError(
                "Connection closed before packet completed"
            )

        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def main() -> None:
    output_path = Path(
        "/home/lab/alpasim/received_vector_map.json"
    )

    server = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM,
    )
    server.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1,
    )
    server.bind(("127.0.0.1", 15003))
    server.listen(1)

    print(
        "Waiting for Runtime map exporter at "
        "tcp://127.0.0.1:15003"
    )

    connection, address = server.accept()

    print("Runtime connected from:", address)

    with connection:
        payload_length = struct.unpack(
            "!I",
            read_exact(connection, 4),
        )[0]

        print("Expected payload bytes:", payload_length)

        payload = read_exact(
            connection,
            payload_length,
        )

    message = json.loads(
        payload.decode("utf-8")
    )

    output_path.write_text(
        json.dumps(
            message,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("Received map:")
    print("  scene_id:", message["scene_id"])
    print("  map_id:", message["map_id"])
    print("  lanes:", len(message["lanes"]))
    print(
        "  road_edges:",
        len(message["road_edges"]),
    )
    print(
        "  traffic_signs:",
        len(message["traffic_signs"]),
    )
    print(
        "  wait_lines:",
        len(message["wait_lines"]),
    )
    print("  saved:", output_path)

    server.close()


if __name__ == "__main__":
    main()
