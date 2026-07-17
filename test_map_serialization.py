#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alpasim_runtime.map_tcp import serialize_vector_map
from alpasim_utils.artifact import Artifact


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "usdz",
        type=Path,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/lab/alpasim/vector_map_test.json"
        ),
    )

    args = parser.parse_args()

    artifact = Artifact(
        source=str(args.usdz.resolve())
    )

    vector_map = artifact.map

    message = serialize_vector_map(
        vector_map=vector_map,
        scene_id=artifact.scene_id,
        frame_id="map",
    )

    args.output.write_text(
        json.dumps(
            message,
            separators=(",", ":"),
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("Serialized map:")
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
    print(
        "  output bytes:",
        args.output.stat().st_size,
    )
    print("  output:", args.output)


if __name__ == "__main__":
    main()
