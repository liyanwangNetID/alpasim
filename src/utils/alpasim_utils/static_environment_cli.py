# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
"""Command-line entry point for AlpaSim static-environment extraction."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from alpasim_utils.static_environment import (
    StaticEnvironmentConfig,
    extract_static_environment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract compact static-environment data from one AlpaSim USDZ"
    )
    parser.add_argument("usdz", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--grid-size-xy-m", type=float, default=0.4)
    parser.add_argument("--min-points-per-cell", type=int, default=3)
    parser.add_argument("--min-column-height-m", type=float, default=0.10)
    parser.add_argument(
        "--min-allowed-z-m",
        type=float,
        default=-2.0,
        help="Discard mesh vertices below this map-frame Z value",
    )
    parser.add_argument("--ground-height-tolerance-m", type=float, default=0.20)
    parser.add_argument("--ground-query-radius-m", type=float, default=0.60)
    parser.add_argument("--tall-column-threshold-m", type=float, default=30.0)
    parser.add_argument("--vertex-chunk-size", type=int, default=250_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = StaticEnvironmentConfig(
        grid_size_xy_m=args.grid_size_xy_m,
        min_points_per_cell=args.min_points_per_cell,
        min_column_height_m=args.min_column_height_m,
        min_allowed_z_m=args.min_allowed_z_m,
        ground_height_tolerance_m=args.ground_height_tolerance_m,
        ground_query_radius_m=args.ground_query_radius_m,
        tall_column_threshold_m=args.tall_column_threshold_m,
        vertex_chunk_size=args.vertex_chunk_size,
    )
    result = extract_static_environment(
        args.usdz,
        args.output_dir,
        config,
        overwrite=args.overwrite,
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Output: {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
