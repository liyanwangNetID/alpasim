# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
"""Static-environment extraction from AlpaSim USDZ artifacts.

The first implementation produces one vertical interval per occupied XY cell.
It reads PLY members directly from the USDZ archive, avoids loading face data,
and preserves supported ClipGT static labels independently from mesh-derived
``unknown_static`` columns.
"""
from __future__ import annotations

import io
import json
import logging
import math
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "0.1"
FLAG_GROUND_REFERENCE_AVAILABLE = np.uint16(1 << 0)
FLAG_OUTSIDE_GROUND_COVERAGE = np.uint16(1 << 1)
FLAG_TALL_COLUMN = np.uint16(1 << 2)


@dataclass(frozen=True)
class StaticEnvironmentConfig:
    grid_size_xy_m: float = 0.4
    min_points_per_cell: int = 3
    min_column_height_m: float = 0.10
    min_allowed_z_m: float = -2.0
    ground_height_tolerance_m: float = 0.20
    ground_query_radius_m: float = 0.60
    tall_column_threshold_m: float = 30.0
    vertex_chunk_size: int = 250_000

    def validate(self) -> None:
        positive = {
            "grid_size_xy_m": self.grid_size_xy_m,
            "min_column_height_m": self.min_column_height_m,
            "ground_height_tolerance_m": self.ground_height_tolerance_m,
            "ground_query_radius_m": self.ground_query_radius_m,
            "tall_column_threshold_m": self.tall_column_threshold_m,
            "vertex_chunk_size": self.vertex_chunk_size,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.min_points_per_cell <= 0:
            raise ValueError(
                "min_points_per_cell must be positive, "
                f"got {self.min_points_per_cell}"
            )
        if not math.isfinite(self.min_allowed_z_m):
            raise ValueError(
                "min_allowed_z_m must be finite, "
                f"got {self.min_allowed_z_m}"
            )


@dataclass(frozen=True)
class PlyVertexHeader:
    vertex_count: int
    vertex_dtype: np.dtype


@dataclass
class ColumnAccumulator:
    min_z: float
    max_z: float
    point_count: int
    ground_reference_count: int

    def update(self, z: float, has_ground_reference: bool) -> None:
        self.min_z = min(self.min_z, z)
        self.max_z = max(self.max_z, z)
        self.point_count += 1
        self.ground_reference_count += int(has_ground_reference)


@dataclass(frozen=True)
class ExtractionResult:
    output_dir: Path
    metadata_path: Path
    columns_path: Path
    summary_path: Path
    column_count: int


def _parse_ply_vertex_header(stream: BinaryIO) -> PlyVertexHeader:
    vertex_count: int | None = None
    current_element: str | None = None
    vertex_properties: list[tuple[str, str]] = []
    format_name: str | None = None
    while True:
        raw = stream.readline()
        if not raw:
            raise ValueError("Unexpected EOF while reading PLY header")
        line = raw.decode("ascii", errors="strict").rstrip("\r\n")
        fields = line.split()
        if fields[:1] == ["format"]:
            format_name = fields[1]
        elif len(fields) == 3 and fields[0] == "element":
            current_element = fields[1]
            if current_element == "vertex":
                vertex_count = int(fields[2])
        elif fields[:1] == ["property"] and current_element == "vertex":
            if len(fields) != 3:
                raise ValueError(f"Unsupported PLY vertex property: {line}")
            vertex_properties.append((fields[1], fields[2]))
        elif line == "end_header":
            break
    if format_name != "binary_little_endian":
        raise ValueError(f"Only binary_little_endian PLY is supported, got {format_name}")
    if vertex_count is None:
        raise ValueError("PLY vertex count is missing")
    expected = [("float", "x"), ("float", "y"), ("float", "z")]
    if vertex_properties != expected:
        raise ValueError(
            "Expected PLY vertex properties x/y/z as float32, "
            f"got {vertex_properties}"
        )
    return PlyVertexHeader(vertex_count=vertex_count, vertex_dtype=np.dtype("<f4"))


def iter_ply_vertices(
    archive: zipfile.ZipFile,
    member: str,
    *,
    chunk_size: int,
) -> Iterator[np.ndarray]:
    """Yield ``(N, 3)`` float32 vertex chunks without reading PLY faces."""
    with archive.open(member, "r") as stream:
        header = _parse_ply_vertex_header(stream)
        remaining = header.vertex_count
        while remaining:
            count = min(chunk_size, remaining)
            expected_bytes = count * 3 * header.vertex_dtype.itemsize
            raw = stream.read(expected_bytes)
            if len(raw) != expected_bytes:
                raise ValueError(
                    f"Short vertex read from {member}: "
                    f"expected {expected_bytes}, got {len(raw)}"
                )
            yield np.frombuffer(raw, dtype=header.vertex_dtype).reshape(-1, 3)
            remaining -= count


def _load_ply_vertices(
    archive: zipfile.ZipFile,
    member: str,
    *,
    chunk_size: int,
) -> np.ndarray:
    chunks = [chunk.copy() for chunk in iter_ply_vertices(
        archive, member, chunk_size=chunk_size
    )]
    if not chunks:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _minimum_z_keep_mask(
    points: np.ndarray,
    min_allowed_z_m: float,
) -> np.ndarray:
    """Return a mask that rejects implausibly deep reconstruction geometry."""
    return points[:, 2] >= min_allowed_z_m

def _grid_indices(points_xy: np.ndarray, grid_size: float) -> np.ndarray:
    return np.floor(points_xy / grid_size).astype(np.int32)


def _build_ground_reference(
    ground_vertices: np.ndarray,
    grid_size: float,
) -> tuple[np.ndarray, np.ndarray, Any]:
    if len(ground_vertices) == 0:
        raise ValueError("mesh_ground.ply contains no vertices")
    indices = _grid_indices(ground_vertices[:, :2], grid_size)
    unique, inverse = np.unique(indices, axis=0, return_inverse=True)
    sums = np.zeros(len(unique), dtype=np.float64)
    counts = np.zeros(len(unique), dtype=np.int64)
    np.add.at(sums, inverse, ground_vertices[:, 2])
    np.add.at(counts, inverse, 1)
    mean_z = (sums / counts).astype(np.float32)
    centers = (unique.astype(np.float64) + 0.5) * grid_size
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError(
            "Static-environment extraction requires scipy for ground lookup"
        ) from exc
    return unique, mean_z, cKDTree(centers)


def _accumulate_columns(
    archive: zipfile.ZipFile,
    config: StaticEnvironmentConfig,
) -> tuple[dict[tuple[int, int], ColumnAccumulator], dict[str, int]]:
    ground = _load_ply_vertices(
        archive,
        "mesh_ground.ply",
        chunk_size=config.vertex_chunk_size,
    )
    _, ground_z, ground_tree = _build_ground_reference(
        ground,
        config.grid_size_xy_m,
    )
    columns: dict[tuple[int, int], ColumnAccumulator] = {}
    stats = Counter()
    for points in iter_ply_vertices(
        archive,
        "mesh.ply",
        chunk_size=config.vertex_chunk_size,
    ):
        stats["full_mesh_vertices"] += len(points)
        distances, nearest = ground_tree.query(
            points[:, :2],
            k=1,
            distance_upper_bound=config.ground_query_radius_m,
            workers=-1,
        )
        has_reference = np.isfinite(distances)
        nearest_ground_z = np.full(len(points), np.nan, dtype=np.float32)
        nearest_ground_z[has_reference] = ground_z[nearest[has_reference]]
        above_min_z = _minimum_z_keep_mask(
            points,
            config.min_allowed_z_m,
        )
        is_ground = has_reference & (
            np.abs(points[:, 2] - nearest_ground_z)
            <= config.ground_height_tolerance_m
        )
        keep = ~is_ground & above_min_z
        stats["ground_filtered_vertices"] += int(is_ground.sum())
        stats["below_min_z_filtered_vertices"] += int((~above_min_z).sum())
        stats["candidate_vertices"] += int(keep.sum())
        stats["candidate_vertices_with_ground_reference"] += int(
            (keep & has_reference).sum()
        )
        stats["candidate_vertices_outside_ground_coverage"] += int(
            (keep & ~has_reference).sum()
        )
        candidate_points = points[keep]
        candidate_has_reference = has_reference[keep]
        candidate_cells = _grid_indices(
            candidate_points[:, :2],
            config.grid_size_xy_m,
        )
        for cell, point, has_ref in zip(
            candidate_cells,
            candidate_points,
            candidate_has_reference,
            strict=True,
        ):
            key = (int(cell[0]), int(cell[1]))
            z = float(point[2])
            existing = columns.get(key)
            if existing is None:
                columns[key] = ColumnAccumulator(
                    min_z=z,
                    max_z=z,
                    point_count=1,
                    ground_reference_count=int(has_ref),
                )
            else:
                existing.update(z, bool(has_ref))
    stats["raw_candidate_cells"] = len(columns)
    return columns, dict(stats)


def _finalize_columns(
    columns: dict[tuple[int, int], ColumnAccumulator],
    config: StaticEnvironmentConfig,
) -> tuple[dict[str, np.ndarray], dict[str, int | float]]:
    rows: list[tuple[int, int, float, float, int, int]] = []
    rejected_point_count = 0
    rejected_height = 0
    tall_count = 0
    outside_count = 0
    for (grid_i, grid_j), value in sorted(columns.items()):
        if value.point_count < config.min_points_per_cell:
            rejected_point_count += 1
            continue
        height = value.max_z - value.min_z
        if height < config.min_column_height_m:
            rejected_height += 1
            continue
        flags = np.uint16(0)
        if value.ground_reference_count > 0:
            flags |= FLAG_GROUND_REFERENCE_AVAILABLE
        if value.ground_reference_count < value.point_count:
            flags |= FLAG_OUTSIDE_GROUND_COVERAGE
            outside_count += 1
        if height > config.tall_column_threshold_m:
            flags |= FLAG_TALL_COLUMN
            tall_count += 1
        rows.append((
            grid_i,
            grid_j,
            value.min_z,
            value.max_z,
            value.point_count,
            int(flags),
        ))
    if rows:
        arr = np.asarray(rows, dtype=np.float64)
        arrays = {
            "grid_i": arr[:, 0].astype(np.int32),
            "grid_j": arr[:, 1].astype(np.int32),
            "min_z": arr[:, 2].astype(np.float32),
            "max_z": arr[:, 3].astype(np.float32),
            "point_count": arr[:, 4].astype(np.int32),
            "flags": arr[:, 5].astype(np.uint16),
        }
    else:
        arrays = {
            "grid_i": np.empty(0, dtype=np.int32),
            "grid_j": np.empty(0, dtype=np.int32),
            "min_z": np.empty(0, dtype=np.float32),
            "max_z": np.empty(0, dtype=np.float32),
            "point_count": np.empty(0, dtype=np.int32),
            "flags": np.empty(0, dtype=np.uint16),
        }
    heights = arrays["max_z"] - arrays["min_z"]
    summary: dict[str, int | float] = {
        "column_count": len(arrays["grid_i"]),
        "rejected_by_min_points": rejected_point_count,
        "rejected_by_min_height": rejected_height,
        "outside_ground_coverage_column_count": outside_count,
        "tall_column_count": tall_count,
        "height_min_m": float(heights.min()) if len(heights) else 0.0,
        "height_max_m": float(heights.max()) if len(heights) else 0.0,
        "height_mean_m": float(heights.mean()) if len(heights) else 0.0,
    }
    return arrays, summary


def _load_metadata(archive: zipfile.ZipFile) -> dict[str, Any]:
    import yaml

    metadata = yaml.safe_load(archive.read("metadata.yaml"))
    return metadata if isinstance(metadata, dict) else {}


def extract_static_environment(
    usdz_path: str | Path,
    output_dir: str | Path,
    config: StaticEnvironmentConfig | None = None,
    *,
    overwrite: bool = False,
) -> ExtractionResult:
    """Extract one scene into a compact single-interval 2.5D representation."""
    config = config or StaticEnvironmentConfig()
    config.validate()
    usdz_path = Path(usdz_path)
    output_dir = Path(output_dir)
    if not usdz_path.is_file():
        raise FileNotFoundError(usdz_path)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use overwrite=True."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(usdz_path, "r") as archive:
        names = set(archive.namelist())
        required = {"metadata.yaml", "mesh.ply", "mesh_ground.ply"}
        missing = sorted(required - names)
        if missing:
            raise ValueError(f"USDZ is missing required members: {missing}")
        source_metadata = _load_metadata(archive)
        columns, vertex_stats = _accumulate_columns(archive, config)
        column_arrays, column_summary = _finalize_columns(columns, config)

    columns_path = output_dir / "columns.npz"
    np.savez_compressed(columns_path, **column_arrays)
    scene_id = str(source_metadata.get("scene_id", usdz_path.stem))
    scene_uuid = str(source_metadata.get("uuid", usdz_path.stem))
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "representation": "single_interval_2p5d_static_occupancy",
        "frame_id": "map",
        "scene_id": scene_id,
        "scene_uuid": scene_uuid,
        "source_usdz_name": usdz_path.name,
        "source_version_string": source_metadata.get("version_string"),
        "grid_origin_xy_m": [0.0, 0.0],
        "config": asdict(config),
        "column_encoding": {
            "grid_i": "int32",
            "grid_j": "int32",
            "min_z": "float32_m",
            "max_z": "float32_m",
            "point_count": "int32",
            "flags": "uint16",
            "center_x": "(grid_i + 0.5) * grid_size_xy_m",
            "center_y": "(grid_j + 0.5) * grid_size_xy_m",
            "vertical_gap_preserved": False,
        },
        "flags": {
            str(int(FLAG_GROUND_REFERENCE_AVAILABLE)): "ground_reference_available",
            str(int(FLAG_OUTSIDE_GROUND_COVERAGE)): "at_least_one_source_point_outside_ground_coverage",
            str(int(FLAG_TALL_COLUMN)): "height_exceeds_tall_column_threshold",
        },
        "occupancy_semantics": (
            "static_scene_geometry_excluding_ground where ground reference "
            "is available; known traffic participants are represented separately"
        ),
        "semantic_labels": False,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "scene_id": scene_id,
        "scene_uuid": scene_uuid,
        **vertex_stats,
        **column_summary,
        "columns_file_bytes": columns_path.stat().st_size,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return ExtractionResult(
        output_dir=output_dir,
        metadata_path=metadata_path,
        columns_path=columns_path,
        summary_path=summary_path,
        column_count=int(column_summary["column_count"]),
    )
