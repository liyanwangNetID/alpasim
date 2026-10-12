# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import io
import json
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest

from alpasim_utils.static_environment import (
    FLAG_GROUND_REFERENCE_AVAILABLE,
    FLAG_OUTSIDE_GROUND_COVERAGE,
    FLAG_TALL_COLUMN,
    StaticEnvironmentConfig,
    _minimum_z_keep_mask,
    extract_static_environment,
    iter_ply_vertices,
)


def _ply(vertices: list[tuple[float, float, float]]) -> bytes:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "element face 0\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    body = b"".join(struct.pack("<fff", *vertex) for vertex in vertices)
    return header + body


def _write_usdz(path: Path, *, include_clipgt: bool = False) -> None:
    ground = [
        (0.05, 0.05, 0.0),
        (0.35, 0.05, 0.0),
        (0.05, 0.35, 0.0),
        (0.35, 0.35, 0.0),
    ]
    full = [
        *ground,
        (0.05, 0.05, 0.3),
        (0.15, 0.15, 0.8),
        (0.25, 0.25, 1.3),
        (0.35, 0.35, 1.8),
        (2.05, 2.05, 0.0),
        (2.15, 2.15, 0.5),
        (2.25, 2.25, 1.0),
    ]
    metadata = {
        "scene_id": "clipgt-test-static-environment",
        "uuid": "test-static-environment",
        "version_string": "test/0.1",
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("metadata.yaml", json.dumps(metadata))
        archive.writestr("mesh.ply", _ply(full))
        archive.writestr("mesh_ground.ply", _ply(ground))
        if include_clipgt:
            pd = pytest.importorskip("pandas")
            buffer = io.BytesIO()
            pd.DataFrame({
                "key": [{
                    "clip_id": "test",
                    "label_class_id": "minimap:poles:autolabels:v0",
                    "map_id": "7",
                    "map_id_version": "1",
                }],
                "pole": [{
                    "category": "TREE",
                    "location": [
                        {"x": 2.0, "y": 2.0, "z": 0.0},
                        {"x": 2.0, "y": 2.0, "z": 3.0},
                    ],
                    "egomotion_label_class_id": "gtaas:test:v0",
                }],
                "version": np.asarray([0], dtype=np.uint64),
            }).to_parquet(buffer, index=False)
            archive.writestr("clipgt/pole.parquet", buffer.getvalue())


def test_iter_ply_vertices_reads_only_vertices(tmp_path: Path) -> None:
    usdz = tmp_path / "scene.usdz"
    _write_usdz(usdz)
    with zipfile.ZipFile(usdz) as archive:
        chunks = list(iter_ply_vertices(archive, "mesh.ply", chunk_size=4))
    vertices = np.concatenate(chunks)
    assert vertices.shape == (11, 3)
    np.testing.assert_allclose(vertices[-1], [2.25, 2.25, 1.0])


def test_extract_columns_filters_ground_and_sets_flags(tmp_path: Path) -> None:
    usdz = tmp_path / "scene.usdz"
    output = tmp_path / "output"
    _write_usdz(usdz)
    result = extract_static_environment(
        usdz,
        output,
        StaticEnvironmentConfig(
            grid_size_xy_m=0.4,
            min_points_per_cell=3,
            min_column_height_m=0.1,
            ground_height_tolerance_m=0.2,
            ground_query_radius_m=0.6,
            tall_column_threshold_m=1.0,
            vertex_chunk_size=3,
        ),
    )
    arrays = np.load(result.columns_path)
    assert result.column_count == 2
    assert arrays["grid_i"].tolist() == [0, 5]
    flags = arrays["flags"].astype(int).tolist()
    assert flags[0] & int(FLAG_GROUND_REFERENCE_AVAILABLE)
    assert flags[0] & int(FLAG_TALL_COLUMN)
    assert flags[1] & int(FLAG_OUTSIDE_GROUND_COVERAGE)
    summary = json.loads(result.summary_path.read_text())
    assert summary["ground_filtered_vertices"] == 4
    assert summary["candidate_vertices"] == 7


def test_refuses_nonempty_output_without_overwrite(tmp_path: Path) -> None:
    usdz = tmp_path / "scene.usdz"
    output = tmp_path / "output"
    _write_usdz(usdz)
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        extract_static_environment(usdz, output)


def test_minimum_z_filter_rejects_deep_geometry() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, -20.0],
            [0.0, 0.0, -5.01],
            [0.0, 0.0, -5.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    mask = _minimum_z_keep_mask(points, min_allowed_z_m=-5.0)
    assert mask.tolist() == [False, False, True, True]



def test_default_output_has_no_semantic_label_file(tmp_path: Path) -> None:
    usdz = tmp_path / "scene.usdz"
    output = tmp_path / "output"
    _write_usdz(usdz)
    result = extract_static_environment(usdz, output)
    assert result.column_count > 0
    assert not (output / "labeled_elements.jsonl").exists()
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["representation"] == "single_interval_2p5d_static_occupancy"
    assert metadata["semantic_labels"] is False
