#!/usr/bin/env python3

"""Inspect the VectorMap loaded from an AlpaSim/NuRec USDZ scene."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from alpasim_utils.artifact import Artifact
from trajdata.maps.vec_map_elements import MapElementType


def public_attributes(value: Any) -> list:
    """Return public attribute names for debugging."""
    return [
        name
        for name in dir(value)
        if not name.startswith("_")
    ]


def safe_len(value: Any) -> int | None:
    """Return len(value), or None if the object has no length."""
    try:
        return len(value)
    except (TypeError, AttributeError):
        return None
    
_MISSING = object()
def safe_getattr(
    value: Any,
    attribute: str,
    default: Any = _MISSING,
) -> Any:
    """Read an attribute without allowing a property exception to stop inspection."""
    try:
        return getattr(value, attribute)
    except Exception as exc:
        if default is not _MISSING:
            return default

        print(
            f"  Could not read {type(value).__name__}.{attribute}: "
            f"{exc!r}"
        )
        return _MISSING


def summarize_array(
    name: str,
    value: Any,
    max_rows: int = 5,
) -> None:
    """Print basic information and a small sample from an array-like value."""
    if value is None:
        print(f"  {name}: None")
        return

    try:
        array = np.asarray(value)
    except Exception as exc:
        print(f"  {name}: conversion failed: {exc!r}")
        return

    print(
        f"  {name}: "
        f"type={type(value).__name__}, "
        f"shape={array.shape}, "
        f"dtype={array.dtype}"
    )

    if array.size > 0:
        print(f"  {name} sample:")
        print(array[:max_rows])


def summarize_polyline(
    name: str,
    polyline: Any,
) -> None:
    """Inspect a trajdata polyline-like object safely."""
    if polyline is None:
        print(f"  {name}: None")
        return

    print(f"  {name} type: {type(polyline)}")
    print(f"  {name} attributes: {public_attributes(polyline)}")

    points_array = None

    try:
        points = polyline.points
        points_array = np.asarray(points)

        summarize_array(
            f"{name}.points",
            points,
        )
    except Exception as exc:
        print(
            f"  {name}.points inspection failed: {exc!r}"
        )

    # In trajdata, Polyline.h reads points[..., 3].
    # Lane centerlines may contain [x, y, z, heading], while lane
    # boundaries often contain only [x, y, z]. Therefore do not call
    # polyline.h unless the fourth column exists.
    if (
        points_array is not None
        and points_array.ndim >= 2
        and points_array.shape[-1] >= 4
    ):
        try:
            summarize_array(
                f"{name}.h",
                polyline.h,
            )
        except Exception as exc:
            print(
                f"  {name}.h inspection failed: {exc!r}"
            )
    else:
        print(
            f"  {name}.h: unavailable "
            "(polyline points do not contain a heading column)"
        )

    # Also inspect safe convenience views where applicable.
    for attribute in ["xy", "xyz", "xyh", "xyzh"]:
        try:
            value = getattr(polyline, attribute)
        except (AttributeError, IndexError, ValueError) as exc:
            print(
                f"  {name}.{attribute}: unavailable ({exc})"
            )
            continue
        except Exception as exc:
            print(
                f"  {name}.{attribute}: unavailable ({exc!r})"
            )
            continue

        summarize_array(
            f"{name}.{attribute}",
            value,
            max_rows=3,
        )


def summarize_polygon(
    name: str,
    polygon: Any,
) -> None:
    """Inspect a trajdata polygon-like object."""
    if polygon is None:
        print(f"  {name}: None")
        return

    print(f"  {name} type: {type(polygon)}")
    print(f"  {name} attributes: {public_attributes(polygon)}")

    if hasattr(polygon, "points"):
        summarize_array(
            f"{name}.points",
            polygon.points,
        )

    # Some geometry implementations expose an exterior ring.
    if hasattr(polygon, "exterior"):
        exterior = polygon.exterior
        print(f"  {name}.exterior type: {type(exterior)}")

        if hasattr(exterior, "coords"):
            summarize_array(
                f"{name}.exterior.coords",
                exterior.coords,
            )


def collection_values(collection: Any) -> list:
    """Convert a list-like or dict-like collection into a list of elements."""
    if collection is None:
        return []

    if hasattr(collection, "values"):
        try:
            return list(collection.values())
        except (TypeError, AttributeError):
            return []

    try:
        return list(collection)
    except TypeError:
        return []


def print_object_metadata(item: Any) -> None:
    """Print common metadata fields when present."""
    for attribute in [
        "id",
        "type",
        "lane_type",
        "edge_type",
        "wait_line_type",
        "sign_type",
        "category",
        "speed_limit_mps",
        "is_intersection",
    ]:
        value = safe_getattr(
            item,
            attribute,
            _MISSING,
        )

        if value is _MISSING:
            continue

        print(f"  {attribute}: {value!r}")


def print_relation_metadata(item: Any) -> None:
    """Print lane/topology relation fields when present."""
    for attribute in [
        "next_lanes",
        "prev_lanes",
        "adj_lanes_left",
        "adj_lanes_right",
        "associated_lane_ids",
        "lane_ids",
    ]:
        if not hasattr(item, attribute):
            continue

        try:
            value = getattr(item, attribute)

            if value is None:
                rendered = None
            elif isinstance(value, (str, bytes)):
                rendered = value
            else:
                rendered = sorted(str(entry) for entry in value)

            print(f"  {attribute}: {rendered}")
        except Exception as exc:
            print(f"  {attribute}: <error: {exc!r}>")


def print_generic_sample(
    collection_name: str,
    collection: Any,
) -> None:
    """Print collection statistics and inspect its first element."""
    values = collection_values(collection)

    print()
    print(f"========== {collection_name} ==========")
    print("collection type:", type(collection))
    print("collection len:", safe_len(collection))

    type_counts = Counter(
        type(item).__name__
        for item in values
    )
    print("element type counts:", dict(type_counts))

    if not values:
        print("collection is empty")
        return

    item = values[0]

    print("sample type:", type(item))
    print("sample attributes:", public_attributes(item))
    print("sample repr:", repr(item)[:3000])

    if hasattr(item, "__dict__"):
        print("sample __dict__:", item.__dict__)

    print_object_metadata(item)
    print_relation_metadata(item)

    for attribute in [
        "polyline",
        "center",
        "left_edge",
        "right_edge",
    ]:
        if hasattr(item, attribute):
            try:
                geometry = getattr(item, attribute)
            except Exception as exc:
                print(f"  {attribute}: <error: {exc!r}>")
                continue

            summarize_polyline(
                attribute,
                geometry,
            )

    if hasattr(item, "polygon"):
        try:
            summarize_polygon(
                "polygon",
                item.polygon,
            )
        except Exception as exc:
            print(f"  polygon inspection failed: {exc!r}")

    if hasattr(item, "position"):
        try:
            summarize_array(
                "position",
                item.position,
            )
        except Exception as exc:
            print(f"  position inspection failed: {exc!r}")


def inspect_lanes(vector_map: Any) -> None:
    """Inspect lane collections and one representative lane."""
    lanes = collection_values(vector_map.lanes)

    print()
    print("========== LANES ==========")
    print("lane collection type:", type(vector_map.lanes))
    print("lane count:", len(lanes))

    if not lanes:
        print("lane collection is empty")
        return

    lane = lanes[0]

    print("sample lane type:", type(lane))
    print("sample lane attributes:", public_attributes(lane))
    print("sample lane repr:", repr(lane)[:4000])

    if hasattr(lane, "__dict__"):
        print("sample lane __dict__:", lane.__dict__)

    print_object_metadata(lane)
    print_relation_metadata(lane)

    for attribute in [
        "center",
        "left_edge",
        "right_edge",
    ]:
        if not hasattr(lane, attribute):
            print(f"  {attribute}: attribute unavailable")
            continue

        try:
            geometry = getattr(lane, attribute)
        except Exception as exc:
            print(f"  {attribute}: <error: {exc!r}>")
            continue

        summarize_polyline(
            attribute,
            geometry,
        )


def inspect_map_elements(vector_map: Any) -> None:
    """Inspect all map element collections registered in VectorMap.elements."""
    elements = vector_map.elements

    print()
    print("========== ELEMENT DICTIONARY ==========")
    print("elements type:", type(elements))
    print("element keys:", list(elements.keys()))

    for element_type, collection in elements.items():
        type_name = getattr(
            element_type,
            "name",
            str(element_type),
        )

        print_generic_sample(
            f"MapElementType.{type_name}",
            collection,
        )


def print_known_counts(vector_map: Any) -> None:
    """Print counts for map element types known to be useful for AV tasks."""
    elements = vector_map.elements

    known_types = [
        "ROAD_LANE",
        "ROAD_EDGE",
        "ROAD_AREA",
        "TRAFFIC_SIGN",
        "TRAFFIC_LIGHT",
        "WAIT_LINE",
        "PED_WALKWAY",
        "PED_CROSSWALK",
    ]

    print()
    print("========== KNOWN ELEMENT COUNTS ==========")

    for type_name in known_types:
        if not hasattr(MapElementType, type_name):
            print(
                f"{type_name}: "
                "not defined in this trajdata version"
            )
            continue

        element_type = getattr(
            MapElementType,
            type_name,
        )
        collection = elements.get(
            element_type,
            {},
        )

        print(
            f"{type_name}: "
            f"{safe_len(collection)}"
        )


def inspect_top_level_collections(vector_map: Any) -> None:
    """Print top-level collection types and lengths."""
    print()
    print("========== TOP-LEVEL COLLECTIONS ==========")

    names = [
        "lanes",
        "road_edges",
        "road_areas",
        "ped_crosswalks",
        "ped_walkways",
    ]

    for name in names:
        if not hasattr(vector_map, name):
            print(f"{name}: attribute unavailable")
            continue

        try:
            collection = getattr(vector_map, name)
        except Exception as exc:
            print(f"{name}: <error: {exc!r}>")
            continue

        print(
            f"{name}: "
            f"type={type(collection).__name__}, "
            f"len={safe_len(collection)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the trajdata VectorMap loaded from "
            "an AlpaSim/NuRec USDZ scene."
        ),
    )

    parser.add_argument(
        "usdz",
        type=Path,
        help="Path to one NuRec USDZ scene",
    )

    args = parser.parse_args()
    usdz_path = args.usdz.expanduser().resolve()

    if not usdz_path.is_file():
        raise FileNotFoundError(
            f"USDZ file does not exist: {usdz_path}"
        )

    print("Loading artifact:", usdz_path)

    artifact = Artifact(
        source=str(usdz_path),
    )

    vector_map = artifact.map

    if vector_map is None:
        raise RuntimeError(
            "The selected artifact did not provide a VectorMap"
        )

    print()
    print("========== MAP OBJECT ==========")
    print("scene_id:", artifact.scene_id)
    print("map type:", type(vector_map))
    print("map attributes:", public_attributes(vector_map))
    print("map_id:", getattr(vector_map, "map_id", None))
    print("extent:", getattr(vector_map, "extent", None))

    inspect_top_level_collections(vector_map)
    print_known_counts(vector_map)

    inspect_lanes(vector_map)

    if hasattr(vector_map, "road_edges"):
        print_generic_sample(
            "ROAD EDGES",
            vector_map.road_edges,
        )

    if hasattr(vector_map, "road_areas"):
        print_generic_sample(
            "ROAD AREAS",
            vector_map.road_areas,
        )

    inspect_map_elements(vector_map)

    print()
    print("========== FINAL MAP STATISTICS ==========")

    # Traffic sign type distribution.
    traffic_signs = vector_map.elements.get(
        MapElementType.TRAFFIC_SIGN,
        {},
    )

    sign_type_counts = Counter(
        str(sign.sign_type)
        for sign in traffic_signs.values()
    )

    print("traffic sign type counts:")
    for sign_type, count in sorted(
        sign_type_counts.items()
    ):
        print(f"  {sign_type}: {count}")

    # Wait-line type and implicit/explicit distribution.
    wait_lines = vector_map.elements.get(
        MapElementType.WAIT_LINE,
        {},
    )

    wait_line_type_counts = Counter(
        str(wait_line.wait_line_type)
        for wait_line in wait_lines.values()
    )

    implicit_counts = Counter(
        bool(wait_line.is_implicit)
        for wait_line in wait_lines.values()
    )

    print("wait-line type counts:")
    for wait_line_type, count in sorted(
        wait_line_type_counts.items()
    ):
        print(f"  {wait_line_type}: {count}")

    print("wait-line implicit counts:")
    print(f"  explicit (False): {implicit_counts[False]}")
    print(f"  implicit (True):  {implicit_counts[True]}")

    # Inspect lanes with non-empty sign or wait-line associations.
    lanes_with_signs = [
        lane
        for lane in vector_map.lanes
        if lane.traffic_sign_ids
    ]

    lanes_with_wait_lines = [
        lane
        for lane in vector_map.lanes
        if lane.wait_line_ids
    ]

    print(
        "lanes with traffic-sign associations:",
        len(lanes_with_signs),
    )
    print(
        "lanes with wait-line associations:",
        len(lanes_with_wait_lines),
    )

    if lanes_with_signs:
        lane = lanes_with_signs[0]

        print("sample lane with traffic signs:")
        print("  lane id:", lane.id)
        print(
            "  traffic_sign_ids type:",
            type(lane.traffic_sign_ids),
        )
        print(
            "  traffic_sign_ids value:",
            repr(lane.traffic_sign_ids),
        )

    if lanes_with_wait_lines:
        lane = lanes_with_wait_lines[0]

        print("sample lane with wait lines:")
        print("  lane id:", lane.id)
        print(
            "  wait_line_ids type:",
            type(lane.wait_line_ids),
        )
        print(
            "  wait_line_ids value:",
            repr(lane.wait_line_ids),
        )

    # Topology statistics.
    no_successor_count = sum(
        not {
            lane_id
            for lane_id in lane.next_lanes
            if str(lane_id) != "-1"
        }
        for lane in vector_map.lanes
    )

    no_predecessor_count = sum(
        not {
            lane_id
            for lane_id in lane.prev_lanes
            if str(lane_id) != "-1"
        }
        for lane in vector_map.lanes
    )

    left_adjacent_count = sum(
        any(
            str(lane_id) != "-1"
            for lane_id in lane.adj_lanes_left
        )
        for lane in vector_map.lanes
    )

    right_adjacent_count = sum(
        any(
            str(lane_id) != "-1"
            for lane_id in lane.adj_lanes_right
        )
        for lane in vector_map.lanes
    )

    print("lane topology summary:")
    print(f"  lanes without successor:   {no_successor_count}")
    print(f"  lanes without predecessor: {no_predecessor_count}")
    print(f"  lanes with left adjacent:  {left_adjacent_count}")
    print(f"  lanes with right adjacent: {right_adjacent_count}")

    print("==========================================")


if __name__ == "__main__":
    main()
