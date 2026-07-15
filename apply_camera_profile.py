#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


MANAGED_CAMERA_IDS = {
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_front_tele_30fov",
    "camera_cross_right_120fov",
    "camera_rear_left_70fov",
}


def find_camera_parent(
    value: Any,
    path: str = "root",
) -> tuple[dict[str, Any], str] | None:
    """Find the dictionary containing simulation_config.cameras."""
    if isinstance(value, dict):
        cameras = value.get("cameras")

        if isinstance(cameras, list):
            logical_ids = [
                item.get("logical_id")
                for item in cameras
                if isinstance(item, dict)
            ]

            if logical_ids:
                return value, path

        for key, child in value.items():
            result = find_camera_parent(
                child,
                f"{path}.{key}",
            )

            if result is not None:
                return result

    elif isinstance(value, list):
        for index, child in enumerate(value):
            result = find_camera_parent(
                child,
                f"{path}[{index}]",
            )

            if result is not None:
                return result

    return None


def choose_template_camera(
    cameras: list[dict[str, Any]],
) -> dict[str, Any]:
    """Prefer front-wide as timing/resolution template."""
    for camera in cameras:
        if camera.get("logical_id") == "camera_front_wide_120fov":
            return camera

    if not cameras:
        raise RuntimeError(
            "No existing camera configuration is available as a template"
        )

    return cameras[0]


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--profile",
        required=True,
        help="Camera profile name",
    )

    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="AlpaSim run directory containing generated-user-config-0.yaml",
    )

    parser.add_argument(
        "--profiles-file",
        type=Path,
        default=Path(
            "/home/lab/alpasim/config/camera_profiles.yaml"
        ),
    )

    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    user_config_path = run_dir / "generated-user-config-0.yaml"

    if not user_config_path.exists():
        raise FileNotFoundError(
            f"User config does not exist: {user_config_path}"
        )

    with args.profiles_file.open("r", encoding="utf-8") as stream:
        profile_document = yaml.safe_load(stream)

    profiles = profile_document.get("profiles", {})

    if args.profile not in profiles:
        available = ", ".join(sorted(profiles))

        raise ValueError(
            f"Unknown camera profile {args.profile!r}. "
            f"Available profiles: {available}"
        )

    profile = profiles[args.profile]
    selected_ids = list(profile["cameras"])

    unknown_ids = set(selected_ids) - MANAGED_CAMERA_IDS

    if unknown_ids:
        raise ValueError(
            f"Profile contains unsupported camera IDs: "
            f"{sorted(unknown_ids)}"
        )

    with user_config_path.open("r", encoding="utf-8") as stream:
        user_config = yaml.safe_load(stream)

    result = find_camera_parent(user_config)

    if result is None:
        raise RuntimeError(
            "Could not find a cameras list in generated-user-config-0.yaml"
        )

    camera_parent, camera_parent_path = result

    existing_cameras = camera_parent["cameras"]
    template = choose_template_camera(existing_cameras)

    existing_by_id = {
        camera.get("logical_id"): camera
        for camera in existing_cameras
        if isinstance(camera, dict)
    }

    new_cameras = []

    for logical_id in selected_ids:
        if logical_id in existing_by_id:
            camera = deepcopy(existing_by_id[logical_id])
        else:
            camera = deepcopy(template)
            camera["logical_id"] = logical_id

        new_cameras.append(camera)

    camera_parent["cameras"] = new_cameras

    # extra_cameras belongs at the top level of UserSimulatorConfig.
    #
    # Preserve overrides only for cameras selected by the active profile.
    # front-tele normally uses the scene/NRE native camera definition unless
    # an explicit override already exists.
    existing_extra = user_config.get("extra_cameras", [])

    retained_extra = [
        item
        for item in existing_extra
        if isinstance(item, dict)
        and (
            item.get("logical_id") not in MANAGED_CAMERA_IDS
            or item.get("logical_id") in selected_ids
        )
    ]

    user_config["extra_cameras"] = retained_extra

    backup_path = (
        run_dir
        / "generated-user-config-0.yaml.before-camera-profile"
    )

    if not backup_path.exists():
        shutil.copy2(user_config_path, backup_path)

    with user_config_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            user_config,
            stream,
            sort_keys=False,
            default_flow_style=False,
        )

    active_profile = {
        "profile": args.profile,
        "description": profile.get("description", ""),
        "camera_logical_ids": selected_ids,
        "config_path": str(user_config_path),
        "camera_parent_path": camera_parent_path,
    }

    active_profile_path = run_dir / "active_camera_profile.json"

    active_profile_path.write_text(
        json.dumps(active_profile, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Applied camera profile: {args.profile}")
    print(f"Config: {user_config_path}")
    print(f"Active profile metadata: {active_profile_path}")
    print()

    for camera in new_cameras:
        print(
            f"  {camera['logical_id']}: "
            f"{camera.get('width')}x{camera.get('height')}, "
            f"interval={camera.get('frame_interval_us')} us"
        )

    if retained_extra:
        print()
        print("Retained camera overrides:")

        for camera in retained_extra:
            print(f"  {camera.get('logical_id')}")


if __name__ == "__main__":
    main()
