from copy import deepcopy
from pathlib import Path
import shutil

import yaml


CONFIG_PATH = Path(
    "/home/lab/alpasim/runs/ros2_pub/generated-user-config-0.yaml"
)

CAMERA_IDS = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_rear_left_70fov",
]

FRONT_ID = "camera_front_wide_120fov"

EXTRINSICS = {
    "camera_cross_left_120fov": {
        "translation_m": [1.646354, 0.143369, 1.521469],
        "rotation_xyzw": [0.679354, -0.207915, 0.215233, -0.670018],
    },
    "camera_front_wide_120fov": {
        "translation_m": [1.670100, -0.025875, 1.522623],
        "rotation_xyzw": [0.509222, -0.503331, 0.495086, -0.492180],
    },
    "camera_cross_right_120fov": {
        "translation_m": [1.626168, -0.161517, 1.526269],
        "rotation_xyzw": [0.205424, -0.674057, 0.676355, -0.214458],
    },
    "camera_rear_left_70fov": {
        "translation_m": [-0.486641, -0.000595, 1.486321],
        "rotation_xyzw": [0.503851, 0.497823, -0.499723, -0.498582],
    },
}


def find_camera_parent(value, path="root"):
    """Find a dict containing a cameras list with the current front camera."""
    if isinstance(value, dict):
        cameras = value.get("cameras")

        if isinstance(cameras, list):
            logical_ids = [
                item.get("logical_id")
                for item in cameras
                if isinstance(item, dict)
            ]

            print(f"Found cameras list at {path}.cameras: {logical_ids}")

            if FRONT_ID in logical_ids:
                return value, path

        for key, child in value.items():
            result = find_camera_parent(child, f"{path}.{key}")
            if result is not None:
                return result

    elif isinstance(value, list):
        for index, child in enumerate(value):
            result = find_camera_parent(child, f"{path}[{index}]")
            if result is not None:
                return result

    return None


with CONFIG_PATH.open("r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)

result = find_camera_parent(config)

if result is None:
    print()
    print("ERROR: No cameras list containing the front camera was found.")
    print("Top-level keys:", list(config.keys()))
    print()
    print("The configuration was not modified.")
    raise SystemExit(1)

camera_parent, camera_parent_path = result
existing_cameras = camera_parent["cameras"]

front_camera = next(
    item
    for item in existing_cameras
    if item.get("logical_id") == FRONT_ID
)

# Copy the current front-camera resolution and timing to all four cameras.
new_cameras = []

for logical_id in CAMERA_IDS:
    camera = deepcopy(front_camera)
    camera["logical_id"] = logical_id
    new_cameras.append(camera)

camera_parent["cameras"] = new_cameras

# extra_cameras is expected next to cameras in simulation_config.
existing_extra = config.get("extra_cameras", [])

unrelated_extra = [
    item
    for item in existing_extra
    if isinstance(item, dict)
    and item.get("logical_id") not in CAMERA_IDS
]

new_extra = [
    {
        "logical_id": logical_id,
        "rig_to_camera": EXTRINSICS[logical_id],
    }
    for logical_id in CAMERA_IDS
]

config["extra_cameras"] = unrelated_extra + new_extra

backup_path = CONFIG_PATH.with_suffix(
    CONFIG_PATH.suffix + ".before_four_cameras"
)

if not backup_path.exists():
    shutil.copy2(CONFIG_PATH, backup_path)

with CONFIG_PATH.open("w", encoding="utf-8") as stream:
    yaml.safe_dump(
        config,
        stream,
        sort_keys=False,
        default_flow_style=False,
    )

print()
print(f"Updated camera configuration at: {camera_parent_path}")
print(f"Updated file: {CONFIG_PATH}")
print(f"Backup file:  {backup_path}")
print()
print("Configured cameras:")

for camera in new_cameras:
    print(
        f"  {camera['logical_id']}: "
        f"{camera.get('width')}x{camera.get('height')}, "
        f"interval={camera.get('frame_interval_us')} us"
    )
