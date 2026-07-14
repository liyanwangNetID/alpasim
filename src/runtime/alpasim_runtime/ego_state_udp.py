from __future__ import annotations

import json
import math
import socket

import numpy as np


def yaw_from_quaternion_xyzw(quaternion) -> float:
    qx, qy, qz, qw = (float(value) for value in quaternion)

    sin_yaw = 2.0 * (qw * qz + qx * qy)
    cos_yaw = 1.0 - 2.0 * (qy * qy + qz * qz)

    return math.atan2(sin_yaw, cos_yaw)


class EgoStateUdpExporter:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 15000,
    ):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.destination = (host, port)

    def publish(self, ego_trajectory) -> None:
        timestamps = np.asarray(ego_trajectory.timestamps_us)
        positions = np.asarray(ego_trajectory.positions)
        quaternions = np.asarray(ego_trajectory.quaternions)
        dynamics = np.asarray(ego_trajectory.dynamics)

        if len(timestamps) == 0:
            return

        timestamp_us = int(timestamps[-1])
        position = positions[-1]
        quaternion = quaternions[-1]
        dynamic_state = dynamics[-1]

        # DynamicTrajectory dynamics layout:
        # [linear_velocity(3),
        #  angular_velocity(3),
        #  linear_acceleration(3),
        #  angular_acceleration(3)]
        linear_velocity = dynamic_state[0:3]
        angular_velocity = dynamic_state[3:6]
        linear_acceleration = dynamic_state[6:9]
        angular_acceleration = dynamic_state[9:12]

        speed = float(np.linalg.norm(linear_velocity))

        message = {
            "timestamp_us": timestamp_us,
            "frame_id": "alpasim_local",

            "position": {
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2]),
            },

            "orientation": {
                "x": float(quaternion[0]),
                "y": float(quaternion[1]),
                "z": float(quaternion[2]),
                "w": float(quaternion[3]),
            },

            "yaw": yaw_from_quaternion_xyzw(quaternion),

            "linear_velocity": {
                "x": float(linear_velocity[0]),
                "y": float(linear_velocity[1]),
                "z": float(linear_velocity[2]),
            },

            "angular_velocity": {
                "x": float(angular_velocity[0]),
                "y": float(angular_velocity[1]),
                "z": float(angular_velocity[2]),
            },

            "linear_acceleration": {
                "x": float(linear_acceleration[0]),
                "y": float(linear_acceleration[1]),
                "z": float(linear_acceleration[2]),
            },

            "angular_acceleration": {
                "x": float(angular_acceleration[0]),
                "y": float(angular_acceleration[1]),
                "z": float(angular_acceleration[2]),
            },

            "speed": speed,
        }

        payload = json.dumps(
            message,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

        self.socket.sendto(payload, self.destination)


ego_state_udp_exporter = EgoStateUdpExporter()