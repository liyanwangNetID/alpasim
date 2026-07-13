from __future__ import annotations

import json
import math
import socket

import numpy as np


def yaw_from_quaternion_xyzw(q):
    qx, qy, qz, qw = [float(x) for x in q]

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)

    return math.atan2(siny_cosp, cosy_cosp)


class EgoStateUdpExporter:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dst = ("127.0.0.1", 15000)

        self.prev_position = None
        self.prev_timestamp_us = None

    def publish(self, ego_trajectory):

        # 直接使用 DynamicTrajectory 暴露的字段
        timestamps = np.asarray(ego_trajectory.timestamps_us)
        positions = np.asarray(ego_trajectory.positions)
        quaternions = np.asarray(ego_trajectory.quaternions)

        if len(timestamps) == 0:
            return

        timestamp_us = int(timestamps[-1])

        pos = positions[-1]
        quat = quaternions[-1]

        x = float(pos[0])
        y = float(pos[1])
        z = float(pos[2])

        yaw = yaw_from_quaternion_xyzw(quat)

        speed = 0.0

        if (
            self.prev_position is not None
            and self.prev_timestamp_us is not None
            and timestamp_us > self.prev_timestamp_us
        ):
            dt = (timestamp_us - self.prev_timestamp_us) / 1e6

            if dt > 0.0:
                speed = (
                    np.linalg.norm(pos - self.prev_position)
                    / dt
                )

        self.prev_position = np.array(pos)
        self.prev_timestamp_us = timestamp_us

        msg = {
            "timestamp_us": timestamp_us,
            "frame_id": "alpasim_local",
            "x": x,
            "y": y,
            "z": z,
            "yaw": float(yaw),
            "speed": float(speed),
        }

        self.sock.sendto(
            json.dumps(msg).encode("utf-8"),
            self.dst,
        )


ego_state_udp_exporter = EgoStateUdpExporter()