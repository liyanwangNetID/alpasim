# SPDX-License-Identifier: Apache-2.0

"""ROS 2 subscriber for externally planned ego trajectories."""

from __future__ import annotations

import logging
import math

import numpy as np
import rclpy
from alpasim_msgs.msg import EgoTrajectory
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from .external_trajectory_buffer import ExternalTrajectoryBuffer


logger = logging.getLogger(__name__)


class ExternalTrajectoryRosSubscriber(Node):
    """Write ROS planning trajectories into an ExternalTrajectoryBuffer."""

    def __init__(
        self,
        trajectory_buffer: ExternalTrajectoryBuffer,
        topic_name: str = "/alpasim/planning/ego/trajectory",
    ) -> None:
        super().__init__(
            "alpasim_external_trajectory_subscriber"
        )

        self._trajectory_buffer = trajectory_buffer
        self._topic_name = str(topic_name)
        self._message_count = 0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._subscription = self.create_subscription(
            EgoTrajectory,
            self._topic_name,
            self._trajectory_callback,
            qos,
        )

        self.get_logger().info(
            "Listening for external planning trajectories on "
            f"{self._topic_name}"
        )

    def _trajectory_callback(
        self,
        message: EgoTrajectory,
    ) -> None:
        """Validate and copy one ROS trajectory into the shared buffer."""
        if len(message.points) == 0:
            self.get_logger().warning(
                "Ignoring empty external planning trajectory"
            )
            return

        if message.pose_frame_id != "base_link":
            self.get_logger().warning(
                "Ignoring planning trajectory in frame "
                f"{message.pose_frame_id!r}; "
                "ExternalTrajectoryModel currently expects base_link"
            )
            return

        trajectory_xy = np.asarray(
            [
                [
                    float(point.pose.position.x),
                    float(point.pose.position.y),
                ]
                for point in message.points
            ],
            dtype=np.float64,
        )

        headings = np.asarray(
            [
                self._point_heading(point)
                for point in message.points
            ],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(trajectory_xy)):
            self.get_logger().warning(
                "Ignoring trajectory containing non-finite positions"
            )
            return

        if not np.all(np.isfinite(headings)):
            self.get_logger().warning(
                "Ignoring trajectory containing non-finite headings"
            )
            return

        try:
            sequence_id = self._trajectory_buffer.update(
                trajectory_xy=trajectory_xy,
                headings=headings,
            )
        except ValueError as exc:
            self.get_logger().warning(
                f"Rejected external planning trajectory: {exc}"
            )
            return

        self._message_count += 1

        if self._message_count == 1:
            self.get_logger().info(
                "Received first external planning trajectory: "
                f"sequence_id={sequence_id}, "
                f"points={len(trajectory_xy)}, "
                f"final=({trajectory_xy[-1, 0]:.3f}, "
                f"{trajectory_xy[-1, 1]:.3f})"
            )

    @staticmethod
    def _point_heading(point) -> float:
        """Read yaw directly, with quaternion fallback."""
        yaw = float(point.yaw)

        if math.isfinite(yaw):
            return yaw

        quaternion = point.pose.orientation

        x = float(quaternion.x)
        y = float(quaternion.y)
        z = float(quaternion.z)
        w = float(quaternion.w)

        sin_yaw = 2.0 * (w * z + x * y)
        cos_yaw = 1.0 - 2.0 * (y * y + z * z)

        return math.atan2(
            sin_yaw,
            cos_yaw,
        )


def start_external_trajectory_ros_subscriber(
    trajectory_buffer: ExternalTrajectoryBuffer,
    topic_name: str = "/alpasim/planning/ego/trajectory",
) -> tuple[
    ExternalTrajectoryRosSubscriber,
    rclpy.executors.SingleThreadedExecutor,
]:
    """Create the ROS node and executor.

    The caller is responsible for spinning the returned executor in
    a dedicated thread.
    """
    if not rclpy.ok():
        rclpy.init(args=None)

    node = ExternalTrajectoryRosSubscriber(
        trajectory_buffer=trajectory_buffer,
        topic_name=topic_name,
    )

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    return node, executor
