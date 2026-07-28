# SPDX-License-Identifier: Apache-2.0

"""External trajectory model.

The model reads a trajectory from a thread-safe trajectory buffer.

This initial version places one fixed trajectory into the buffer when the
model starts. Later, a ROS 2 subscription callback will update the same
buffer while the Driver is running.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from ..external_trajectory_buffer import ExternalTrajectoryBuffer
from ..schema import ModelConfig
from .base import (
    BaseTrajectoryModel,
    DriveCommand,
    ModelPrediction,
    PredictionInput,
)

logger = logging.getLogger(__name__)


class ExternalTrajectoryModel(BaseTrajectoryModel):
    """Trajectory model backed by an external trajectory buffer."""

    DEFAULT_HORIZON_S = 4.0
    DEFAULT_SPEED_MPS = 2.0
    INITIAL_BUFFER_TIMEOUT_S = 3600.0

    @classmethod
    def from_config(
        cls,
        model_cfg: ModelConfig,
        device: torch.device,
        camera_ids: list[str],
        context_length: int | None,
        output_frequency_hz: int,
    ) -> "ExternalTrajectoryModel":
        """Create an ExternalTrajectoryModel from Driver configuration."""

        del model_cfg

        return cls(
            device=device,
            camera_ids=camera_ids,
            context_length=context_length,
            output_frequency_hz=output_frequency_hz,
            horizon_s=cls.DEFAULT_HORIZON_S,
            target_speed_mps=cls.DEFAULT_SPEED_MPS,
        )

    def __init__(
        self,
        device: torch.device,
        camera_ids: list[str],
        context_length: int | None,
        output_frequency_hz: int,
        horizon_s: float = DEFAULT_HORIZON_S,
        target_speed_mps: float = DEFAULT_SPEED_MPS,
    ) -> None:
        """Initialize the external trajectory model."""

        self._device = device
        self._camera_ids = list(camera_ids)

        if context_length is None:
            self._context_length = 1
        else:
            self._context_length = int(context_length)

        self._output_frequency_hz = int(output_frequency_hz)
        self._horizon_s = float(horizon_s)
        self._target_speed_mps = float(target_speed_mps)

        if self._context_length <= 0:
            raise ValueError(
                "context_length must be positive"
            )

        if self._output_frequency_hz <= 0:
            raise ValueError(
                "output_frequency_hz must be positive"
            )

        if self._horizon_s <= 0.0:
            raise ValueError(
                "horizon_s must be positive"
            )

        waypoint_count_value = np.multiply(
            self._horizon_s,
            self._output_frequency_hz,
        )

        self._num_waypoints = max(
            1,
            int(round(float(waypoint_count_value))),
        )

        self._trajectory_buffer = ExternalTrajectoryBuffer(
            timeout_s=self.INITIAL_BUFFER_TIMEOUT_S,
        )

        initial_xy, initial_headings = (
            self._build_initial_fixed_trajectory()
        )

        sequence_id = self._trajectory_buffer.update(
            trajectory_xy=initial_xy,
            headings=initial_headings,
        )

        logger.info(
            "Initialized ExternalTrajectoryModel: "
            "speed=%.2f m/s, horizon=%.2f s, "
            "frequency=%d Hz, points=%d, sequence_id=%d",
            self._target_speed_mps,
            self._horizon_s,
            self._output_frequency_hz,
            self._num_waypoints,
            sequence_id,
        )

    @property
    def trajectory_buffer(
        self,
    ) -> ExternalTrajectoryBuffer:
        """Return the trajectory buffer shared with external producers."""
        return self._trajectory_buffer

    @property
    def camera_ids(self) -> list:
        """Return camera IDs requested by this model."""

        return self._camera_ids

    @property
    def context_length(self) -> int:
        """Return the required observation history length."""

        return self._context_length

    @property
    def output_frequency_hz(self) -> int:
        """Return the trajectory waypoint frequency."""

        return self._output_frequency_hz

    def _encode_command(
        self,
        command: DriveCommand,
    ) -> torch.Tensor:
        """Encode a semantic command.

        The current external trajectory source does not use route commands.
        This method is implemented because BaseTrajectoryModel requires it.
        """

        del command

        return torch.zeros(
            1,
            dtype=torch.float32,
            device=self._device,
        )

    def _build_initial_fixed_trajectory(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build the initial straight trajectory.

        The trajectory is represented in the current AlpaSim ego rig frame.

        Coordinate convention:
            positive x means forward
            positive y means left
            heading is measured in radians
        """

        waypoint_indices = np.arange(
            1,
            self._num_waypoints + 1,
            dtype=np.float64,
        )

        time_from_start_s = np.divide(
            waypoint_indices,
            float(self._output_frequency_hz),
        )

        x = np.multiply(
            self._target_speed_mps,
            time_from_start_s,
        )

        y = np.zeros_like(x)

        trajectory_xy = np.column_stack(
            (x, y)
        )

        headings = np.zeros(
            self._num_waypoints,
            dtype=np.float64,
        )

        return trajectory_xy, headings

    def _build_stop_trajectory(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a trajectory that keeps the ego vehicle stationary."""

        trajectory_xy = np.zeros(
            (self._num_waypoints, 2),
            dtype=np.float64,
        )

        headings = np.zeros(
            self._num_waypoints,
            dtype=np.float64,
        )

        return trajectory_xy, headings

    def predict(
        self,
        inputs: PredictionInput,
    ) -> ModelPrediction:
        """Return the latest trajectory stored in the buffer."""

        del inputs

        snapshot = self._trajectory_buffer.snapshot()

        if snapshot is None:
            logger.warning(
                "No valid external trajectory is available. "
                "Returning a stop trajectory."
            )

            trajectory_xy, headings = (
                self._build_stop_trajectory()
            )

            return ModelPrediction(
                trajectory_xy=trajectory_xy,
                headings=headings,
            )

        logger.debug(
            "Using external trajectory: "
            "sequence_id=%d, age=%.3f s, points=%d, "
            "final_x=%.2f, final_y=%.2f",
            snapshot.sequence_id,
            snapshot.age_s,
            len(snapshot.trajectory_xy),
            snapshot.trajectory_xy[-1, 0],
            snapshot.trajectory_xy[-1, 1],
        )

        return ModelPrediction(
            trajectory_xy=snapshot.trajectory_xy,
            headings=snapshot.headings,
        )