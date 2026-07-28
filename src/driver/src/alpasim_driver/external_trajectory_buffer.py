# SPDX-License-Identifier: Apache-2.0

"""Thread-safe storage for an externally supplied ego trajectory."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectorySnapshot:
    """A snapshot of the latest trajectory."""

    trajectory_xy: np.ndarray
    headings: np.ndarray
    sequence_id: int
    received_monotonic_s: float
    age_s: float


class ExternalTrajectoryBuffer:
    """Thread-safe trajectory buffer.

    A producer updates the trajectory through update().
    ExternalTrajectoryModel reads the trajectory through snapshot().
    """

    def __init__(self, timeout_s: float = 0.5) -> None:
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")

        self._timeout_s = float(timeout_s)
        self._lock = threading.Lock()

        self._trajectory_xy: np.ndarray | None = None
        self._headings: np.ndarray | None = None
        self._received_monotonic_s: float | None = None
        self._sequence_id = 0

    @property
    def timeout_s(self) -> float:
        """Return the trajectory timeout."""

        return self._timeout_s

    def update(
        self,
        trajectory_xy: np.ndarray,
        headings: np.ndarray,
    ) -> int:
        """Validate and store a trajectory.

        Args:
            trajectory_xy:
                Array with shape (N, 2). Coordinates are expressed in the
                current AlpaSim ego rig frame.

            headings:
                Array with shape (N,). Heading values are in radians.

        Returns:
            The assigned sequence ID.
        """

        xy = np.asarray(trajectory_xy, dtype=np.float64)
        yaw = np.asarray(headings, dtype=np.float64)

        if xy.ndim != 2:
            raise ValueError(
                "trajectory_xy must be a two-dimensional array"
            )

        if xy.shape[1] != 2:
            raise ValueError(
                f"trajectory_xy must have shape (N, 2), got {xy.shape}"
            )

        if yaw.ndim != 1:
            raise ValueError(
                f"headings must have shape (N,), got {yaw.shape}"
            )

        if len(xy) == 0:
            raise ValueError(
                "trajectory must contain at least one point"
            )

        if len(xy) != len(yaw):
            raise ValueError(
                "trajectory_xy and headings must have the same length"
            )

        if not np.all(np.isfinite(xy)):
            raise ValueError(
                "trajectory_xy contains NaN or infinity"
            )

        if not np.all(np.isfinite(yaw)):
            raise ValueError(
                "headings contains NaN or infinity"
            )

        stored_xy = np.ascontiguousarray(xy.copy())
        stored_headings = np.ascontiguousarray(yaw.copy())
        received_time = time.monotonic()

        with self._lock:
            self._sequence_id += 1
            self._trajectory_xy = stored_xy
            self._headings = stored_headings
            self._received_monotonic_s = received_time

            return self._sequence_id

    def snapshot(
        self,
        allow_stale: bool = False,
    ) -> TrajectorySnapshot | None:
        """Return a copy of the latest trajectory.

        Returns None if no trajectory has been stored.

        Returns None if the trajectory has expired and allow_stale is False.
        """

        current_time = time.monotonic()

        with self._lock:
            if self._trajectory_xy is None:
                return None

            if self._headings is None:
                return None

            if self._received_monotonic_s is None:
                return None

            age_s = current_time - self._received_monotonic_s

            if age_s > self._timeout_s and not allow_stale:
                return None

            return TrajectorySnapshot(
                trajectory_xy=self._trajectory_xy.copy(),
                headings=self._headings.copy(),
                sequence_id=self._sequence_id,
                received_monotonic_s=self._received_monotonic_s,
                age_s=age_s,
            )

    def clear(self) -> None:
        """Remove the currently stored trajectory."""

        with self._lock:
            self._trajectory_xy = None
            self._headings = None
            self._received_monotonic_s = None