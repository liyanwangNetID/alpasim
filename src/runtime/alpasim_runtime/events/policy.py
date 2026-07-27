# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Policy event for event-based simulation loop.

Opens the per-step StepContext, gathers observations (egopose, route,
recording ground truth), queries the driver, and writes the transformed
trajectory into the context for downstream pipeline events.
"""

from __future__ import annotations

import logging

import numpy as np
from alpasim_runtime.events.base import (
    EndSimulationException,
    EventPriority,
    EventQueue,
    RecurringEvent,
)
from alpasim_runtime.events.force_gt_utils import controller_reference_trajectory
from alpasim_runtime.events.state import RolloutState, ServiceBundle
from alpasim_runtime.route_generator import RouteGenerator
from alpasim_utils import geometry

from alpasim_runtime.ground_truth_tcp import (
    ground_truth_tcp_exporter,
)
from alpasim_runtime.navigation_tcp import (
    navigation_tcp_exporter,
)

logger = logging.getLogger(__name__)


class PolicyEvent(RecurringEvent):
    """Open per-step context, gather observations, query driver.

    Handles egopose submission, sync tracking, route updates, and recording
    ground-truth submission.  Everything after the driver query (controller,
    physics, traffic, commit) is handled by downstream pipeline events.
    """

    priority: int = EventPriority.POLICY

    def __init__(
        self,
        timestamp_us: int,
        policy_timestep_us: int,
        services: ServiceBundle,
        camera_ids: list[str],
        route_generator: RouteGenerator | None,
        send_recording_ground_truth: bool,
    ):
        super().__init__(timestamp_us=timestamp_us)
        self.interval_us = policy_timestep_us
        self.services = services
        self.camera_ids = camera_ids
        self.route_generator = route_generator
        self.send_recording_ground_truth = send_recording_ground_truth
        self._ground_truth_export_requested = False

    async def run(self, state: RolloutState, queue: EventQueue) -> None:
        step_start_us = self.timestamp_us
        target_time_us = step_start_us + self.interval_us
        svc = self.services

        if not self._ground_truth_export_requested:
            self._ground_truth_export_requested = True

            ground_truth_tcp_exporter.publish(
                trajectory=state.unbound.gt_ego_trajectory,
                scene_id=state.unbound.scene_id,
                frame_id="map",
            )

        # --- Step boundary: fill timing on existing StepContext ---
        assert (
            state.step_context is not None
        ), "StepContext must exist before PolicyEvent (created by StepEvent)"
        state.step_context.step_start_us = step_start_us
        state.step_context.target_time_us = target_time_us
        state.step_context.force_gt = target_time_us in state.unbound.force_gt_period

        # --- Sensor sync validation ---
        if state.unbound.assert_zero_decision_delay:
            assert_sensors_up_to_date(state, step_start_us, self.camera_ids)

        # --- Submit observations concurrently ---
        # Send all egomotion observations since the last update, not just the
        # latest one.  When pose_reporting_interval_us > 0 the controller
        # produces intermediate poses that StepEvent appends to the estimated
        # trajectory.  The driver should receive every one of them.
        all_ts = state.ego_trajectory_estimate.timestamps_us
        if state.last_egopose_update_us is None:
            mask = all_ts <= step_start_us
        else:
            mask = (all_ts > state.last_egopose_update_us) & (all_ts <= step_start_us)
        ts_arr = all_ts[mask]
        if len(ts_arr) == 0:
            ts_arr = np.array([step_start_us], dtype=np.uint64)

        ego_trajectory = state.ego_trajectory_estimate.trajectory().interpolate(ts_arr)
        dynamics_arr = state.ego_trajectory_estimate.interpolate_dynamics(ts_arr)
        dynamic_states_in_rig = geometry.array_to_dynamic_states(dynamics_arr)

        if (self.route_generator is not None or self.send_recording_ground_truth) and (
            state.ego_trajectory.timestamps_us[-1] != step_start_us
        ):
            raise ValueError(
                f"Timestamp mismatch: {state.ego_trajectory.timestamps_us[-1]} "
                f"!= {step_start_us}"
            )

        ctx = state.step_context
        ctx.track_task(
            svc.driver.submit_trajectory(ego_trajectory, dynamic_states_in_rig)
        )

        # if not hasattr(self, "_printed_gt_debug"):
        #     self._printed_gt_debug = True

        #     gt_traj = state.unbound.gt_ego_trajectory

        #     timestamps = gt_traj.timestamps_us.astype(
        #         np.int64
        #     )

        #     intervals_us = np.diff(timestamps)

        #     current_index = int(
        #         np.searchsorted(
        #             timestamps,
        #             step_start_us,
        #             side="left",
        #         )
        #     )

        #     print("\n========== GT EGO AUDIT ==========")
        #     print("type:", type(gt_traj))
        #     print("len:", len(gt_traj))
        #     print("time range:", gt_traj.time_range_us)
        #     print("timestamps shape:", timestamps.shape)
        #     print("positions shape:", gt_traj.positions.shape)
        #     print(
        #         "quaternions shape:",
        #         gt_traj.quaternions.shape,
        #     )

        #     print("first timestamp:", int(timestamps[0]))
        #     print("last timestamp:", int(timestamps[-1]))

        #     print(
        #         "duration sec:",
        #         (int(timestamps[-1]) - int(timestamps[0]))
        #         / 1e6,
        #     )

        #     if len(intervals_us) > 0:
        #         print(
        #             "sample interval us:"
        #             f" min={intervals_us.min()},"
        #             f" median={np.median(intervals_us)},"
        #             f" max={intervals_us.max()}"
        #         )

        #     print("current step timestamp:", step_start_us)
        #     print(
        #         "available future sec:",
        #         (
        #             int(timestamps[-1])
        #             - int(step_start_us)
        #         )
        #         / 1e6,
        #     )
        #     print("current insertion index:", current_index)

        #     if current_index < len(gt_traj):
        #         print(
        #             "current/future position sample:",
        #             gt_traj.positions[
        #                 current_index:
        #                 current_index + 5
        #             ],
        #         )

        #     try:
        #         velocities = gt_traj.velocities()
        #         print(
        #             "velocities shape:",
        #             velocities.shape,
        #         )
        #         print(
        #             "velocities sample:",
        #             velocities[
        #                 current_index:
        #                 current_index + 5
        #             ],
        #         )
        #     except Exception as exc:
        #         print(
        #             "velocities unavailable:",
        #             repr(exc),
        #         )

        #     try:
        #         accelerations = gt_traj.accelerations()
        #         print(
        #             "accelerations shape:",
        #             accelerations.shape,
        #         )
        #     except Exception as exc:
        #         print(
        #             "accelerations unavailable:",
        #             repr(exc),
        #         )

        #     try:
        #         print(
        #             "yaw_rates shape:",
        #             gt_traj.yaw_rates().shape,
        #         )
        #     except Exception as exc:
        #         print(
        #             "yaw rates unavailable:",
        #             repr(exc),
        #         )

        #     print("==================================\n")

        route_model_input = None
        route_map = None

        if self.route_generator is not None:
            pose_local_to_rig = (
                state.ego_trajectory.last_pose
            )

            route_generated = (
                self.route_generator.generate_route(
                    step_start_us,
                    pose_local_to_rig,
                )
            )

            route_model_input = (
                RouteGenerator.prepare_for_policy(
                    route_generated
                )
            )

            # Preserve exactly the same waypoint indexing and
            # validity as the actual Driver input, but express it
            # in the true local/map frame.
            route_map = route_model_input.transform(
                pose_local_to_rig
            )

            ctx.track_task(
                svc.driver.submit_route(
                    step_start_us,
                    route_model_input,
                )
            )

            # route_generated = self.route_generator.generate_route(
            #     step_start_us,
            #     pose_local_to_rig,
            # )

            # route_model_input = RouteGenerator.prepare_for_policy(
            #     route_generated,
            # )

            # ctx.track_task(
            #     svc.driver.submit_route(
            #         step_start_us,
            #         route_model_input,
            #     )
            # )

            # if not hasattr(self, "_printed_route_debug"):
            #     self._printed_route_debug = True

            #     route_map = route_model_input.transform(
            #         pose_local_to_rig
            #     )

            #     print("\n========== ROUTE AUDIT ==========")
            #     print(
            #         "route generator:",
            #         type(self.route_generator).__name__,
            #     )
            #     print(
            #         "configured generator type:",
            #         state.unbound.route_generator_type,
            #     )
            #     print(
            #         "route start offset m:",
            #         state.unbound.route_start_offset_m,
            #     )
            #     print("step_start_us:", step_start_us)

            #     print("\nInternal full route in local/map:")
            #     full_route = (
            #         self.route_generator.route_polyline_in_local
            #     )
            #     print("  type:", type(full_route))
            #     print("  len:", len(full_route))
            #     print("  points shape:", full_route.points.shape)
            #     print("  dimension:", full_route.dimension)
            #     print("  first points:", full_route.points[:5])
            #     print("  last points:", full_route.points[-5:])

            #     print("\nGenerated route in rig:")
            #     print("  type:", type(route_generated))
            #     print("  len:", len(route_generated))
            #     print(
            #         "  points shape:",
            #         route_generated.points.shape,
            #     )
            #     print(
            #         "  points:",
            #         route_generated.points,
            #     )

            #     print("\nPrepared model-input route:")
            #     print("  type:", type(route_model_input))
            #     print("  len:", len(route_model_input))
            #     print(
            #         "  points shape:",
            #         route_model_input.points.shape,
            #     )
            #     print(
            #         "  points:",
            #         route_model_input.points,
            #     )

            #     print("\nPrepared route transformed to map:")
            #     print("  len:", len(route_map))
            #     print("  points shape:", route_map.points.shape)
            #     print("  points:", route_map.points)

            #     print("=================================\n")

        if self.send_recording_ground_truth:
            gt_traj = state.unbound.gt_ego_trajectory
            pose_local_to_rig = state.ego_trajectory.last_pose
            traj_in_rig = gt_traj.transform(pose_local_to_rig.inverse())
            ctx.track_task(
                svc.driver.submit_recording_ground_truth(step_start_us, traj_in_rig)
            )

        # Barrier: all observations (images + egopose + route + GT) must
        # reach the driver before we call drive().
        await ctx.drain_outstanding_tasks()
        state.last_egopose_update_us = step_start_us

        if (
            ctx.force_gt
            and state.unbound.skip_driver_during_force_gt
        ):
            state.data_sensorsim_to_driver = None

            controller_reference = (
                controller_reference_trajectory(
                    state.force_gt_trajectory,
                    step_start_us,
                )
            )

            state.step_context.driver_trajectory = (
                controller_reference
            )

            navigation_tcp_exporter.publish_update(
                reference_timestamp_us=step_start_us,
                route_generator_type=(
                    state.unbound.route_generator_type.name
                ),
                force_gt_active=True,
                route_map=route_map,
                route_model_input=route_model_input,
                # planned_trajectory=controller_reference,
                # plan_source="CONTROLLER_REFERENCE",
                # plan_producer=(
                #     "alpasim_force_gt_controller_reference"
                # ),
                # is_model_generated=False,
            )

            return

        # --- Driver query ---
        drive_trajectory_noisy, terminate_session = await svc.driver.drive(
            time_now_us=step_start_us,
            time_query_us=target_time_us,
            renderer_data=state.data_sensorsim_to_driver,
        )
        state.data_sensorsim_to_driver = None  # Consumed

        if terminate_session:
            logger.info(
                "Driver requested session termination at sim_time=%d us", step_start_us
            )
            raise EndSimulationException()

        # --- Transform from noisy to true local frame ---
        drive_trajectory = (
            transform_trajectory_from_noisy_to_true_local_frame(
                state,
                drive_trajectory_noisy,
            )
        )

        state.step_context.driver_trajectory = (
            drive_trajectory
        )

        navigation_tcp_exporter.publish_update(
            reference_timestamp_us=step_start_us,
            route_generator_type=(
                state.unbound.route_generator_type.name
            ),
            force_gt_active=bool(ctx.force_gt),
            route_map=route_map,
            route_model_input=route_model_input,
            # planned_trajectory=drive_trajectory,
            # plan_source="MODEL_PLANNING",
            # plan_producer="alpasim_driver",
            # is_model_generated=True,
        )

        # if not hasattr(self, "_printed_plan_debug"):
        #     self._printed_plan_debug = True

        #     print("\n========== MODEL PLAN AUDIT ==========")
        #     print("step_start_us:", step_start_us)
        #     print("target_time_us:", target_time_us)
        #     print("force_gt:", ctx.force_gt)

        #     print("\nNoisy driver output:")
        #     print("  type:", type(drive_trajectory_noisy))
        #     print("  len:", len(drive_trajectory_noisy))
        #     print(
        #         "  time range:",
        #         drive_trajectory_noisy.time_range_us,
        #     )
        #     print(
        #         "  timestamps:",
        #         drive_trajectory_noisy.timestamps_us,
        #     )
        #     print(
        #         "  positions shape:",
        #         drive_trajectory_noisy.positions.shape,
        #     )
        #     print(
        #         "  positions sample:",
        #         drive_trajectory_noisy.positions[:5],
        #     )
        #     print(
        #         "  quaternions sample:",
        #         drive_trajectory_noisy.quaternions[:5],
        #     )

        #     print("\nTrue-local model plan:")
        #     print("  len:", len(drive_trajectory))
        #     print(
        #         "  time range:",
        #         drive_trajectory.time_range_us,
        #     )
        #     print(
        #         "  timestamps:",
        #         drive_trajectory.timestamps_us,
        #     )
        #     print(
        #         "  positions sample:",
        #         drive_trajectory.positions[:5],
        #     )
        #     print(
        #         "  quaternions sample:",
        #         drive_trajectory.quaternions[:5],
        #     )

        #     try:
        #         print(
        #             "  velocities shape:",
        #             drive_trajectory.velocities().shape,
        #         )
        #     except Exception as exc:
        #         print(
        #             "  velocities unavailable:",
        #             repr(exc),
        #         )

        #     print("======================================\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def assert_sensors_up_to_date(
    state: RolloutState, step_start_us: int, camera_ids: list[str]
) -> None:
    """Validate that egopose and all camera frames are current before the policy decision."""
    # --- egopose freshness ---
    latest_ego_us = int(state.ego_trajectory_estimate.timestamps_us[-1])
    if latest_ego_us != step_start_us:
        raise ValueError(
            f"Egopose not up to date at {step_start_us}: "
            f"ego_trajectory_estimate latest timestamp is {latest_ego_us}"
        )

    # --- camera freshness ---
    stale = []
    for cid in camera_ids:
        frame_us = state.last_camera_frame_us.get(cid)
        if frame_us is None or frame_us > step_start_us:
            stale.append(cid)
    if stale:
        raise ValueError(f"Cameras not up to date at {step_start_us}: {stale}")


def transform_trajectory_from_noisy_to_true_local_frame(
    state: RolloutState, drive_trajectory_noisy: geometry.Trajectory
) -> geometry.Trajectory:
    """Transform trajectory from noisy local frame to true local frame.

    The driver operates in the "noisy" (estimated) rig frame. To map its output
    into the true local frame we:

    1. Undo the estimated rig frame:  ``T_estimate_inv * traj``
    2. Apply the true rig frame:      ``T_true * result``

    When no egomotion noise model is active, ``ego_trajectory_estimate`` tracks
    ``ego_trajectory`` exactly and the transform is identity.  When noise is
    present the two trajectories diverge and this mapping compensates for the
    drift the driver doesn't know about.
    """
    return drive_trajectory_noisy.transform(
        state.ego_trajectory_estimate.last_pose.inverse()
    ).transform(state.ego_trajectory.last_pose)
