# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Close the current step and open the next.

Commits all computed state to RolloutState, logs actor poses, and clears
StepContext. This is the only event that mutates trajectory state.
"""

from alpasim_runtime.ego_state_udp import ego_state_udp_exporter
from alpasim_runtime.actor_tcp import actor_tcp_exporter
from alpasim_runtime.map_tcp import map_tcp_exporter

import logging
import time

import numpy as np
from alpasim_grpc.v0.logging_pb2 import ActorPoses, LogEntry
from alpasim_runtime.broadcaster import MessageBroadcaster
from alpasim_runtime.events.base import Event, EventPriority, EventQueue, RecurringEvent
from alpasim_runtime.events.state import RolloutState, ServiceBundle, StepContext
from alpasim_runtime.telemetry.telemetry_context import try_get_context
from alpasim_utils import geometry
from numpy.typing import NDArray


logger = logging.getLogger(__name__)

class StepEvent(RecurringEvent):
    """Close the current step and open the next.

    Commits all computed state to RolloutState, logs actor poses, and clears
    StepContext. This is the only event that mutates trajectory state.
    """

    priority: int = EventPriority.STEP

    def __init__(
        self,
        timestamp_us: int,
        control_timestep_us: int,
        services: ServiceBundle,
    ):
        super().__init__(timestamp_us=timestamp_us)
        self.interval_us = control_timestep_us
        self.services = services

        # Avoid checking and serializing the same map on every simulation step.
        self._map_export_requested = False

    async def run(self, state: RolloutState, queue: EventQueue) -> None:
        if not self._map_export_requested:
            self._map_export_requested = True

            vector_map = state.unbound.vector_map

            if vector_map is None:
                logger.warning(
                    "Scene %s has no VectorMap; "
                    "skipping ROS map export",
                    state.unbound.scene_id,
                )
            else:
                map_tcp_exporter.publish_vector_map(
                    scene_id=state.unbound.scene_id,
                    vector_map=vector_map,
                    frame_id="map",
                )


        ctx = state.step_context

        if ctx is not None and ctx.ego_true is not None:
            # --- Normal case: commit pipeline results ---
            assert (
                ctx.step_start_us == self.timestamp_us
            ), f"StepEvent timestamp mismatch: {ctx.step_start_us} != {self.timestamp_us}"
            assert ctx.ego_estimated is not None
            assert ctx.corrected_ego_trajectory is not None

            # Commit ego trajectories (bulk concat)
            corrected_ego = geometry.DynamicTrajectory.from_trajectory_and_dynamics(
                ctx.corrected_ego_trajectory,
                ctx.ego_true.dynamics,
            )
            state.ego_trajectory = state.ego_trajectory.concat(corrected_ego)
            state.ego_trajectory_estimate = state.ego_trajectory_estimate.concat(
                ctx.ego_estimated
            )

            ego_state_udp_exporter.publish(state.ego_trajectory)

            # Commit accumulated traffic trajectories
            for obj_id, accumulated_traj in ctx.traffic_trajectories.items():
                if obj_id == "EGO":
                    continue
                for i in range(len(accumulated_traj)):
                    ts = int(accumulated_traj.timestamps_us[i])
                    pose = accumulated_traj.get_pose(i)
                    state.traffic_objs[obj_id].trajectory.update_absolute(ts, pose)


            # Export current actors after physics corrections are committed.
            export_actor_states(
                state,
                self.timestamp_us,
            )


            # if not hasattr(self, "_printed_actor_debug"):
            #     self._printed_actor_debug = True

            #     traffic_objs = state.traffic_objs

            #     print("\n========== TRAFFIC OBJECTS DEBUG ==========")
            #     print("traffic_objs type:", type(traffic_objs))
            #     print("traffic_objs len:", len(traffic_objs))
            #     print("track IDs:", list(traffic_objs.keys())[:20])

            #     # Find the first actor with a non-empty trajectory.
            #     selected = None

            #     # Prefer a dynamic actor whose trajectory is valid at the current time.
            #     for track_id, actor in traffic_objs.items():
            #         trajectory = actor.trajectory

            #         if actor.is_static:
            #             continue

            #         if len(trajectory) == 0:
            #             continue

            #         time_range = trajectory.time_range_us

            #         if self.timestamp_us in time_range:
            #             selected = (track_id, actor)
            #             break

            #     # Fallback: take any actor with a non-empty trajectory.
            #     if selected is None:
            #         for track_id, actor in traffic_objs.items():
            #             if len(actor.trajectory) > 0:
            #                 selected = (track_id, actor)
            #                 break

            #     if selected is None:
            #         print("No actor with a non-empty trajectory was found.")
            #     else:
            #         track_id, actor = selected
            #         trajectory = actor.trajectory

            #         print("\n========== ONE ACTIVE TRAFFIC ACTOR ==========")
            #         print("track_id:", track_id)
            #         print("actor type:", type(actor))
            #         print("label_class:", actor.label_class)
            #         print("is_static:", actor.is_static)

            #         print(
            #             "aabb:",
            #             {
            #                 "x": actor.aabb.x,
            #                 "y": actor.aabb.y,
            #                 "z": actor.aabb.z,
            #             },
            #         )

            #         print("trajectory type:", type(trajectory))
            #         print("trajectory len:", len(trajectory))
            #         print("time range:", trajectory.time_range_us)
            #         print("timestamps shape:", trajectory.timestamps_us.shape)
            #         print("positions shape:", trajectory.positions.shape)
            #         print("quaternions shape:", trajectory.quaternions.shape)

            #         # Select the pose at the current StepEvent timestamp.
            #         time_range = trajectory.time_range_us

            #         if self.timestamp_us in time_range:
            #             query_ts = np.array(
            #                 [self.timestamp_us],
            #                 dtype=np.uint64,
            #             )

            #             sampled = trajectory.interpolate(query_ts)

            #             print("sample timestamp:", self.timestamp_us)
            #             print("sample position:", sampled.positions[0])
            #             print("sample quaternion:", sampled.quaternions[0])

            #             # Inspect derived trajectory dynamics.
            #             try:
            #                 velocities = trajectory.velocities()

            #                 print("velocities type:", type(velocities))
            #                 print("velocities shape:", velocities.shape)
            #                 print("velocities:")
            #                 print(velocities)
            #             except Exception as exc:
            #                 print("velocity unavailable:", repr(exc))

            #             try:
            #                 accelerations = trajectory.accelerations()

            #                 print("accelerations type:", type(accelerations))
            #                 print("accelerations shape:", accelerations.shape)
            #                 print("accelerations:")
            #                 print(accelerations)
            #             except Exception as exc:
            #                 print("acceleration unavailable:", repr(exc))

            #             try:
            #                 yaws = trajectory.yaws

            #                 print("yaws type:", type(yaws))
            #                 print("yaws shape:", yaws.shape)
            #                 print("yaws:")
            #                 print(yaws)
            #             except Exception as exc:
            #                 print("yaw unavailable:", repr(exc))

            #             try:
            #                 yaw_rates = trajectory.yaw_rates()

            #                 print("yaw_rates type:", type(yaw_rates))
            #                 print("yaw_rates shape:", yaw_rates.shape)
            #                 print("yaw_rates:")
            #                 print(yaw_rates)
            #             except Exception as exc:
            #                 print("yaw rate unavailable:", repr(exc))

            #             try:
            #                 yaw_accelerations = trajectory.yaw_accelerations()

            #                 print(
            #                     "yaw_accelerations type:",
            #                     type(yaw_accelerations),
            #                 )
            #                 print(
            #                     "yaw_accelerations shape:",
            #                     yaw_accelerations.shape,
            #                 )
            #                 print("yaw_accelerations:")
            #                 print(yaw_accelerations)
            #             except Exception as exc:
            #                 print(
            #                     "yaw acceleration unavailable:",
            #                     repr(exc),
            #                 )
                            
            #         else:
            #             print(
            #                 "Selected trajectory is not valid at current timestamp:",
            #                 self.timestamp_us,
            #             )

            #         print("==============================================")

            #     # Also inspect several label classes and static/dynamic counts.
            #     label_counts = {}
            #     static_count = 0
            #     dynamic_count = 0
            #     nonempty_count = 0

            #     for actor in traffic_objs.values():
            #         label_counts[actor.label_class] = (
            #             label_counts.get(actor.label_class, 0) + 1
            #         )

            #         if actor.is_static:
            #             static_count += 1
            #         else:
            #             dynamic_count += 1

            #         if len(actor.trajectory) > 0:
            #             nonempty_count += 1

            #     print("\n========== ACTOR SUMMARY ==========")
            #     print("static actor count:", static_count)
            #     print("dynamic actor count:", dynamic_count)
            #     print("non-empty trajectory count:", nonempty_count)
            #     print("label counts:", label_counts)
            #     print("===================================\n")



            # Log actor poses at each intermediate timestamp
            await log_actor_poses(
                state, ctx.ego_true.timestamps_us, self.services.broadcaster
            )

            # Record step duration telemetry
            step_duration = time.perf_counter() - state.step_wall_start
            telemetry_ctx = try_get_context()
            if telemetry_ctx is not None:
                telemetry_ctx.step_duration.observe(step_duration)
        else:
            # --- Initial case: log initial actor poses ---
            t0_us = state.unbound.egomotion_context_start_us
            t1_us = state.unbound.first_policy_timestamp_us
            await log_actor_poses(
                state,
                np.array([t0_us, t1_us], dtype=np.uint64),
                self.services.broadcaster,
            )

        # --- Always: create fresh StepContext for next step ---
        state.step_context = StepContext()
        state.step_wall_start = time.perf_counter()


class InitialStepEvent(Event):
    """Log the initial rollout state before the first policy step."""

    priority: int = EventPriority.STEP

    def __init__(self, timestamp_us: int, services: ServiceBundle):
        super().__init__(timestamp_us=timestamp_us)
        self.services = services

    async def handle(self, state: RolloutState, queue: EventQueue) -> None:
        del queue
        await log_actor_poses(
            state,
            np.array(
                [
                    state.unbound.egomotion_context_start_us,
                    state.unbound.first_policy_timestamp_us,
                ],
                dtype=np.uint64,
            ),
            self.services.broadcaster,
        )
        state.step_context = StepContext()
        state.step_wall_start = time.perf_counter()

def _actor_point(
    trajectory,
    index: int,
) -> dict:
    timestamps = trajectory.timestamps_us
    positions = trajectory.positions
    quaternions = trajectory.quaternions

    velocities = trajectory.velocities()
    accelerations = trajectory.accelerations()
    yaws = trajectory.yaws
    yaw_rates = trajectory.yaw_rates()
    yaw_accelerations = trajectory.yaw_accelerations()

    velocity = velocities[index]

    return {
        "timestamp_us": int(timestamps[index]),

        "position": {
            "x": float(positions[index][0]),
            "y": float(positions[index][1]),
            "z": float(positions[index][2]),
        },

        "orientation": {
            "x": float(quaternions[index][0]),
            "y": float(quaternions[index][1]),
            "z": float(quaternions[index][2]),
            "w": float(quaternions[index][3]),
        },

        "linear_velocity": {
            "x": float(velocity[0]),
            "y": float(velocity[1]),
            "z": float(velocity[2]),
        },

        "linear_acceleration": {
            "x": float(accelerations[index][0]),
            "y": float(accelerations[index][1]),
            "z": float(accelerations[index][2]),
        },

        "yaw": float(yaws[index]),
        "yaw_rate": float(yaw_rates[index]),
        "yaw_acceleration": float(yaw_accelerations[index]),

        "speed": float(np.linalg.norm(velocity)),
    }

def export_actor_states(
    state: RolloutState,
    timestamp_us: int,
) -> None:
    actors = []

    for track_id, actor in state.traffic_objs.items():
        trajectory = actor.trajectory

        if len(trajectory) == 0:
            continue

        time_range = trajectory.time_range_us

        if timestamp_us not in time_range:
            continue

        timestamps = trajectory.timestamps_us.astype(np.int64)

        # Closest trajectory sample to current simulation time.
        current_index = int(
            np.argmin(
                np.abs(timestamps - int(timestamp_us))
            )
        )

        current_state = _actor_point(
            trajectory,
            current_index,
        )

        # Send all currently available points from t onward.
        # ROS Bridge will crop according to its YAML parameters.
        future_indices = np.nonzero(
            timestamps >= int(timestamp_us)
        )[0]

        future_points = [
            _actor_point(trajectory, int(index))
            for index in future_indices
        ]

        actors.append(
            {
                "track_id": str(track_id),
                "label_class": str(actor.label_class),
                "is_static": bool(actor.is_static),

                "dimensions": {
                    "x": float(actor.aabb.x),
                    "y": float(actor.aabb.y),
                    "z": float(actor.aabb.z),
                },

                "current_state": current_state,
                "available_future_points": future_points,
            }
        )

    actor_tcp_exporter.publish(
        {
            "message_type": "actor_snapshot",
            "timestamp_us": int(timestamp_us),

            "pose_frame_id": "map",
            "dynamics_frame_id": "map",

            "actors": actors,
        }
    )

async def log_actor_poses(
    state: RolloutState,
    timestamps_us: NDArray[np.uint64],
    broadcaster: MessageBroadcaster,
) -> None:
    """Log actor poses (ego + traffic) to the ASL file.

    Builds the trajectory dict and ego coordinate transform once, then
    batch-interpolates each trajectory at all requested timestamps before
    assembling and broadcasting per-timestamp ``LogEntry`` messages.
    """
    trajectories = {
        obj_id: obj.trajectory for obj_id, obj in state.traffic_objs.items()
    }
    trajectories["EGO"] = state.ego_trajectory.trajectory().transform(
        state.unbound.transform_ego_coords_ds_to_aabb,
        is_relative=True,
    )

    # Pre-allocate per-timestamp actor pose lists
    poses_by_ts: dict[int, list[ActorPoses.ActorPose]] = {
        int(ts): [] for ts in timestamps_us
    }

    for obj_id, trajectory in trajectories.items():
        time_range = trajectory.time_range_us

        if obj_id == "EGO":
            if (
                int(timestamps_us[0]) not in time_range
                or int(timestamps_us[-1]) not in time_range
            ):
                raise AssertionError("Ego trajectory ended early.")
            valid_ts = timestamps_us
        else:
            valid_ts = timestamps_us[
                (timestamps_us >= time_range.start) & (timestamps_us < time_range.stop)
            ]

        if len(valid_ts) == 0:
            continue

        poses = trajectory.interpolate_poses_list(valid_ts)
        for ts, pose in zip(valid_ts, poses, strict=True):
            poses_by_ts[int(ts)].append(
                ActorPoses.ActorPose(
                    actor_id=obj_id,
                    actor_pose=geometry.pose_to_grpc(pose),
                )
            )

    for ts in timestamps_us:
        ts_int = int(ts)
        poses_message = LogEntry(
            actor_poses=ActorPoses(
                timestamp_us=ts_int,
                actor_poses=poses_by_ts[ts_int],
            )
        )
        await broadcaster.broadcast(poses_message)
