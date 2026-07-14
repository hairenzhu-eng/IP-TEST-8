"""Unified COLREG controller with embedded strategy implementations."""

from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
from types import MethodType

import numpy as np
from behavior_tree import BTAction, BTCondition, BTSelector, BTSequence, BTStatus
from colreg_apf import classify_colreg_zone, obstacle_stern_waypoint, smooth_ellipse_repulsion


_HERE = Path(__file__).resolve().parent


def _load_source_module(name, filename):
    path = _HERE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_overtaking = _load_source_module("_laptop_overtaking_source", "laptop-overtaking.py")
_head_on = _load_source_module("_laptop_head_on_source", "laptop-headon.py")
_crossing = _load_source_module("_laptop_crossing_source", "laptop-crossing.py")

_OvertakingController = _overtaking.LaptopController
_HeadOnController = _head_on.LaptopController
_CrossingController = _crossing.LaptopController

cluster_principal_dimensions = _crossing.cluster_principal_dimensions

ellipse_level_and_away = _crossing.ellipse_level_and_away

def _switches_from_combination(value):
    value = str(value).strip().lower()
    match = re.fullmatch(r"ekf_(on|off)_cluster_(on|off)", value)
    if not match:
        raise ValueError(
            "SWITCH_COMBINATION must be ekf_<on|off>_cluster_<on|off>"
        )
    return value, match.group(1) == "on", match.group(2) == "on"


SWITCH_COMBINATION, ENABLE_OBSTACLE_EKF_PREDICTION, ENABLE_CLUSTER_BASED_APF_RANGE = (
    _switches_from_combination(os.environ.get("SWITCH_COMBINATION", "ekf_on_cluster_on"))
)


def _mode_value(robot_value, simulation_value):
    return lambda controller: simulation_value if controller.OPERATING_MODE == 2 else robot_value


# APF parameters whose values are identical in all three strategy files.
# Parameters with different original values stay
# strategy-specific and are not listed here.
UNIFIED_APF_PARAMS = {
    "apf_risk_pc_scale": 2.5,
    "apf_avoidance_pc_scale": 2.5,
    "apf_direction_pc_scale": 2.5,
    "apf_virtual_pc_scale": 4.0,
    "apf_activation_front_half_angle_rad": np.deg2rad(150.0),
    "apf_priority_front_half_angle_rad": np.deg2rad(90.0),
    "apf_goal_gain": 5.5,
    "apf_path_gain": 7.5,
    # Single calibrated gain for both dynamic ellipse fields.  Their geometry
    # remains 2.5x (current) and 4x (predicted).
    "apf_repulsive_gain": 3.0,
    "apf_attraction_saturation_m": 3.5,
    "apf_path_threshold_m": 0.10,
    "apf_route_lookahead_m": _mode_value(1.8, 2.1),
    "apf_collision_horizon_s": _mode_value(6.0, 10.0),
    "apf_prediction_dt_s": 0.5,
    "apf_heading_gain": 0.9,
    "apf_heading_step_limit_rad": np.deg2rad(60.0),
    "apf_constant_descent_speed_m_s": lambda controller: controller.route_tracking_speed_m_s,
    "apf_dynamic_speed_threshold_m_s": _mode_value(0.05, 0.06),
    "apf_dynamic_exit_speed_threshold_m_s": _mode_value(0.03, 0.04),
    "apf_crossing_pass_ahead_surge_m_s": lambda controller: (
        controller.route_tracking_speed_m_s + (0.18 if controller.OPERATING_MODE == 2 else 0.08)
    ),
    "apf_crossing_pass_ahead_safe_dcpa_m": _mode_value(0.45, 0.60),
    "apf_track_association_m": _mode_value(0.60, 0.80),
    "apf_track_timeout_s": _mode_value(1.0, 1.5),
    "obstacle_min_pc1_m": _mode_value(0.30, 0.36),
    "obstacle_min_pc2_m": _mode_value(0.16, 0.20),
    "apf_own_equivalent_radius_m": _mode_value(0.25, 0.30),
    "apf_side_lock_s": _mode_value(5.0, 8.0),
    "apf_side_lock_exit_level": 1.15,
    "apf_visual_hold_s": 2.0,
}


def _sync_strategy_switches():
    for module in (_overtaking, _head_on, _crossing):
        module.ENABLE_OBSTACLE_EKF_PREDICTION = bool(ENABLE_OBSTACLE_EKF_PREDICTION)
        module.ENABLE_CLUSTER_BASED_APF_RANGE = bool(ENABLE_CLUSTER_BASED_APF_RANGE)


_sync_strategy_switches()


_CSV_CONTEXT_COLUMNS = (
    "WebotsEnvironment",
    "SwitchCombination",
    "EKFPredictionEnabled",
    "ClusterSizeAPFEnabled",
)


def _apply_unified_apf_params(controller):
    for name, value in UNIFIED_APF_PARAMS.items():
        setattr(controller, name, value(controller) if callable(value) else value)

    controller.obstacle_min_equivalent_radius_m = 0.5 * controller.obstacle_min_pc1_m

    if hasattr(controller, "apf_build_encounter_params"):
        controller.apf_crossing_params = controller.apf_build_encounter_params()
        controller.apf_overtaking_params = controller.apf_build_encounter_params()
        controller.apf_head_on_params = controller.apf_build_encounter_params()


def _world_name_from_text(text):
    match = re.search(r'["\']([^"\']+\.wbt)["\']|(\S+\.wbt)', str(text), flags=re.IGNORECASE)
    if not match:
        return None
    return Path(match.group(1) or match.group(2)).name


def _detect_webots_environment(operating_mode):
    if operating_mode != 2:
        return "not_webots"

    for name in (
        "WEBOTS_WORLD",
        "WEBOTS_WORLD_FILE",
        "WEBOTS_CURRENT_WORLD",
        "WEBOTS_SCENARIO",
        "WORLD_FILE",
    ):
        value = os.environ.get(name)
        if not value:
            continue
        return _world_name_from_text(value) or Path(value).name or value

    try:
        if os.name == "nt":
            result = subprocess.run(
                ["wmic", "process", "where", "name like '%webots%'", "get", "CommandLine", "/value"],
                capture_output=True,
                text=True,
                timeout=0.8,
            )
        else:
            result = subprocess.run(
                ["ps", "-eo", "args"],
                capture_output=True,
                text=True,
                timeout=0.8,
            )
        detected = _world_name_from_text(result.stdout)
        if detected:
            return detected
    except Exception:
        pass

    return "WEBOTS_UNKNOWN"


def __getattr__(name):
    return getattr(_overtaking, name)


_MISSING = object()

_CROSSING_METHODS = (
    "apf_obstacle_endpoint_direction_body",
    "apf_stern_direction_body",
    "apf_bow_direction_body",
    "apf_crossing_strategy_from_velocity",
    "apf_encounter_speed_m_s",
    "apf_obstacle_in_forward_half_plane",
)

_HEAD_ON_METHODS = (
    "apf_track_for_obstacle",
    "apf_build_encounter_params",
    "apf_profile_name_for_encounter",
    "apf_params_for_encounter",
    "apf_cpa_metrics",
    "own_prediction_velocity_ne",
    "obstacle_track_state_at",
    "virtual_collision_visuals",
    "apf_default_side_from_obstacle",
    "apf_obstacle_in_priority_front_sector",
    "apf_lock_side",
    "refresh_apf_side_lock",
    "route_progress_and_point",
    "goal_distance_m",
    "final_approach_active",
    "apf_path_attraction_body",
    "apf_goal_attraction_body",
)


class LaptopController(_OvertakingController):
    """One live controller instance with COLREG strategy dispatch."""

    _HEAD_ON_RULES = {"head_on"}
    _OVERTAKING_RULES = {"overtaking", "being_overtaken"}
    _CROSSING_RULES = {"crossing", "crossing_from_starboard", "crossing_from_port"}

    def __init__(self, OPERATING_MODE):
        _sync_strategy_switches()
        super().__init__(OPERATING_MODE)
        # Keep the live controller state tied to laptop.py's unified switches.
        self.obstacle_ekf_prediction_enabled = bool(ENABLE_OBSTACLE_EKF_PREDICTION)
        self.apf_cluster_range_enabled = bool(ENABLE_CLUSTER_BASED_APF_RANGE)
        _apply_unified_apf_params(self)
        self.apf_selected_controller = "default_apf"
        self._last_colreg_decision = None
        self.webots_environment = _detect_webots_environment(self.OPERATING_MODE)
        self.apf_current_field_size_scale = 4.0
        self.apf_predicted_field_size_scale = 7.0
        # ponytail: constant route acceleration; replace with the propulsion
        # model only if measured acceleration materially improves CPA timing.
        self.apf_own_acceleration_m_s2 = 0.25 if self.OPERATING_MODE == 2 else 0.15
        self.apf_stand_on_emergency_tcpa_s = 3.0
        self.apf_last_dynamic_field_s = -np.inf
        self._csv_context_last_patched_line_end = None
        self._ensure_csv_context_header()
        # Defaults used directly by laptop-crossing.py helpers when they run on
        # this single overtaking-initialised controller instance.
        self.apf_boundary_activation_level = getattr(self, "apf_boundary_activation_level", 1.35)
        self.apf_boundary_inside_boost = getattr(self, "apf_boundary_inside_boost", 3.0)
        self.apf_min_detour_offset_m = getattr(
            self,
            "apf_min_detour_offset_m",
            1.0 if self.OPERATING_MODE == 2 else 0.8,
        )
        self.apf_pass_ahead_gain = getattr(self, "apf_pass_ahead_gain", 2.4)
        self.apf_crossing_min_forward_speed = getattr(
            self,
            "apf_crossing_min_forward_speed",
            0.14 if self.OPERATING_MODE == 2 else 0.16,
        )
        self.apf_crossing_close_quarters_surge_m_s = getattr(
            self,
            "apf_crossing_close_quarters_surge_m_s",
            0.18 if self.OPERATING_MODE == 2 else 0.16,
        )
        self.obstacle_track_confirmation_hits = getattr(
            self,
            "obstacle_track_confirmation_hits",
            2,
        )
        self.obstacle_prediction_min_samples = getattr(
            self,
            "obstacle_prediction_min_samples",
            2,
        )
        self.obstacle_prediction_min_hits = getattr(
            self,
            "obstacle_prediction_min_hits",
            2,
        )
        # A full two-second EKF window rejects the repeating LiDAR visible-face
        # jump before a target course is trusted for future-field placement.
        self.obstacle_prediction_min_time_span_s = 2.0
        self.obstacle_prediction_min_displacement_m = getattr(
            self,
            "obstacle_prediction_min_displacement_m",
            0.03,
        )
        self.apf_crossing_longitudinal_scale = getattr(
            self,
            "apf_crossing_longitudinal_scale",
            1.3,
        )
        self.apf_crossing_lateral_scale = getattr(
            self,
            "apf_crossing_lateral_scale",
            5.2,
        )
        self._mission_waypoints = list(getattr(self, "waypoints", []))
        # Match the strategy controllers so position noise near the goal does
        # not keep the vessel circling a point it has effectively reached.
        self.goal_tolerance_m = 0.20
        self.apf_waypoint_detour_enabled = True
        self.apf_waypoint_acceptance_m = 0.45 if self.OPERATING_MODE == 2 else 0.30
        self.apf_waypoint_replan_interval_s = 0.6 if self.OPERATING_MODE == 2 else 0.8
        self.apf_waypoint_entry_margin_m = max(
            self.route_tracking_lookahead_m,
            1.2 if self.OPERATING_MODE == 2 else 0.8,
        )
        self.apf_waypoint_merge_margin_m = 1.8 if self.OPERATING_MODE == 2 else 1.1
        self.apf_waypoint_lateral_margin_m = 0.45 if self.OPERATING_MODE == 2 else 0.28
        self.apf_waypoint_path_ne = []
        self.apf_waypoint_index = 0
        self.apf_waypoint_side_sign = 0.0
        self.apf_waypoint_merge_along_m = 0.0
        self.apf_waypoint_planned_at_s = -np.inf
        self.apf_waypoint_step_m = 0.90 if self.OPERATING_MODE == 2 else 0.55
        self.apf_waypoint_preview_points = 6
        self.apf_waypoint_force_blend = 0.65
        self.apf_waypoint_speed_m_s = float(self.route_tracking_speed_m_s)
        self._colreg_recognition_bt = self._build_colreg_recognition_behavior_tree()
        self._colreg_avoidance_bt = self._build_colreg_avoidance_behavior_tree()
        self._colreg_bt = self._build_colreg_behavior_tree()

    def _current_ne(self):
        return np.array([float(self.North), float(self.East)], dtype=float)

    def apf_obstacle_level_and_away(self, obstacle, *_args, **_kwargs):
        centre_body = self._obstacle_body_position(obstacle)
        axis_body = self.earth_vector_to_body(self.obstacle_length_axis_ne(obstacle))
        pc1_m, pc2_m = self.obstacle_pc_dimensions(obstacle)
        return ellipse_level_and_away(
            -centre_body,
            axis_body,
            0.5 * pc1_m + self.apf_own_equivalent_radius_m,
            0.5 * pc2_m + self.apf_own_equivalent_radius_m,
        )

    def obstacle_pc_dimensions(self, obstacle):
        if not self.apf_cluster_range_enabled:
            return float(self.obstacle_min_pc1_m), float(self.obstacle_min_pc2_m)
        return _CrossingController.obstacle_pc_dimensions(self, obstacle)

    def obstacle_length_axis_ne(self, obstacle):
        if not self.apf_cluster_range_enabled:
            return np.array([1.0, 0.0], dtype=float)
        return _CrossingController.obstacle_length_axis_ne(self, obstacle)

    def apf_direction_clearance_m(self, obstacle, *_args, **_kwargs):
        _, pc2_m = self.obstacle_pc_dimensions(obstacle)
        outer_scale = (
            self.apf_predicted_field_size_scale
            if bool(obstacle.get("virtual", False))
            else self.apf_current_field_size_scale
        )
        return outer_scale * (0.5 * pc2_m + self.apf_own_equivalent_radius_m)

    def apf_pass_astern_side_from_velocity(self, obs_vel_body):
        # Body y and planner side are both positive to port/left; the stern is
        # therefore on the opposite side of the obstacle's lateral velocity.
        return -_OvertakingController.apf_pass_astern_side_from_velocity(self, obs_vel_body)

    # The APF has exactly two repulsive fields: the measured LiDAR ellipse and
    # the EKF/CPA-predicted ellipse.  COLREG still selects the controller, but
    # never injects an additional force into either field.
    def _dynamic_ellipse_repulsion(self, obstacle, outer_scale):
        centre_body = self._obstacle_body_position(obstacle)
        if not np.isfinite(centre_body).all():
            return np.zeros(2, dtype=float), False
        axis_body = self.earth_vector_to_body(
            self.obstacle_length_axis_ne(obstacle)
        )
        pc1_m, pc2_m = self.obstacle_pc_dimensions(obstacle)
        evaluation_offset_body = -centre_body
        if bool(obstacle.get("virtual", False)):
            # A future field must act on the robot's unavoided position at
            # TCPA, rather than waiting for the present robot to reach it.
            predicted_own_ne = np.asarray(
                obstacle.get("collision_position_ne", [np.nan, np.nan]),
                dtype=float,
            ).reshape(2)
            centre_ne = np.asarray(
                obstacle.get("centre_ne", [np.nan, np.nan]),
                dtype=float,
            ).reshape(2)
            if np.isfinite(predicted_own_ne).all() and np.isfinite(centre_ne).all():
                evaluation_offset_body = self.earth_vector_to_body(
                    predicted_own_ne - centre_ne
                )
        level, away = ellipse_level_and_away(
            evaluation_offset_body,
            axis_body,
            0.5 * pc1_m + self.apf_own_equivalent_radius_m,
            0.5 * pc2_m + self.apf_own_equivalent_radius_m,
        )
        if level >= float(outer_scale):
            return np.zeros(2, dtype=float), False
        # Smoothly fades from the hull boundary (level=1) to zero at the
        # requested size multiple; inside the measured ellipse remains unsafe.
        return smooth_ellipse_repulsion(
            level,
            away,
            self.apf_repulsive_gain,
            outer_scale,
        )

    def apf_repulsion_for_obstacle(self, obstacle, target_body, own_vel_body):
        del target_body, own_vel_body
        force, active = self._dynamic_ellipse_repulsion(
            obstacle,
            float(getattr(
                self,
                "apf_predicted_field_size_scale"
                if bool(obstacle.get("virtual", False))
                else "apf_current_field_size_scale",
                4.0 if bool(obstacle.get("virtual", False)) else 2.5,
            )),
        )
        if active:
            self.apf_last_dynamic_field_s = (
                float(self.timefromstart) if self.timefromstart is not None else 0.0
            )
            obs_pos_body = self._obstacle_body_position(obstacle)
            obs_vel_body = self._obstacle_body_velocity(obstacle)
            own_vel_body = self.current_velocity_body()
            encounter, requested_side, action = self.apf_classify_encounter(
                obs_pos_body,
                obs_vel_body,
                own_vel_body,
            )
            if encounter in self._CROSSING_RULES:
                stern_side = self.apf_pass_astern_side_from_velocity(obs_vel_body)
                if stern_side != 0.0:
                    requested_side = stern_side
                    action = "Crossing: pass astern of obstacle ship"
            tcpa_s, dcpa_m = self.apf_cpa_metrics(
                obs_pos_body,
                obs_vel_body,
                own_vel_body,
            )
            locked_side = self.apf_lock_side(requested_side, 1.0)
            self.apf_encounter_mode = encounter
            self.apf_colreg_rule = action
            self.apf_avoidance_side_sign = locked_side
            self.apf_colreg_dcpa_m = dcpa_m
            self.apf_colreg_tcpa_s = tcpa_s
            self.apf_colreg_active = locked_side != 0.0

        # Overtaking/head-on legacy loops also consume their profile mapping.
        # It is retained solely for that call signature; it adds no force.
        return force, active, getattr(self, "apf_overtaking_params", {})

    def _predict_unavoided_route(self, times_s):
        """Predict the configured start-to-goal motion before APF avoidance."""
        times_s = np.asarray(times_s, dtype=float)
        route = np.asarray(self.route_path_unit_ne, dtype=float).reshape(2)
        current_ne = self._current_ne()
        along_now = float(np.dot(current_ne - self.start_ne, route))
        velocity_ne = np.asarray(getattr(self, "v_robot", np.zeros((3, 1))), dtype=float).reshape(-1)[:2]
        speed = max(float(np.dot(velocity_ne, route)), 0.0)
        target_speed = float(self.route_tracking_speed_m_s)
        acceleration = max(float(self.apf_own_acceleration_m_s2), 0.0)
        if speed >= target_speed or acceleration <= 1e-9:
            distance = target_speed * times_s
        else:
            ramp_s = (target_speed - speed) / acceleration
            ramp_distance = speed * ramp_s + 0.5 * acceleration * ramp_s * ramp_s
            distance = np.where(
                times_s <= ramp_s,
                speed * times_s + 0.5 * acceleration * times_s * times_s,
                ramp_distance + target_speed * (times_s - ramp_s),
            )
        along = np.clip(along_now + distance, 0.0, self.route_path_length_m)
        return self.start_ne + along[:, None] * route

    def obstacle_track_state_at(self, track, dt_s):
        """Constant-velocity EKF prediction using its short-window estimate."""
        dt_s = max(float(dt_s), 0.0)
        state = np.asarray(
            track.get("state", [np.nan, np.nan, 0.0, 0.0]),
            dtype=float,
        ).reshape(4)
        if not np.isfinite(state).all():
            return None, None
        velocity_ne = np.asarray(
            track.get("velocity_mean_ne", state[2:4]),
            dtype=float,
        ).reshape(2)
        if not np.isfinite(velocity_ne).all():
            velocity_ne = state[2:4].copy()
        return state[0:2] + velocity_ne * dt_s, velocity_ne.copy()

    def update_apf_virtual_obstacles(self):
        self.apf_virtual_obstacles = []
        if not self.obstacle_ekf_prediction_enabled:
            return self.apf_virtual_obstacles

        now = float(getattr(self, "latest_lidar_received_s", 0.0) or 0.0)
        step_s = max(float(self.apf_prediction_dt_s), 0.05)
        times_s = np.arange(0.0, float(self.apf_collision_horizon_s) + 0.5 * step_s, step_s)
        own_trajectory_ne = self._predict_unavoided_route(times_s)
        for track in self.apf_obstacle_tracks:
            if now - float(track.get("last_seen_s", now)) > self.apf_track_timeout_s:
                continue
            if not self.obstacle_track_motion_is_stable(track):
                continue

            predicted_states = [self.obstacle_track_state_at(track, t) for t in times_s]
            obstacle_trajectory_ne = np.asarray([state[0] for state in predicted_states], dtype=float)
            distances = np.linalg.norm(own_trajectory_ne - obstacle_trajectory_ne, axis=1)
            closest_index = int(np.argmin(distances))
            tcpa_s = float(times_s[closest_index])
            if tcpa_s <= 0.0:
                continue

            predicted_ne, predicted_vel_ne = predicted_states[closest_index]
            separation = own_trajectory_ne[closest_index] - predicted_ne
            pc1_m, pc2_m = self.obstacle_pc_dimensions(track)
            level, _ = ellipse_level_and_away(
                separation,
                self.obstacle_length_axis_ne(track),
                0.5 * pc1_m + self.apf_own_equivalent_radius_m,
                0.5 * pc2_m + self.apf_own_equivalent_radius_m,
            )
            if level >= self.apf_predicted_field_size_scale:
                continue

            self.apf_virtual_obstacles.append({
                "label": -1000 - int(track.get("id", 0)),
                "virtual": True,
                "centre_ne": predicted_ne.tolist(),
                "centre_body": self.earth_point_to_body(predicted_ne).tolist(),
                "pc1_m": pc1_m,
                "pc2_m": pc2_m,
                "length_axis_ne": self.obstacle_length_axis_ne(track).tolist(),
                "velocity_ne": predicted_vel_ne.tolist(),
                "tcpa_s": tcpa_s,
                "dcpa_m": float(distances[closest_index]),
                "collision_position_ne": own_trajectory_ne[closest_index].tolist(),
            })
        return self.apf_virtual_obstacles

    def _route_normal_left_ne(self):
        unit = np.asarray(self.route_path_unit_ne, dtype=float).reshape(2)
        norm = float(np.linalg.norm(unit))
        if not np.isfinite(unit).all() or norm < 1e-9:
            return np.array([0.0, 1.0], dtype=float)
        unit = unit / norm
        return np.array([-unit[1], unit[0]], dtype=float)

    def _point_on_main_route(self, along_m):
        along_m = float(np.clip(along_m, 0.0, self.route_path_length_m))
        return self.start_ne + along_m * self.route_path_unit_ne

    def _project_to_main_route(self, point_ne):
        point_ne = np.asarray(point_ne, dtype=float).reshape(2)
        if self.route_path_length_m < 1e-9:
            return 0.0, 0.0, self.goal_ne.copy()
        delta_ne = point_ne - self.start_ne
        along_m = float(np.dot(delta_ne, self.route_path_unit_ne))
        along_m = float(np.clip(along_m, 0.0, self.route_path_length_m))
        closest_ne = self._point_on_main_route(along_m)
        lateral_m = float(np.dot(point_ne - closest_ne, self._route_normal_left_ne()))
        return along_m, lateral_m, closest_ne

    def _apf_waypoint_path_active(self):
        return self.apf_waypoint_index < len(self.apf_waypoint_path_ne)

    def _rejoined_predicted_trajectory(self):
        merge_along_m = getattr(self, "apf_waypoint_merge_along_m", None)
        if not self._apf_waypoint_path_active() or merge_along_m is None:
            return False
        along_m, lateral_m, _ = self._project_to_main_route(self._current_ne())
        return (
            along_m >= merge_along_m - self.apf_waypoint_acceptance_m
            and abs(lateral_m) <= self.apf_waypoint_acceptance_m
        )

    def _restore_display_waypoints(self):
        if getattr(self, "_mission_waypoints", None) is not None:
            self.waypoints = list(self._mission_waypoints)

    def _update_display_waypoints(self):
        if not self._apf_waypoint_path_active():
            self._restore_display_waypoints()
            return
        if not getattr(self, "_mission_waypoints", None):
            return

        waypoint_type = type(self._mission_waypoints[0])
        detour_waypoints = []
        for point_ne in self.apf_waypoint_path_ne[self.apf_waypoint_index:]:
            waypoint = waypoint_type()
            waypoint.y = float(point_ne[0])
            waypoint.x = float(point_ne[1])
            detour_waypoints.append(waypoint)
        self.waypoints = detour_waypoints + list(self._mission_waypoints)

    def _clear_apf_waypoint_path(self):
        self.apf_waypoint_path_ne = []
        self.apf_waypoint_index = 0
        self.apf_waypoint_side_sign = 0.0
        self.apf_waypoint_merge_along_m = 0.0
        self.apf_waypoint_planned_at_s = -np.inf
        self.apf_waypoint_speed_m_s = float(self.route_tracking_speed_m_s)
        self._restore_display_waypoints()

    def _advance_apf_waypoint_progress(self, current_ne=None):
        if current_ne is None:
            current_ne = self._current_ne()
        current_ne = np.asarray(current_ne, dtype=float).reshape(2)

        while self._apf_waypoint_path_active():
            waypoint_ne = np.asarray(
                self.apf_waypoint_path_ne[self.apf_waypoint_index],
                dtype=float,
            ).reshape(2)
            acceptance_m = (
                self.goal_tolerance_m
                if self.apf_waypoint_index == len(self.apf_waypoint_path_ne) - 1
                and np.allclose(waypoint_ne, self.goal_ne)
                else self.apf_waypoint_acceptance_m
            )
            if float(np.linalg.norm(waypoint_ne - current_ne)) > acceptance_m:
                break
            self.apf_waypoint_index += 1

        if not self._apf_waypoint_path_active():
            self._clear_apf_waypoint_path()
        else:
            self._update_display_waypoints()

    def _detour_side_sign(self):
        if self.apf_side_lock_sign != 0.0:
            return float(np.sign(self.apf_side_lock_sign))

        repulsive_force = np.asarray(self.apf_repulsive_force_body, dtype=float).reshape(2)
        if np.isfinite(repulsive_force).all() and abs(float(repulsive_force[1])) > 1e-6:
            return float(np.sign(repulsive_force[1]))

        if self.left_clearance_m > self.right_clearance_m + 0.05:
            return 1.0
        if self.right_clearance_m > self.left_clearance_m + 0.05:
            return -1.0
        return -1.0

    def _waypoint_detour_side(self):
        """Keep an active detour on its original side of the planned route."""
        if self._apf_waypoint_path_active() and self.apf_waypoint_side_sign != 0.0:
            return float(np.sign(self.apf_waypoint_side_sign))
        return self._detour_side_sign()

    def _detour_candidate_obstacles(self):
        current_ne = self._current_ne()
        current_along_m, _, _ = self._project_to_main_route(current_ne)
        route_normal_left_ne = self._route_normal_left_ne()
        candidates = []

        try:
            virtual_obstacles = list(self.update_apf_virtual_obstacles() or [])
        except Exception:
            virtual_obstacles = []

        for obstacle in list(getattr(self, "lidar_obstacles", []) or []) + virtual_obstacles:
            centre_ne = np.asarray(
                obstacle.get("centre_ne", [np.nan, np.nan]),
                dtype=float,
            ).reshape(2)
            if not np.isfinite(centre_ne).all():
                continue

            centre_body = np.asarray(
                obstacle.get("centre_body", [np.nan, np.nan]),
                dtype=float,
            ).reshape(2)
            if np.isfinite(centre_body).all() and centre_body[0] < -0.25:
                continue

            along_m, _, closest_ne = self._project_to_main_route(centre_ne)
            if along_m < current_along_m - self.route_tracking_lookahead_m:
                continue

            obstacle_to_route_ne = centre_ne - closest_ne
            lateral_m = float(np.dot(obstacle_to_route_ne, route_normal_left_ne))
            pc1_m = max(
                float(obstacle.get("pc1_m", self.obstacle_min_pc1_m)),
                float(self.obstacle_min_pc1_m),
            )
            pc2_m = max(
                float(obstacle.get("pc2_m", self.obstacle_min_pc2_m)),
                float(self.obstacle_min_pc2_m),
            )
            candidates.append(
                {
                    "along_m": along_m,
                    "lateral_m": lateral_m,
                    "pc1_m": pc1_m,
                    "pc2_m": pc2_m,
                    "virtual": bool(obstacle.get("virtual", False)),
                    "centre_ne": centre_ne,
                    "axis_ne": self.obstacle_length_axis_ne(obstacle),
                    "velocity_ne": np.asarray(
                        obstacle.get("velocity_mean_ne", obstacle.get("velocity_ne", [np.nan, np.nan])),
                        dtype=float,
                    ).reshape(2),
                }
            )

        for candidate in candidates:
            field_scale = float(getattr(
                self,
                "apf_predicted_field_size_scale" if candidate["virtual"] else "apf_current_field_size_scale",
                8.0 if candidate["virtual"] else 3.0,
            ))
            own_radius_m = float(getattr(self, "apf_own_equivalent_radius_m", 0.0))
            candidate["field_long_m"] = candidate["field_half_along_m"] = field_scale * (
                0.5 * candidate["pc1_m"] + own_radius_m
            )
            candidate["field_lateral_m"] = candidate["field_half_lateral_m"] = field_scale * (
                0.5 * candidate["pc2_m"] + own_radius_m
            )

        return candidates

    def _keep_waypoints_outside_apf_fields(self, points_ne, candidates, side_sign):
        """Project waypoint samples outside every measured and predicted APF ellipse."""
        margin_m = float(self.apf_waypoint_lateral_margin_m)
        route_normal_ne = self._route_normal_left_ne()
        safe_points = []
        for point_ne in points_ne:
            point_ne = np.asarray(point_ne, dtype=float).reshape(2)
            for obstacle in candidates:
                axis_ne = np.asarray(obstacle["axis_ne"], dtype=float).reshape(2)
                axis_norm = float(np.linalg.norm(axis_ne))
                if not np.isfinite(axis_ne).all() or axis_norm < 1e-9:
                    axis_ne = self.route_path_unit_ne
                else:
                    axis_ne = axis_ne / axis_norm
                lateral_ne = np.array([-axis_ne[1], axis_ne[0]])
                long_m = float(obstacle.get("field_long_m", obstacle.get("field_half_along_m", 0.0)))
                lateral_m = float(obstacle.get("field_lateral_m", obstacle.get("field_half_lateral_m", 0.0)))
                if long_m <= 0.0 or lateral_m <= 0.0:
                    continue
                offset = point_ne - np.asarray(obstacle["centre_ne"], dtype=float).reshape(2)
                scaled = np.array([np.dot(offset, axis_ne) / long_m, np.dot(offset, lateral_ne) / lateral_m])
                level = float(np.linalg.norm(scaled))
                if level >= 1.0 + margin_m / min(long_m, lateral_m):
                    continue
                direction = offset if level > 1e-9 else float(side_sign) * route_normal_ne
                direction_norm = float(np.linalg.norm(direction))
                if direction_norm < 1e-9:
                    direction = lateral_ne
                    direction_norm = 1.0
                # ponytail: radial projection is conservative for rotated ellipses; use a full local planner only for dense obstacle fields.
                direction = direction / direction_norm
                boundary_distance_m = 1.0 / np.hypot(
                    np.dot(direction, axis_ne) / long_m,
                    np.dot(direction, lateral_ne) / lateral_m,
                )
                point_ne = np.asarray(obstacle["centre_ne"], dtype=float).reshape(2) + direction * (
                    boundary_distance_m + margin_m
                )
            safe_points.append(point_ne)
        return safe_points

    def _keep_waypoints_on_detour_side(self, points_ne, side_sign):
        kept_points = []
        for point_ne in points_ne:
            along_m, lateral_m, route_point_ne = self._project_to_main_route(point_ne)
            del along_m
            kept_points.append(
                np.asarray(point_ne, dtype=float).reshape(2)
                if side_sign * lateral_m >= 0.0 else route_point_ne
            )
        return kept_points

    def _activate_apf_waypoint_path(self, path_ne, side_sign, merge_along_m, current_ne=None):
        if current_ne is None:
            current_ne = self._current_ne()

        filtered_path = []
        previous_point = np.asarray(current_ne, dtype=float).reshape(2)
        previous_along_m, _, _ = self._project_to_main_route(previous_point)
        for point_ne in path_ne:
            point_ne = np.asarray(point_ne, dtype=float).reshape(2)
            if not np.isfinite(point_ne).all():
                continue
            point_along_m, _, _ = self._project_to_main_route(point_ne)
            if point_along_m <= previous_along_m + 1e-6:
                continue
            if float(np.linalg.norm(point_ne - previous_point)) < 0.25:
                continue
            filtered_path.append(point_ne)
            previous_point = point_ne
            previous_along_m = point_along_m

        if not filtered_path:
            return False

        self.apf_waypoint_path_ne = filtered_path
        self.apf_waypoint_index = 0
        self.apf_waypoint_side_sign = float(side_sign)
        self.apf_waypoint_merge_along_m = float(merge_along_m)
        self.apf_waypoint_planned_at_s = (
            float(self.timefromstart) if self.timefromstart is not None else 0.0
        )
        self._advance_apf_waypoint_progress(current_ne)
        self._update_display_waypoints()
        return self._apf_waypoint_path_active()

    def _force_guidance_direction_ne(self):
        guidance_body = np.asarray(self.apf_force_body, dtype=float).reshape(2)
        if not np.isfinite(guidance_body).all() or float(np.linalg.norm(guidance_body)) < 1e-6:
            guidance_body = np.asarray(self.apf_repulsive_force_body, dtype=float).reshape(2)
        if not np.isfinite(guidance_body).all() or float(np.linalg.norm(guidance_body)) < 1e-6:
            return None

        guidance_ne = np.asarray(self.body_vector_to_earth(guidance_body), dtype=float).reshape(2)
        guidance_norm = float(np.linalg.norm(guidance_ne))
        if not np.isfinite(guidance_ne).all() or guidance_norm < 1e-6:
            return None
        return guidance_ne / guidance_norm

    def _plan_apf_waypoint_path(self):
        if not self.apf_waypoint_detour_enabled or self.route_path_length_m < 1e-9:
            return False

        current_ne = self._current_ne()
        current_along_m, _, _ = self._project_to_main_route(current_ne)
        side_sign = self._waypoint_detour_side()
        if side_sign == 0.0:
            return False

        candidates = self._detour_candidate_obstacles()
        repulsive_force = np.asarray(self.apf_repulsive_force_body, dtype=float).reshape(2)
        repulsive_norm = (
            float(np.linalg.norm(repulsive_force))
            if np.isfinite(repulsive_force).all()
            else 0.0
        )
        if not candidates and repulsive_norm < 1e-6:
            return False

        lateral_offset_m = max(
            float(getattr(self, "apf_min_detour_offset_m", 0.0)),
            float(self.apf_waypoint_lateral_margin_m + self.apf_own_equivalent_radius_m),
        )
        lateral_offset_m = max(
            lateral_offset_m,
            float(getattr(self, "obstacle_min_pc2_m", 0.0)) + self.apf_waypoint_lateral_margin_m,
            lateral_offset_m + 0.35 * min(repulsive_norm, 2.0),
        )
        furthest_along_m = current_along_m + self.apf_waypoint_entry_margin_m

        for obstacle in candidates:
            half_length_m = 0.5 * obstacle["pc1_m"] + self.apf_waypoint_merge_margin_m
            half_width_m = (
                0.5 * obstacle["pc2_m"]
                + self.apf_own_equivalent_radius_m
                + self.apf_waypoint_lateral_margin_m
            )
            furthest_along_m = max(furthest_along_m, obstacle["along_m"] + half_length_m)
            lateral_offset_m = max(
                lateral_offset_m,
                side_sign * obstacle["lateral_m"] + half_width_m,
            )

        lateral_offset_m = max(lateral_offset_m, self.apf_waypoint_acceptance_m + 0.05)
        entry_along_m = float(
            np.clip(
                min(
                    current_along_m + self.apf_waypoint_entry_margin_m,
                    furthest_along_m,
                ),
                0.0,
                self.route_path_length_m,
            )
        )
        offset_along_m = float(
            np.clip(
                max(entry_along_m + self.route_tracking_lookahead_m, furthest_along_m),
                0.0,
                self.route_path_length_m,
            )
        )
        merge_along_m = float(
            np.clip(
                offset_along_m + self.apf_waypoint_merge_margin_m,
                0.0,
                self.route_path_length_m,
            )
        )

        route_normal_left_ne = self._route_normal_left_ne()
        force_dir_ne = self._force_guidance_direction_ne()
        if force_dir_ne is None:
            force_dir_ne = self.route_path_unit_ne.copy()

        blend_dir_ne = self.route_path_unit_ne + self.apf_waypoint_force_blend * force_dir_ne
        blend_norm = float(np.linalg.norm(blend_dir_ne))
        if not np.isfinite(blend_dir_ne).all() or blend_norm < 1e-6:
            blend_dir_ne = self.route_path_unit_ne.copy()
        else:
            blend_dir_ne = blend_dir_ne / blend_norm

        preview_points = max(int(getattr(self, "apf_waypoint_preview_points", 3)), 2)
        waypoint_step_m = max(
            float(getattr(self, "apf_waypoint_step_m", self.route_tracking_lookahead_m)),
            self.apf_waypoint_acceptance_m + 0.05,
        )
        first_guided_point = (
            current_ne
            + waypoint_step_m * blend_dir_ne
            + side_sign * min(lateral_offset_m, waypoint_step_m) * 0.55 * route_normal_left_ne
        )
        candidate_points = [first_guided_point]
        for index in range(preview_points):
            alpha = float(index + 1) / float(preview_points)
            along_m = entry_along_m + alpha * max(offset_along_m - entry_along_m, 0.0)
            route_point_ne = self._point_on_main_route(along_m)
            force_bias_m = waypoint_step_m * (0.35 + 0.45 * alpha)
            candidate_points.append(
                route_point_ne
                + side_sign * lateral_offset_m * route_normal_left_ne
                + force_bias_m * blend_dir_ne
            )
        if self.apf_encounter_mode in self._CROSSING_RULES and candidates:
            stern_candidates = [obstacle for obstacle in candidates if obstacle["virtual"]] or candidates
            obstacle = min(stern_candidates, key=lambda item: item["along_m"])
            stern_ne = obstacle_stern_waypoint(
                obstacle["centre_ne"],
                obstacle["velocity_ne"],
                obstacle["field_long_m"] + self.apf_waypoint_lateral_margin_m,
            )
            if stern_ne is not None:
                candidate_points.append(stern_ne)
                candidate_points.sort(
                    key=lambda point: float(np.dot(point - self.start_ne, self.route_path_unit_ne))
                )
        candidate_points.append(self._point_on_main_route(merge_along_m))
        resume_along_m = min(
            merge_along_m + waypoint_step_m,
            self.route_path_length_m - self.apf_waypoint_acceptance_m,
        )
        if resume_along_m > merge_along_m + 1e-6:
            candidate_points.append(self._point_on_main_route(resume_along_m))
        candidate_points = self._keep_waypoints_on_detour_side(candidate_points, side_sign)
        candidate_points = self._keep_waypoints_outside_apf_fields(
            candidate_points, candidates, side_sign,
        )
        candidate_points = [
            point for point in candidate_points
            if self._project_to_main_route(point)[0] < self.route_path_length_m - 1e-6
        ]
        candidate_points.append(self.goal_ne.copy())

        return self._activate_apf_waypoint_path(
            candidate_points,
            side_sign,
            merge_along_m,
            current_ne=current_ne,
        )

    def _ensure_apf_waypoint_path(self):
        now_s = float(self.timefromstart) if self.timefromstart is not None else 0.0
        if self._apf_waypoint_path_active():
            self._advance_apf_waypoint_progress()
            if self._apf_waypoint_path_active():
                target_ne = np.asarray(
                    self.apf_waypoint_path_ne[self.apf_waypoint_index],
                    dtype=float,
                ).reshape(2)
                target_blocked = any(
                    ellipse_level_and_away(
                        target_ne - obstacle["centre_ne"],
                        obstacle["axis_ne"],
                        obstacle["field_long_m"],
                        obstacle["field_lateral_m"],
                    )[0] < 1.0
                    for obstacle in self._detour_candidate_obstacles()
                )
                if target_blocked:
                    self._clear_apf_waypoint_path()
                elif now_s - self.apf_waypoint_planned_at_s < self.apf_waypoint_replan_interval_s:
                    return True
        return self._plan_apf_waypoint_path() or self._apf_waypoint_path_active()

    def _compute_waypoint_tracking_control(self, navigation_mode):
        current_ne = self._current_ne()
        self._advance_apf_waypoint_progress(current_ne)
        if not self._apf_waypoint_path_active():
            return None

        target_ne = np.asarray(
            self.apf_waypoint_path_ne[self.apf_waypoint_index],
            dtype=float,
        ).reshape(2)
        to_target_ne = target_ne - current_ne
        target_distance_m = float(np.linalg.norm(to_target_ne))
        if target_distance_m < 1e-9:
            return None

        desired_heading = float(np.arctan2(to_target_ne[1], to_target_ne[0]))
        heading_error = _overtaking.wrap_angle(float(self.Yaw) - desired_heading)
        speed_fraction = float(
            np.clip(
                target_distance_m / max(self.final_slowdown_distance_m, 1e-3),
                0.35,
                1.0,
            )
        )
        if abs(heading_error) >= self.final_heading_slow_angle_rad:
            heading_speed_scale = 0.0
        else:
            heading_speed_scale = max(float(np.cos(heading_error)), 0.15)

        target_speed_m_s = float(getattr(self, "apf_waypoint_speed_m_s", self.route_tracking_speed_m_s))
        if not np.isfinite(target_speed_m_s) or target_speed_m_s <= 0.0:
            target_speed_m_s = float(self.route_tracking_speed_m_s)

        p_ref = _overtaking.Vector(3)
        p_ref[0, 0] = target_ne[0]
        p_ref[1, 0] = target_ne[1]
        p_ref[2, 0] = desired_heading

        u_ref = _overtaking.Vector(2)
        u_ref[0, 0] = target_speed_m_s * speed_fraction

        u_track = _overtaking.Vector(2)
        if "overtaking" in str(getattr(self, "webots_environment", "")).lower():
            u_track[0, 0] = self.route_tracking_speed_m_s
        else:
            u_track[0, 0] = max(
                self.apf_min_forward_speed,
                target_speed_m_s * speed_fraction * heading_speed_scale,
            )
        u_track[1, 0] = 1.4 * heading_error
        self.apf_target_ne = target_ne.copy()
        self.navigation_mode = navigation_mode
        return p_ref, u_ref, u_track

    def limit_heading_deviation_command(self, yaw_rate_cmd):
        if self._apf_waypoint_path_active():
            return float(yaw_rate_cmd)
        return _OvertakingController.limit_heading_deviation_command(self, yaw_rate_cmd)

    def update_apf_obstacle_tracks(self, stamp_s):
        _OvertakingController.update_apf_obstacle_tracks(self, stamp_s)

    def _run_context(self):
        return {
            "webots_environment": self.webots_environment,
            "switch_combination": SWITCH_COMBINATION,
            "ekf_prediction_enabled": bool(self.obstacle_ekf_prediction_enabled),
            "cluster_size_apf_enabled": bool(self.apf_cluster_range_enabled),
        }

    def _csv_context_values(self):
        context = self._run_context()
        return [
            context["webots_environment"],
            context["switch_combination"],
            int(context["ekf_prediction_enabled"]),
            int(context["cluster_size_apf_enabled"]),
        ]

    def _ensure_csv_context_header(self):
        try:
            lines = self.filename.read_text().splitlines()
            if not lines:
                return
            header = lines[0].split(",")
            if all(column in header for column in _CSV_CONTEXT_COLUMNS):
                return
            lines[0] = lines[0] + "," + ",".join(_CSV_CONTEXT_COLUMNS)
            self.filename.write_text("\n".join(lines) + "\n")
        except Exception:
            return

    def _patch_latest_csv_context_row(self):
        try:
            values = ",".join(str(value) for value in self._csv_context_values())
            with self.filename.open("rb+") as f:
                f.seek(0, os.SEEK_END)
                end = f.tell()
                if end <= 0:
                    return

                pos = end - 1
                while pos >= 0:
                    f.seek(pos)
                    if f.read(1) not in (b"\n", b"\r"):
                        break
                    pos -= 1
                if pos < 0:
                    return

                line_end = pos + 1
                if getattr(self, "_csv_context_last_patched_line_end", None) == line_end:
                    return

                f.seek(line_end)
                f.truncate()
                f.write(("," + values + "\n").encode("utf-8"))
                self._csv_context_last_patched_line_end = f.tell() - 1
        except Exception:
            return

    @contextmanager
    def _crossing_strategy_methods(self):
        """Temporarily use crossing.py APF helpers on this same controller."""
        previous = {}
        for name in _CROSSING_METHODS:
            method = getattr(_CrossingController, name, None)
            if method is None:
                continue
            previous[name] = self.__dict__.get(name, _MISSING)
            setattr(self, name, MethodType(method, self))

        try:
            yield
        finally:
            for name, value in previous.items():
                if value is _MISSING:
                    self.__dict__.pop(name, None)
                else:
                    setattr(self, name, value)

    @contextmanager
    def _head_on_strategy_methods(self):
        """Temporarily use headon.py APF helpers on this same controller."""
        previous = {}
        for name in _HEAD_ON_METHODS:
            method = getattr(_HeadOnController, name, None)
            if method is None:
                continue
            previous[name] = self.__dict__.get(name, _MISSING)
            setattr(self, name, MethodType(method, self))

        try:
            yield
        finally:
            for name, value in previous.items():
                if value is _MISSING:
                    self.__dict__.pop(name, None)
                else:
                    setattr(self, name, value)

    def _obstacle_body_position(self, obstacle):
        centre_body = np.asarray(
            obstacle.get("centre_body", [np.nan, np.nan]),
            dtype=float,
        ).reshape(2)
        if np.isfinite(centre_body).all():
            return centre_body

        centre_ne = np.asarray(obstacle.get("centre_ne", [np.nan, np.nan]), dtype=float).reshape(2)
        if np.isfinite(centre_ne).all():
            return self.earth_point_to_body(centre_ne)

        return centre_body

    def _obstacle_body_velocity(self, obstacle):
        velocity_ne = np.asarray(
            obstacle.get("velocity_ne", [np.nan, np.nan]),
            dtype=float,
        ).reshape(2)
        if np.isfinite(velocity_ne).all():
            return self.earth_vector_to_body(velocity_ne)

        track = None if bool(obstacle.get("virtual", False)) else self.apf_track_for_obstacle(obstacle)
        if track is None or not self.obstacle_track_motion_is_stable(track):
            return np.zeros(2, dtype=float)

        velocity_ne = np.asarray(
            track.get("velocity_mean_ne", track.get("vel_ne", [np.nan, np.nan])),
            dtype=float,
        ).reshape(2)
        if not np.isfinite(velocity_ne).all():
            velocity_ne = np.asarray(track.get("vel_ne", [np.nan, np.nan]), dtype=float).reshape(2)
        if np.isfinite(velocity_ne).all():
            return self.earth_vector_to_body(velocity_ne)

        return np.zeros(2, dtype=float)

    def _rule_priority(self, rule):
        if rule == "head_on":
            return 0
        if rule in self._OVERTAKING_RULES:
            return 1
        if rule in self._CROSSING_RULES:
            return 2
        if rule == "static_obstacle":
            return 3
        return 4

    def apf_classify_encounter(self, obs_pos_body, obs_vel_body, own_vel_body):
        """Four-zone COLREG classifier from Li et al. Fig. 3."""
        obs_pos_body = np.asarray(obs_pos_body, dtype=float).reshape(2)
        obs_vel_body = np.asarray(obs_vel_body, dtype=float).reshape(2)
        own_vel_body = np.asarray(own_vel_body, dtype=float).reshape(2)
        obs_speed = float(np.linalg.norm(obs_vel_body))
        if obs_speed < self.apf_dynamic_speed_threshold_m_s:
            return "static_obstacle", 0.0, "none"

        # Body y is port/left, while maritime bearings/headings increase to
        # starboard, so both angles use the same clockwise conversion.
        bearing_deg = (-np.rad2deg(np.arctan2(obs_pos_body[1], obs_pos_body[0]))) % 360.0
        relative_heading_deg = (-np.rad2deg(np.arctan2(obs_vel_body[1], obs_vel_body[0]))) % 360.0
        tcpa_s, dcpa_m = self.apf_cpa_metrics(obs_pos_body, obs_vel_body, own_vel_body)
        emergency = (
            np.isfinite(tcpa_s)
            and np.isfinite(dcpa_m)
            and 0.0 < tcpa_s <= self.apf_stand_on_emergency_tcpa_s
            and dcpa_m <= 2.0 * self.apf_own_equivalent_radius_m
        )
        return classify_colreg_zone(bearing_deg, relative_heading_deg, emergency)

    def _select_colreg_strategy_impl(self):
        """IP_test5 COLREG rule detection, kept intact under the behaviour tree shell."""
        own_vel_body = self.current_velocity_body()
        try:
            virtual_obstacles = self.update_apf_virtual_obstacles()
        except Exception:
            virtual_obstacles = []

        best = {
            "rule": "none",
            "controller": "default_apf",
            "tcpa_s": np.nan,
            "dcpa_m": np.nan,
            "score": (self._rule_priority("none"), np.inf, np.inf),
        }

        for obstacle in list(getattr(self, "lidar_obstacles", []) or []) + list(virtual_obstacles or []):
            try:
                obs_pos_body = self._obstacle_body_position(obstacle)
                if not np.isfinite(obs_pos_body).all():
                    continue

                angle_rad = abs(_overtaking.wrap_angle(float(np.arctan2(obs_pos_body[1], obs_pos_body[0]))))
                if (
                    not bool(obstacle.get("virtual", False))
                    and angle_rad > self.apf_activation_front_half_angle_rad
                ):
                    continue

                obs_vel_body = self._obstacle_body_velocity(obstacle)
                if bool(obstacle.get("virtual", False)):
                    rule = str(obstacle.get("encounter_mode", "crossing"))
                    if rule in {"dynamic_virtual_obstacle", "predicted_collision", "none", ""}:
                        rule, _, _ = self.apf_classify_encounter(obs_pos_body, obs_vel_body, own_vel_body)
                else:
                    rule, _, _ = self.apf_classify_encounter(obs_pos_body, obs_vel_body, own_vel_body)

                if rule not in self._HEAD_ON_RULES | self._OVERTAKING_RULES | self._CROSSING_RULES:
                    rule = "static_obstacle"

                tcpa_s, dcpa_m = self.apf_cpa_metrics(obs_pos_body, obs_vel_body, own_vel_body)
                distance_m = float(np.linalg.norm(obs_pos_body))
                tcpa_score = float(tcpa_s) if np.isfinite(tcpa_s) else np.inf
                distance_score = float(dcpa_m) if np.isfinite(dcpa_m) else distance_m
                score = (self._rule_priority(rule), tcpa_score, distance_score)

                if score < best["score"]:
                    best.update(
                        {
                            "rule": rule,
                            "tcpa_s": tcpa_s,
                            "dcpa_m": dcpa_m,
                            "score": score,
                        }
                    )
            except Exception:
                continue

        rule = best["rule"]
        if rule in self._HEAD_ON_RULES:
            best["controller"] = "head_on"
        elif rule in self._OVERTAKING_RULES:
            best["controller"] = "overtaking"
        elif rule in self._CROSSING_RULES:
            best["controller"] = "crossing"

        self._last_colreg_decision = best
        return rule

    def _build_colreg_recognition_behavior_tree(self):
        def recognize_rule(bb):
            bb["rule"] = self._select_colreg_strategy_impl()
            return BTStatus.SUCCESS

        return BTSequence(BTAction(recognize_rule))

    # COLREG rule detection location.
    def select_colreg_strategy(self):
        """Pick the current primary COLREG rule without creating another controller."""
        blackboard = {}
        status = self._colreg_recognition_bt.tick(blackboard)
        if status != BTStatus.SUCCESS:
            return "none"
        return blackboard.get("rule", "none")

    def apf_primary_encounter_mode(self):
        return self.select_colreg_strategy()

    def _mark_selected_controller(self, rule):
        if rule in self._HEAD_ON_RULES:
            self.apf_selected_controller = "head_on"
            self.apf_active_profile_name = "head_on"
        elif rule in self._OVERTAKING_RULES:
            self.apf_selected_controller = "overtaking"
            self.apf_active_profile_name = "overtaking"
        elif rule in self._CROSSING_RULES:
            self.apf_selected_controller = "crossing"
            self.apf_active_profile_name = "crossing"
        else:
            self.apf_selected_controller = "default_apf"

    def _build_colreg_behavior_tree(self):
        def keep_selected_rule(bb):
            bb["rule"] = bb.get("rule", "none")
            return BTStatus.SUCCESS

        def dispatch_head_on(bb):
            with self._head_on_strategy_methods():
                bb["u_cmd"] = _HeadOnController.compute_apf_control(self, bb["t"], bb["u_track"])
            return BTStatus.SUCCESS

        def dispatch_overtaking(bb):
            bb["u_cmd"] = _OvertakingController.compute_apf_control(self, bb["t"], bb["u_track"])
            return BTStatus.SUCCESS

        def dispatch_crossing(bb):
            with self._crossing_strategy_methods():
                bb["u_cmd"] = _CrossingController.compute_apf_control(self, bb["t"], bb["u_track"])
            return BTStatus.SUCCESS

        def dispatch_default(bb):
            bb["u_cmd"] = _OvertakingController.compute_apf_control(self, bb["t"], bb["u_track"])
            return BTStatus.SUCCESS

        def is_head_on(bb):
            return bb.get("rule") in self._HEAD_ON_RULES

        def is_overtaking(bb):
            return bb.get("rule") in self._OVERTAKING_RULES

        def is_crossing(bb):
            return bb.get("rule") in self._CROSSING_RULES

        return BTSequence(
            BTAction(keep_selected_rule),
            BTSelector(
                BTSequence(BTCondition(is_head_on), BTAction(dispatch_head_on)),
                BTSequence(BTCondition(is_overtaking), BTAction(dispatch_overtaking)),
                BTSequence(BTCondition(is_crossing), BTAction(dispatch_crossing)),
                BTAction(dispatch_default),
            ),
        )

    def _tick_colreg_bt(self, rule, t, u_track):
        blackboard = {"rule": rule, "t": t, "u_track": u_track}
        self._mark_selected_controller(rule)
        status = self._colreg_bt.tick(blackboard)
        if status != BTStatus.SUCCESS:
            return None
        return blackboard.get("u_cmd")

    def _apf_avoidance_needed_impl(self, selected_rule):
        self._mark_selected_controller(selected_rule)
        if selected_rule in self._HEAD_ON_RULES:
            with self._head_on_strategy_methods():
                return _HeadOnController.apf_avoidance_needed(self)
        if selected_rule in self._CROSSING_RULES:
            with self._crossing_strategy_methods():
                return _CrossingController.apf_avoidance_needed(self)
        return _OvertakingController.apf_avoidance_needed(self)

    def _build_colreg_avoidance_behavior_tree(self):
        def evaluate_avoidance(bb):
            bb["result"] = self._apf_avoidance_needed_impl(bb.get("rule", "none"))
            return BTStatus.SUCCESS

        return BTSequence(BTAction(evaluate_avoidance))

    def _waypoint_speed_from_command(self, u_cmd):
        if "overtaking" in str(getattr(self, "webots_environment", "")).lower():
            return float(self.route_tracking_speed_m_s)
        target_speed_m_s = float(self.route_tracking_speed_m_s)
        if u_cmd is None:
            return target_speed_m_s

        try:
            values = np.asarray(u_cmd, dtype=float).reshape(-1)
        except Exception:
            return target_speed_m_s

        if values.size <= 0 or not np.isfinite(values[0]):
            return target_speed_m_s

        return float(np.clip(values[0], self.apf_min_forward_speed, self.v_max))

    # crossing/overtaking/head_on dispatch location.
    def compute_apf_control(self, t, u_track):
        selected_rule = self.select_colreg_strategy()
        u_cmd = self._tick_colreg_bt(selected_rule, t, u_track)
        if u_cmd is None:
            u_cmd = _OvertakingController.compute_apf_control(self, t, u_track)
        if "overtaking" in str(getattr(self, "webots_environment", "")).lower():
            u_cmd[0, 0] = self.route_tracking_speed_m_s
        self.apf_waypoint_speed_m_s = self._waypoint_speed_from_command(u_cmd)

        self._mark_selected_controller(selected_rule)
        if self.apf_encounter_mode in ("none", "static_obstacle") and selected_rule != "none":
            self.apf_encounter_mode = selected_rule
        if self._last_colreg_decision is not None:
            self.apf_colreg_tcpa_s = self._last_colreg_decision.get("tcpa_s", self.apf_colreg_tcpa_s)
            self.apf_colreg_dcpa_m = self._last_colreg_decision.get("dcpa_m", self.apf_colreg_dcpa_m)
        if self._apf_waypoint_path_active() and not (self.apf_colreg_active or self.apf_side_lock_active):
            if "overtaking" in str(getattr(self, "webots_environment", "")).lower():
                self._ensure_apf_waypoint_path()
            tracking = self._compute_waypoint_tracking_control("apf_return")
            if tracking is not None:
                return tracking[2]
        if self._ensure_apf_waypoint_path():
            tracking = self._compute_waypoint_tracking_control(
                "apf_colreg" if self.apf_colreg_active else "apf_waypoint",
            )
            if tracking is not None:
                return tracking[2]
        self._clear_apf_waypoint_path()
        return u_cmd

    def apf_avoidance_needed(self):
        selected_rule = self.select_colreg_strategy()
        blackboard = {"rule": selected_rule}
        status = self._colreg_avoidance_bt.tick(blackboard)
        if status != BTStatus.SUCCESS:
            return self._apf_waypoint_path_active()
        return bool(blackboard.get("result", False)) or self._apf_waypoint_path_active()

    def compute_route_tracking_control(self, t):
        tracking = self._compute_waypoint_tracking_control("apf_return")
        if tracking is not None:
            return tracking
        self._clear_apf_waypoint_path()
        return _OvertakingController.compute_route_tracking_control(self, t)

    def write_obstacle_snapshot(self):
        stamp_s = self.latest_lidar_received_s
        _OvertakingController.write_obstacle_snapshot(self)
        self._patch_latest_csv_context_row()
        if stamp_s is None:
            return

        path = self.obstacle_log_dir / f"obstacle_{int(round(float(stamp_s) * 1000.0))}.json"
        if not path.exists():
            return

        try:
            with path.open("r") as f:
                payload = json.load(f)
            payload["run_context"] = self._run_context()
            apf = payload.setdefault("apf", {})
            apf["selected_controller"] = self.apf_selected_controller
            apf["colreg_active"] = self.apf_colreg_active
            apf["active_profile"] = self.apf_active_profile_name
            with path.open("w") as f:
                json.dump(self.json_safe(payload), f, indent=2)
        except Exception:
            return


for _crossing_method_name in _CROSSING_METHODS:
    if not hasattr(LaptopController, _crossing_method_name):
        setattr(
            LaptopController,
            _crossing_method_name,
            getattr(_CrossingController, _crossing_method_name),
        )
