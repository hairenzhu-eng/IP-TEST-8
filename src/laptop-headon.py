"""
Copyright (c) 2025 The uos_sess6072_build Authors.
Authors: Blair Thornton, Alec O'Loughlin, Miquel Massot
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""
import numpy as np
import time
from colreg_apf import straight_line_cpa
from math_sess6072 import Vector
from math_sess6072 import Vector

def Vector(dim):
    return np.zeros((dim, 1), dtype=float)

def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

class LaptopController:

    def earth_vector_to_body(self, vector_ne):
        pose = np.asarray(self.p_robot, dtype=float).reshape(-1)
        yaw = pose[2]
        c = np.cos(yaw)
        s = np.sin(yaw)
        vector_ne = np.asarray(vector_ne, dtype=float).reshape(2)
        return np.array([c * vector_ne[0] + s * vector_ne[1], -s * vector_ne[0] + c * vector_ne[1]])

    def earth_point_to_body(self, point_ne):
        pose = np.asarray(self.p_robot, dtype=float).reshape(-1)
        return self.earth_vector_to_body(np.asarray(point_ne, dtype=float).reshape(2) - pose[0:2])

    def reset_apf_diagnostics(self, clear_visual=True):
        if clear_visual:
            self.apf_force_body = np.zeros(2, dtype=float)
            self.apf_repulsive_force_body = np.zeros(2, dtype=float)
            self.apf_attractive_force_body = np.zeros(2, dtype=float)
            self.apf_steering_force_body = np.zeros(2, dtype=float)
            self.apf_target_ne = np.array([np.nan, np.nan], dtype=float)
            self.apf_virtual_obstacles = []
            self.apf_visual_hold_until_s = 0.0
        self.apf_active_profile_name = 'crossing'
        self.apf_encounter_mode = 'none'
        self.apf_colreg_rule = 'none'
        self.apf_avoidance_side_sign = 0.0
        self.apf_colreg_dcpa_m = np.nan
        self.apf_colreg_tcpa_s = np.nan
        self.apf_colreg_active = False

    def current_velocity_body(self):
        vel_ne = np.asarray(self.v_robot[0:2], dtype=float).reshape(2)
        vel_body = self.earth_vector_to_body(vel_ne)
        if not np.isfinite(vel_body).all():
            return np.zeros(2, dtype=float)
        return vel_body

    def obstacle_ekf_process_noise(self, dt):
        dt = max(float(dt), 0.001)
        q = self.obstacle_ekf_accel_std_m_s2 ** 2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        return q * np.array([[0.25 * dt4, 0.0, 0.5 * dt3, 0.0], [0.0, 0.25 * dt4, 0.0, 0.5 * dt3], [0.5 * dt3, 0.0, dt2, 0.0], [0.0, 0.5 * dt3, 0.0, dt2]], dtype=float)

    def obstacle_ekf_predict(self, state, covariance, dt):
        dt = max(float(dt), 0.0)
        state = np.asarray(state, dtype=float).reshape(4)
        covariance = np.asarray(covariance, dtype=float).reshape(4, 4)
        F = np.array([[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=float)
        predicted_state = F @ state
        predicted_covariance = F @ covariance @ F.T + self.obstacle_ekf_process_noise(dt)
        return (predicted_state, predicted_covariance)

    def obstacle_track_prediction_accel_ne(self, track):
        if not bool(track.get('motion_stable', False)) or int(track.get('stats_sample_count', 0)) < self.obstacle_prediction_accel_min_samples:
            return np.zeros(2, dtype=float)
        accel_ne = np.asarray(track.get('accel_ne', [0.0, 0.0]), dtype=float).reshape(2)
        accel_var_ne = np.asarray(track.get('accel_var_ne', [np.inf, np.inf]), dtype=float).reshape(2)
        if not np.isfinite(accel_ne).all() or not np.isfinite(accel_var_ne).all():
            return np.zeros(2, dtype=float)
        accel_std_m_s2 = float(np.sqrt(max(float(np.max(accel_var_ne)), 0.0)))
        if accel_std_m_s2 > self.obstacle_prediction_max_accel_std_m_s2:
            return np.zeros(2, dtype=float)
        return accel_ne

    def apf_track_for_obstacle(self, obstacle):
        if not self.obstacle_ekf_prediction_enabled:
            return None
        centre_ne = np.asarray(obstacle.get('centre_ne', [np.nan, np.nan]), dtype=float).reshape(2)
        if not np.isfinite(centre_ne).all():
            return None
        now = float(self.latest_lidar_received_s if self.latest_lidar_received_s is not None else time.time())
        best_track = None
        best_distance = np.inf
        for track in self.apf_obstacle_tracks:
            if now - float(track.get('last_seen_s', track.get('stamp_s', now))) > self.apf_track_timeout_s:
                continue
            predicted_state, _ = self.obstacle_ekf_predict(track.get('state', np.r_[track.get('pos_ne', [np.nan, np.nan]), track.get('vel_ne', [0.0, 0.0])]), track.get('covariance', np.eye(4, dtype=float)), max(now - float(track.get('stamp_s', now)), 0.0))
            predicted_pos = predicted_state[0:2]
            distance = float(np.linalg.norm(centre_ne - predicted_pos))
            if distance < best_distance:
                best_distance = distance
                best_track = track
        if best_distance <= max(self.apf_track_association_m * 1.5, self.lidar_dbscan_eps_m * 2.0):
            return best_track
        return None

    def apf_build_encounter_params(self, **overrides):
        params = {'cluster_range_enabled': bool(self.apf_cluster_range_enabled), 'risk_pc_scale': float(self.apf_risk_pc_scale), 'avoidance_pc_scale': float(self.apf_avoidance_pc_scale), 'direction_pc_scale': float(self.apf_direction_pc_scale), 'virtual_pc_scale': float(self.apf_virtual_pc_scale), 'repulsive_gain': float(self.apf_repulsive_gain), 'route_lookahead_m': float(self.apf_route_lookahead_m), 'collision_horizon_s': float(self.apf_collision_horizon_s), 'constant_descent_speed_m_s': float(self.apf_constant_descent_speed_m_s), 'dynamic_speed_threshold_m_s': float(self.apf_dynamic_speed_threshold_m_s)}
        params.update(overrides)
        return params

    def apf_profile_name_for_encounter(self, encounter):
        if encounter == 'overtaking':
            return 'overtaking'
        if encounter == 'head_on':
            return 'head_on'
        return 'crossing'

    def apf_params_for_encounter(self, encounter=None):
        profile_name = self.apf_profile_name_for_encounter(self.apf_encounter_mode if encounter is None else encounter)
        if profile_name == 'overtaking':
            return self.apf_overtaking_params
        if profile_name == 'head_on':
            return self.apf_head_on_params
        return self.apf_crossing_params

    def apf_cpa_metrics(self, obs_pos_body, obs_vel_body, own_vel_body):
        if not self.obstacle_ekf_prediction_enabled:
            return (np.nan, np.nan)
        tcpa_s, dcpa_m, _, _ = straight_line_cpa([0.0, 0.0], own_vel_body, obs_pos_body, obs_vel_body)
        return (tcpa_s, dcpa_m)

    def own_prediction_velocity_ne(self):
        current_ne = np.array([float(self.North), float(self.East)], dtype=float)
        to_goal_ne = self.goal_ne - current_ne
        to_goal_distance_m = float(np.linalg.norm(to_goal_ne))
        route_direction_ne = to_goal_ne / to_goal_distance_m if self.final_approach_active(to_goal_distance_m) and to_goal_distance_m >= 1e-06 else self.route_path_unit_ne
        return route_direction_ne * max(float(self.route_tracking_speed_m_s), self.apf_min_forward_speed)

    def obstacle_track_state_at(self, track, dt_s):
        dt_s = max(float(dt_s), 0.0)
        state = np.asarray(track.get('state', [np.nan, np.nan, 0.0, 0.0]), dtype=float).reshape(4)
        if not np.isfinite(state).all():
            return (None, None)
        velocity_ne = state[2:4].copy()
        accel_ne = self.obstacle_track_prediction_accel_ne(track)
        pos_ne = state[0:2] + velocity_ne * dt_s + 0.5 * accel_ne * dt_s * dt_s
        vel_ne = velocity_ne + accel_ne * dt_s
        return (pos_ne, vel_ne)

    def virtual_collision_visuals(self):
        visuals = []
        for obstacle in self.apf_virtual_obstacles:
            if not bool(obstacle.get('predicted_risk_active', False)):
                continue
            collision_position_ne = np.asarray(obstacle.get('collision_position_ne', [np.nan, np.nan]), dtype=float).reshape(2)
            if not np.isfinite(collision_position_ne).all():
                continue
            visuals.append({'track_id': int(obstacle.get('track_id', 0)), 'collision_position_ne': collision_position_ne.copy(), 'own_prediction_ne': np.asarray(obstacle.get('own_prediction_ne', [np.nan, np.nan]), dtype=float).reshape(2), 'tcpa_s': float(obstacle.get('tcpa_s', np.nan)), 'predicted_separation_m': float(obstacle.get('dcpa_m', np.nan)), 'collision_level': float(obstacle.get('collision_level', np.nan))})
        return visuals

    def apf_default_side_from_obstacle(self, obs_pos_body):
        body_angle_rad = float(np.arctan2(obs_pos_body[1], obs_pos_body[0]))
        if abs(wrap_angle(body_angle_rad)) > self.apf_activation_front_half_angle_rad:
            return 0.0
        if abs(body_angle_rad) <= np.deg2rad(5.0):
            if self.left_clearance_m > self.right_clearance_m + 0.05:
                return 1.0
            if self.right_clearance_m > self.left_clearance_m + 0.05:
                return -1.0
            return -1.0
        return -1.0 if body_angle_rad > 0.0 else 1.0

    def apf_obstacle_in_priority_front_sector(self, obstacle):
        obs_pos_body = np.asarray(obstacle.get('centre_body', [np.nan, np.nan]), dtype=float).reshape(2)
        if not np.isfinite(obs_pos_body).all():
            return False
        body_angle_rad = float(np.arctan2(obs_pos_body[1], obs_pos_body[0]))
        return abs(wrap_angle(body_angle_rad)) <= self.apf_priority_front_half_angle_rad

    def apf_lock_side(self, requested_side, obstacle_level):
        now_s = float(self.timefromstart) if self.timefromstart is not None else 0.0
        obstacle_close = np.isfinite(obstacle_level) and obstacle_level <= self.apf_side_lock_exit_level
        if requested_side == 0.0:
            if self.apf_side_lock_sign != 0.0 and (obstacle_close or now_s < self.apf_side_lock_until_s):
                self.apf_side_lock_active = True
                return self.apf_side_lock_sign
            self.apf_side_lock_sign = 0.0
            self.apf_side_lock_active = False
            return 0.0
        requested_side = float(np.sign(requested_side))
        if self.apf_side_lock_sign != 0.0 and (obstacle_close or now_s < self.apf_side_lock_until_s):
            self.apf_side_lock_active = True
            return self.apf_side_lock_sign
        self.apf_side_lock_sign = requested_side
        self.apf_side_lock_until_s = now_s + self.apf_side_lock_s
        self.apf_side_lock_active = True
        return self.apf_side_lock_sign

    def refresh_apf_side_lock(self, nearest_forward_level=np.inf):
        if self.apf_side_lock_sign == 0.0:
            self.apf_side_lock_active = False
            return False
        now_s = float(self.timefromstart) if self.timefromstart is not None else 0.0
        obstacle_close = np.isfinite(nearest_forward_level) and nearest_forward_level <= self.apf_side_lock_exit_level
        if obstacle_close or now_s < self.apf_side_lock_until_s:
            self.apf_side_lock_active = True
            return True
        self.apf_side_lock_sign = 0.0
        self.apf_side_lock_active = False
        return False

    def route_progress_and_point(self, lookahead_m=0.0):
        current_ne = np.array([float(self.North), float(self.East)], dtype=float)
        if self.route_path_length_m < 1e-09:
            return (0.0, self.goal_ne.copy())
        along_m = float(np.dot(current_ne - self.start_ne, self.route_path_unit_ne))
        closest_along_m = float(np.clip(along_m, 0.0, self.route_path_length_m))
        target_along_m = float(np.clip(along_m + lookahead_m, 0.0, self.route_path_length_m))
        target_ne = self.start_ne + target_along_m * self.route_path_unit_ne
        return (closest_along_m, target_ne)

    def goal_distance_m(self):
        current_ne = np.array([float(self.North), float(self.East)], dtype=float)
        return float(np.linalg.norm(self.goal_ne - current_ne))

    def final_approach_active(self, final_distance_m=None):
        if final_distance_m is None:
            final_distance_m = self.goal_distance_m()
        if not np.isfinite(final_distance_m):
            return False
        if final_distance_m <= self.final_approach_distance_m:
            return True
        if self.route_path_length_m < 1e-09:
            return True
        current_ne = np.array([float(self.North), float(self.East)], dtype=float)
        along_m = float(np.dot(current_ne - self.start_ne, self.route_path_unit_ne))
        return along_m >= self.route_path_length_m - self.final_approach_distance_m

    def apf_path_attraction_body(self):
        path_vec = self.goal_ne - self.start_ne
        path_len_sq = float(np.dot(path_vec, path_vec))
        if path_len_sq < 1e-09:
            return np.zeros(2, dtype=float)
        current_ne = np.array([float(self.North), float(self.East)], dtype=float)
        ratio = float(np.clip(np.dot(current_ne - self.start_ne, path_vec) / path_len_sq, 0.0, 1.0))
        closest_ne = self.start_ne + ratio * path_vec
        to_path_body = self.earth_point_to_body(closest_ne)
        distance_m = float(np.linalg.norm(to_path_body))
        if distance_m < self.apf_path_threshold_m or distance_m < 1e-06:
            return np.zeros(2, dtype=float)
        return self.apf_path_gain * to_path_body

    def apf_goal_attraction_body(self, target_body):
        target_body = np.asarray(target_body, dtype=float).reshape(2)
        distance_m = float(np.linalg.norm(target_body))
        if distance_m < 1e-06:
            return np.zeros(2, dtype=float)
        magnitude = self.apf_goal_gain * min(distance_m, self.apf_attraction_saturation_m)
        return magnitude * target_body / distance_m

    def apf_avoidance_needed(self):
        virtual_obstacles = self.update_apf_virtual_obstacles()
        for obstacle in self.lidar_obstacles + virtual_obstacles:
            obs_pos_body = np.asarray(obstacle.get('centre_body', [np.nan, np.nan]), dtype=float).reshape(2)
            if not np.isfinite(obs_pos_body).all():
                continue
            angle_rad = abs(wrap_angle(float(np.arctan2(obs_pos_body[1], obs_pos_body[0]))))
            if not bool(obstacle.get('virtual', False)) and angle_rad > self.apf_activation_front_half_angle_rad:
                continue
            if self.apf_repulsion_for_obstacle(obstacle, None, None)[1]:
                return True
        return False

    def compute_apf_control(self, t, u_track):
        active_params = self.apf_head_on_params
        final_approach = self.final_approach_active()
        if final_approach:
            target_ne = self.goal_ne.copy()
        else:
            _, target_ne = self.route_progress_and_point(float(active_params.get('route_lookahead_m', self.apf_route_lookahead_m)))
        target_body = self.earth_point_to_body(target_ne)
        goal_body = self.earth_point_to_body(self.goal_ne)
        path_force = np.zeros(2, dtype=float) if final_approach else self.apf_path_attraction_body()
        attractive_force = self.apf_goal_attraction_body(goal_body) + path_force
        force_body = attractive_force.copy()
        repulsive_force = np.zeros(2, dtype=float)
        own_vel_body = self.current_velocity_body()
        any_repulsion = False
        clearance_offset_m = 0.0
        self.reset_apf_diagnostics()
        self.apf_target_ne = target_ne.copy()
        self.apf_attractive_force_body = attractive_force
        virtual_obstacles = self.update_apf_virtual_obstacles()
        obstacles = self.lidar_obstacles + virtual_obstacles
        priority_obstacles = []
        secondary_obstacles = []
        for obstacle in obstacles:
            if self.apf_obstacle_in_priority_front_sector(obstacle):
                priority_obstacles.append(obstacle)
            else:
                secondary_obstacles.append(obstacle)
        for obstacle in priority_obstacles:
            repulsion, active, obstacle_params = self.apf_repulsion_for_obstacle(obstacle, target_body, own_vel_body)
            repulsive_force += repulsion
            force_body += repulsion
            any_repulsion = any_repulsion or active
            if active:
                active_params = obstacle_params
                clearance_offset_m = max(clearance_offset_m, self.apf_direction_clearance_m(obstacle, params=obstacle_params))
        if not any_repulsion:
            for obstacle in secondary_obstacles:
                repulsion, active, obstacle_params = self.apf_repulsion_for_obstacle(obstacle, target_body, own_vel_body)
                repulsive_force += repulsion
                force_body += repulsion
                any_repulsion = any_repulsion or active
                if active:
                    active_params = obstacle_params
                    clearance_offset_m = max(clearance_offset_m, self.apf_direction_clearance_m(obstacle, params=obstacle_params))
        if any_repulsion and self.apf_side_lock_sign != 0.0:
            route_normal_left_ne = np.array([self.route_path_unit_ne[1], -self.route_path_unit_ne[0]], dtype=float)
            offset_target_ne = target_ne + self.apf_side_lock_sign * max(clearance_offset_m, self.obstacle_min_pc2_m) * route_normal_left_ne
            offset_target_body = self.earth_point_to_body(offset_target_ne)
            offset_distance = float(np.linalg.norm(offset_target_body))
            if offset_distance > 1e-06:
                offset_force = self.apf_clearance_gain * min(offset_distance, self.apf_attraction_saturation_m) * offset_target_body / offset_distance
                force_body += offset_force
                attractive_force += offset_force
                self.apf_target_ne = offset_target_ne.copy()
        self.apf_attractive_force_body = attractive_force
        force_norm = float(np.linalg.norm(force_body))
        if not np.isfinite(force_norm) or force_norm < 1e-06:
            force_body = np.array([0.001, 0.0], dtype=float)
            force_norm = float(np.linalg.norm(force_body))
        self.apf_force_body = force_body
        self.apf_repulsive_force_body = repulsive_force
        steering_force = force_body.copy()
        if any_repulsion and steering_force[0] <= 0.0:
            side_sign = float(np.sign(steering_force[1]))
            if side_sign == 0.0:
                side_sign = self.apf_side_lock_sign
            if side_sign == 0.0:
                if self.left_clearance_m > self.right_clearance_m + 0.05:
                    side_sign = 1.0
                else:
                    side_sign = -1.0
            lateral_mag = max(abs(float(steering_force[1])), 0.5 * abs(float(force_body[0])), 0.2)
            steering_force[1] = side_sign * lateral_mag
            steering_force[0] = max(0.25 * lateral_mag, 0.05)
        self.apf_steering_force_body = steering_force
        now_s = float(self.timefromstart) if self.timefromstart is not None else 0.0
        self.apf_visual_hold_until_s = now_s + self.apf_visual_hold_s
        force_angle = wrap_angle(float(np.arctan2(steering_force[1], steering_force[0])))
        force_angle = float(np.clip(force_angle, -self.apf_heading_step_limit_rad, self.apf_heading_step_limit_rad))
        u_cmd = Vector(2)
        u_cmd[1, 0] = np.clip(-self.apf_heading_gain * force_angle / max(self.lastdt, 0.001), -self.w_max, self.w_max)
        u_cmd[0, 0] = float(np.clip(float(active_params.get('constant_descent_speed_m_s', self.apf_constant_descent_speed_m_s)), 0.0, self.v_max))
        if any_repulsion or self.apf_colreg_active or self.apf_side_lock_active:
            if self.apf_colreg_active:
                self.navigation_mode = 'apf_colreg'
            else:
                self.navigation_mode = 'apf_avoid'
        else:
            self.navigation_mode = 'apf_track'
        return u_cmd
