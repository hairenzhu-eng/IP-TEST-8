"""
Copyright (c) 2025 The uos_sess6072_build Authors.
Authors: Blair Thornton, Alec O'Loughlin, Miquel Massot
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""
import numpy as np
import time
from colreg_apf import constrain_velocity_to_axis, smooth_undirected_axis, straight_line_cpa
from math_sess6072 import Vector
from math_sess6072 import Vector
G = 2

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

    def apf_visual_hold_active(self):
        now_s = float(self.timefromstart) if self.timefromstart is not None else 0.0
        return now_s < self.apf_visual_hold_until_s

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

    def obstacle_measurement_covariance(self, measurement_ne, reference_pose=None):
        measurement_ne = np.asarray(measurement_ne, dtype=float).reshape(2)
        if reference_pose is None:
            reference_pose = getattr(self, 'lidar_reference_pose', None)
        if reference_pose is None:
            reference_pose = np.asarray(self.p_robot, dtype=float).reshape(3)
        else:
            reference_pose = np.asarray(reference_pose, dtype=float).reshape(3)
        base_variance = float(self.obstacle_ekf_measurement_std_m) ** 2
        yaw_variance = 0.0
        sigma = getattr(self, 'Sigma', None)
        if sigma is not None:
            sigma = np.asarray(sigma, dtype=float)
            if sigma.ndim == 2 and sigma.shape[0] > G and (sigma.shape[1] > G):
                yaw_variance = max(float(sigma[G, G]), 0.0)
        yaw_rate_rad_s = getattr(self, 'sensed_imu_yaw_rate_rad_s', None)
        if yaw_rate_rad_s is None or not np.isfinite(yaw_rate_rad_s):
            velocity = np.asarray(getattr(self, 'v_robot', np.zeros((3, 1))), dtype=float).reshape(-1)
            yaw_rate_rad_s = velocity[2] if len(velocity) > 2 else 0.0
        timing_std_s = max(float(getattr(self, 'lidar_time_sync_std_s', 0.02)), float(getattr(self, 'lidar_scan_time_s', 0.0)) / np.sqrt(12.0))
        yaw_variance += (float(yaw_rate_rad_s) * timing_std_s) ** 2
        relative_ne = measurement_ne - reference_pose[0:2]
        yaw_jacobian = np.array([-relative_ne[1], relative_ne[0]], dtype=float)
        return base_variance * np.eye(2, dtype=float) + yaw_variance * np.outer(yaw_jacobian, yaw_jacobian)

    def obstacle_ekf_update(self, state, covariance, measurement_ne, measurement_covariance=None):
        state = np.asarray(state, dtype=float).reshape(4)
        covariance = np.asarray(covariance, dtype=float).reshape(4, 4)
        measurement_ne = np.asarray(measurement_ne, dtype=float).reshape(2)
        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
        R_obs = self.obstacle_measurement_covariance(measurement_ne) if measurement_covariance is None else np.asarray(measurement_covariance, dtype=float).reshape(2, 2)
        innovation = measurement_ne - H @ state
        innovation_covariance = H @ covariance @ H.T + R_obs
        try:
            kalman_gain = covariance @ H.T @ np.linalg.inv(innovation_covariance)
        except np.linalg.LinAlgError:
            kalman_gain = covariance @ H.T @ np.linalg.pinv(innovation_covariance)
        corrected_state = state + kalman_gain @ innovation
        identity = np.eye(4, dtype=float)
        correction = identity - kalman_gain @ H
        corrected_covariance = correction @ covariance @ correction.T + kalman_gain @ R_obs @ kalman_gain.T
        return (corrected_state, corrected_covariance)

    def obstacle_track_motion_is_stable(self, track):
        sample_count = int(track.get('stats_sample_count', 0))
        hit_count = int(track.get('hit_count', 0))
        if sample_count < self.obstacle_prediction_min_samples or hit_count < self.obstacle_prediction_min_hits or int(track.get('miss_count', 0)) > 0:
            return False
        samples = track.get('motion_window', [])
        if len(samples) < 2:
            return False
        first_stamp = float(samples[0].get('stamp_s', 0.0))
        last_stamp = float(samples[-1].get('stamp_s', first_stamp))
        if last_stamp - first_stamp < self.obstacle_prediction_min_time_span_s:
            return False
        speed_m_s = float(track.get('speed_mean_m_s', 0.0))
        was_stable = bool(track.get('motion_stable', False))
        speed_threshold = self.apf_dynamic_exit_speed_threshold_m_s if was_stable else self.apf_dynamic_speed_threshold_m_s
        if not np.isfinite(speed_m_s) or speed_m_s < speed_threshold:
            return False
        return True

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

    def obstacle_track_prediction_ne(self, track):
        if not self.obstacle_ekf_prediction_enabled:
            return np.empty((0, 2), dtype=float)
        state = np.asarray(track.get('state', [np.nan, np.nan, 0.0, 0.0]), dtype=float).reshape(4)
        if not np.isfinite(state).all():
            return np.empty((0, 2), dtype=float)
        velocity_ne = state[2:4].copy()
        accel_ne = self.obstacle_track_prediction_accel_ne(track)
        step_s = max(float(self.obstacle_prediction_step_s), 0.001)
        horizon_s = float(self.obstacle_prediction_horizon_s)
        if not np.isfinite(horizon_s) or horizon_s <= 0.0:
            return np.empty((0, 2), dtype=float)
        times = np.arange(0.0, horizon_s, step_s, dtype=float)
        times = np.r_[times, horizon_s]
        return np.column_stack([state[0] + velocity_ne[0] * times + 0.5 * accel_ne[0] * times * times, state[1] + velocity_ne[1] * times + 0.5 * accel_ne[1] * times * times])

    def sync_obstacle_track_fields(self, track):
        state = np.asarray(track.get('state', [np.nan, np.nan, 0.0, 0.0]), dtype=float).reshape(4)
        track['pos_ne'] = state[0:2].copy()
        track['vel_ne'] = state[2:4].copy()
        pc1_m = float(track.get('pc1_m', self.obstacle_min_pc1_m))
        pc2_m = float(track.get('pc2_m', self.obstacle_min_pc2_m))
        if not np.isfinite(pc1_m) or pc1_m <= 0.0:
            pc1_m = self.obstacle_min_pc1_m
        if not np.isfinite(pc2_m) or pc2_m <= 0.0:
            pc2_m = self.obstacle_min_pc2_m
        track['pc1_m'] = max(pc1_m, self.obstacle_min_pc1_m)
        track['pc2_m'] = max(min(pc2_m, track['pc1_m']), self.obstacle_min_pc2_m)
        track['radius_m'] = 0.5 * track['pc1_m']
        track['equivalent_radius_m'] = track['radius_m']
        length_axis_ne = np.asarray(track.get('length_axis_ne', [1.0, 0.0]), dtype=float).reshape(2)
        axis_norm = float(np.linalg.norm(length_axis_ne))
        track['length_axis_ne'] = length_axis_ne / axis_norm if np.isfinite(length_axis_ne).all() and axis_norm >= 1e-06 else np.array([1.0, 0.0], dtype=float)
        if bool(track.get('motion_stable', False)) and track['pc1_m'] >= self.obstacle_axis_min_aspect_ratio * track['pc2_m']:
            state[2:4] = constrain_velocity_to_axis(state[2:4], track['length_axis_ne'], self.obstacle_axis_max_velocity_gap_rad)
            track['state'] = state
            track['vel_ne'] = state[2:4].copy()
        ekf_velocity_ne = track['vel_ne'].copy()
        speed_m_s = float(np.linalg.norm(ekf_velocity_ne))
        track['speed_m_s'] = speed_m_s
        heading_hold_speed_m_s = float(getattr(self, 'obstacle_heading_hold_speed_m_s', 0.03))
        if speed_m_s >= heading_hold_speed_m_s:
            heading_rad = float(np.arctan2(ekf_velocity_ne[1], ekf_velocity_ne[0]))
            track['heading_rad'] = heading_rad
            track['heading_deg'] = float(np.rad2deg(heading_rad))
            track['heading_axis_ne'] = ekf_velocity_ne / speed_m_s
        else:
            previous_heading_rad = float(track.get('heading_rad', np.nan))
            if not np.isfinite(previous_heading_rad):
                previous_heading_rad = float(np.arctan2(track['length_axis_ne'][1], track['length_axis_ne'][0]))
            track['heading_rad'] = previous_heading_rad
            track['heading_deg'] = float(np.rad2deg(previous_heading_rad))
            track['heading_axis_ne'] = np.array([np.cos(previous_heading_rad), np.sin(previous_heading_rad)])
        if not self.obstacle_ekf_prediction_enabled:
            track['prediction_model'] = 'disabled'
            track['prediction_ne'] = np.empty((0, 2), dtype=float)
            return
        if np.linalg.norm(self.obstacle_track_prediction_accel_ne(track)) > 0.0:
            track['prediction_model'] = 'sliding_window_acceleration'
        elif bool(track.get('motion_stable', False)):
            track['prediction_model'] = 'stable_constant_velocity'
        else:
            track['prediction_model'] = 'ekf_constant_velocity'
        track['prediction_ne'] = self.obstacle_track_prediction_ne(track)

    def make_obstacle_track(self, detection_ne, stamp_s, pc1_m=np.nan, pc2_m=np.nan, length_axis_ne=None):
        detection_ne = np.asarray(detection_ne, dtype=float).reshape(2)
        pc1_m = float(pc1_m) if np.isfinite(pc1_m) and pc1_m > 0.0 else self.obstacle_min_pc1_m
        pc2_m = float(pc2_m) if np.isfinite(pc2_m) and pc2_m > 0.0 else self.obstacle_min_pc2_m
        length_axis_ne = np.asarray([1.0, 0.0] if length_axis_ne is None else length_axis_ne, dtype=float).reshape(2)
        covariance = np.diag([self.obstacle_ekf_initial_position_std_m ** 2, self.obstacle_ekf_initial_position_std_m ** 2, self.obstacle_ekf_initial_velocity_std_m_s ** 2, self.obstacle_ekf_initial_velocity_std_m_s ** 2])
        track = {'id': self.apf_next_track_id, 'state': np.array([detection_ne[0], detection_ne[1], 0.0, 0.0], dtype=float), 'covariance': covariance, 'stamp_s': float(stamp_s), 'last_seen_s': float(stamp_s), 'hit_count': 1, 'miss_count': 0, 'history_ne': [], 'lidar_history_ne': [], 'motion_window': [], 'pc1_m': pc1_m, 'pc2_m': pc2_m, 'length_axis_ne': length_axis_ne}
        self.sync_obstacle_track_fields(track)
        self.append_obstacle_track_history(track, pc1_m=pc1_m, pc2_m=pc2_m, length_axis_ne=length_axis_ne, detection_ne=detection_ne, stamp_s=stamp_s)
        self.apf_next_track_id += 1
        return track

    def make_obstacle_track_candidate(self, detection_ne, stamp_s, pc1_m=np.nan, pc2_m=np.nan, length_axis_ne=None):
        detection_ne = np.asarray(detection_ne, dtype=float).reshape(2)
        length_axis_ne = np.asarray([1.0, 0.0] if length_axis_ne is None else length_axis_ne, dtype=float).reshape(2)
        pc1_m = float(pc1_m) if np.isfinite(pc1_m) and pc1_m > 0.0 else self.obstacle_min_pc1_m
        pc2_m = float(pc2_m) if np.isfinite(pc2_m) and pc2_m > 0.0 else self.obstacle_min_pc2_m
        return {'centre_ne': detection_ne.copy(), 'stamp_s': float(stamp_s), 'last_seen_s': float(stamp_s), 'hit_count': 1, 'pc1_m': pc1_m, 'pc2_m': pc2_m, 'length_axis_ne': length_axis_ne}

    def predict_obstacle_track_to_time(self, track, stamp_s):
        now = float(stamp_s)
        dt = max(now - float(track.get('stamp_s', now)), 0.0)
        state, covariance = self.obstacle_ekf_predict(track.get('state', np.r_[track.get('pos_ne', [np.nan, np.nan]), track.get('vel_ne', [0.0, 0.0])]), track.get('covariance', np.eye(4, dtype=float)), dt)
        track['state'] = state
        track['covariance'] = covariance
        track['stamp_s'] = now
        self.sync_obstacle_track_fields(track)

    def append_obstacle_track_history(self, track, pc1_m=np.nan, pc2_m=np.nan, length_axis_ne=None, detection_ne=None, stamp_s=None):
        stamp_s = float(stamp_s if stamp_s is not None else track.get('stamp_s', time.time()))
        pos_ne = np.asarray(track['pos_ne'], dtype=float).reshape(2).copy()
        vel_ne = np.asarray(track.get('vel_ne', [0.0, 0.0]), dtype=float).reshape(2).copy()
        if not np.isfinite(vel_ne).all():
            vel_ne = np.zeros(2, dtype=float)
        pc1_m = float(pc1_m) if np.isfinite(pc1_m) and pc1_m > 0.0 else float(track.get('pc1_m', self.obstacle_min_pc1_m))
        pc2_m = float(pc2_m) if np.isfinite(pc2_m) and pc2_m > 0.0 else float(track.get('pc2_m', self.obstacle_min_pc2_m))
        track['pc1_m'] = max(pc1_m, self.obstacle_min_pc1_m)
        track['pc2_m'] = max(min(pc2_m, track['pc1_m']), self.obstacle_min_pc2_m)
        if length_axis_ne is not None:
            length_axis_ne = np.asarray(length_axis_ne, dtype=float).reshape(2)
            axis_norm = float(np.linalg.norm(length_axis_ne))
            if np.isfinite(length_axis_ne).all() and axis_norm >= 1e-06:
                length_axis_ne = length_axis_ne / axis_norm
                previous_axis = np.asarray(track.get('length_axis_ne', length_axis_ne), dtype=float).reshape(2)
                if float(np.dot(length_axis_ne, previous_axis)) < 0.0:
                    length_axis_ne = -length_axis_ne
                track['length_axis_ne'] = smooth_undirected_axis(previous_axis, length_axis_ne, self.obstacle_axis_smoothing_alpha)
        history = track.setdefault('history_ne', [])
        history.append(pos_ne)
        if len(history) > self.obstacle_history_len:
            del history[:-self.obstacle_history_len]
        if detection_ne is not None:
            detection_ne = np.asarray(detection_ne, dtype=float).reshape(2)
            if np.isfinite(detection_ne).all():
                lidar_history = track.setdefault('lidar_history_ne', [])
                lidar_history.append(detection_ne.copy())
                if len(lidar_history) > self.obstacle_history_len:
                    del lidar_history[:-self.obstacle_history_len]
        speed_m_s = float(np.linalg.norm(vel_ne))
        heading_rad = float(np.arctan2(vel_ne[1], vel_ne[0])) if speed_m_s >= self.obstacle_heading_hold_speed_m_s else float(track.get('heading_rad', np.nan))
        motion_window = track.setdefault('motion_window', [])
        motion_window.append({'stamp_s': stamp_s, 'pos_ne': pos_ne, 'vel_ne': vel_ne, 'speed_m_s': speed_m_s, 'heading_rad': heading_rad, 'pc1_m': track['pc1_m'], 'pc2_m': track['pc2_m']})
        cutoff_s = stamp_s - max(float(self.obstacle_stats_window_s), 0.0)
        track['motion_window'] = [sample for sample in motion_window if float(sample.get('stamp_s', stamp_s)) >= cutoff_s][-self.obstacle_history_len:]
        self.update_obstacle_track_statistics(track)

    def update_obstacle_track_statistics(self, track):
        samples = track.get('motion_window', [])
        samples = [sample for sample in samples if np.isfinite(np.asarray(sample.get('pos_ne', [np.nan, np.nan]), dtype=float)).all()]
        track['stats_sample_count'] = int(len(samples))
        if not samples:
            track['velocity_mean_ne'] = np.asarray(track.get('vel_ne', [0.0, 0.0]), dtype=float).reshape(2)
            track['velocity_var_ne'] = np.zeros(2, dtype=float)
            track['speed_mean_m_s'] = float(np.linalg.norm(track['velocity_mean_ne']))
            track['speed_var_m2_s2'] = 0.0
            track['heading_mean_rad'] = np.nan
            track['heading_var_rad2'] = np.nan
            track['heading_circular_variance'] = np.nan
            track['pc1_mean_m'] = track.get('pc1_m', self.obstacle_min_pc1_m)
            track['pc2_mean_m'] = track.get('pc2_m', self.obstacle_min_pc2_m)
            track['pc1_var_m2'] = 0.0
            track['pc2_var_m2'] = 0.0
            track['accel_ne'] = np.zeros(2, dtype=float)
            track['accel_var_ne'] = np.zeros(2, dtype=float)
            track['motion_stable'] = False
            self.sync_obstacle_track_fields(track)
            return
        velocities = np.asarray([sample['vel_ne'] for sample in samples], dtype=float)
        finite_vel = np.isfinite(velocities).all(axis=1)
        velocities = velocities[finite_vel]
        if len(velocities) > 0:
            track['velocity_mean_ne'] = np.mean(velocities, axis=0)
        else:
            track['velocity_mean_ne'] = np.asarray(track.get('vel_ne', [0.0, 0.0]), dtype=float).reshape(2)
        if len(velocities) > 0:
            track['velocity_var_ne'] = np.var(velocities, axis=0)
        else:
            track['velocity_var_ne'] = np.zeros(2, dtype=float)
        speeds = np.asarray([sample['speed_m_s'] for sample in samples], dtype=float)
        speeds = speeds[np.isfinite(speeds)]
        fitted_speed_m_s = float(np.linalg.norm(track['velocity_mean_ne']))
        if len(speeds) > 0:
            track['speed_mean_m_s'] = fitted_speed_m_s
            track['speed_var_m2_s2'] = float(np.var(speeds))
        else:
            track['speed_mean_m_s'] = fitted_speed_m_s
            track['speed_var_m2_s2'] = 0.0
        headings = np.asarray([sample['heading_rad'] for sample in samples], dtype=float)
        headings = headings[np.isfinite(headings)]
        if len(headings) > 0:
            sin_mean = float(np.mean(np.sin(headings)))
            cos_mean = float(np.mean(np.cos(headings)))
            heading_mean = float(np.arctan2(sin_mean, cos_mean))
            heading_error = wrap_angle(headings - heading_mean)
            resultant_length = float(np.hypot(sin_mean, cos_mean))
            track['heading_mean_rad'] = heading_mean
            track['heading_var_rad2'] = float(np.var(heading_error))
            track['heading_circular_variance'] = float(1.0 - np.clip(resultant_length, 0.0, 1.0))
        else:
            track['heading_mean_rad'] = np.nan
            track['heading_var_rad2'] = np.nan
            track['heading_circular_variance'] = np.nan
        pc1_values = np.asarray([sample['pc1_m'] for sample in samples], dtype=float)
        pc2_values = np.asarray([sample['pc2_m'] for sample in samples], dtype=float)
        pc1_values = pc1_values[np.isfinite(pc1_values) & (pc1_values > 0.0)]
        pc2_values = pc2_values[np.isfinite(pc2_values) & (pc2_values > 0.0)]
        track['pc1_mean_m'] = max(float(np.mean(pc1_values)), self.obstacle_min_pc1_m) if len(pc1_values) > 0 else float(track.get('pc1_m', self.obstacle_min_pc1_m))
        track['pc2_mean_m'] = max(float(np.mean(pc2_values)), self.obstacle_min_pc2_m) if len(pc2_values) > 0 else float(track.get('pc2_m', self.obstacle_min_pc2_m))
        track['pc2_mean_m'] = min(track['pc2_mean_m'], track['pc1_mean_m'])
        track['pc1_var_m2'] = float(np.var(pc1_values)) if len(pc1_values) > 0 else 0.0
        track['pc2_var_m2'] = float(np.var(pc2_values)) if len(pc2_values) > 0 else 0.0
        track['pc1_m'] = track['pc1_mean_m']
        track['pc2_m'] = track['pc2_mean_m']
        velocity_samples = [sample for sample in samples if np.isfinite(np.asarray(sample.get('vel_ne', [np.nan, np.nan]), dtype=float)).all()]
        velocity_samples.sort(key=lambda sample: float(sample.get('stamp_s', 0.0)))
        accels = []
        for prev_sample, next_sample in zip(velocity_samples[:-1], velocity_samples[1:]):
            dt = float(next_sample.get('stamp_s', 0.0)) - float(prev_sample.get('stamp_s', 0.0))
            if dt <= 0.001:
                continue
            prev_vel = np.asarray(prev_sample['vel_ne'], dtype=float).reshape(2)
            next_vel = np.asarray(next_sample['vel_ne'], dtype=float).reshape(2)
            accels.append((next_vel - prev_vel) / dt)
        if accels:
            accel_ne = np.mean(np.asarray(accels, dtype=float), axis=0)
            accel_norm = float(np.linalg.norm(accel_ne))
            if accel_norm > self.obstacle_max_accel_m_s2:
                accel_ne *= self.obstacle_max_accel_m_s2 / max(accel_norm, 1e-06)
            track['accel_ne'] = accel_ne
            track['accel_var_ne'] = np.var(np.asarray(accels, dtype=float), axis=0)
        else:
            track['accel_ne'] = np.zeros(2, dtype=float)
            track['accel_var_ne'] = np.zeros(2, dtype=float)
        track['motion_stable'] = self.obstacle_track_motion_is_stable(track)
        self.sync_obstacle_track_fields(track)

    def annotate_lidar_obstacle_with_track(self, obstacle, track):
        prediction_ne = np.asarray(track.get('prediction_ne', np.empty((0, 2))), dtype=float)
        obstacle['track_id'] = int(track['id'])
        obstacle['velocity_ne'] = np.asarray(track['vel_ne'], dtype=float).reshape(2).tolist()
        obstacle['velocity_mean_ne'] = np.asarray(track.get('velocity_mean_ne', track['vel_ne']), dtype=float).reshape(2).tolist()
        obstacle['velocity_var_ne'] = np.asarray(track.get('velocity_var_ne', [0.0, 0.0]), dtype=float).reshape(2).tolist()
        obstacle['accel_ne'] = np.asarray(track.get('accel_ne', [0.0, 0.0]), dtype=float).reshape(2).tolist()
        obstacle['speed_m_s'] = float(track.get('speed_m_s', 0.0))
        obstacle['speed_mean_m_s'] = float(track.get('speed_mean_m_s', obstacle['speed_m_s']))
        obstacle['speed_var_m2_s2'] = float(track.get('speed_var_m2_s2', 0.0))
        obstacle['heading_rad'] = float(track.get('heading_rad', np.nan))
        obstacle['heading_deg'] = float(track.get('heading_deg', np.nan))
        obstacle['heading_mean_rad'] = float(track.get('heading_mean_rad', np.nan))
        obstacle['heading_var_rad2'] = float(track.get('heading_var_rad2', np.nan))
        obstacle['pc1_m'] = float(track.get('pc1_mean_m', track.get('pc1_m', self.obstacle_min_pc1_m)))
        obstacle['pc2_m'] = float(track.get('pc2_mean_m', track.get('pc2_m', self.obstacle_min_pc2_m)))
        obstacle['pc1_var_m2'] = float(track.get('pc1_var_m2', 0.0))
        obstacle['pc2_var_m2'] = float(track.get('pc2_var_m2', 0.0))
        obstacle['length_axis_ne'] = np.asarray(track.get('heading_axis_ne', track.get('length_axis_ne', [1.0, 0.0])), dtype=float).reshape(2).tolist()
        obstacle['equivalent_radius_m'] = float(track.get('equivalent_radius_m', self.obstacle_min_equivalent_radius_m))
        obstacle['stats_sample_count'] = int(track.get('stats_sample_count', 0))
        obstacle['motion_stable'] = bool(track.get('motion_stable', False))
        obstacle['prediction_model'] = track.get('prediction_model', 'ekf_constant_velocity')
        obstacle['predicted_trajectory_ne'] = prediction_ne.tolist()
        lidar_points_ne = np.asarray(obstacle.get('points_ne', []), dtype=float)
        if lidar_points_ne.ndim == 2 and lidar_points_ne.shape[1] == 2:
            track['lidar_points_ne'] = lidar_points_ne.tolist()

    def prune_obstacle_tracks(self, now):
        self.apf_obstacle_tracks = [track for track in self.apf_obstacle_tracks if now - float(track.get('last_seen_s', track.get('stamp_s', now))) <= self.apf_track_timeout_s and int(track.get('miss_count', 0)) <= 5]

    def update_apf_obstacle_tracks(self, stamp_s):
        if not self.obstacle_ekf_prediction_enabled:
            self.apf_obstacle_tracks = []
            self.apf_virtual_obstacles = []
            tracked_fields = {'track_id', 'velocity_ne', 'velocity_mean_ne', 'velocity_var_ne', 'accel_ne', 'speed_m_s', 'speed_mean_m_s', 'speed_var_m2_s2', 'heading_rad', 'heading_deg', 'heading_mean_rad', 'heading_var_rad2', 'pc1_m', 'pc2_m', 'pc1_var_m2', 'pc2_var_m2', 'stats_sample_count', 'motion_stable', 'prediction_model', 'predicted_trajectory_ne'}
            for obstacle in self.lidar_obstacles:
                for field in tracked_fields:
                    obstacle.pop(field, None)
            return
        now = float(stamp_s if stamp_s is not None else time.time())
        confirmation_hits = max(int(getattr(self, 'obstacle_track_confirmation_hits', 2)), 2)
        detections = []
        for obstacle_index, obstacle in enumerate(self.lidar_obstacles):
            centre_ne = np.asarray(obstacle.get('centre_ne', [np.nan, np.nan]), dtype=float).reshape(2)
            if np.isfinite(centre_ne).all():
                pc1_m = float(obstacle.get('pc1_m', self.obstacle_min_pc1_m))
                pc2_m = float(obstacle.get('pc2_m', self.obstacle_min_pc2_m))
                length_axis_ne = np.asarray(obstacle.get('length_axis_ne', [1.0, 0.0]), dtype=float).reshape(2)
                measurement_covariance = np.asarray(obstacle.get('measurement_covariance', self.obstacle_measurement_covariance(centre_ne)), dtype=float).reshape(2, 2)
                detections.append((obstacle_index, centre_ne, pc1_m, pc2_m, length_axis_ne, measurement_covariance))
        if not detections:
            self.apf_obstacle_track_candidates = [candidate for candidate in self.apf_obstacle_track_candidates if now - float(candidate.get('last_seen_s', candidate.get('stamp_s', now))) <= self.apf_track_timeout_s]
            for track in self.apf_obstacle_tracks:
                self.predict_obstacle_track_to_time(track, now)
                track['miss_count'] = int(track.get('miss_count', 0)) + 1
            self.prune_obstacle_tracks(now)
            return
        detection_positions = np.asarray([detection for _, detection, _, _, _, _ in detections], dtype=float)
        predicted_tracks = []
        candidates = []
        matched_candidates = set()
        promoted_candidates = set()
        for track_index, track in enumerate(self.apf_obstacle_tracks):
            dt = max(now - float(track.get('stamp_s', now)), 0.0)
            predicted_state, predicted_covariance = self.obstacle_ekf_predict(track.get('state', np.r_[track.get('pos_ne', [np.nan, np.nan]), track.get('vel_ne', [0.0, 0.0])]), track.get('covariance', np.eye(4, dtype=float)), dt)
            predicted_tracks.append((predicted_state, predicted_covariance))
            predicted_pos = predicted_state[0:2]
            for detection_index, detection in enumerate(detection_positions):
                distance = float(np.linalg.norm(detection - predicted_pos))
                candidates.append((distance, track_index, detection_index))
        candidates.sort(key=lambda item: item[0])
        assigned_tracks = set()
        assigned_detections = set()
        detection_track = {}
        for distance, track_index, detection_index in candidates:
            if distance > self.apf_track_association_m:
                break
            if track_index in assigned_tracks or detection_index in assigned_detections:
                continue
            track = self.apf_obstacle_tracks[track_index]
            detection = detection_positions[detection_index]
            predicted_state, predicted_covariance = predicted_tracks[track_index]
            corrected_state, corrected_covariance = self.obstacle_ekf_update(predicted_state, predicted_covariance, detection, detections[detection_index][5])
            track['state'] = corrected_state
            track['covariance'] = corrected_covariance
            track['stamp_s'] = now
            track['last_seen_s'] = now
            track['hit_count'] = int(track.get('hit_count', 0)) + 1
            track['miss_count'] = 0
            self.sync_obstacle_track_fields(track)
            self.append_obstacle_track_history(track, pc1_m=detections[detection_index][2], pc2_m=detections[detection_index][3], length_axis_ne=detections[detection_index][4], detection_ne=detection, stamp_s=now)
            assigned_tracks.add(track_index)
            assigned_detections.add(detection_index)
            detection_track[detection_index] = track
        for track_index, track in enumerate(self.apf_obstacle_tracks):
            if track_index not in assigned_tracks:
                predicted_state, predicted_covariance = predicted_tracks[track_index]
                track['state'] = predicted_state
                track['covariance'] = predicted_covariance
                track['stamp_s'] = now
                track['miss_count'] = int(track.get('miss_count', 0)) + 1
                self.sync_obstacle_track_fields(track)
        for detection_index, detection in enumerate(detections):
            if detection_index in assigned_detections:
                continue
            _, detection_ne, pc1_m, pc2_m, length_axis_ne, _ = detection
            best_candidate_index = None
            best_candidate_distance = np.inf
            for candidate_index, candidate in enumerate(self.apf_obstacle_track_candidates):
                if candidate_index in matched_candidates:
                    continue
                candidate_pos = np.asarray(candidate.get('centre_ne', [np.nan, np.nan]), dtype=float).reshape(2)
                if not np.isfinite(candidate_pos).all():
                    continue
                distance = float(np.linalg.norm(detection_ne - candidate_pos))
                if distance < best_candidate_distance:
                    best_candidate_distance = distance
                    best_candidate_index = candidate_index
            if best_candidate_index is not None and best_candidate_distance <= self.apf_track_association_m:
                candidate = self.apf_obstacle_track_candidates[best_candidate_index]
                candidate['centre_ne'] = detection_ne.copy()
                candidate['last_seen_s'] = now
                candidate['hit_count'] = int(candidate.get('hit_count', 0)) + 1
                candidate['pc1_m'] = pc1_m
                candidate['pc2_m'] = pc2_m
                candidate['length_axis_ne'] = length_axis_ne
                matched_candidates.add(best_candidate_index)
                if int(candidate['hit_count']) >= confirmation_hits:
                    track = self.make_obstacle_track(detection_ne, now, pc1_m=pc1_m, pc2_m=pc2_m, length_axis_ne=length_axis_ne)
                    self.apf_obstacle_tracks.append(track)
                    assigned_detections.add(detection_index)
                    detection_track[detection_index] = track
                    promoted_candidates.add(best_candidate_index)
                continue
            self.apf_obstacle_track_candidates.append(self.make_obstacle_track_candidate(detection_ne, now, pc1_m=pc1_m, pc2_m=pc2_m, length_axis_ne=length_axis_ne))
        for detection_index, track in detection_track.items():
            obstacle_index, _, _, _, _, _ = detections[detection_index]
            self.annotate_lidar_obstacle_with_track(self.lidar_obstacles[obstacle_index], track)
        self.apf_obstacle_track_candidates = [candidate for candidate_index, candidate in enumerate(self.apf_obstacle_track_candidates) if candidate_index not in promoted_candidates if now - float(candidate.get('last_seen_s', candidate.get('stamp_s', now))) <= self.apf_track_timeout_s]
        self.prune_obstacle_tracks(now)

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

    def apf_pass_astern_side_from_velocity(self, obs_vel_body):
        obs_vel_body = np.asarray(obs_vel_body, dtype=float).reshape(2)
        if not np.isfinite(obs_vel_body).all():
            return 0.0
        if float(np.linalg.norm(obs_vel_body)) < self.apf_dynamic_speed_threshold_m_s:
            return 0.0
        lateral_speed = float(obs_vel_body[1])
        if abs(lateral_speed) < self.apf_dynamic_speed_threshold_m_s:
            return 0.0
        return float(np.sign(lateral_speed))

    def apf_classify_encounter(self, obs_pos_body, obs_vel_body, own_vel_body):
        obs_pos_body = np.asarray(obs_pos_body, dtype=float).reshape(2)
        obs_vel_body = np.asarray(obs_vel_body, dtype=float).reshape(2)
        own_vel_body = np.asarray(own_vel_body, dtype=float).reshape(2)
        body_angle_rad = float(np.arctan2(obs_pos_body[1], obs_pos_body[0]))
        bearing_starboard_deg = float(np.rad2deg(wrap_angle(-body_angle_rad)))
        obs_speed = float(np.linalg.norm(obs_vel_body))
        own_speed = float(np.linalg.norm(own_vel_body))
        dynamic_obstacle = obs_speed >= self.apf_dynamic_speed_threshold_m_s
        relative_heading_deg = np.nan
        if 'overtaking' in str(getattr(self, 'webots_environment', '')).lower():
            return ('overtaking', -1.0, 'COLREG Rule 13: overtake on starboard side')
        if dynamic_obstacle:
            relative_heading_deg = abs(float(np.rad2deg(wrap_angle(np.arctan2(obs_vel_body[1], obs_vel_body[0])))))
        if np.isfinite(relative_heading_deg) and abs(bearing_starboard_deg) <= 22.5 and (relative_heading_deg >= 157.5):
            return ('head_on', -1.0, 'COLREG Rule 14: alter to starboard')
        if np.isfinite(relative_heading_deg) and abs(bearing_starboard_deg) <= 67.5 and (relative_heading_deg <= 67.5) and (own_speed > obs_speed + self.apf_dynamic_speed_threshold_m_s):
            return ('overtaking', -1.0, 'COLREG Rule 13: overtake on starboard side')
        pass_astern_side = self.apf_pass_astern_side_from_velocity(obs_vel_body)
        if dynamic_obstacle and 0.0 < bearing_starboard_deg <= 112.5:
            requested_side = pass_astern_side if pass_astern_side != 0.0 else -1.0
            return ('crossing_from_starboard', requested_side, 'COLREG Rule 15: give way, pass astern')
        if dynamic_obstacle and -112.5 <= bearing_starboard_deg < 0.0:
            return ('crossing_from_port', 0.0, 'COLREG Rule 17: stand on')
        return ('static_obstacle', 0.0, 'none')

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

    def apf_primary_encounter_mode(self):
        own_vel_body = self.current_velocity_body()
        obstacles = self.lidar_obstacles + self.update_apf_virtual_obstacles()
        saw_crossing = False
        for obstacle in obstacles:
            obs_pos_body = np.asarray(obstacle.get('centre_body', [np.nan, np.nan]), dtype=float).reshape(2)
            if not np.isfinite(obs_pos_body).all():
                continue
            angle_rad = abs(wrap_angle(float(np.arctan2(obs_pos_body[1], obs_pos_body[0]))))
            if not bool(obstacle.get('virtual', False)) and angle_rad > self.apf_activation_front_half_angle_rad:
                continue
            if bool(obstacle.get('virtual', False)):
                encounter = str(obstacle.get('encounter_mode', 'crossing'))
            else:
                track = self.apf_track_for_obstacle(obstacle)
                if track is None or not self.obstacle_track_motion_is_stable(track):
                    continue
                track_velocity_ne = np.asarray(track.get('vel_ne', [np.nan, np.nan]), dtype=float).reshape(2)
                if not np.isfinite(track_velocity_ne).all():
                    continue
                obs_vel_body = self.earth_vector_to_body(track_velocity_ne)
                encounter, _, _ = self.apf_classify_encounter(obs_pos_body, obs_vel_body, own_vel_body)
            if encounter == 'head_on':
                return 'head_on'
            if encounter == 'overtaking':
                return 'overtaking'
            if encounter in {'crossing_from_starboard', 'crossing_from_port'}:
                saw_crossing = True
        return 'crossing' if saw_crossing else 'crossing'

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

    def apf_overtaking_close_turn_command(self, encounter, nearest_obstacle_distance_m, force_angle, surge_speed):
        overtaking_world = 'overtaking' in str(getattr(self, 'webots_environment', '')).lower()
        if encounter not in {'overtaking', 'static_obstacle', 'none'} and (not overtaking_world) or nearest_obstacle_distance_m > self.apf_overtaking_close_distance_m:
            return (force_angle, surge_speed)
        force_angle = max(abs(float(force_angle)), self.apf_overtaking_close_turn_angle_rad)
        heading_offset = abs(wrap_angle(float(self.Yaw) - self.route_heading_rad))
        if heading_offset < self.apf_overtaking_turn_release_angle_rad:
            surge_speed = min(surge_speed, self.apf_overtaking_close_turn_speed_m_s)
        return (force_angle, surge_speed)

    def compute_apf_control(self, t, u_track):
        primary_encounter = 'overtaking'
        active_params = self.apf_overtaking_params
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
        nearest_obstacle_distance_m = np.inf
        priority_obstacles = []
        secondary_obstacles = []
        for obstacle in obstacles:
            obs_pos_body = np.asarray(obstacle.get('centre_body', [np.nan, np.nan]), dtype=float).reshape(2)
            if not bool(obstacle.get('virtual', False)) and np.isfinite(obs_pos_body).all() and (obs_pos_body[0] > 0.0) and (abs(float(np.arctan2(obs_pos_body[1], obs_pos_body[0]))) <= np.deg2rad(22.5)):
                nearest_obstacle_distance_m = min(nearest_obstacle_distance_m, float(np.linalg.norm(obs_pos_body)))
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
        surge_speed = float(active_params.get('constant_descent_speed_m_s', self.apf_constant_descent_speed_m_s))
        force_angle, surge_speed = self.apf_overtaking_close_turn_command(primary_encounter, nearest_obstacle_distance_m, force_angle, surge_speed)
        u_cmd = Vector(2)
        u_cmd[1, 0] = np.clip(-self.apf_heading_gain * force_angle / max(self.lastdt, 0.001), -self.w_max, self.w_max)
        u_cmd[0, 0] = float(np.clip(surge_speed, 0.0, self.v_max))
        if any_repulsion or self.apf_colreg_active or self.apf_side_lock_active:
            if self.apf_colreg_active:
                self.navigation_mode = 'apf_colreg'
            else:
                self.navigation_mode = 'apf_avoid'
        else:
            self.navigation_mode = 'apf_track'
        return u_cmd
