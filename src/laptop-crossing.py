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

def Vector(dim):
    return np.zeros((dim, 1), dtype=float)

def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def ellipse_level_and_away(offset, length_axis, semi_length_m, semi_width_m):
    offset = np.asarray(offset, dtype=float).reshape(2)
    length_axis = np.asarray(length_axis, dtype=float).reshape(2)
    axis_norm = float(np.linalg.norm(length_axis))
    if axis_norm < 1e-09:
        length_axis = np.array([1.0, 0.0], dtype=float)
    else:
        length_axis = length_axis / axis_norm
    width_axis = np.array([-length_axis[1], length_axis[0]], dtype=float)
    semi_length_m = max(float(semi_length_m), 1e-06)
    semi_width_m = max(float(semi_width_m), 1e-06)
    along = float(np.dot(offset, length_axis))
    across = float(np.dot(offset, width_axis))
    level = float(np.hypot(along / semi_length_m, across / semi_width_m))
    gradient = along / (semi_length_m * semi_length_m) * length_axis + across / (semi_width_m * semi_width_m) * width_axis
    gradient_norm = float(np.linalg.norm(gradient))
    if gradient_norm < 1e-09:
        offset_norm = float(np.linalg.norm(offset))
        away = offset / offset_norm if offset_norm >= 1e-09 else -length_axis
    else:
        away = gradient / gradient_norm
    return (level, away)

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

    def body_vector_to_earth(self, vector_body):
        pose = np.asarray(self.p_robot, dtype=float).reshape(-1)
        yaw = pose[2]
        c = np.cos(yaw)
        s = np.sin(yaw)
        vector_body = np.asarray(vector_body, dtype=float).reshape(2)
        return np.array([c * vector_body[0] - s * vector_body[1], s * vector_body[0] + c * vector_body[1]])

    def reset_apf_diagnostics(self, clear_visual=True):
        if clear_visual:
            self.apf_force_body = np.zeros(2, dtype=float)
            self.apf_repulsive_force_body = np.zeros(2, dtype=float)
            self.apf_attractive_force_body = np.zeros(2, dtype=float)
            self.apf_steering_force_body = np.zeros(2, dtype=float)
            self.apf_target_ne = np.array([np.nan, np.nan], dtype=float)
            self.apf_virtual_obstacles = []
            self.apf_visual_hold_until_s = 0.0
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
        q = float(self.obstacle_ekf_accel_std_m_s2) ** 2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        return q * np.array([[0.25 * dt4, 0.0, 0.5 * dt3, 0.0], [0.0, 0.25 * dt4, 0.0, 0.5 * dt3], [0.5 * dt3, 0.0, dt2, 0.0], [0.0, 0.5 * dt3, 0.0, dt2]], dtype=float)

    def obstacle_measurement_covariance(self, centre_body):
        centre_body = np.asarray(centre_body, dtype=float).reshape(2)
        range_m = float(np.linalg.norm(centre_body)) if np.isfinite(centre_body).all() else 0.0
        yaw_rate_rad_s = float(abs(getattr(self, 'sensed_imu_yaw_rate_rad_s', 0.0) or 0.0))
        measurement_std_m = float(self.obstacle_ekf_measurement_std_m)
        lateral_std_m = measurement_std_m * (1.0 + 0.08 * range_m + 0.25 * range_m * yaw_rate_rad_s)
        return np.diag([measurement_std_m ** 2, lateral_std_m ** 2])

    def obstacle_ekf_predict(self, state, covariance, dt):
        dt = max(float(dt), 0.0)
        state = np.asarray(state, dtype=float).reshape(4)
        covariance = np.asarray(covariance, dtype=float).reshape(4, 4)
        transition = np.array([[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=float)
        return (transition @ state, transition @ covariance @ transition.T + self.obstacle_ekf_process_noise(dt))

    def obstacle_ekf_update(self, state, covariance, measurement_ne, measurement_covariance=None):
        state = np.asarray(state, dtype=float).reshape(4)
        covariance = np.asarray(covariance, dtype=float).reshape(4, 4)
        measurement_ne = np.asarray(measurement_ne, dtype=float).reshape(2)
        observation = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
        noise = self.obstacle_ekf_measurement_std_m ** 2 * np.eye(2, dtype=float) if measurement_covariance is None else np.asarray(measurement_covariance, dtype=float).reshape(2, 2)
        innovation = measurement_ne - observation @ state
        innovation_covariance = observation @ covariance @ observation.T + noise
        gain = covariance @ observation.T @ np.linalg.pinv(innovation_covariance)
        corrected_state = state + gain @ innovation
        correction = np.eye(4, dtype=float) - gain @ observation
        corrected_covariance = correction @ covariance @ correction.T + gain @ noise @ gain.T
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
        if not np.isfinite(speed_m_s) or speed_m_s < max(speed_threshold, getattr(self, 'obstacle_prediction_min_speed_m_s', 0.0)):
            return False
        displacement_m = float(track.get('displacement_m', 0.0))
        if displacement_m < float(getattr(self, 'obstacle_prediction_min_displacement_m', 0.0)):
            return False
        return True

    def obstacle_track_prediction_ne(self, track):
        if not self.obstacle_ekf_prediction_enabled:
            return np.empty((0, 2), dtype=float)
        state = np.asarray(track.get('state', [np.nan] * 4), dtype=float).reshape(4)
        if not np.isfinite(state).all():
            return np.empty((0, 2), dtype=float)
        horizon_s = float(track.get('collision_time_s', np.nan))
        if not np.isfinite(horizon_s) or horizon_s <= 0.0:
            horizon_s = float(getattr(self, 'obstacle_prediction_horizon_s', 0.0))
        if horizon_s <= 0.0:
            return np.empty((0, 2), dtype=float)
        step_s = max(float(self.obstacle_prediction_step_s), 0.001)
        times_s = np.arange(0.0, horizon_s, step_s, dtype=float)
        times_s = np.r_[times_s, horizon_s]
        return state[:2] + times_s[:, None] * state[2:4]

    def obstacle_track_regularize_velocity(self, track):
        state = np.asarray(track.get('state', [np.nan] * 4), dtype=float).reshape(4)
        speed_m_s = float(np.linalg.norm(state[2:4]))
        if speed_m_s < float(self.obstacle_ekf_static_speed_reset_m_s):
            track['state'][2:4] = 0.0
        elif speed_m_s > float(self.obstacle_v_max_m_s):
            track['state'][2:4] *= float(self.obstacle_v_max_m_s) / speed_m_s

    def sync_obstacle_track_fields(self, track):
        state = np.asarray(track.get('state', [np.nan] * 4), dtype=float).reshape(4)
        covariance = np.asarray(track.get('covariance', np.eye(4)), dtype=float).reshape(4, 4)
        track['state'] = state
        track['covariance'] = covariance
        track['pos_ne'] = state[0:2].copy()
        track['vel_ne'] = state[2:4].copy()
        track['position_uncertainty_m2'] = float(np.trace(covariance[0:2, 0:2]))
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
        display_velocity_ne = np.asarray(track['vel_ne'], dtype=float).reshape(2)
        if not np.isfinite(display_velocity_ne).all():
            display_velocity_ne = track['vel_ne']
        speed_m_s = float(np.linalg.norm(display_velocity_ne))
        track['speed_m_s'] = speed_m_s
        previous_heading_rad = float(track.get('heading_rad', np.nan))
        if speed_m_s >= float(getattr(self, 'obstacle_heading_hold_speed_m_s', 0.03)):
            heading_rad = float(np.arctan2(display_velocity_ne[1], display_velocity_ne[0]))
            track['heading_rad'] = heading_rad
            track['heading_deg'] = float(np.rad2deg(heading_rad))
            track['heading_axis_ne'] = display_velocity_ne / speed_m_s
        else:
            if not np.isfinite(previous_heading_rad):
                previous_heading_rad = float(np.arctan2(track['length_axis_ne'][1], track['length_axis_ne'][0]))
            track['heading_rad'] = wrap_angle(previous_heading_rad)
            track['heading_deg'] = float(np.rad2deg(track['heading_rad']))
            track['heading_axis_ne'] = track['length_axis_ne'].copy()
        if not self.obstacle_ekf_prediction_enabled:
            track['prediction_model'] = 'disabled'
            track['prediction_ne'] = np.empty((0, 2), dtype=float)
            return
        if bool(track.get('motion_stable', False)):
            track['prediction_model'] = 'cv_ekf'
        else:
            track['prediction_model'] = 'cv_ekf_warmup'
        track['prediction_ne'] = self.obstacle_track_prediction_ne(track)

    def make_obstacle_track(self, detection_ne, stamp_s, pc1_m=np.nan, pc2_m=np.nan, length_axis_ne=None):
        detection_ne = np.asarray(detection_ne, dtype=float).reshape(2)
        pc1_m = float(pc1_m) if np.isfinite(pc1_m) and pc1_m > 0.0 else self.obstacle_min_pc1_m
        pc2_m = float(pc2_m) if np.isfinite(pc2_m) and pc2_m > 0.0 else self.obstacle_min_pc2_m
        length_axis_ne = np.asarray([1.0, 0.0] if length_axis_ne is None else length_axis_ne, dtype=float).reshape(2)
        covariance = np.diag([self.obstacle_ekf_initial_position_std_m ** 2, self.obstacle_ekf_initial_position_std_m ** 2, self.obstacle_ekf_initial_velocity_std_m_s ** 2, self.obstacle_ekf_initial_velocity_std_m_s ** 2])
        track = {'id': self.apf_next_track_id, 'state': np.array([detection_ne[0], detection_ne[1], 0.0, 0.0], dtype=float), 'covariance': covariance, 'stamp_s': float(stamp_s), 'last_seen_s': float(stamp_s), 'hit_count': 1, 'miss_count': 0, 'history_ne': [], 'lidar_history_ne': [], 'motion_window': [], 'pc1_m': pc1_m, 'pc2_m': pc2_m, 'length_axis_ne': length_axis_ne, 'raw_detection_ne': detection_ne.copy()}
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
        if not np.isfinite(dt) or dt < 0.001:
            return
        state, covariance = self.obstacle_ekf_predict(track.get('state', np.r_[track.get('pos_ne', [np.nan, np.nan]), 0.0, 0.0]), track.get('covariance', np.eye(4, dtype=float)), dt)
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
                lidar_times = track.setdefault('lidar_history_timestamps_s', [])
                lidar_history.append(detection_ne.copy())
                lidar_times.append(float(stamp_s))
                if len(lidar_history) > self.obstacle_history_len:
                    del lidar_history[:-self.obstacle_history_len]
                    del lidar_times[:-self.obstacle_history_len]
        speed_m_s = float(np.linalg.norm(vel_ne))
        heading_rad = float(np.arctan2(vel_ne[1], vel_ne[0])) if speed_m_s >= self.apf_dynamic_speed_threshold_m_s else np.nan
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
            track['displacement_m'] = 0.0
            track['motion_stable'] = False
            self.sync_obstacle_track_fields(track)
            return
        velocities = np.asarray([sample['vel_ne'] for sample in samples], dtype=float)
        finite_vel = np.isfinite(velocities).all(axis=1)
        velocities = velocities[finite_vel]
        sample_times = np.asarray([float(sample.get('stamp_s', 0.0)) for sample in samples], dtype=float)
        sample_positions = np.asarray([sample['pos_ne'] for sample in samples], dtype=float)
        time_span_s = float(np.ptp(sample_times)) if len(sample_times) > 1 else 0.0
        if len(samples) >= 3 and time_span_s >= 0.4:
            centred_times = sample_times - float(np.mean(sample_times))
            design = np.column_stack([centred_times, np.ones_like(centred_times)])
            fit, _, _, _ = np.linalg.lstsq(design, sample_positions, rcond=None)
            track['velocity_mean_ne'] = fit[0]
        elif len(velocities) > 0:
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
        track['displacement_m'] = float(np.linalg.norm(sample_positions[-1] - sample_positions[0])) if len(sample_positions) >= 2 else 0.0
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
        track['motion_stable'] = self.obstacle_track_motion_is_stable(track)
        self.obstacle_track_regularize_velocity(track)
        self.sync_obstacle_track_fields(track)

    def annotate_lidar_obstacle_with_track(self, obstacle, track):
        prediction_ne = np.asarray(track.get('prediction_ne', np.empty((0, 2))), dtype=float)
        raw_centre_ne = np.asarray(obstacle.get('centre_ne', [np.nan, np.nan]), dtype=float).reshape(2)
        obstacle['track_id'] = int(track['id'])
        obstacle['raw_centre_ne'] = raw_centre_ne.tolist()
        obstacle['raw_centre_body'] = np.asarray(obstacle.get('centre_body', [np.nan, np.nan]), dtype=float).reshape(2).tolist()
        obstacle['raw_px'] = float(raw_centre_ne[0])
        obstacle['raw_py'] = float(raw_centre_ne[1])
        obstacle['filtered_px'] = float(track['pos_ne'][0])
        obstacle['filtered_py'] = float(track['pos_ne'][1])
        predicted_position_ne = prediction_ne[-1] if prediction_ne.ndim == 2 and len(prediction_ne) > 0 else np.asarray(track.get('virtual_position_ne', [np.nan, np.nan]), dtype=float).reshape(2)
        obstacle['predicted_px'] = float(predicted_position_ne[0])
        obstacle['predicted_py'] = float(predicted_position_ne[1])
        obstacle['centre_ne'] = track['pos_ne'].tolist()
        obstacle['centre_body'] = self.earth_point_to_body(track['pos_ne']).tolist()
        obstacle['distance_m'] = float(np.linalg.norm(obstacle['centre_body']))
        obstacle['min_distance_m'] = float(min(obstacle.get('min_distance_m', obstacle['distance_m']), obstacle['distance_m']))
        obstacle['angle_rad'] = float(np.arctan2(obstacle['centre_body'][1], obstacle['centre_body'][0]))
        obstacle['angle_deg'] = float(np.rad2deg(obstacle['angle_rad']))
        obstacle['velocity_ne'] = np.asarray(track['vel_ne'], dtype=float).reshape(2).tolist()
        obstacle['velocity_mean_ne'] = np.asarray(track.get('velocity_mean_ne', track['vel_ne']), dtype=float).reshape(2).tolist()
        obstacle['velocity_var_ne'] = np.asarray(track.get('velocity_var_ne', [0.0, 0.0]), dtype=float).reshape(2).tolist()
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
        obstacle['position_uncertainty_m2'] = float(track.get('position_uncertainty_m2', np.nan))
        obstacle['covariance_trace'] = float(np.trace(np.asarray(track.get('covariance', np.eye(4, dtype=float)), dtype=float)))
        obstacle['prediction_model'] = track.get('prediction_model', 'cv_ekf')
        obstacle['predicted_trajectory_ne'] = prediction_ne.tolist()
        lidar_points_ne = np.asarray(obstacle.get('points_ne', []), dtype=float)
        if lidar_points_ne.ndim == 2 and lidar_points_ne.shape[1] == 2:
            track['lidar_points_ne'] = lidar_points_ne.tolist()

    def prune_obstacle_tracks(self, now):
        self.apf_obstacle_tracks = [track for track in self.apf_obstacle_tracks if now - float(track.get('last_seen_s', track.get('stamp_s', now))) <= self.apf_track_timeout_s and int(track.get('miss_count', 0)) <= 5]

    def update_apf_obstacle_tracks(self, stamp_s):
        tracking_enabled = bool(getattr(self, 'obstacle_ekf_tracking_enabled', True))
        if not tracking_enabled:
            self.apf_obstacle_tracks = []
            self.apf_virtual_obstacles = []
            tracked_fields = {'track_id', 'raw_px', 'raw_py', 'filtered_px', 'filtered_py', 'predicted_px', 'predicted_py', 'velocity_ne', 'velocity_mean_ne', 'velocity_var_ne', 'speed_m_s', 'speed_mean_m_s', 'speed_var_m2_s2', 'heading_rad', 'heading_deg', 'heading_mean_rad', 'heading_var_rad2', 'pc1_m', 'pc2_m', 'pc1_var_m2', 'pc2_var_m2', 'stats_sample_count', 'motion_stable', 'position_uncertainty_m2', 'covariance_trace', 'prediction_model', 'predicted_trajectory_ne'}
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
                measurement_covariance = np.asarray(obstacle.get('measurement_covariance', self.obstacle_measurement_covariance(obstacle.get('centre_body', [0.0, 0.0]))), dtype=float).reshape(2, 2)
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
            predicted_state, predicted_covariance = self.obstacle_ekf_predict(track.get('state', np.r_[track.get('pos_ne', [np.nan, np.nan]), 0.0, 0.0]), track.get('covariance', np.eye(4, dtype=float)), dt)
            predicted_tracks.append((predicted_state, predicted_covariance))
            predicted_pos = predicted_state[0:2]
            for detection_index, detection in enumerate(detection_positions):
                distance = float(np.linalg.norm(detection - predicted_pos))
                H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
                innovation = detection - predicted_pos
                measurement_covariance = detections[detection_index][5]
                innovation_covariance = H @ np.asarray(predicted_covariance, dtype=float) @ H.T + measurement_covariance
                try:
                    mahalanobis_sq = float(innovation.T @ np.linalg.inv(innovation_covariance) @ innovation)
                except np.linalg.LinAlgError:
                    mahalanobis_sq = float(innovation.T @ np.linalg.pinv(innovation_covariance) @ innovation)
                candidates.append((mahalanobis_sq, distance, track_index, detection_index))
        candidates.sort(key=lambda item: (item[0], item[1]))
        assigned_tracks = set()
        assigned_detections = set()
        detection_track = {}
        for mahalanobis_sq, distance, track_index, detection_index in candidates:
            if distance > self.apf_track_association_m or mahalanobis_sq > 9.21:
                continue
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
            track['raw_detection_ne'] = detection.copy()
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

    def obstacle_pc_dimensions(self, obstacle):
        min_pc1_m = float(getattr(self, 'obstacle_min_pc1_m', 0.3))
        min_pc2_m = float(getattr(self, 'obstacle_min_pc2_m', 0.16))
        pc1_m = float(obstacle.get('pc1_m', min_pc1_m))
        pc2_m = float(obstacle.get('pc2_m', min_pc2_m))
        if not np.isfinite(pc1_m) or pc1_m <= 0.0:
            pc1_m = min_pc1_m
        if not np.isfinite(pc2_m) or pc2_m <= 0.0:
            pc2_m = min_pc2_m
        pc1_m = max(pc1_m, min_pc1_m)
        pc2_m = max(min(pc2_m, pc1_m), min_pc2_m)
        return (pc1_m, pc2_m)

    def obstacle_length_axis_ne(self, obstacle):
        state = np.asarray(obstacle.get('state', []), dtype=float).reshape(-1)
        if state.size == 4 and np.isfinite(state).all():
            velocity_ne = state[2:4]
            velocity_norm = float(np.linalg.norm(velocity_ne))
            if velocity_norm >= self.apf_dynamic_speed_threshold_m_s:
                return velocity_ne / velocity_norm
        for key in ('velocity_ne', 'vel_ne', 'velocity_mean_ne'):
            velocity_ne = np.asarray(obstacle.get(key, [np.nan, np.nan]), dtype=float).reshape(2)
            velocity_norm = float(np.linalg.norm(velocity_ne))
            if np.isfinite(velocity_ne).all() and velocity_norm >= self.apf_dynamic_speed_threshold_m_s:
                return velocity_ne / velocity_norm
        axis_ne = np.asarray(obstacle.get('length_axis_ne', [1.0, 0.0]), dtype=float).reshape(2)
        axis_norm = float(np.linalg.norm(axis_ne))
        if not np.isfinite(axis_ne).all() or axis_norm < 1e-06:
            return np.array([1.0, 0.0], dtype=float)
        return axis_ne / axis_norm

    def update_apf_virtual_obstacles(self):
        """Log the same EKF/own-ship trajectories used for predicted CPA."""
        self.apf_virtual_obstacles = []
        if not self.obstacle_ekf_prediction_enabled:
            return self.apf_virtual_obstacles
        horizon_s = float(self.apf_collision_horizon_s)
        step_s = max(float(self.apf_prediction_dt_s), 0.05)
        times_s = np.arange(0.0, horizon_s + 0.5 * step_s, step_s)
        own_start_ne = np.array([float(self.North), float(self.East)], dtype=float)
        measured_velocity_ne = np.asarray(self.v_robot[0:2], dtype=float).reshape(2)
        speed_m_s = float(np.linalg.norm(measured_velocity_ne))
        if not np.isfinite(speed_m_s) or speed_m_s < 1e-06:
            speed_m_s = float(self.route_tracking_speed_m_s)
        own_velocity_ne = self.body_vector_to_earth(np.array([speed_m_s, 0.0]))
        own_trajectory_ne = own_start_ne + times_s[:, None] * own_velocity_ne
        for track in self.apf_obstacle_tracks:
            if not self.obstacle_track_motion_is_stable(track):
                continue
            start_ne = np.asarray(track.get('pos_ne', track.get('position_ne', [np.nan, np.nan])), dtype=float).reshape(2)
            predicted_start_ne, velocity_ne = self.obstacle_track_state_at(track, 0.0)
            if predicted_start_ne is not None:
                start_ne = np.asarray(predicted_start_ne, dtype=float).reshape(2)
            if not np.isfinite(start_ne).all() or velocity_ne is None:
                continue
            velocity_ne = np.asarray(velocity_ne, dtype=float).reshape(2)
            speed_m_s = float(np.linalg.norm(velocity_ne))
            if not np.isfinite(velocity_ne).all() or speed_m_s < self.apf_dynamic_speed_threshold_m_s:
                continue
            tcpa_s, dcpa_m, own_cpa_ne, obstacle_cpa_ne = straight_line_cpa(own_start_ne, own_velocity_ne, start_ne, velocity_ne, horizon_s)
            if tcpa_s <= 0.0:
                continue
            pc1_m, pc2_m = self.obstacle_pc_dimensions(track)
            ellipse_axis_ne = velocity_ne / max(float(np.linalg.norm(velocity_ne)), 1e-06)
            collision_level, _ = ellipse_level_and_away(own_cpa_ne - obstacle_cpa_ne, ellipse_axis_ne, 0.5 * pc1_m + self.apf_own_equivalent_radius_m, 0.5 * pc2_m + self.apf_own_equivalent_radius_m)
            if collision_level >= float(self.apf_risk_pc_scale):
                continue
            centre_trajectory_ne = start_ne + times_s[:, None] * velocity_ne
            self.apf_virtual_obstacles.append({'label': -1000 - int(track.get('id', 0)), 'virtual': True, 'track_id': int(track.get('id', 0)), 'centre_ne': obstacle_cpa_ne.tolist(), 'centre_body': self.earth_point_to_body(obstacle_cpa_ne).tolist(), 'pc1_m': pc1_m, 'pc2_m': pc2_m, 'length_axis_ne': ellipse_axis_ne.tolist(), 'apf_ellipse_axis_ne': ellipse_axis_ne.tolist(), 'lidar_motion_axis_ne': ellipse_axis_ne.tolist(), 'apf_ellipse_pc1_m': pc1_m, 'apf_ellipse_pc2_m': pc2_m, 'velocity_ne': velocity_ne.tolist(), 'tcpa_s': tcpa_s, 'heading_rad': float(np.arctan2(velocity_ne[1], velocity_ne[0])), 'dcpa_m': dcpa_m, 'collision_position_ne': own_cpa_ne.tolist(), 'predicted_risk_active': True, 'collision_level': float(collision_level), 'obstacle_dcpa_position_ne': obstacle_cpa_ne.tolist(), 'obstacle_min_separation_point_ne': obstacle_cpa_ne.tolist(), 'own_prediction_velocity_ne': own_velocity_ne.tolist(), 'own_prediction_heading_rad': float(self.Yaw), 'own_prediction_ne': own_trajectory_ne.tolist(), 'obstacle_prediction_ne': centre_trajectory_ne.tolist()})
        return self.apf_virtual_obstacles

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

    def apf_obstacle_endpoint_direction_body(self, obs_pos_body, obs_vel_body, obstacle, along_sign):
        obs_pos_body = np.asarray(obs_pos_body, dtype=float).reshape(2)
        obs_vel_body = np.asarray(obs_vel_body, dtype=float).reshape(2)
        if not np.isfinite(obs_pos_body).all() or not np.isfinite(obs_vel_body).all():
            return np.zeros(2, dtype=float)
        obs_speed = float(np.linalg.norm(obs_vel_body))
        if obs_speed < self.apf_dynamic_speed_threshold_m_s:
            return np.zeros(2, dtype=float)
        axis_body = obs_vel_body / max(obs_speed, 1e-06)
        pc1_m, _ = self.obstacle_pc_dimensions(obstacle)
        endpoint_body = obs_pos_body + float(along_sign) * 0.5 * pc1_m * axis_body
        endpoint_distance = float(np.linalg.norm(endpoint_body))
        if endpoint_distance < 1e-06:
            return float(along_sign) * axis_body
        return endpoint_body / endpoint_distance

    def apf_stern_direction_body(self, obs_pos_body, obs_vel_body, obstacle):
        return self.apf_obstacle_endpoint_direction_body(obs_pos_body, obs_vel_body, obstacle, -1.0)

    def apf_bow_direction_body(self, obs_pos_body, obs_vel_body, obstacle):
        return self.apf_obstacle_endpoint_direction_body(obs_pos_body, obs_vel_body, obstacle, 1.0)

    def apf_crossing_strategy_from_velocity(self, obs_vel_body):
        obs_vel_body = np.asarray(obs_vel_body, dtype=float).reshape(2)
        if not np.isfinite(obs_vel_body).all():
            return ('none', 0.0)
        if float(np.linalg.norm(obs_vel_body)) < self.apf_dynamic_speed_threshold_m_s:
            return ('none', 0.0)
        lateral_speed = float(obs_vel_body[1])
        if abs(lateral_speed) < self.apf_dynamic_speed_threshold_m_s:
            return ('none', 0.0)
        return ('pass_astern', self.apf_pass_astern_side_from_velocity(obs_vel_body))

    def apf_encounter_speed_m_s(self, encounter, nearest_active_level, force_angle):
        base_speed = float(getattr(self, 'apf_constant_descent_speed_m_s', self.route_tracking_speed_m_s))
        if encounter == 'overtaking':
            return float(np.clip(max(base_speed, getattr(self, 'apf_overtaking_surge_m_s', base_speed)), 0.0, self.v_max))
        if isinstance(encounter, str) and encounter.startswith('crossing'):
            pass_ahead_active = 'ahead' in str(getattr(self, 'apf_colreg_rule', '')).lower()
            if np.isfinite(nearest_active_level) and nearest_active_level <= 1.0:
                target_speed = max(getattr(self, 'apf_crossing_min_forward_speed', base_speed), getattr(self, 'apf_crossing_close_quarters_surge_m_s', base_speed))
            else:
                target_speed = max(base_speed, getattr(self, 'apf_crossing_min_forward_speed', base_speed))
            if pass_ahead_active:
                pass_ahead_speed = float(getattr(self, 'apf_crossing_pass_ahead_surge_m_s', target_speed))
                target_speed = max(target_speed, pass_ahead_speed)
                tcpa_s = float(getattr(self, 'apf_colreg_tcpa_s', np.nan))
                dcpa_m = float(getattr(self, 'apf_colreg_dcpa_m', np.nan))
                horizon_s = max(float(getattr(self, 'apf_collision_horizon_s', 1.0)), 1e-06)
                safe_dcpa_m = max(float(getattr(self, 'apf_crossing_pass_ahead_safe_dcpa_m', 0.0)), 1e-06)
                tcpa_urgency = 0.0 if not np.isfinite(tcpa_s) else float(np.clip((horizon_s - max(tcpa_s, 0.0)) / horizon_s, 0.0, 1.0))
                dcpa_urgency = 0.0 if not np.isfinite(dcpa_m) else float(np.clip((safe_dcpa_m - max(dcpa_m, 0.0)) / safe_dcpa_m, 0.0, 1.0))
                pass_ahead_urgency = max(tcpa_urgency, dcpa_urgency)
                target_speed = max(target_speed, base_speed + pass_ahead_urgency * max(pass_ahead_speed - base_speed, 0.0))
            return float(np.clip(target_speed, 0.0, self.v_max))
        if encounter == 'head_on':
            turn_scale = float(np.clip(1.0 - abs(force_angle) / max(self.apf_heading_step_limit_rad, 1e-06), 0.35, 1.0))
            target_speed = max(getattr(self, 'apf_min_forward_speed', 0.0), getattr(self, 'apf_head_on_surge_m_s', base_speed) * turn_scale)
            return float(np.clip(target_speed, 0.0, self.v_max))
        if np.isfinite(nearest_active_level) and nearest_active_level <= 1.0:
            target_speed = max(getattr(self, 'apf_min_forward_speed', base_speed), getattr(self, 'apf_close_quarters_surge_m_s', base_speed))
            return float(np.clip(target_speed, 0.0, self.v_max))
        return float(np.clip(base_speed, 0.0, self.v_max))

    def obstacle_track_state_at(self, track, dt_s):
        dt_s = max(float(dt_s), 0.0)
        state = np.asarray(track.get('state', [np.nan] * 4), dtype=float).reshape(4)
        if not np.isfinite(state).all():
            return (None, None)
        return (state[:2] + state[2:4] * dt_s, state[2:4].copy())

    def apf_obstacle_in_priority_front_sector(self, obstacle):
        obs_pos_body = np.asarray(obstacle.get('centre_body', [np.nan, np.nan]), dtype=float).reshape(2)
        if not np.isfinite(obs_pos_body).all():
            return False
        body_angle_rad = float(np.arctan2(obs_pos_body[1], obs_pos_body[0]))
        return abs(wrap_angle(body_angle_rad)) <= self.apf_priority_front_half_angle_rad

    def apf_obstacle_in_forward_half_plane(self, obstacle):
        obs_pos_body = np.asarray(obstacle.get('centre_body', [np.nan, np.nan]), dtype=float).reshape(2)
        if not np.isfinite(obs_pos_body).all():
            return False
        return float(obs_pos_body[0]) >= 0.0

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
            if not bool(obstacle.get('virtual', False)) and (not self.apf_obstacle_in_forward_half_plane(obstacle)):
                continue
            angle_rad = abs(wrap_angle(float(np.arctan2(obs_pos_body[1], obs_pos_body[0]))))
            if not bool(obstacle.get('virtual', False)) and angle_rad > self.apf_activation_front_half_angle_rad:
                continue
            if self.apf_repulsion_for_obstacle(obstacle, None, None)[1]:
                return True
        return False

    def compute_apf_control(self, t, u_track):
        final_approach = self.final_approach_active()
        if final_approach:
            target_ne = self.goal_ne.copy()
        else:
            _, target_ne = self.route_progress_and_point(self.apf_route_lookahead_m)
        target_body = self.earth_point_to_body(target_ne)
        goal_body = self.earth_point_to_body(self.goal_ne)
        path_force = np.zeros(2, dtype=float) if final_approach else self.apf_path_attraction_body()
        attractive_force = self.apf_goal_attraction_body(goal_body) + path_force
        force_body = attractive_force.copy()
        repulsive_force = np.zeros(2, dtype=float)
        own_vel_body = self.current_velocity_body()
        any_repulsion = False
        clearance_offset_m = 0.0
        nearest_active_level = np.inf
        avoidance_pc_scale = float(getattr(self, 'apf_avoidance_pc_scale', 2.0))
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
            repulsion, active = self.apf_repulsion_for_obstacle(obstacle, target_body, own_vel_body)[:2]
            repulsive_force += repulsion
            force_body += repulsion
            any_repulsion = any_repulsion or active
            if active:
                obstacle_encounter = obstacle.get('encounter_mode', self.apf_encounter_mode)
                clearance_offset_m = max(clearance_offset_m, self.apf_direction_clearance_m(obstacle, obstacle_encounter))
                nearest_active_level = min(nearest_active_level, self.apf_obstacle_level_and_away(obstacle, avoidance_pc_scale, encounter=obstacle_encounter)[0])
        if not any_repulsion:
            for obstacle in secondary_obstacles:
                repulsion, active = self.apf_repulsion_for_obstacle(obstacle, target_body, own_vel_body)[:2]
                repulsive_force += repulsion
                force_body += repulsion
                any_repulsion = any_repulsion or active
                if active:
                    obstacle_encounter = obstacle.get('encounter_mode', self.apf_encounter_mode)
                    clearance_offset_m = max(clearance_offset_m, self.apf_direction_clearance_m(obstacle, obstacle_encounter))
                    nearest_active_level = min(nearest_active_level, self.apf_obstacle_level_and_away(obstacle, avoidance_pc_scale, encounter=obstacle_encounter)[0])
        if any_repulsion and self.apf_side_lock_sign != 0.0:
            route_normal_left_ne = np.array([self.route_path_unit_ne[1], -self.route_path_unit_ne[0]], dtype=float)
            offset_target_ne = target_ne + self.apf_side_lock_sign * max(clearance_offset_m, float(getattr(self, 'obstacle_min_pc2_m', 0.16))) * route_normal_left_ne
            offset_target_body = self.earth_point_to_body(offset_target_ne)
            offset_distance = float(np.linalg.norm(offset_target_body))
            if offset_distance > 1e-06:
                offset_force = self.apf_clearance_gain * min(offset_distance, self.apf_attraction_saturation_m) * offset_target_body / offset_distance
                force_body += offset_force
                attractive_force += offset_force
                self.apf_target_ne = offset_target_ne.copy()
        if any_repulsion and np.isfinite(nearest_active_level):
            close_quarters_scale = float(np.clip(nearest_active_level, float(getattr(self, 'apf_close_quarters_force_scale', 0.15)), 1.0))
            force_body = repulsive_force + close_quarters_scale * attractive_force
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
        if self.apf_encounter_mode == 'head_on':
            force_angle = min(force_angle, -abs(np.deg2rad(8.0)))
        u_cmd = Vector(2)
        u_cmd[1, 0] = np.clip(-self.apf_heading_gain * force_angle / max(self.lastdt, 0.001), -self.w_max, self.w_max)
        surge_cmd = self.apf_encounter_speed_m_s(self.apf_encounter_mode, nearest_active_level, force_angle)
        u_cmd[0, 0] = float(np.clip(surge_cmd, 0.0, self.v_max))
        if any_repulsion or self.apf_colreg_active or self.apf_side_lock_active:
            if self.apf_colreg_active:
                self.navigation_mode = 'apf_colreg'
            else:
                self.navigation_mode = 'apf_avoid'
        else:
            self.navigation_mode = 'apf_track'
        return u_cmd
