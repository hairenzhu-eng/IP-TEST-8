"""
Copyright (c) 2025 The uos_sess6072_build Authors.
Authors: Blair Thornton, Alec O'Loughlin, Miquel Massot
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import numpy as np
import json
import os
from datetime import datetime
import time
from pathlib import Path
import subprocess
import platform
import copy
from dataclasses import dataclass

from zeroros import Publisher, Subscriber
from zeroros.messages import String, Vector3, Vector3Stamped, Pose, PoseStamped, RBLaserScan
from zeroros.datalogger import DataLogger

from drivers.aruco import ArUcoUDPDriver
from drivers.rpi import Console, Rate
from drivers import __version__
from scipy.spatial.transform import Rotation as R

# ---------------- LiDAR + DBSCAN imports ----------------
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
from sklearn.cluster import DBSCAN

from model_sess6072 import TAM, Vehicle2D_e, dynamics_translation_e, dynamics_rotation_e, TrajectoryGenerate, RangeAngleKinematics
from math_sess6072 import l2m, HomogeneousTransformation, Vector, HomogeneousTransformation, Matrix, Identity
from model_sess6072 import rigid_body_kinematics # tried to remove <existing libraries>
from math_sess6072 import Inverse, Vector # tried to remove <existing libraries>

# enter additional library imports here

# define global variables 
N = 0
E = 1
G = 2
DOTN = 3
DOTE = 4
DOTG = 5

# Obstacle ship EKF tracking and CPA switch.
# True: enable EKF tracking, CPA, predicted trajectories, and virtual collision points.
# False: use current LiDAR obstacles only; disable EKF tracking, CPA, and prediction.
ENABLE_OBSTACLE_EKF_PREDICTION = True

# APF range is always derived from each DBSCAN cluster's PCA dimensions.
ENABLE_CLUSTER_BASED_APF_RANGE = True


# define global functions
def get_wifi_name():
    result = subprocess.run(["netsh", "wlan", "show", "interfaces"], capture_output=True, text=True)
    for line in result.stdout.split("\n"):
        if "SSID" in line and "BSSID" not in line:
            return line.split(":")[1].strip()
    return 0

def Vector(dim): return np.zeros((dim, 1), dtype=float)

def rpm2N(x, fwd_lim = 2000, rev_lim = -2000): 
    tol = 10
    if x>fwd_lim: x=fwd_lim
    if x<rev_lim: x=rev_lim
    if abs(x) <= tol: return 0
    elif x>tol: return 1.541571428571430076E-7*x**2+3.293357142857142252E-4*x-1.401428571428424679E-3
    else: return -7.35749999999999954E-8*x**2+1.716749999999999581E-4*x-1.054478382732365536E-16

def N2rpm(x, fwd_lim = 1.2753, rev_lim = -0.63765): 
    tol = 10E-4
    if x>fwd_lim: x=fwd_lim
    if x<rev_lim: x=rev_lim    
    if abs(x) <= tol: return 0
    elif x>tol: return -5.727416623043567370E2*x**2+2.268233085499708977E3*x+2.958718669408357371E1
    else: return 2.397948698765131667E3*x**2+4.665568885752373717E3*x - -6.685183692128883879E-14

# Keep heading errors continuous for route tracking.
def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def heading_to_velocity_ne(speed_m_s, heading_rad):
    speed_m_s = float(speed_m_s)
    heading_rad = float(heading_rad)
    return np.array(
        [
            speed_m_s * np.cos(heading_rad),
            speed_m_s * np.sin(heading_rad),
        ],
        dtype=float,
    )


def velocity_ne_to_speed_heading(
    velocity_ne,
    fallback_heading_rad=0.0,
    hold_speed_m_s=1e-3,
):
    velocity_ne = np.asarray(velocity_ne, dtype=float).reshape(2)
    speed_m_s = float(np.linalg.norm(velocity_ne))
    if speed_m_s < max(float(hold_speed_m_s), 0.0):
        return speed_m_s, wrap_angle(float(fallback_heading_rad))
    return speed_m_s, wrap_angle(float(np.arctan2(velocity_ne[1], velocity_ne[0])))


@dataclass
class ObstacleEkfConfig:
    process_noise_px: float
    process_noise_py: float
    process_noise_v: float
    process_noise_phi: float
    process_noise_phi_dot: float
    measurement_noise_px: float
    measurement_noise_py: float
    initial_position_std_m: float
    initial_speed_std_m_s: float
    initial_heading_std_rad: float
    initial_turn_rate_std_rad_s: float
    v_max_m_s: float
    phi_dot_max_rad_s: float
    heading_hold_speed_m_s: float
    static_speed_reset_m_s: float
    turn_rate_epsilon_rad_s: float
    min_dt_s: float
    max_dt_s: float


class ObstacleEKF:
    STATE_SIZE = 5

    def __init__(self, state, covariance, config):
        self.config = config
        self.state = self.normalise_state(state, config.heading_hold_speed_m_s)
        self.covariance = self.normalise_covariance(covariance)
        self.last_predict_dt = 0.0
        self.last_pre_predict_state = self.state.copy()
        self._clamp_state()

    @staticmethod
    def normalise_state(state, hold_speed_m_s=1e-3):
        state = np.asarray(state, dtype=float).reshape(-1)
        if state.size == 5:
            px, py, v, phi, phi_dot = state
            return np.array(
                [
                    float(px),
                    float(py),
                    max(float(v), 0.0),
                    wrap_angle(float(phi)),
                    float(phi_dot),
                ],
                dtype=float,
            )

        if state.size == 4:
            px, py, vx, vy = state
            speed_m_s, heading_rad = velocity_ne_to_speed_heading(
                [vx, vy],
                fallback_heading_rad=0.0,
                hold_speed_m_s=hold_speed_m_s,
            )
            return np.array(
                [float(px), float(py), speed_m_s, heading_rad, 0.0],
                dtype=float,
            )

        raise ValueError(f"Unsupported obstacle state size: {state.size}")

    @staticmethod
    def normalise_covariance(covariance):
        covariance = np.asarray(covariance, dtype=float)
        if covariance.shape == (5, 5):
            return covariance.copy()
        if covariance.shape == (4, 4):
            expanded = np.zeros((5, 5), dtype=float)
            expanded[0:2, 0:2] = covariance[0:2, 0:2]
            expanded[2, 2] = max(float(covariance[2, 2]), float(covariance[3, 3]))
            expanded[3, 3] = np.pi ** 2
            expanded[4, 4] = 1.0
            return expanded
        if covariance.size == 0:
            return np.eye(5, dtype=float)
        raise ValueError(f"Unsupported obstacle covariance shape: {covariance.shape}")

    @classmethod
    def initial_covariance(cls, config):
        return np.diag(
            [
                config.initial_position_std_m ** 2,
                config.initial_position_std_m ** 2,
                config.initial_speed_std_m_s ** 2,
                config.initial_heading_std_rad ** 2,
                config.initial_turn_rate_std_rad_s ** 2,
            ]
        )

    def copy(self):
        return ObstacleEKF(
            self.state.copy(),
            self.covariance.copy(),
            self.config,
        )

    def velocity_ne(self):
        return heading_to_velocity_ne(self.state[2], self.state[3])

    def _clamp_state(self):
        self.state[2] = float(np.clip(self.state[2], 0.0, self.config.v_max_m_s))
        self.state[3] = wrap_angle(float(self.state[3]))
        self.state[4] = float(
            np.clip(
                self.state[4],
                -self.config.phi_dot_max_rad_s,
                self.config.phi_dot_max_rad_s,
            )
        )
        if self.state[2] < self.config.static_speed_reset_m_s:
            self.state[2] = 0.0
            self.state[4] = 0.0

    def _ctrv_step(self, state, dt):
        px, py, v, phi, phi_dot = self.normalise_state(
            state,
            self.config.heading_hold_speed_m_s,
        )
        dt = float(dt)
        phi_dot = float(
            np.clip(
                phi_dot,
                -self.config.phi_dot_max_rad_s,
                self.config.phi_dot_max_rad_s,
            )
        )
        v = float(np.clip(v, 0.0, self.config.v_max_m_s))

        if abs(phi_dot) >= self.config.turn_rate_epsilon_rad_s:
            next_phi = wrap_angle(phi + phi_dot * dt)
            px_next = px + v / phi_dot * (np.sin(next_phi) - np.sin(phi))
            py_next = py + v / phi_dot * (-np.cos(next_phi) + np.cos(phi))
        else:
            next_phi = phi if v < self.config.heading_hold_speed_m_s else wrap_angle(phi)
            px_next = px + v * np.cos(phi) * dt
            py_next = py + v * np.sin(phi) * dt

        next_state = np.array(
            [
                px_next,
                py_next,
                v,
                next_phi,
                phi_dot,
            ],
            dtype=float,
        )
        if next_state[2] < self.config.static_speed_reset_m_s:
            next_state[2] = 0.0
            next_state[4] = 0.0
        next_state[3] = wrap_angle(next_state[3])
        return next_state

    def _numerical_jacobian(self, state, dt):
        state = np.asarray(state, dtype=float).reshape(5)
        base = self._ctrv_step(state, dt)
        jacobian = np.zeros((5, 5), dtype=float)
        steps = np.array([1e-4, 1e-4, 1e-4, 1e-5, 1e-5], dtype=float)
        for index, step in enumerate(steps):
            perturbed = state.copy()
            perturbed[index] += step
            diff = self._ctrv_step(perturbed, dt) - base
            if index == 3:
                diff[3] = wrap_angle(diff[3])
            jacobian[:, index] = diff / step
        return jacobian

    def process_noise(self, dt):
        dt = max(float(dt), self.config.min_dt_s)
        return np.diag(
            [
                (self.config.process_noise_px * dt) ** 2,
                (self.config.process_noise_py * dt) ** 2,
                (self.config.process_noise_v * np.sqrt(dt)) ** 2,
                (self.config.process_noise_phi * np.sqrt(dt)) ** 2,
                (self.config.process_noise_phi_dot * np.sqrt(dt)) ** 2,
            ]
        )

    def measurement_noise(self):
        return np.diag(
            [
                self.config.measurement_noise_px ** 2,
                self.config.measurement_noise_py ** 2,
            ]
        )

    def predict(self, dt):
        dt = float(dt)
        if not np.isfinite(dt) or dt < self.config.min_dt_s:
            return self.get_state(), self.covariance.copy()
        dt = min(dt, self.config.max_dt_s)
        prior_state = self.state.copy()
        self.last_pre_predict_state = prior_state.copy()
        self.last_predict_dt = dt
        self.state = self._ctrv_step(prior_state, dt)
        F = self._numerical_jacobian(prior_state, dt)
        self.covariance = F @ self.covariance @ F.T + self.process_noise(dt)
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self._clamp_state()
        return self.get_state(), self.covariance.copy()

    def update(self, px_meas, py_meas, measurement_covariance=None):
        z = np.array([float(px_meas), float(py_meas)], dtype=float)
        H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        R = self.measurement_noise() if measurement_covariance is None else np.asarray(
            measurement_covariance,
            dtype=float,
        ).reshape(2, 2)
        innovation = z - H @ self.state
        innovation_covariance = H @ self.covariance @ H.T + R
        try:
            innovation_inverse = np.linalg.inv(innovation_covariance)
        except np.linalg.LinAlgError:
            innovation_inverse = np.linalg.pinv(innovation_covariance)
        kalman_gain = self.covariance @ H.T @ innovation_inverse
        predicted_heading = float(self.state[3])
        self.state = self.state + kalman_gain @ innovation
        self.state[3] = predicted_heading if self.state[2] < self.config.heading_hold_speed_m_s else wrap_angle(float(self.state[3]))

        # Bootstrap speed, heading, and turn rate from multi-frame position
        # changes without overwriting the EKF state outright.
        if self.last_predict_dt >= self.config.min_dt_s:
            displacement_ne = z - self.last_pre_predict_state[0:2]
            observed_velocity_ne = displacement_ne / self.last_predict_dt
            observed_speed_m_s, observed_heading_rad = velocity_ne_to_speed_heading(
                observed_velocity_ne,
                fallback_heading_rad=predicted_heading,
                hold_speed_m_s=self.config.heading_hold_speed_m_s,
            )
            if observed_speed_m_s >= self.config.heading_hold_speed_m_s:
                speed_alpha = 0.55 if self.state[2] < self.config.heading_hold_speed_m_s else 0.25
                heading_alpha = 0.60 if self.state[2] < self.config.heading_hold_speed_m_s else 0.25
                self.state[2] = (1.0 - speed_alpha) * self.state[2] + speed_alpha * observed_speed_m_s
                heading_error = wrap_angle(observed_heading_rad - self.state[3])
                self.state[3] = wrap_angle(self.state[3] + heading_alpha * heading_error)
                previous_speed = float(self.last_pre_predict_state[2])
                if previous_speed >= self.config.heading_hold_speed_m_s:
                    observed_phi_dot = wrap_angle(observed_heading_rad - self.last_pre_predict_state[3]) / self.last_predict_dt
                    turn_alpha = 0.20
                    self.state[4] = (1.0 - turn_alpha) * self.state[4] + turn_alpha * observed_phi_dot

        identity = np.eye(self.STATE_SIZE, dtype=float)
        correction = identity - kalman_gain @ H
        self.covariance = correction @ self.covariance @ correction.T + kalman_gain @ R @ kalman_gain.T
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self._clamp_state()
        return self.get_state(), self.covariance.copy()

    def get_predicted_position(self, t_horizon):
        predicted_state = self._ctrv_step(self.state, max(float(t_horizon), 0.0))
        return predicted_state[0:2].copy()

    def get_predicted_trajectory(self, t_horizon, steps):
        t_horizon = max(float(t_horizon), 0.0)
        steps = max(int(steps), 1)
        if t_horizon <= 0.0:
            return np.asarray([self.state[0:2].copy()], dtype=float)
        times = np.linspace(0.0, t_horizon, steps + 1)
        return np.asarray(
            [
                self._ctrv_step(self.state, dt)[0:2]
                for dt in times
            ],
            dtype=float,
        )

    def get_state(self):
        return self.state.copy()


def cluster_principal_dimensions(points, min_pc1_m, min_pc2_m):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    centre = np.mean(points, axis=0)
    relative = points - centre

    if len(points) < 2 or np.allclose(relative, 0.0):
        return centre, float(min_pc1_m), float(min_pc2_m), np.array([1.0, 0.0])

    covariance = relative.T @ relative / max(len(relative), 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    length_axis = eigenvectors[:, -1]
    if length_axis[0] < 0.0:
        length_axis = -length_axis
    width_axis = np.array([-length_axis[1], length_axis[0]], dtype=float)

    pc1_m = max(float(np.ptp(relative @ length_axis)), float(min_pc1_m))
    pc2_m = max(float(np.ptp(relative @ width_axis)), float(min_pc2_m))
    if pc2_m > pc1_m:
        pc1_m, pc2_m = pc2_m, pc1_m
        length_axis = width_axis

    return centre, pc1_m, pc2_m, length_axis

def closest_point_on_segment(point, start, end):
    point = np.asarray(point, dtype=float).reshape(2)
    start = np.asarray(start, dtype=float).reshape(2)
    end = np.asarray(end, dtype=float).reshape(2)
    segment = end - start
    segment_length_sq = float(np.dot(segment, segment))
    if segment_length_sq < 1e-12:
        return start.copy()
    ratio = float(np.clip(np.dot(point - start, segment) / segment_length_sq, 0.0, 1.0))
    return start + ratio * segment


def ellipse_level_and_away(offset, length_axis, semi_length_m, semi_width_m):
    offset = np.asarray(offset, dtype=float).reshape(2)
    length_axis = np.asarray(length_axis, dtype=float).reshape(2)
    axis_norm = float(np.linalg.norm(length_axis))
    if axis_norm < 1e-9:
        length_axis = np.array([1.0, 0.0], dtype=float)
    else:
        length_axis = length_axis / axis_norm
    width_axis = np.array([-length_axis[1], length_axis[0]], dtype=float)
    semi_length_m = max(float(semi_length_m), 1e-6)
    semi_width_m = max(float(semi_width_m), 1e-6)

    along = float(np.dot(offset, length_axis))
    across = float(np.dot(offset, width_axis))
    level = float(np.hypot(along / semi_length_m, across / semi_width_m))
    gradient = (
        along / (semi_length_m * semi_length_m) * length_axis
        + across / (semi_width_m * semi_width_m) * width_axis
    )
    gradient_norm = float(np.linalg.norm(gradient))
    if gradient_norm < 1e-9:
        offset_norm = float(np.linalg.norm(offset))
        away = offset / offset_norm if offset_norm >= 1e-9 else -length_axis
    else:
        away = gradient / gradient_norm
    return level, away

def extended_kalman_filter_predict(mu, Sigma, u, f, Q, dt):
    # (1) Project the state forward
    pred_mu, F = f(mu, u , dt)
      
    # (2) Project the error forward: 
    pred_Sigma = F@Sigma@F.T+Q
    
    # Return the predicted state and the covariance
    return pred_mu, pred_Sigma

def extended_kalman_filter_update(mu, Sigma, z, h, R, wrap_index = None):
    
    # Prepare the estimated measurement
    pred_z, H = h(mu)
 
    # (3) Compute the Kalman gain
    K = Sigma@ H.T@ Inverse(H@Sigma@H.T + R)
    
    # (4) Compute the updated state estimate
    delta_z = z- pred_z        
    if wrap_index != None: delta_z[wrap_index] = (delta_z[wrap_index] + np.pi) % (2 * np.pi) - np.pi    
    cor_mu = mu + K@(delta_z)

    # (5) Compute the updated state covariance
    cor_Sigma = (Identity(mu.shape[0]) - K @ H) @ Sigma
    
    # Return the state and the covariance
    return cor_mu, cor_Sigma

def h_pose_update(x):
    est_measurement = Vector(6)
    est_measurement[N] = x[N]
    est_measurement[E] = x[E]
    est_measurement[G] = x[G]
    H = Matrix(6,6)
    H[N, N] = 1
    H[E, E] = 1
    H[G, G] = 1
    return est_measurement, H

def h_grate_update(x):
    est_measurement = Vector(6)
    est_measurement[DOTG] = x[DOTG]

    H=Matrix(6,6)
    H[DOTG,DOTG]=1
    return est_measurement, H 

# main class
class LaptopController:
    def __init__(self, OPERATING_MODE):
        
        ########### DEFINE ARUCO MARKER ID ###################                     
        MARKER_ID = 24 # <<< CHANGE TO YOUR ROBOT'S ARUCO ID

        ########### SET NETWORK CONDITIONS ###################             
        if OPERATING_MODE != 2: # robot
            self.robot_ip = "192.168.10.1"
            self.robot_available = False
            self.sim_init = False
            aruco_params = {
                "port": 50001,  # Port to listen to (DO NOT CHANGE)
                "marker_id": MARKER_ID,  # Marker ID to listen to
            }                     
            if platform.system() == "Windows": wifi_name = get_wifi_name()
            else: wifi_name = "SmartCatXX"

        elif OPERATING_MODE == 2: # webots
            self.robot_ip = "127.0.0.1"          
            aruco_params = {
                "port": 50000,  # Port to listen to (DO NOT CHANGE)
                "marker_id": 0,  # Overide for WEBOTS (DO NOT CHANGE)
            }
            wifi_name = "WEBOTS"
            self.sim_init = True # Deal with webots timestamps

        self.sim_time_offset = 0.0
                            
        Console.info("Connecting to:", self.robot_ip, "")
        if wifi_name:            
            Console.info(f"You are connected to {wifi_name}")
        else:
            Console.info("No WiFi connection detected")

        # store operating mode
        self.OPERATING_MODE = OPERATING_MODE
        self.obstacle_ekf_tracking_enabled = True
        self.obstacle_ekf_prediction_enabled = ENABLE_OBSTACLE_EKF_PREDICTION
        Console.info(
            "Obstacle EKF prediction:",
            "enabled" if self.obstacle_ekf_prediction_enabled else "disabled",
        )

        ########### INITIALISE DATA LOGS ###################                     
        filename_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path("logs/run_" + filename_time)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.filename = self.run_dir / f"log_{filename_time}.csv"
        self.obstacle_log_dir = self.run_dir
        self.last_obstacle_snapshot_stamp_s = None
        
        with self.filename.open('w') as f:
            f.write("EpochTime(s),TimeFromStart(s),right_prop_rate(rad/s),left_prop_rate(rad/s),LastDT(s),Yaw(rad),North(m),East(m),IMUSensedYawRate(rad/s),IMUIntegratedYaw(rad),IMUSensedTimeStamp(s),ARUCOSensedNorth(m),ARUCOSensedEast(m),ARUCOSensedYaw(rad),ArucoSensedTimeStamp(s),DepthTimeStamp(s),Depth(m),NavigationMode,APFEncounter,APFSide,APFDCPA(m),APFTCPA(s),APFForceX,APFForceY,NearestObstacleNorth(m),NearestObstacleEast(m),NearestObstacleDistance(m)\n")
        global file 
        file = self.filename

        ########### ENTER WAYPOINT VARIABLES ###############
        # Start waypoint: (North, East) in metres
        start_north, start_east = 0, 1

        # Goal waypoint: (North, East) in metres
        goal_north, goal_east = 10, 1

        north_path = [start_north, goal_north]
        east_path = [start_east, goal_east]
        
        self.waypoints = []
        
        for i in range(len(north_path)):
            waypoint = Vector3()
            
            waypoint.y = north_path[i]
            waypoint.x = east_path[i]
            
            self.waypoints.append(waypoint)
            print("WAYPOINTS: ", self.waypoints)
            print("WAYPOINTS TYPE: ", type(self.waypoints))
        
        ########### INITIALISE ROBOT VARIABLES #############        
        rate = 5.0  # Hz
        self.r = Rate(rate)
        self.lastdt = 1/rate        
        self.starttime = time.time()
        self.timefromstart = None
        self.prev_sensed = None # originally None
        
        self.sensed_imu_yaw_rate_rad_s = None
        self.sensed_imu_stamp_s = None
        self.sensed_imu_prev_stamp_s = None
        self.sensed_yaw_rate = None
        self.integrated_yaw = 0

        self.sensed_pos_northings_m = None
        self.sensed_pos_eastings_m = None
        self.sensed_pos_yaw_rad = None
        self.sensed_pos_stamp_s = None
        self.sensed_bottom_depth_m = None        
        self.sensed_bottom_depth_stamp_s = None        

        # ---------------- LiDAR definitions ----------------
        self.lidar_data = None
        self.lidar_data_rb = None
        self.lidar_timestamp_s = None
        self.latest_lidar_received_s = None
        self.lidar_new = False
        self.lidar_x_bl = 0.1
        self.lidar_y_bl = 0.0
        self.lidar_gamma_bl = 0.0
        self.lidar = RangeAngleKinematics(self.lidar_x_bl, self.lidar_y_bl, self.lidar_gamma_bl)

        # ---------------- LiDAR DBSCAN parameters and outputs ----------------
        self.lidar_dbscan_eps_m = 0.20
        self.lidar_dbscan_min_points = 3
        self.lidar_points_body = np.empty((0, 2))
        self.lidar_cluster_labels = np.array([], dtype=int)
        self.lidar_obstacles = []
        self.lidar_obstacle_centres_body = np.empty((0, 2))
        self.lidar_obstacle_centres_ne = np.empty((0, 2))
        self.lidar_obstacle_distances_m = np.array([])
        self.lidar_obstacle_angles_rad = np.array([])
        self.nearest_lidar_obstacle = None

        # ---------------- LiDAR sector parameters ----------------
        self.front_angle_limit = np.deg2rad(25)
        self.front_block_threshold = 0.8
        self.min_front_close_beams = 5
        self.side_angle_min = np.deg2rad(45)
        self.side_angle_max = np.deg2rad(120)

        self.map_observation_class = "unknown"
        self.front_clearance_m = np.inf
        self.left_clearance_m = np.inf
        self.right_clearance_m = np.inf

        # ---------------- Navigation parameters ----------------
        self.start_ne = np.array([start_north, start_east], dtype=float)
        self.goal_ne = np.array([goal_north, goal_east], dtype=float)
        self.goal_tolerance_m = 0.40
        self.navigation_mode = "track"
        self.goal_reached = False
        self.route_path_vec_ne = self.goal_ne - self.start_ne
        self.route_path_length_m = float(np.linalg.norm(self.route_path_vec_ne))
        if self.route_path_length_m > 1e-9:
            self.route_path_unit_ne = self.route_path_vec_ne / self.route_path_length_m
        else:
            self.route_path_unit_ne = np.array([1.0, 0.0], dtype=float)
        self.route_heading_rad = float(np.arctan2(self.route_path_unit_ne[1], self.route_path_unit_ne[0]))
        self.route_tracking_speed_m_s = 0.90 if self.OPERATING_MODE == 2 else 0.42
        self.route_tracking_lookahead_m = 1.0 if self.OPERATING_MODE == 2 else 0.7
        self.final_approach_distance_m = 1.0 if self.OPERATING_MODE == 2 else 0.7
        self.final_slowdown_distance_m = 1.0 if self.OPERATING_MODE == 2 else 0.8
        self.final_heading_slow_angle_rad = np.deg2rad(75.0)
        self.max_heading_deviation_rad = np.deg2rad(60.0)
        self.heading_deviation_guard_rad = np.deg2rad(55.0)
        self.heading_deviation_return_gain = 3.0

        # ----------------  APF parameters ----------------
        self.apf_cluster_range_enabled = ENABLE_CLUSTER_BASED_APF_RANGE
        self.apf_risk_pc_scale = 2.0
        self.apf_avoidance_pc_scale = 2.0
        self.apf_direction_pc_scale = 2.0
        self.apf_boundary_pc_scale = 20.0
        self.apf_virtual_pc_scale = 3.0
        self.apf_activation_front_half_angle_rad = np.deg2rad(150.0)
        self.apf_priority_front_half_angle_rad = np.deg2rad(90.0)
        self.apf_goal_gain = 6.5
        self.apf_path_gain = 6.5
        self.apf_repulsive_gain = 0.35
        self.apf_boundary_activation_level = 1.35
        self.apf_boundary_inside_boost = 3.0
        self.apf_attraction_saturation_m = 3.5
        self.apf_path_threshold_m = 0.10
        self.apf_route_lookahead_m = 2.1 if self.OPERATING_MODE == 2 else 1.8
        self.apf_clearance_gain = 3.4
        self.apf_collision_horizon_s = 6.0 if self.OPERATING_MODE != 2 else 10.0
        self.apf_prediction_dt_s = 0.5
        self.apf_heading_gain = 0.9
        self.apf_heading_step_limit_rad = np.deg2rad(60.0)
        self.apf_min_detour_offset_m = 1.0 if self.OPERATING_MODE == 2 else 0.8
        self.apf_min_forward_speed = 0.10 if self.OPERATING_MODE == 2 else 0.12
        # Fixed-step APF descent: the potential-field gradient determines only
        # the travel direction while avoidance uses a constant surge speed.
        self.apf_constant_descent_speed_m_s = self.route_tracking_speed_m_s
        self.apf_pass_ahead_gain = 2.4
        self.apf_crossing_repulsive_gain_scale = 3.2
        self.apf_virtual_crossing_repulsive_gain_scale = 1.8
        self.apf_crossing_min_forward_speed = 0.14 if self.OPERATING_MODE == 2 else 0.16
        self.apf_crossing_close_quarters_surge_m_s = 0.18 if self.OPERATING_MODE == 2 else 0.16
        self.apf_crossing_pass_ahead_surge_m_s = (
            self.route_tracking_speed_m_s + (0.18 if self.OPERATING_MODE == 2 else 0.08)
        )
        self.apf_crossing_pass_ahead_safe_dcpa_m = 0.60 if self.OPERATING_MODE == 2 else 0.45
        self.apf_overtaking_longitudinal_scale = 0.75
        self.apf_overtaking_lateral_scale = 0.65
        self.apf_overtaking_repulsive_gain_scale = 0.85
        self.apf_overtaking_side_gain_scale = 0.55
        self.apf_overtaking_surge_m_s = (
            self.route_tracking_speed_m_s + (1.0 if self.OPERATING_MODE == 2 else 0.4)
        )
        self.apf_crossing_longitudinal_scale = 1.1
        self.apf_crossing_lateral_scale = 3.2
        self.apf_head_on_longitudinal_scale = 1.3
        self.apf_head_on_lateral_scale = 1.2
        self.apf_head_on_surge_m_s = max(
            self.apf_min_forward_speed,
            self.route_tracking_speed_m_s - 0.02,
        )
        self.apf_close_quarters_force_scale = 0.15
        self.apf_close_quarters_surge_m_s = 0.16 if self.OPERATING_MODE == 2 else 0.14
        self.apf_force_body = np.zeros(2, dtype=float)
        self.apf_repulsive_force_body = np.zeros(2, dtype=float)
        self.apf_attractive_force_body = np.zeros(2, dtype=float)
        self.apf_steering_force_body = np.zeros(2, dtype=float)
        self.apf_target_ne = np.array([np.nan, np.nan], dtype=float)
        self.apf_encounter_mode = "none"
        self.apf_colreg_rule = "none"
        self.apf_avoidance_side_sign = 0.0
        self.apf_colreg_dcpa_m = np.nan
        self.apf_colreg_tcpa_s = np.nan
        self.apf_colreg_active = False
        # A target is only treated as dynamic after its EKF motion estimate has
        # remained coherent for several observations.  The previous 0.03 m/s
        # threshold allowed cluster jitter to change the COLREG encounter type.
        self.apf_dynamic_speed_threshold_m_s = 0.06 if self.OPERATING_MODE == 2 else 0.05
        self.apf_dynamic_exit_speed_threshold_m_s = 0.04 if self.OPERATING_MODE == 2 else 0.03
        self.apf_track_association_m = 0.80 if self.OPERATING_MODE == 2 else 0.60
        self.apf_track_timeout_s = 1.5 if self.OPERATING_MODE == 2 else 1.0
        self.apf_next_track_id = 1
        self.apf_obstacle_tracks = []
        self.apf_obstacle_track_candidates = []
        self.apf_virtual_obstacles = []
        self.obstacle_ekf_measurement_std_m = 0.08 if self.OPERATING_MODE == 2 else 0.12
        self.obstacle_ekf_initial_position_std_m = 0.20
        self.obstacle_ekf_initial_velocity_std_m_s = 0.35
        self.obstacle_ekf_initial_heading_std_rad = np.deg2rad(90.0)
        self.obstacle_ekf_initial_phi_dot_std_rad_s = np.deg2rad(45.0)
        self.process_noise_px = 0.04 if self.OPERATING_MODE == 2 else 0.06
        self.process_noise_py = 0.04 if self.OPERATING_MODE == 2 else 0.06
        self.process_noise_v = 0.10 if self.OPERATING_MODE == 2 else 0.14
        self.process_noise_phi = np.deg2rad(8.0)
        self.process_noise_phi_dot = np.deg2rad(12.0)
        self.measurement_noise_px = self.obstacle_ekf_measurement_std_m
        self.measurement_noise_py = self.obstacle_ekf_measurement_std_m
        self.obstacle_heading_hold_speed_m_s = 0.04 if self.OPERATING_MODE == 2 else 0.03
        self.obstacle_ekf_static_speed_reset_m_s = 0.03 if self.OPERATING_MODE == 2 else 0.02
        self.obstacle_turn_rate_epsilon_rad_s = 1e-3
        self.obstacle_track_min_dt_s = 1e-3
        self.obstacle_track_max_dt_s = 1.0
        self.obstacle_v_max_m_s = 2.0 if self.OPERATING_MODE == 2 else 1.5
        self.obstacle_phi_dot_max_rad_s = np.deg2rad(60.0)
        self.obstacle_prediction_horizon_s = 10.0 if self.OPERATING_MODE == 2 else 4.0
        self.obstacle_prediction_step_s = 0.8
        self.obstacle_history_len = 60
        self.obstacle_stats_window_s = 2.5
        self.obstacle_prediction_min_samples = 3
        self.obstacle_prediction_min_hits = 3
        self.obstacle_prediction_min_time_span_s = 0.8
        self.obstacle_prediction_min_displacement_m = 0.06
        self.obstacle_prediction_min_speed_m_s = 0.05
        self.obstacle_prediction_max_speed_std_m_s = 0.10
        self.obstacle_prediction_max_heading_var_rad2 = np.deg2rad(35.0) ** 2
        self.obstacle_min_pc1_m = 0.36 if self.OPERATING_MODE == 2 else 0.30
        self.obstacle_min_pc2_m = 0.20 if self.OPERATING_MODE == 2 else 0.16
        self.obstacle_min_equivalent_radius_m = 0.5 * self.obstacle_min_pc1_m
        self.apf_own_equivalent_radius_m = 0.30 if self.OPERATING_MODE == 2 else 0.25
        self.apf_virtual_repulsive_gain = 0.40
        self.apf_side_lock_sign = 0.0
        self.apf_side_lock_until_s = 0.0
        self.apf_side_lock_s = 5.0 if self.OPERATING_MODE != 2 else 8.0
        self.apf_side_lock_exit_level = 1.15
        self.apf_side_lock_active = False
        self.apf_visual_hold_s = 2.0
        self.apf_visual_hold_until_s = 0.0
        Console.info(
            "APF influence range:",
            "PCA ellipse 2.5x current / 4x EKF-predicted",
        )
        
        self.initial_state = Vector(6)
        self.initial_state[N] = start_north
        self.initial_state[E] = start_east
        self.initial_state[G] = 0
        self.initial_state[DOTN] = 0
        self.initial_state[DOTE] = 0
        self.initial_state[DOTG] = 0
        
        self.North = self.initial_state[N][0]
        self.East = self.initial_state[E][0]
        self.Yaw = self.initial_state[G][0]         

        self.right_rate = 0
        self.left_rate = 0

        ############################# MOTION MODEL VARIABLES #######################
        # Body-force model: positive force from either thruster acts forward.
        # The right propeller command sign is handled at the RPM conversion.
        phi=l2m([0,0])        
        x=l2m([0,0])
        # Body-frame lateral offsets: right thruster is negative y, left is positive y.
        y=l2m([-0.09,0.09])
        
        self.G=TAM(phi,x,y)
        print('G = ',self.G)
        
        # hull, water properties
        rho = 1000 # density of water in kg/m3
        draft = 0.07 #m
        beam = 0.04 #m of the immersed hull section
        length = 0.5 #m
        width = 0.4 #m # of the whole hull

        # from ESDU 71016. Fluid forces, pressures and moments on rectangular blocks. ESDU 71016 ESDU International, London
        CD = 7#1.5 # approximation for block from Newman (0.9 to 2.75) 
        A = 2*beam*draft #catamaran cross section in surge
        k_drag = 0.5*rho*CD*A

        # Added mass from Imlay 1961, Technical Report DTMB - assuming a prolate spheroid
        mass = 3
        e = 1 - (beam/length)**2
        alpha = (2*(1-e**2)/e**3)*(0.5*np.log((1+e)/(1-e))-e) #note np.log() = ln(), np.log10()=log()

        mass_add = 2*alpha*mass/(2-alpha)  # kg of water pushed by hull with, note this is for an infinite

        m_tot = mass + mass_add

        # Added mass from Imlay 1961, Technical Report DTMB - assuming a prolate spheroid
        I_66 = mass*((length/2)**2+(width/2)**2)/4 # rough approximation as rectangle
        e = 1 - (beam/length)**2
        alpha = (2*(1-e**2)/e**3)*(0.5*np.log((1+e)/(1-e))-e) #note np.log() = ln(), np.log10()=log()
        beta = 1/e**2 - ((1-e**2)/(2*e**3)) * np.log((1+e)/(1-e))  # kg of water pushed by hull with, note this is for an infinite

        I66_add = 2*(1/5)*mass*((draft**2-length**2)**2*(alpha-beta)/(2*(draft**2-length**2)+(draft**2+length**2)/(beta-alpha)))

        I_tot = I_66+I66_add

        # drag B_66
        B_66 = 0.12#0.12
        
        self.initial_pose = True # Set false after pose is initialised
        

        # read these into our vehicle class
        self.robot = Vehicle2D_e(m_tot,I_tot,k_drag,B_66)
        self.robot.info()
        
        self.v_robot = Vector(3) # initially stationary velocity vector in e frame
        self.p_robot = Vector(3); self.p_robot[0] = start_north; self.p_robot[1] = start_east; self.p_robot[2] = np.deg2rad(0) # pose in the e frame
        
        ############################# CONTROL VARIABLES #######################
        # Setup control parameters
        #################################################################
        tau_s = 2 #s to remove along track error # 0.5
        self.L = 0.3#m distance to remove normal and angular error
        self.ks =  1/tau_s
        self.kn = None 
        self.kg = None
        
        self.v_max = 0.85 #fastest the robot can go # 0.2
        self.w_max = np.deg2rad(80) #fastest the robot can turn # 30
        self.prop_rate_limit_rad_s = 200.0
        # setup a contranor to store controls
        self.U = Vector(2).T
        ################################################################
        # Setup trajectory
        #################################################################
        v = self.route_tracking_speed_m_s
        a = 0.4 # 0.1 
        self.s = TrajectoryGenerate(north_path,east_path)
        self.s.path_to_trajectory(v, a)

        # Generate turning arcs trajectory
        self.arc_radius = 0.02
        self.s.turning_arcs(self.arc_radius) 
        self.s.wp_id = len(self.s.P_arc) - 1
        self.trajectory_duration_s = float(self.s.Tp_arc[-1][0])

        ############################# EKF VARIABLES ####################
        # State x = [N, E, G, Ndot, Edot, Gdot]^T
        self.mu = Vector(6)
        self.mu[N]    = self.initial_state[N]
        self.mu[E]    = self.initial_state[E]
        self.mu[G]    = self.initial_state[G]
        self.mu[DOTN] = self.initial_state[DOTN]
        self.mu[DOTE] = self.initial_state[DOTE]
        self.mu[DOTG] = self.initial_state[DOTG]
        
        # Initial covariance
        self.Sigma = Identity(6)
        # Position uncertainty (m^2)
        self.Sigma[N, N]   = 0.01      # 0.1 m std
        self.Sigma[E, E]   = 0.01
        # Heading uncertainty (rad^2)
        self.Sigma[G, G]   = np.deg2rad(5.0)**2
        # Velocity uncertainty ((m/s)^2 and (rad/s)^2)
        self.Sigma[DOTN, DOTN] = 0.01
        self.Sigma[DOTE, DOTE] = 0.01
        self.Sigma[DOTG, DOTG] = np.deg2rad(10.0)**2
        
        # Process noise Q (very simple diagonal)
        self.Q = Identity(6)
        q_pos = 1e-4
        q_vel = 1e-3
        self.Q[N, N]   = q_pos
        self.Q[E, E]   = q_pos
        self.Q[G, G]   = 1e-5
        self.Q[DOTN, DOTN] = q_vel
        self.Q[DOTE, DOTE] = q_vel
        self.Q[DOTG, DOTG] = 1e-4
        
        # Measurement noise for ArUco pose (N, E, G)
        self.R_pose = Identity(6)
        self.R_pose[N, N] = 0.02**2                 # 2 cm std
        self.R_pose[E, E] = 0.02**2
        self.R_pose[G, G] = np.deg2rad(2.0)**2      # 2 deg std
        
        # Measurement noise for IMU yaw rate (Gdot)
        self.R_grate = Identity(6)
        self.R_grate[DOTG, DOTG] = np.deg2rad(1.0)**2
        
        # Time bookkeeping for EKF (not strictly needed, but handy)
        self.last_nav_t = self.starttime
                    
        ############################# DECLARE PUBLISHERS AND SUBSCRIBERS ######         
        self.control_pub = Publisher("/control", Vector3, ip=self.robot_ip)
        self.config_pub = Publisher("/config", String, ip=self.robot_ip)
        self.imu_sub = Subscriber("/imu", Vector3, self.imu_cb, ip=self.robot_ip)
        self.sonar_sub = Subscriber("/sonar", Vector3, self.sonar_cb, ip=self.robot_ip)
        self.lidar_sub = Subscriber("/lidar", RBLaserScan, self.lidar_callback, ip=self.robot_ip)
        self.console_sub = Subscriber("/command", String, self.command_cb, ip=self.robot_ip)
        self.aruco_driver = ArUcoUDPDriver(aruco_params, parent=self)        
        # a callback only used by WEBOTS to fake Aruco readings 
        self.groundtruth_sub = Subscriber("/groundtruth", PoseStamped, self.groundtruth_callback, ip=self.robot_ip) 
        self.pseudo_aruco_counter = 0
        
        ########### CONNECT TO ROBOT ###########
        if OPERATING_MODE != 2: # not a simulation
            # waits for robot to respond to configure
            count = 0
            Console.info("Connecting to robot")               
            while not self.robot_available:
                self.config_pub.publish(String("Configure"))
                time.sleep(1.0)
                count += 1
            time.sleep(5.0)
        else: # WEBOTS create fake ARUCO logs
            self.sensed_imu_stamp_s = 0 
            self.groundtruth_log = self.run_dir / f"log_{filename_time}_pseudo_aruco.csv"
            with self.groundtruth_log.open('w') as f:
                f.write("epoch [s],elapsed [s],x [m],y [m],z [m],roll [deg],pitch [deg],yaw [deg],broadcast\n")

        ########### INITIALISE THRUSTERS ###########
        for i in range(10): #  rad/s
            self.control_pub.publish(Vector3())           
            self.r.sleep()  
            self.initialise_pose = True # Will set to false once the pose is initialised               


        ######## Setup EXIT key if show_laptop not used #####
        if OPERATING_MODE == 0: # robot without show_laptop - stopped via <Ctrl+C> 
            while True:
                try:
                    self.loop()
                except KeyboardInterrupt:
                    Console.info("Ctrl+C pressed. Stopping...")
                    if self.OPERATING_MODE == 0:
                        self.imu_sub.stop()
                        self.sonar_sub.stop()
                        self.lidar_sub.stop()
                    break
                self.r.sleep() 
        ############################## END OF INITIALISATION ##################

    ######## DEFINE FUNCTIONS HERE ##################
    def stopcommand(self):        
        Console.info("Thrusters stopping")
        control_msg = Vector3() # initially 0
        for i in range(10):
            self.control_pub.publish(control_msg)
            self.imu_sub.stop()
            self.r.sleep()
        self.sonar_sub.stop()
        self.lidar_sub.stop()
        Console.info("Thrusters stopped")
        Console.info("Data saved in ",self.filename)
        self.r.sleep()

    ######## DEFINE CALLBACKS HERE ##################
    def imu_cb(self, msg: Vector3): 
        self.sensed_imu_yaw_rate_rad_s = msg.z
        self.sensed_imu_stamp_s = time.time()
        self.robot_available = True
        
    def sonar_cb(self,msg: Vector3):
        self.sensed_bottom_depth_m = msg.z/1000        
        self.sensed_bottom_depth_stamp_s = time.time()
        self.robot_available = True

    def command_cb(self,msg: String):
        Console.info(f"Response from robot: {msg.data}")

    # ---------------- LiDAR callback ----------------
    def lidar_callback(self, msg: RBLaserScan):
        if self.sim_init:
            self.sim_time_offset = time.time() - msg.header.stamp
            self.sim_init = False

        self.lidar_timestamp_s = msg.header.stamp + self.sim_time_offset
        self.latest_lidar_received_s = self.lidar_timestamp_s

        ranges = np.array(msg.ranges, dtype=float)
        angles = np.array(msg.angles, dtype=float)

        if len(ranges) != len(angles):
            count = min(len(ranges), len(angles))
            ranges = ranges[:count]
            angles = angles[:count]

        ranges = np.where((ranges > 0.0) & np.isfinite(ranges), ranges, np.nan)
        self.lidar_data_rb = np.column_stack([ranges, angles])

        pose = np.asarray(self.p_robot, dtype=float).reshape(-1)
        p_eb = Vector(3)
        p_eb[0] = pose[0]
        p_eb[1] = pose[1]
        p_eb[2] = pose[2]

        self.lidar_data = np.full((len(ranges), 2), np.nan)
        z_lm = Vector(2)

        for i, range_m in enumerate(ranges):
            if np.isfinite(range_m):
                z_lm[0] = range_m
                z_lm[1] = angles[i]
                t_em = self.lidar.rangeangle_to_loc(p_eb, z_lm)

                self.lidar_data[i, 0] = t_em[0]
                self.lidar_data[i, 1] = t_em[1]

        self.lidar_data = self.lidar_data[~np.isnan(self.lidar_data).any(axis=1)]
        self.update_lidar_obstacle_clusters()
        self.update_apf_obstacle_tracks(self.lidar_timestamp_s)
        self.update_lidar_sectors()
        self.lidar_new = True
        self.robot_available = True

    # ---------------- LiDAR coordinate transforms ----------------
    def earth_vector_to_body(self, vector_ne):
        # Body frame convention: x is forward, y is left, gamma is yaw in earth frame.
        pose = np.asarray(self.p_robot, dtype=float).reshape(-1)
        yaw = pose[2]
        c = np.cos(yaw)
        s = np.sin(yaw)
        vector_ne = np.asarray(vector_ne, dtype=float).reshape(2)

        return np.array([
            c * vector_ne[0] + s * vector_ne[1],
            -s * vector_ne[0] + c * vector_ne[1],
        ])

    def earth_point_to_body(self, point_ne):
        # Point transform is a vector transform after subtracting the EKF robot position.
        pose = np.asarray(self.p_robot, dtype=float).reshape(-1)
        return self.earth_vector_to_body(np.asarray(point_ne, dtype=float).reshape(2) - pose[0:2])

    def body_vector_to_earth(self, vector_body):
        pose = np.asarray(self.p_robot, dtype=float).reshape(-1)
        yaw = pose[2]
        c = np.cos(yaw)
        s = np.sin(yaw)
        vector_body = np.asarray(vector_body, dtype=float).reshape(2)

        return np.array([
            c * vector_body[0] - s * vector_body[1],
            s * vector_body[0] + c * vector_body[1],
        ])

    def body_point_to_earth(self, point_body):
        pose = np.asarray(self.p_robot, dtype=float).reshape(-1)
        yaw = pose[2]
        c = np.cos(yaw)
        s = np.sin(yaw)
        point_body = np.asarray(point_body, dtype=float).reshape(2)

        return np.array([
            pose[0] + c * point_body[0] - s * point_body[1],
            pose[1] + s * point_body[0] + c * point_body[1],
        ])

    # ---------------- LiDAR DBSCAN clustering ----------------
    def clear_lidar_obstacle_clusters(self):
        self.lidar_points_body = np.empty((0, 2))
        self.lidar_cluster_labels = np.array([], dtype=int)
        self.lidar_obstacles = []
        self.lidar_obstacle_centres_body = np.empty((0, 2))
        self.lidar_obstacle_centres_ne = np.empty((0, 2))
        self.lidar_obstacle_distances_m = np.array([])
        self.lidar_obstacle_angles_rad = np.array([])
        self.nearest_lidar_obstacle = None

    def update_lidar_obstacle_clusters(self):
        if self.lidar_data_rb is None:
            self.clear_lidar_obstacle_clusters()
            return

        ranges = self.lidar_data_rb[:, 0]
        angles = self.lidar_data_rb[:, 1]
        valid = np.isfinite(ranges) & np.isfinite(angles)

        if np.count_nonzero(valid) < self.lidar_dbscan_min_points:
            self.clear_lidar_obstacle_clusters()
            return

        valid_ranges = ranges[valid]
        valid_angles = angles[valid] + self.lidar_gamma_bl

        self.lidar_points_body = np.column_stack([
            self.lidar_x_bl + valid_ranges * np.cos(valid_angles),
            self.lidar_y_bl + valid_ranges * np.sin(valid_angles),
        ])

        labels = DBSCAN(
            eps=self.lidar_dbscan_eps_m,
            min_samples=self.lidar_dbscan_min_points,
            n_jobs=1,
        ).fit_predict(self.lidar_points_body)
        self.lidar_cluster_labels = labels

        obstacles = []
        for label in sorted(set(labels)):
            if label == -1:
                continue

            cluster_points = self.lidar_points_body[labels == label]
            centre_body, pc1_m, pc2_m, length_axis_body = cluster_principal_dimensions(
                cluster_points,
                self.obstacle_min_pc1_m,
                self.obstacle_min_pc2_m,
            )
            relative_points = cluster_points - centre_body
            cluster_radius_m = float(np.max(np.linalg.norm(relative_points, axis=1)))
            cluster_extent_xy_m = np.ptp(cluster_points, axis=0)
            cluster_size_m = float(np.linalg.norm(cluster_extent_xy_m))
            equivalent_radius_m = 0.5 * max(pc1_m, pc2_m)
            centre_ne = self.body_point_to_earth(centre_body)
            length_axis_ne = self.body_vector_to_earth(length_axis_body)
            centre_distance_m = float(np.linalg.norm(centre_body))
            centre_angle_rad = float(np.arctan2(centre_body[1], centre_body[0]))
            min_distance_m = float(np.min(np.linalg.norm(cluster_points, axis=1)))
            measurement_covariance = self.obstacle_measurement_covariance(centre_body)

            obstacles.append({
                "label": int(label),
                "point_count": int(len(cluster_points)),
                "centre_body": centre_body.tolist(),
                "centre_ne": centre_ne.tolist(),
                "distance_m": centre_distance_m,
                "angle_rad": centre_angle_rad,
                "angle_deg": float(np.rad2deg(centre_angle_rad)),
                "min_distance_m": min_distance_m,
                "cluster_radius_m": cluster_radius_m,
                "cluster_size_m": cluster_size_m,
                "pc1_m": pc1_m,
                "pc2_m": pc2_m,
                "length_axis_body": length_axis_body.tolist(),
                "length_axis_ne": length_axis_ne.tolist(),
                "equivalent_radius_m": equivalent_radius_m,
                "measurement_covariance": measurement_covariance.tolist(),
            })

        obstacles.sort(key=lambda obstacle: obstacle["distance_m"])
        self.lidar_obstacles = obstacles
        self.nearest_lidar_obstacle = obstacles[0] if obstacles else None

        if obstacles:
            self.lidar_obstacle_centres_body = np.array(
                [obstacle["centre_body"] for obstacle in obstacles],
                dtype=float,
            )
            self.lidar_obstacle_centres_ne = np.array(
                [obstacle["centre_ne"] for obstacle in obstacles],
                dtype=float,
            )
            self.lidar_obstacle_distances_m = np.array(
                [obstacle["distance_m"] for obstacle in obstacles],
                dtype=float,
            )
            self.lidar_obstacle_angles_rad = np.array(
                [obstacle["angle_rad"] for obstacle in obstacles],
                dtype=float,
            )
        else:
            self.lidar_obstacle_centres_body = np.empty((0, 2))
            self.lidar_obstacle_centres_ne = np.empty((0, 2))
            self.lidar_obstacle_distances_m = np.array([])
            self.lidar_obstacle_angles_rad = np.array([])

    # ---------------- LiDAR sector helpers ----------------
    def sector_ranges(self, angle_min, angle_max):
        if self.lidar_data_rb is None:
            return np.array([])

        ranges = self.lidar_data_rb[:, 0]
        angles = self.lidar_data_rb[:, 1]

        mask = (
            (angles > angle_min)
            & (angles < angle_max)
            & np.isfinite(ranges)
        )

        return ranges[mask]

    def sector_min_range(self, angle_min, angle_max):
        vals = self.sector_ranges(angle_min, angle_max)

        if len(vals) == 0:
            return np.inf

        val = np.nanmin(vals)

        if not np.isfinite(val):
            return np.inf

        return val

    def update_lidar_sectors(self):
        self.front_clearance_m = self.sector_min_range(
            -self.front_angle_limit,
            self.front_angle_limit,
        )

        self.left_clearance_m = self.sector_min_range(
            self.side_angle_min,
            self.side_angle_max,
        )

        self.right_clearance_m = self.sector_min_range(
            -self.side_angle_max,
            -self.side_angle_min,
        )

        front_ranges = self.sector_ranges(
            -self.front_angle_limit,
            self.front_angle_limit,
        )

        if len(front_ranges) == 0:
            front_blocked = False
        else:
            front_blocked = np.sum(front_ranges < self.front_block_threshold) >= self.min_front_close_beams

        left_open = self.left_clearance_m > self.front_block_threshold
        right_open = self.right_clearance_m > self.front_block_threshold

        if front_blocked and left_open and right_open:
            self.map_observation_class = "t_junction_or_end_wall"
        elif front_blocked and left_open:
            self.map_observation_class = "right_angle_left_turn"
        elif front_blocked and right_open:
            self.map_observation_class = "right_angle_right_turn"
        elif front_blocked:
            self.map_observation_class = "blocked_front"
        else:
            self.map_observation_class = "straight_section"

    def front_blocked(self):
        self.update_lidar_sectors()

        front_ranges = self.sector_ranges(
            -self.front_angle_limit,
            self.front_angle_limit,
        )

        if len(front_ranges) == 0:
            return False

        close_count = int(np.sum(front_ranges < self.front_block_threshold))
        return close_count >= self.min_front_close_beams

    def json_safe(self, value):
        if value is None:
            return None

        if isinstance(value, (str, bool)):
            return value

        if isinstance(value, np.ndarray):
            return self.json_safe(value.tolist())

        if isinstance(value, np.generic):
            return self.json_safe(value.item())

        if isinstance(value, int):
            return value

        if isinstance(value, float):
            if not np.isfinite(value):
                return None
            return value

        if isinstance(value, dict):
            return {str(key): self.json_safe(item) for key, item in value.items()}

        if isinstance(value, (list, tuple)):
            return [self.json_safe(item) for item in value]

        return value

    def write_obstacle_snapshot(self):
        stamp_s = self.latest_lidar_received_s
        if stamp_s is None:
            return

        stamp_s = float(stamp_s)
        if self.last_obstacle_snapshot_stamp_s == stamp_s:
            return

        self.last_obstacle_snapshot_stamp_s = stamp_s
        payload = {
            "t": self.timefromstart,
            "timestamp_s": stamp_s,
            "robot_pos": [self.North, self.East],
            "robot_yaw_rad": self.Yaw,
            "cloud": self.lidar_data if self.lidar_data is not None else [],
            "clusters": self.lidar_obstacles,
            "tracks": self.obstacle_track_visuals(),
            "virtual_obstacles": self.apf_virtual_obstacles,
            "apf": {
                "navigation_mode": self.navigation_mode,
                "encounter": self.apf_encounter_mode,
                "colreg_rule": self.apf_colreg_rule,
                "side": self.apf_avoidance_side_sign,
                "dcpa_m": self.apf_colreg_dcpa_m,
                "tcpa_s": self.apf_colreg_tcpa_s,
                "force_body": self.apf_force_body,
                "repulsive_force_body": self.apf_repulsive_force_body,
                "attractive_force_body": self.apf_attractive_force_body,
                "target_ne": self.apf_target_ne,
                "goal_ne": self.goal_ne,
                "path_start_ne": self.start_ne,
                "path_end_ne": self.goal_ne,
                "path_unit_ne": self.route_path_unit_ne,
                "goal_gain": self.apf_goal_gain,
                "path_gain": self.apf_path_gain,
                "path_threshold_m": self.apf_path_threshold_m,
                "attraction_saturation_m": self.apf_attraction_saturation_m,
            },
            "apf_settings": {
                "own_equivalent_radius_m": self.apf_own_equivalent_radius_m,
                "collision_horizon_s": self.apf_collision_horizon_s,
                "prediction_dt_s": self.apf_prediction_dt_s,
                "constant_descent_speed_m_s": self.apf_constant_descent_speed_m_s,
                "obstacle_ekf_prediction_enabled": self.obstacle_ekf_prediction_enabled,
                "obstacle_prediction_horizon_s": self.obstacle_prediction_horizon_s,
                "obstacle_prediction_step_s": self.obstacle_prediction_step_s,
                "dynamic_speed_enter_m_s": self.apf_dynamic_speed_threshold_m_s,
                "dynamic_speed_exit_m_s": self.apf_dynamic_exit_speed_threshold_m_s,
                "heading_hold_speed_m_s": self.obstacle_heading_hold_speed_m_s,
                "static_speed_reset_m_s": self.obstacle_ekf_static_speed_reset_m_s,
                "v_max_m_s": self.obstacle_v_max_m_s,
                "phi_dot_max_rad_s": self.obstacle_phi_dot_max_rad_s,
                "prediction_min_samples": self.obstacle_prediction_min_samples,
                "prediction_min_time_span_s": self.obstacle_prediction_min_time_span_s,
                "prediction_min_displacement_m": self.obstacle_prediction_min_displacement_m,
                "prediction_min_speed_m_s": self.obstacle_prediction_min_speed_m_s,
                "prediction_max_speed_std_m_s": self.obstacle_prediction_max_speed_std_m_s,
                "prediction_max_heading_var_rad2": self.obstacle_prediction_max_heading_var_rad2,
                "process_noise_px": self.process_noise_px,
                "process_noise_py": self.process_noise_py,
                "process_noise_v": self.process_noise_v,
                "process_noise_phi": self.process_noise_phi,
                "process_noise_phi_dot": self.process_noise_phi_dot,
                "measurement_noise_px": self.measurement_noise_px,
                "measurement_noise_py": self.measurement_noise_py,
                "cluster_range_enabled": self.apf_cluster_range_enabled,
                "risk_pc_scale": self.apf_risk_pc_scale,
                "avoidance_pc_scale": self.apf_avoidance_pc_scale,
                "direction_pc_scale": self.apf_direction_pc_scale,
                "virtual_pc_scale": self.apf_virtual_pc_scale,
                "minimum_pc1_m": self.obstacle_min_pc1_m,
                "minimum_pc2_m": self.obstacle_min_pc2_m,
            },
            "dbscan": {
                "eps_m": self.lidar_dbscan_eps_m,
                "min_samples": self.lidar_dbscan_min_points,
            },
        }

        filename = f"obstacle_{int(round(stamp_s * 1000.0))}.json"
        with (self.obstacle_log_dir / filename).open("w") as f:
            json.dump(self.json_safe(payload), f, indent=2)

    # ---------------- COLREGS-compliant modified APF ----------------
    def reset_apf_diagnostics(self, clear_visual=True):
        if clear_visual:
            self.apf_force_body = np.zeros(2, dtype=float)
            self.apf_repulsive_force_body = np.zeros(2, dtype=float)
            self.apf_attractive_force_body = np.zeros(2, dtype=float)
            self.apf_steering_force_body = np.zeros(2, dtype=float)
            self.apf_target_ne = np.array([np.nan, np.nan], dtype=float)
            self.apf_virtual_obstacles = []
            self.apf_visual_hold_until_s = 0.0

        self.apf_encounter_mode = "none"
        self.apf_colreg_rule = "none"
        self.apf_avoidance_side_sign = 0.0
        self.apf_colreg_dcpa_m = np.nan
        self.apf_colreg_tcpa_s = np.nan
        self.apf_colreg_active = False

    def apf_visual_hold_active(self):
        now_s = float(self.timefromstart) if self.timefromstart is not None else 0.0
        return now_s < self.apf_visual_hold_until_s

    def limit_heading_deviation_command(self, yaw_rate_cmd):
        yaw_rate_cmd = float(yaw_rate_cmd)
        heading_offset = wrap_angle(float(self.Yaw) - self.route_heading_rad)
        max_offset = self.max_heading_deviation_rad
        guard_offset = self.heading_deviation_guard_rad
        dt = max(float(self.lastdt), 1e-3)
        offset_sign = float(np.sign(heading_offset))

        if abs(heading_offset) >= max_offset:
            return offset_sign * self.w_max

        if abs(heading_offset) >= guard_offset and yaw_rate_cmd * offset_sign < 0.0:
            return offset_sign * min(
                self.w_max,
                self.heading_deviation_return_gain * (abs(heading_offset) - guard_offset),
            )

        predicted_offset = heading_offset - yaw_rate_cmd * dt
        if predicted_offset > max_offset:
            return 0.0

        if predicted_offset < -max_offset:
            return 0.0

        return yaw_rate_cmd

    def current_velocity_body(self):
        vel_ne = np.asarray(self.v_robot[0:2], dtype=float).reshape(2)
        vel_body = self.earth_vector_to_body(vel_ne)

        if not np.isfinite(vel_body).all():
            return np.zeros(2, dtype=float)

        return vel_body

    def obstacle_ekf_config(self):
        measurement_std_m = float(getattr(self, "obstacle_ekf_measurement_std_m", 0.08))
        initial_speed_std_m_s = float(getattr(self, "obstacle_ekf_initial_velocity_std_m_s", 0.35))
        return ObstacleEkfConfig(
            process_noise_px=float(getattr(self, "process_noise_px", measurement_std_m)),
            process_noise_py=float(getattr(self, "process_noise_py", measurement_std_m)),
            process_noise_v=float(getattr(self, "process_noise_v", initial_speed_std_m_s)),
            process_noise_phi=float(getattr(self, "process_noise_phi", np.deg2rad(8.0))),
            process_noise_phi_dot=float(getattr(self, "process_noise_phi_dot", np.deg2rad(12.0))),
            measurement_noise_px=float(getattr(self, "measurement_noise_px", measurement_std_m)),
            measurement_noise_py=float(getattr(self, "measurement_noise_py", measurement_std_m)),
            initial_position_std_m=float(getattr(self, "obstacle_ekf_initial_position_std_m", 0.2)),
            initial_speed_std_m_s=initial_speed_std_m_s,
            initial_heading_std_rad=float(getattr(self, "obstacle_ekf_initial_heading_std_rad", np.deg2rad(90.0))),
            initial_turn_rate_std_rad_s=float(getattr(self, "obstacle_ekf_initial_phi_dot_std_rad_s", np.deg2rad(45.0))),
            v_max_m_s=float(getattr(self, "obstacle_v_max_m_s", 1.5)),
            phi_dot_max_rad_s=float(getattr(self, "obstacle_phi_dot_max_rad_s", np.deg2rad(60.0))),
            heading_hold_speed_m_s=float(getattr(self, "obstacle_heading_hold_speed_m_s", 0.03)),
            static_speed_reset_m_s=float(getattr(self, "obstacle_ekf_static_speed_reset_m_s", 0.02)),
            turn_rate_epsilon_rad_s=float(getattr(self, "obstacle_turn_rate_epsilon_rad_s", 1e-3)),
            min_dt_s=float(getattr(self, "obstacle_track_min_dt_s", 1e-3)),
            max_dt_s=float(getattr(self, "obstacle_track_max_dt_s", 1.0)),
        )

    def obstacle_ekf_process_noise(self, dt):
        return ObstacleEKF(
            np.zeros(5, dtype=float),
            np.eye(5, dtype=float),
            self.obstacle_ekf_config(),
        ).process_noise(dt)

    def obstacle_measurement_covariance(self, centre_body):
        centre_body = np.asarray(centre_body, dtype=float).reshape(2)
        config = self.obstacle_ekf_config()
        range_m = float(np.linalg.norm(centre_body)) if np.isfinite(centre_body).all() else 0.0
        yaw_rate_rad_s = float(abs(getattr(self, "sensed_imu_yaw_rate_rad_s", 0.0) or 0.0))
        lateral_std_m = config.measurement_noise_py * (1.0 + 0.08 * range_m + 0.25 * range_m * yaw_rate_rad_s)
        return np.diag(
            [
                config.measurement_noise_px ** 2,
                lateral_std_m ** 2,
            ]
        )

    def obstacle_ekf_predict(self, state, covariance, dt):
        ekf = ObstacleEKF(state, covariance, self.obstacle_ekf_config())
        return ekf.predict(dt)

    def obstacle_ekf_update(self, state, covariance, measurement_ne, measurement_covariance=None):
        measurement_ne = np.asarray(measurement_ne, dtype=float).reshape(2)
        ekf = ObstacleEKF(state, covariance, self.obstacle_ekf_config())
        return ekf.update(
            measurement_ne[0],
            measurement_ne[1],
            measurement_covariance=measurement_covariance,
        )

    def obstacle_track_motion_is_stable(self, track):
        sample_count = int(track.get("stats_sample_count", 0))
        hit_count = int(track.get("hit_count", 0))
        if (
            sample_count < self.obstacle_prediction_min_samples
            or hit_count < self.obstacle_prediction_min_hits
            or int(track.get("miss_count", 0)) > 0
        ):
            return False

        samples = track.get("motion_window", [])
        if len(samples) < 2:
            return False

        first_stamp = float(samples[0].get("stamp_s", 0.0))
        last_stamp = float(samples[-1].get("stamp_s", first_stamp))
        if last_stamp - first_stamp < self.obstacle_prediction_min_time_span_s:
            return False

        speed_m_s = float(track.get("speed_mean_m_s", 0.0))
        was_stable = bool(track.get("motion_stable", False))
        speed_threshold = (
            self.apf_dynamic_exit_speed_threshold_m_s
            if was_stable
            else self.apf_dynamic_speed_threshold_m_s
        )
        if not np.isfinite(speed_m_s) or speed_m_s < max(speed_threshold, getattr(self, "obstacle_prediction_min_speed_m_s", 0.0)):
            return False

        displacement_m = float(track.get("displacement_m", 0.0))
        if displacement_m < float(getattr(self, "obstacle_prediction_min_displacement_m", 0.0)):
            return False

        return True

    def obstacle_track_prediction_velocity_ne(self, track, dt_s=0.0):
        _, velocity_ne = self.obstacle_track_state_at(track, dt_s)
        if velocity_ne is None:
            return np.zeros(2, dtype=float)
        return np.asarray(velocity_ne, dtype=float).reshape(2)

    def obstacle_track_prediction_ne(self, track):
        if not self.obstacle_ekf_prediction_enabled:
            return np.empty((0, 2), dtype=float)

        ekf = track.get("ekf")
        if ekf is not None:
            horizon_s = float(track.get("collision_time_s", np.nan))
            if not np.isfinite(horizon_s) or horizon_s <= 0.0:
                horizon_s = float(getattr(self, "obstacle_prediction_horizon_s", 0.0))
            if horizon_s <= 0.0:
                return np.empty((0, 2), dtype=float)
            step_s = max(float(self.obstacle_prediction_step_s), 1e-3)
            steps = max(int(np.ceil(horizon_s / step_s)), 1)
            return ekf.get_predicted_trajectory(horizon_s, steps)

        state = ObstacleEKF.normalise_state(
            track.get("state", [np.nan, np.nan, 0.0, 0.0, 0.0]),
            getattr(self, "obstacle_heading_hold_speed_m_s", 0.03),
        )
        if not np.isfinite(state).all():
            return np.empty((0, 2), dtype=float)
        horizon_s = float(track.get("collision_time_s", np.nan))
        if not np.isfinite(horizon_s) or horizon_s <= 0.0:
            horizon_s = float(getattr(self, "obstacle_prediction_horizon_s", 0.0))
        if horizon_s <= 0.0:
            return np.empty((0, 2), dtype=float)
        step_s = max(float(self.obstacle_prediction_step_s), 1e-3)
        steps = max(int(np.ceil(horizon_s / step_s)), 1)
        return ObstacleEKF(
            state,
            track.get("covariance", np.eye(5, dtype=float)),
            self.obstacle_ekf_config(),
        ).get_predicted_trajectory(horizon_s, steps)

    def obstacle_track_regularize_velocity(self, track):
        state = np.asarray(track.get("state", [np.nan, np.nan, 0.0, 0.0]), dtype=float).reshape(-1)
        if state.size == 4:
            speed_m_s = float(np.linalg.norm(state[2:4]))
            if speed_m_s < float(getattr(self, "obstacle_ekf_static_speed_reset_m_s", 0.0)):
                track["state"][2:4] = 0.0
            return

        if state.size != 5:
            return

        if not bool(track.get("motion_stable", False)):
            track["state"][2] = float(np.clip(track["state"][2], 0.0, self.obstacle_ekf_config().v_max_m_s))
            return

        predicted_velocity_ne = np.asarray(
            self.obstacle_track_prediction_velocity_ne(track),
            dtype=float,
        ).reshape(2)
        predicted_speed_m_s, predicted_heading_rad = velocity_ne_to_speed_heading(
            predicted_velocity_ne,
            fallback_heading_rad=float(track["state"][3]),
            hold_speed_m_s=float(getattr(self, "obstacle_heading_hold_speed_m_s", 0.03)),
        )
        if predicted_speed_m_s >= float(getattr(self, "obstacle_heading_hold_speed_m_s", 0.03)):
            track["state"][2] = predicted_speed_m_s
            track["state"][3] = predicted_heading_rad

    def sync_obstacle_track_fields(self, track):
        if track.get("ekf") is None:
            track["ekf"] = ObstacleEKF(
                track.get("state", np.zeros(5, dtype=float)),
                track.get("covariance", np.eye(5, dtype=float)),
                self.obstacle_ekf_config(),
            )
        else:
            track["ekf"].state = ObstacleEKF.normalise_state(
                track.get("state", track["ekf"].get_state()),
                getattr(self, "obstacle_heading_hold_speed_m_s", 0.03),
            )
            track["ekf"].covariance = ObstacleEKF.normalise_covariance(
                track.get("covariance", track["ekf"].covariance),
            )
            track["ekf"]._clamp_state()
        track["state"] = track["ekf"].get_state()
        track["covariance"] = track["ekf"].covariance.copy()

        state = ObstacleEKF.normalise_state(
            track.get("state", [np.nan, np.nan, 0.0, 0.0, 0.0]),
            getattr(self, "obstacle_heading_hold_speed_m_s", 0.03),
        )
        track["state"] = state
        track["pos_ne"] = state[0:2].copy()
        track["vel_ne"] = heading_to_velocity_ne(state[2], state[3])
        track["v_m_s"] = float(state[2])
        track["phi_rad"] = float(state[3])
        track["phi_dot_rad_s"] = float(state[4])
        track["position_uncertainty_m2"] = float(np.trace(np.asarray(track.get("covariance", np.eye(5, dtype=float)), dtype=float)[0:2, 0:2]))

        pc1_m = float(track.get("pc1_m", self.obstacle_min_pc1_m))
        pc2_m = float(track.get("pc2_m", self.obstacle_min_pc2_m))
        if not np.isfinite(pc1_m) or pc1_m <= 0.0:
            pc1_m = self.obstacle_min_pc1_m
        if not np.isfinite(pc2_m) or pc2_m <= 0.0:
            pc2_m = self.obstacle_min_pc2_m
        track["pc1_m"] = max(pc1_m, self.obstacle_min_pc1_m)
        track["pc2_m"] = max(min(pc2_m, track["pc1_m"]), self.obstacle_min_pc2_m)
        track["radius_m"] = 0.5 * track["pc1_m"]
        track["equivalent_radius_m"] = track["radius_m"]

        length_axis_ne = np.asarray(track.get("length_axis_ne", [1.0, 0.0]), dtype=float).reshape(2)
        axis_norm = float(np.linalg.norm(length_axis_ne))
        track["length_axis_ne"] = (
            length_axis_ne / axis_norm
            if np.isfinite(length_axis_ne).all() and axis_norm >= 1e-6
            else np.array([1.0, 0.0], dtype=float)
        )

        if bool(track.get("motion_stable", False)):
            display_velocity_ne = np.asarray(track.get("velocity_mean_ne", track["vel_ne"]), dtype=float).reshape(2)
        else:
            display_velocity_ne = np.asarray(track["vel_ne"], dtype=float).reshape(2)
        if not np.isfinite(display_velocity_ne).all():
            display_velocity_ne = track["vel_ne"]

        speed_m_s = float(np.linalg.norm(display_velocity_ne))
        track["speed_m_s"] = speed_m_s
        previous_heading_rad = float(track.get("heading_rad", state[3]))
        if speed_m_s >= float(getattr(self, "obstacle_heading_hold_speed_m_s", 0.03)):
            heading_rad = float(state[3]) if np.isfinite(state[3]) else float(np.arctan2(display_velocity_ne[1], display_velocity_ne[0]))
            track["heading_rad"] = heading_rad
            track["heading_deg"] = float(np.rad2deg(heading_rad))
            track["heading_axis_ne"] = (
                display_velocity_ne / speed_m_s
                if bool(track.get("motion_stable", False))
                else track["length_axis_ne"].copy()
            )
        else:
            track["heading_rad"] = wrap_angle(previous_heading_rad)
            track["heading_deg"] = float(np.rad2deg(track["heading_rad"]))
            track["heading_axis_ne"] = track["length_axis_ne"].copy()

        if not self.obstacle_ekf_prediction_enabled:
            track["prediction_model"] = "disabled"
            track["prediction_ne"] = np.empty((0, 2), dtype=float)
            return

        if bool(track.get("motion_stable", False)):
            track["prediction_model"] = "ctrv_ekf"
        else:
            track["prediction_model"] = "ctrv_ekf_warmup"
        track["prediction_ne"] = self.obstacle_track_prediction_ne(track)

    def make_obstacle_track(
        self,
        detection_ne,
        stamp_s,
        pc1_m=np.nan,
        pc2_m=np.nan,
        length_axis_ne=None,
    ):
        detection_ne = np.asarray(detection_ne, dtype=float).reshape(2)
        pc1_m = float(pc1_m) if np.isfinite(pc1_m) and pc1_m > 0.0 else self.obstacle_min_pc1_m
        pc2_m = float(pc2_m) if np.isfinite(pc2_m) and pc2_m > 0.0 else self.obstacle_min_pc2_m
        length_axis_ne = np.asarray(
            [1.0, 0.0] if length_axis_ne is None else length_axis_ne,
            dtype=float,
        ).reshape(2)
        config = self.obstacle_ekf_config()
        track = {
            "id": self.apf_next_track_id,
            "ekf": ObstacleEKF(
                np.array([detection_ne[0], detection_ne[1], 0.0, 0.0, 0.0], dtype=float),
                ObstacleEKF.initial_covariance(config),
                config,
            ),
            "stamp_s": float(stamp_s),
            "last_seen_s": float(stamp_s),
            "hit_count": 1,
            "miss_count": 0,
            "history_ne": [],
            "lidar_history_ne": [],
            "motion_window": [],
            "pc1_m": pc1_m,
            "pc2_m": pc2_m,
            "length_axis_ne": length_axis_ne,
            "raw_detection_ne": detection_ne.copy(),
        }
        self.sync_obstacle_track_fields(track)
        self.append_obstacle_track_history(
            track,
            pc1_m=pc1_m,
            pc2_m=pc2_m,
            length_axis_ne=length_axis_ne,
            detection_ne=detection_ne,
            stamp_s=stamp_s,
        )
        self.apf_next_track_id += 1
        return track

    def make_obstacle_track_candidate(
        self,
        detection_ne,
        stamp_s,
        pc1_m=np.nan,
        pc2_m=np.nan,
        length_axis_ne=None,
    ):
        detection_ne = np.asarray(detection_ne, dtype=float).reshape(2)
        length_axis_ne = np.asarray(
            [1.0, 0.0] if length_axis_ne is None else length_axis_ne,
            dtype=float,
        ).reshape(2)
        pc1_m = float(pc1_m) if np.isfinite(pc1_m) and pc1_m > 0.0 else self.obstacle_min_pc1_m
        pc2_m = float(pc2_m) if np.isfinite(pc2_m) and pc2_m > 0.0 else self.obstacle_min_pc2_m
        return {
            "centre_ne": detection_ne.copy(),
            "stamp_s": float(stamp_s),
            "last_seen_s": float(stamp_s),
            "hit_count": 1,
            "pc1_m": pc1_m,
            "pc2_m": pc2_m,
            "length_axis_ne": length_axis_ne,
        }

    def predict_obstacle_track_to_time(self, track, stamp_s):
        now = float(stamp_s)
        dt = max(now - float(track.get("stamp_s", now)), 0.0)
        if not np.isfinite(dt) or dt < float(getattr(self, "obstacle_track_min_dt_s", 1e-3)):
            return
        state, covariance = self.obstacle_ekf_predict(
            track.get("state", np.r_[track.get("pos_ne", [np.nan, np.nan]), 0.0, 0.0, 0.0]),
            track.get("covariance", np.eye(5, dtype=float)),
            dt,
        )
        track["state"] = state
        track["covariance"] = covariance
        track["stamp_s"] = now
        self.sync_obstacle_track_fields(track)

    def append_obstacle_track_history(
        self,
        track,
        pc1_m=np.nan,
        pc2_m=np.nan,
        length_axis_ne=None,
        detection_ne=None,
        stamp_s=None,
    ):
        stamp_s = float(stamp_s if stamp_s is not None else track.get("stamp_s", time.time()))
        pos_ne = np.asarray(track["pos_ne"], dtype=float).reshape(2).copy()
        vel_ne = np.asarray(track.get("vel_ne", [0.0, 0.0]), dtype=float).reshape(2).copy()
        if not np.isfinite(vel_ne).all():
            vel_ne = np.zeros(2, dtype=float)

        pc1_m = float(pc1_m) if np.isfinite(pc1_m) and pc1_m > 0.0 else float(track.get("pc1_m", self.obstacle_min_pc1_m))
        pc2_m = float(pc2_m) if np.isfinite(pc2_m) and pc2_m > 0.0 else float(track.get("pc2_m", self.obstacle_min_pc2_m))
        track["pc1_m"] = max(pc1_m, self.obstacle_min_pc1_m)
        track["pc2_m"] = max(min(pc2_m, track["pc1_m"]), self.obstacle_min_pc2_m)

        if length_axis_ne is not None:
            length_axis_ne = np.asarray(length_axis_ne, dtype=float).reshape(2)
            axis_norm = float(np.linalg.norm(length_axis_ne))
            if np.isfinite(length_axis_ne).all() and axis_norm >= 1e-6:
                length_axis_ne = length_axis_ne / axis_norm
                previous_axis = np.asarray(track.get("length_axis_ne", length_axis_ne), dtype=float).reshape(2)
                if float(np.dot(length_axis_ne, previous_axis)) < 0.0:
                    length_axis_ne = -length_axis_ne
                track["length_axis_ne"] = length_axis_ne

        history = track.setdefault("history_ne", [])
        history.append(pos_ne)
        if len(history) > self.obstacle_history_len:
            del history[:-self.obstacle_history_len]

        if detection_ne is not None:
            detection_ne = np.asarray(detection_ne, dtype=float).reshape(2)
            if np.isfinite(detection_ne).all():
                lidar_history = track.setdefault("lidar_history_ne", [])
                lidar_history.append(detection_ne.copy())
                if len(lidar_history) > self.obstacle_history_len:
                    del lidar_history[:-self.obstacle_history_len]

        speed_m_s = float(np.linalg.norm(vel_ne))
        heading_rad = float(np.arctan2(vel_ne[1], vel_ne[0])) if speed_m_s >= self.apf_dynamic_speed_threshold_m_s else np.nan
        motion_window = track.setdefault("motion_window", [])
        motion_window.append({
            "stamp_s": stamp_s,
            "pos_ne": pos_ne,
            "vel_ne": vel_ne,
            "speed_m_s": speed_m_s,
            "heading_rad": heading_rad,
            "v_m_s": float(track.get("v_m_s", speed_m_s)),
            "phi_rad": float(track.get("phi_rad", heading_rad if np.isfinite(heading_rad) else 0.0)),
            "phi_dot_rad_s": float(track.get("phi_dot_rad_s", 0.0)),
            "pc1_m": track["pc1_m"],
            "pc2_m": track["pc2_m"],
        })

        cutoff_s = stamp_s - max(float(self.obstacle_stats_window_s), 0.0)
        track["motion_window"] = [
            sample for sample in motion_window
            if float(sample.get("stamp_s", stamp_s)) >= cutoff_s
        ][-self.obstacle_history_len:]
        self.update_obstacle_track_statistics(track)

    def update_obstacle_track_statistics(self, track):
        samples = track.get("motion_window", [])
        samples = [
            sample for sample in samples
            if np.isfinite(np.asarray(sample.get("pos_ne", [np.nan, np.nan]), dtype=float)).all()
        ]
        track["stats_sample_count"] = int(len(samples))

        if not samples:
            track["velocity_mean_ne"] = np.asarray(track.get("vel_ne", [0.0, 0.0]), dtype=float).reshape(2)
            track["velocity_var_ne"] = np.zeros(2, dtype=float)
            track["speed_mean_m_s"] = float(np.linalg.norm(track["velocity_mean_ne"]))
            track["speed_var_m2_s2"] = 0.0
            track["heading_mean_rad"] = np.nan
            track["heading_var_rad2"] = np.nan
            track["heading_circular_variance"] = np.nan
            track["pc1_mean_m"] = track.get("pc1_m", self.obstacle_min_pc1_m)
            track["pc2_mean_m"] = track.get("pc2_m", self.obstacle_min_pc2_m)
            track["pc1_var_m2"] = 0.0
            track["pc2_var_m2"] = 0.0
            track["phi_dot_mean_rad_s"] = float(track.get("phi_dot_rad_s", 0.0))
            track["phi_dot_var_rad_s2"] = 0.0
            track["displacement_m"] = 0.0
            track["motion_stable"] = False
            self.sync_obstacle_track_fields(track)
            return

        velocities = np.asarray([sample["vel_ne"] for sample in samples], dtype=float)
        finite_vel = np.isfinite(velocities).all(axis=1)
        velocities = velocities[finite_vel]
        sample_times = np.asarray([float(sample.get("stamp_s", 0.0)) for sample in samples], dtype=float)
        sample_positions = np.asarray([sample["pos_ne"] for sample in samples], dtype=float)
        time_span_s = float(np.ptp(sample_times)) if len(sample_times) > 1 else 0.0
        if len(samples) >= 3 and time_span_s >= 0.4:
            centred_times = sample_times - float(np.mean(sample_times))
            design = np.column_stack([centred_times, np.ones_like(centred_times)])
            fit, _, _, _ = np.linalg.lstsq(design, sample_positions, rcond=None)
            track["velocity_mean_ne"] = fit[0]
        elif len(velocities) > 0:
            track["velocity_mean_ne"] = np.mean(velocities, axis=0)
        else:
            track["velocity_mean_ne"] = np.asarray(track.get("vel_ne", [0.0, 0.0]), dtype=float).reshape(2)

        if len(velocities) > 0:
            track["velocity_var_ne"] = np.var(velocities, axis=0)
        else:
            track["velocity_var_ne"] = np.zeros(2, dtype=float)

        speeds = np.asarray([sample["speed_m_s"] for sample in samples], dtype=float)
        speeds = speeds[np.isfinite(speeds)]
        fitted_speed_m_s = float(np.linalg.norm(track["velocity_mean_ne"]))
        if len(speeds) > 0:
            track["speed_mean_m_s"] = fitted_speed_m_s
            track["speed_var_m2_s2"] = float(np.var(speeds))
        else:
            track["speed_mean_m_s"] = fitted_speed_m_s
            track["speed_var_m2_s2"] = 0.0
        track["displacement_m"] = float(np.linalg.norm(sample_positions[-1] - sample_positions[0])) if len(sample_positions) >= 2 else 0.0

        headings = np.asarray([sample["heading_rad"] for sample in samples], dtype=float)
        headings = headings[np.isfinite(headings)]
        if len(headings) > 0:
            sin_mean = float(np.mean(np.sin(headings)))
            cos_mean = float(np.mean(np.cos(headings)))
            heading_mean = float(np.arctan2(sin_mean, cos_mean))
            heading_error = wrap_angle(headings - heading_mean)
            resultant_length = float(np.hypot(sin_mean, cos_mean))
            track["heading_mean_rad"] = heading_mean
            track["heading_var_rad2"] = float(np.var(heading_error))
            track["heading_circular_variance"] = float(1.0 - np.clip(resultant_length, 0.0, 1.0))
        else:
            track["heading_mean_rad"] = np.nan
            track["heading_var_rad2"] = np.nan
            track["heading_circular_variance"] = np.nan

        pc1_values = np.asarray([sample["pc1_m"] for sample in samples], dtype=float)
        pc2_values = np.asarray([sample["pc2_m"] for sample in samples], dtype=float)
        pc1_values = pc1_values[np.isfinite(pc1_values) & (pc1_values > 0.0)]
        pc2_values = pc2_values[np.isfinite(pc2_values) & (pc2_values > 0.0)]
        track["pc1_mean_m"] = (
            max(float(np.mean(pc1_values)), self.obstacle_min_pc1_m)
            if len(pc1_values) > 0
            else float(track.get("pc1_m", self.obstacle_min_pc1_m))
        )
        track["pc2_mean_m"] = (
            max(float(np.mean(pc2_values)), self.obstacle_min_pc2_m)
            if len(pc2_values) > 0
            else float(track.get("pc2_m", self.obstacle_min_pc2_m))
        )
        track["pc2_mean_m"] = min(track["pc2_mean_m"], track["pc1_mean_m"])
        track["pc1_var_m2"] = float(np.var(pc1_values)) if len(pc1_values) > 0 else 0.0
        track["pc2_var_m2"] = float(np.var(pc2_values)) if len(pc2_values) > 0 else 0.0
        track["pc1_m"] = track["pc1_mean_m"]
        track["pc2_m"] = track["pc2_mean_m"]

        phi_dot_samples = np.asarray(
            [float(sample.get("phi_dot_rad_s", 0.0)) for sample in samples],
            dtype=float,
        )
        phi_dot_samples = phi_dot_samples[np.isfinite(phi_dot_samples)]
        track["phi_dot_mean_rad_s"] = float(np.mean(phi_dot_samples)) if len(phi_dot_samples) > 0 else float(track.get("phi_dot_rad_s", 0.0))
        track["phi_dot_var_rad_s2"] = float(np.var(phi_dot_samples)) if len(phi_dot_samples) > 0 else 0.0

        track["motion_stable"] = self.obstacle_track_motion_is_stable(track)
        self.obstacle_track_regularize_velocity(track)
        self.sync_obstacle_track_fields(track)

    def annotate_lidar_obstacle_with_track(self, obstacle, track):
        prediction_ne = np.asarray(track.get("prediction_ne", np.empty((0, 2))), dtype=float)
        raw_centre_ne = np.asarray(obstacle.get("centre_ne", [np.nan, np.nan]), dtype=float).reshape(2)
        obstacle["track_id"] = int(track["id"])
        obstacle["raw_centre_ne"] = raw_centre_ne.tolist()
        obstacle["raw_centre_body"] = np.asarray(obstacle.get("centre_body", [np.nan, np.nan]), dtype=float).reshape(2).tolist()
        obstacle["raw_px"] = float(raw_centre_ne[0])
        obstacle["raw_py"] = float(raw_centre_ne[1])
        obstacle["filtered_px"] = float(track["pos_ne"][0])
        obstacle["filtered_py"] = float(track["pos_ne"][1])
        predicted_position_ne = (
            prediction_ne[-1]
            if prediction_ne.ndim == 2 and len(prediction_ne) > 0
            else np.asarray(track.get("virtual_position_ne", [np.nan, np.nan]), dtype=float).reshape(2)
        )
        obstacle["predicted_px"] = float(predicted_position_ne[0])
        obstacle["predicted_py"] = float(predicted_position_ne[1])
        obstacle["centre_ne"] = track["pos_ne"].tolist()
        obstacle["centre_body"] = self.earth_point_to_body(track["pos_ne"]).tolist()
        obstacle["distance_m"] = float(np.linalg.norm(obstacle["centre_body"]))
        obstacle["min_distance_m"] = float(min(obstacle.get("min_distance_m", obstacle["distance_m"]), obstacle["distance_m"]))
        obstacle["angle_rad"] = float(np.arctan2(obstacle["centre_body"][1], obstacle["centre_body"][0]))
        obstacle["angle_deg"] = float(np.rad2deg(obstacle["angle_rad"]))
        obstacle["velocity_ne"] = np.asarray(track["vel_ne"], dtype=float).reshape(2).tolist()
        obstacle["velocity_mean_ne"] = np.asarray(track.get("velocity_mean_ne", track["vel_ne"]), dtype=float).reshape(2).tolist()
        obstacle["velocity_var_ne"] = np.asarray(track.get("velocity_var_ne", [0.0, 0.0]), dtype=float).reshape(2).tolist()
        obstacle["v"] = float(track.get("v_m_s", 0.0))
        obstacle["phi"] = float(track.get("phi_rad", np.nan))
        obstacle["phi_dot"] = float(track.get("phi_dot_rad_s", 0.0))
        obstacle["speed_m_s"] = float(track.get("speed_m_s", 0.0))
        obstacle["speed_mean_m_s"] = float(track.get("speed_mean_m_s", obstacle["speed_m_s"]))
        obstacle["speed_var_m2_s2"] = float(track.get("speed_var_m2_s2", 0.0))
        obstacle["heading_rad"] = float(track.get("heading_rad", np.nan))
        obstacle["heading_deg"] = float(track.get("heading_deg", np.nan))
        obstacle["heading_mean_rad"] = float(track.get("heading_mean_rad", np.nan))
        obstacle["heading_var_rad2"] = float(track.get("heading_var_rad2", np.nan))
        obstacle["phi_dot_mean_rad_s"] = float(track.get("phi_dot_mean_rad_s", track.get("phi_dot_rad_s", 0.0)))
        obstacle["phi_dot_var_rad_s2"] = float(track.get("phi_dot_var_rad_s2", 0.0))
        obstacle["pc1_m"] = float(track.get("pc1_mean_m", track.get("pc1_m", self.obstacle_min_pc1_m)))
        obstacle["pc2_m"] = float(track.get("pc2_mean_m", track.get("pc2_m", self.obstacle_min_pc2_m)))
        obstacle["pc1_var_m2"] = float(track.get("pc1_var_m2", 0.0))
        obstacle["pc2_var_m2"] = float(track.get("pc2_var_m2", 0.0))
        obstacle["length_axis_ne"] = np.asarray(
            track.get("heading_axis_ne", track.get("length_axis_ne", [1.0, 0.0])),
            dtype=float,
        ).reshape(2).tolist()
        obstacle["equivalent_radius_m"] = float(track.get("equivalent_radius_m", self.obstacle_min_equivalent_radius_m))
        obstacle["stats_sample_count"] = int(track.get("stats_sample_count", 0))
        obstacle["motion_stable"] = bool(track.get("motion_stable", False))
        obstacle["position_uncertainty_m2"] = float(track.get("position_uncertainty_m2", np.nan))
        obstacle["covariance_trace"] = float(np.trace(np.asarray(track.get("covariance", np.eye(5, dtype=float)), dtype=float)))
        obstacle["prediction_model"] = track.get("prediction_model", "ctrv_ekf")
        obstacle["predicted_trajectory_ne"] = prediction_ne.tolist()

    def prune_obstacle_tracks(self, now):
        self.apf_obstacle_tracks = [
            track for track in self.apf_obstacle_tracks
            if now - float(track.get("last_seen_s", track.get("stamp_s", now))) <= self.apf_track_timeout_s
            and int(track.get("miss_count", 0)) <= 5
        ]

    def update_apf_obstacle_tracks(self, stamp_s):
        tracking_enabled = bool(getattr(self, "obstacle_ekf_tracking_enabled", True))
        if not tracking_enabled:
            self.apf_obstacle_tracks = []
            self.apf_virtual_obstacles = []
            tracked_fields = {
                "track_id",
                "raw_px",
                "raw_py",
                "filtered_px",
                "filtered_py",
                "predicted_px",
                "predicted_py",
                "velocity_ne",
                "velocity_mean_ne",
                "velocity_var_ne",
                "v",
                "phi",
                "phi_dot",
                "speed_m_s",
                "speed_mean_m_s",
                "speed_var_m2_s2",
                "heading_rad",
                "heading_deg",
                "heading_mean_rad",
                "heading_var_rad2",
                "phi_dot_mean_rad_s",
                "phi_dot_var_rad_s2",
                "pc1_m",
                "pc2_m",
                "pc1_var_m2",
                "pc2_var_m2",
                "stats_sample_count",
                "motion_stable",
                "position_uncertainty_m2",
                "covariance_trace",
                "prediction_model",
                "predicted_trajectory_ne",
            }
            for obstacle in self.lidar_obstacles:
                for field in tracked_fields:
                    obstacle.pop(field, None)
            return

        now = float(stamp_s if stamp_s is not None else time.time())
        confirmation_hits = max(int(getattr(self, "obstacle_track_confirmation_hits", 2)), 2)
        detections = []

        for obstacle_index, obstacle in enumerate(self.lidar_obstacles):
            centre_ne = np.asarray(obstacle.get("centre_ne", [np.nan, np.nan]), dtype=float).reshape(2)
            if np.isfinite(centre_ne).all():
                pc1_m = float(obstacle.get("pc1_m", self.obstacle_min_pc1_m))
                pc2_m = float(obstacle.get("pc2_m", self.obstacle_min_pc2_m))
                length_axis_ne = np.asarray(
                    obstacle.get("length_axis_ne", [1.0, 0.0]),
                    dtype=float,
                ).reshape(2)
                measurement_covariance = np.asarray(
                    obstacle.get(
                        "measurement_covariance",
                        self.obstacle_measurement_covariance(obstacle.get("centre_body", [0.0, 0.0])),
                    ),
                    dtype=float,
                ).reshape(2, 2)
                detections.append(
                    (obstacle_index, centre_ne, pc1_m, pc2_m, length_axis_ne, measurement_covariance)
                )

        if not detections:
            self.apf_obstacle_track_candidates = [
                candidate
                for candidate in self.apf_obstacle_track_candidates
                if now - float(candidate.get("last_seen_s", candidate.get("stamp_s", now))) <= self.apf_track_timeout_s
            ]
            for track in self.apf_obstacle_tracks:
                self.predict_obstacle_track_to_time(track, now)
                track["miss_count"] = int(track.get("miss_count", 0)) + 1

            self.prune_obstacle_tracks(now)
            return

        detection_positions = np.asarray(
            [detection for _, detection, _, _, _, _ in detections],
            dtype=float,
        )
        predicted_tracks = []
        candidates = []
        matched_candidates = set()

        for track_index, track in enumerate(self.apf_obstacle_tracks):
            dt = max(now - float(track.get("stamp_s", now)), 0.0)
            predicted_state, predicted_covariance = self.obstacle_ekf_predict(
                track.get("state", np.r_[track.get("pos_ne", [np.nan, np.nan]), 0.0, 0.0, 0.0]),
                track.get("covariance", np.eye(5, dtype=float)),
                dt,
            )
            predicted_tracks.append((predicted_state, predicted_covariance))
            predicted_pos = predicted_state[0:2]

            for detection_index, detection in enumerate(detection_positions):
                distance = float(np.linalg.norm(detection - predicted_pos))
                state_size = int(np.asarray(predicted_state, dtype=float).reshape(-1).size)
                H = np.zeros((2, state_size), dtype=float)
                H[0, 0] = 1.0
                H[1, 1] = 1.0
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
            corrected_state, corrected_covariance = self.obstacle_ekf_update(
                predicted_state,
                predicted_covariance,
                detection,
                detections[detection_index][5],
            )

            track["state"] = corrected_state
            track["covariance"] = corrected_covariance
            track["stamp_s"] = now
            track["last_seen_s"] = now
            track["hit_count"] = int(track.get("hit_count", 0)) + 1
            track["miss_count"] = 0
            track["raw_detection_ne"] = detection.copy()
            self.sync_obstacle_track_fields(track)
            self.append_obstacle_track_history(
                track,
                pc1_m=detections[detection_index][2],
                pc2_m=detections[detection_index][3],
                length_axis_ne=detections[detection_index][4],
                detection_ne=detection,
                stamp_s=now,
            )
            assigned_tracks.add(track_index)
            assigned_detections.add(detection_index)
            detection_track[detection_index] = track

        for track_index, track in enumerate(self.apf_obstacle_tracks):
            if track_index not in assigned_tracks:
                predicted_state, predicted_covariance = predicted_tracks[track_index]
                track["state"] = predicted_state
                track["covariance"] = predicted_covariance
                track["stamp_s"] = now
                track["miss_count"] = int(track.get("miss_count", 0)) + 1
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
                candidate_pos = np.asarray(candidate.get("centre_ne", [np.nan, np.nan]), dtype=float).reshape(2)
                if not np.isfinite(candidate_pos).all():
                    continue
                distance = float(np.linalg.norm(detection_ne - candidate_pos))
                if distance < best_candidate_distance:
                    best_candidate_distance = distance
                    best_candidate_index = candidate_index

            if best_candidate_index is not None and best_candidate_distance <= self.apf_track_association_m:
                candidate = self.apf_obstacle_track_candidates[best_candidate_index]
                candidate["centre_ne"] = detection_ne.copy()
                candidate["last_seen_s"] = now
                candidate["hit_count"] = int(candidate.get("hit_count", 0)) + 1
                candidate["pc1_m"] = pc1_m
                candidate["pc2_m"] = pc2_m
                candidate["length_axis_ne"] = length_axis_ne
                matched_candidates.add(best_candidate_index)
                if int(candidate["hit_count"]) >= confirmation_hits:
                    track = self.make_obstacle_track(
                        detection_ne,
                        now,
                        pc1_m=pc1_m,
                        pc2_m=pc2_m,
                        length_axis_ne=length_axis_ne,
                    )
                    self.apf_obstacle_tracks.append(track)
                    assigned_detections.add(detection_index)
                    detection_track[detection_index] = track
                continue

            self.apf_obstacle_track_candidates.append(
                self.make_obstacle_track_candidate(
                    detection_ne,
                    now,
                    pc1_m=pc1_m,
                    pc2_m=pc2_m,
                    length_axis_ne=length_axis_ne,
                )
            )

        for detection_index, track in detection_track.items():
            obstacle_index, _, _, _, _, _ = detections[detection_index]
            self.annotate_lidar_obstacle_with_track(self.lidar_obstacles[obstacle_index], track)

        self.apf_obstacle_track_candidates = [
            candidate
            for candidate in self.apf_obstacle_track_candidates
            if now - float(candidate.get("last_seen_s", candidate.get("stamp_s", now))) <= self.apf_track_timeout_s
        ]
        self.prune_obstacle_tracks(now)

    def obstacle_track_visuals(self):
        if not self.obstacle_ekf_prediction_enabled:
            return []

        now = float(self.latest_lidar_received_s if self.latest_lidar_received_s is not None else time.time())
        visuals = []

        for track in self.apf_obstacle_tracks:
            if now - float(track.get("last_seen_s", track.get("stamp_s", now))) > self.apf_track_timeout_s:
                continue

            state, _ = self.obstacle_ekf_predict(
                track.get("state", np.r_[track.get("pos_ne", [np.nan, np.nan]), 0.0, 0.0, 0.0]),
                track.get("covariance", np.eye(5, dtype=float)),
                max(now - float(track.get("stamp_s", now)), 0.0),
            )
            velocity_ne = np.asarray(track.get("velocity_mean_ne", heading_to_velocity_ne(state[2], state[3])), dtype=float).reshape(2)
            if not np.isfinite(velocity_ne).all():
                velocity_ne = heading_to_velocity_ne(state[2], state[3])
            speed_m_s = float(np.linalg.norm(velocity_ne))
            heading_rad = float(state[3]) if speed_m_s >= 1e-3 else np.nan

            # Preserve the complete statistical state so the visualised
            # trajectory uses the same acceleration/stability gates as the
            # collision predictor and APF controller.
            prediction_track = dict(track)
            prediction_track["state"] = state
            prediction_track["velocity_mean_ne"] = velocity_ne
            prediction_ne = self.obstacle_track_prediction_ne(prediction_track)
            history_ne = np.asarray(track.get("history_ne", []), dtype=float)
            if history_ne.ndim != 2 or history_ne.shape[1] != 2:
                history_ne = np.empty((0, 2), dtype=float)
            lidar_history_ne = np.asarray(track.get("lidar_history_ne", []), dtype=float)
            if lidar_history_ne.ndim != 2 or lidar_history_ne.shape[1] != 2:
                lidar_history_ne = np.empty((0, 2), dtype=float)

            visuals.append({
                "id": int(track["id"]),
                "position_ne": state[0:2].copy(),
                "velocity_ne": velocity_ne.copy(),
                "speed_m_s": speed_m_s,
                "heading_rad": heading_rad,
                "heading_deg": float(np.rad2deg(heading_rad)) if np.isfinite(heading_rad) else np.nan,
                "phi_dot_rad_s": float(state[4]),
                "prediction_ne": prediction_ne.copy(),
                "history_ne": history_ne.copy(),
                "lidar_history_ne": lidar_history_ne.copy(),
                "speed_var_m2_s2": float(track.get("speed_var_m2_s2", 0.0)),
                "heading_var_rad2": float(track.get("heading_var_rad2", np.nan)),
                "position_uncertainty_m2": float(track.get("position_uncertainty_m2", np.nan)),
                "pc1_m": float(track.get("pc1_mean_m", track.get("pc1_m", self.obstacle_min_pc1_m))),
                "pc2_m": float(track.get("pc2_mean_m", track.get("pc2_m", self.obstacle_min_pc2_m))),
                "virtual_position_ne": np.asarray(
                    track.get("virtual_position_ne", [np.nan, np.nan]),
                    dtype=float,
                ).reshape(2).copy(),
                "collision_time_s": float(track.get("collision_time_s", np.nan)),
                "prediction_model": track.get("prediction_model", "ctrv_ekf"),
                "stats_sample_count": int(track.get("stats_sample_count", 0)),
                "motion_stable": bool(track.get("motion_stable", False)),
                "hit_count": int(track.get("hit_count", 0)),
                "miss_count": int(track.get("miss_count", 0)),
            })

        return visuals

    def apf_track_for_obstacle(self, obstacle):
        if not self.obstacle_ekf_prediction_enabled:
            return None

        centre_ne = np.asarray(obstacle.get("centre_ne", [np.nan, np.nan]), dtype=float).reshape(2)
        if not np.isfinite(centre_ne).all():
            return None

        now = float(self.latest_lidar_received_s if self.latest_lidar_received_s is not None else time.time())
        best_track = None
        best_distance = np.inf

        for track in self.apf_obstacle_tracks:
            if now - float(track.get("last_seen_s", track.get("stamp_s", now))) > self.apf_track_timeout_s:
                continue

            predicted_state, _ = self.obstacle_ekf_predict(
                track.get("state", np.r_[track.get("pos_ne", [np.nan, np.nan]), 0.0, 0.0, 0.0]),
                track.get("covariance", np.eye(5, dtype=float)),
                max(now - float(track.get("stamp_s", now)), 0.0),
            )
            predicted_pos = predicted_state[0:2]
            distance = float(np.linalg.norm(centre_ne - predicted_pos))

            if distance < best_distance:
                best_distance = distance
                best_track = track

        if best_distance <= max(self.apf_track_association_m * 1.5, self.lidar_dbscan_eps_m * 2.0):
            return best_track

        return None

    def obstacle_pc_dimensions(self, obstacle):
        min_pc1_m = float(getattr(self, "obstacle_min_pc1_m", 0.3))
        min_pc2_m = float(getattr(self, "obstacle_min_pc2_m", 0.16))
        pc1_m = float(obstacle.get("pc1_m", min_pc1_m))
        pc2_m = float(obstacle.get("pc2_m", min_pc2_m))
        if not np.isfinite(pc1_m) or pc1_m <= 0.0:
            pc1_m = min_pc1_m
        if not np.isfinite(pc2_m) or pc2_m <= 0.0:
            pc2_m = min_pc2_m
        pc1_m = max(pc1_m, min_pc1_m)
        pc2_m = max(min(pc2_m, pc1_m), min_pc2_m)
        return pc1_m, pc2_m

    def obstacle_length_axis_ne(self, obstacle):
        axis_ne = np.asarray(
            obstacle.get("length_axis_ne", [1.0, 0.0]),
            dtype=float,
        ).reshape(2)
        axis_norm = float(np.linalg.norm(axis_ne))
        if not np.isfinite(axis_ne).all() or axis_norm < 1e-6:
            return np.array([1.0, 0.0], dtype=float)
        return axis_ne / axis_norm

    def apf_cpa_metrics(self, obs_pos_body, obs_vel_body, own_vel_body):
        if not self.obstacle_ekf_prediction_enabled:
            return np.nan, np.nan

        rel_vel = np.asarray(obs_vel_body, dtype=float).reshape(2) - np.asarray(own_vel_body, dtype=float).reshape(2)
        obs_pos_body = np.asarray(obs_pos_body, dtype=float).reshape(2)
        rel_speed_sq = float(np.dot(rel_vel, rel_vel))

        if rel_speed_sq < 1e-9:
            return np.inf, float(np.linalg.norm(obs_pos_body))

        tcpa = max(-float(np.dot(obs_pos_body, rel_vel)) / rel_speed_sq, 0.0)
        dcpa = float(np.linalg.norm(obs_pos_body + rel_vel * tcpa))
        return tcpa, dcpa

    def apf_pass_astern_side_from_velocity(self, obs_vel_body):
        obs_vel_body = np.asarray(obs_vel_body, dtype=float).reshape(2)
        if not np.isfinite(obs_vel_body).all():
            return 0.0

        if float(np.linalg.norm(obs_vel_body)) < self.apf_dynamic_speed_threshold_m_s:
            return 0.0

        lateral_speed = float(obs_vel_body[1])
        if abs(lateral_speed) < self.apf_dynamic_speed_threshold_m_s:
            return 0.0

        # Body y is positive to starboard/right.  The planner side sign is
        # positive to port/left, so the astern side has the same sign here.
        return float(np.sign(lateral_speed))

    def apf_obstacle_endpoint_direction_body(self, obs_pos_body, obs_vel_body, obstacle, along_sign):
        obs_pos_body = np.asarray(obs_pos_body, dtype=float).reshape(2)
        obs_vel_body = np.asarray(obs_vel_body, dtype=float).reshape(2)
        if not np.isfinite(obs_pos_body).all() or not np.isfinite(obs_vel_body).all():
            return np.zeros(2, dtype=float)

        obs_speed = float(np.linalg.norm(obs_vel_body))
        if obs_speed < self.apf_dynamic_speed_threshold_m_s:
            return np.zeros(2, dtype=float)

        axis_body = obs_vel_body / max(obs_speed, 1e-6)
        pc1_m, _ = self.obstacle_pc_dimensions(obstacle)
        endpoint_body = obs_pos_body + float(along_sign) * 0.5 * pc1_m * axis_body
        endpoint_distance = float(np.linalg.norm(endpoint_body))
        if endpoint_distance < 1e-6:
            return float(along_sign) * axis_body
        return endpoint_body / endpoint_distance

    def apf_stern_direction_body(self, obs_pos_body, obs_vel_body, obstacle):
        return self.apf_obstacle_endpoint_direction_body(
            obs_pos_body,
            obs_vel_body,
            obstacle,
            -1.0,
        )

    def apf_bow_direction_body(self, obs_pos_body, obs_vel_body, obstacle):
        return self.apf_obstacle_endpoint_direction_body(
            obs_pos_body,
            obs_vel_body,
            obstacle,
            1.0,
        )

    def apf_crossing_strategy_from_velocity(self, obs_vel_body):
        obs_vel_body = np.asarray(obs_vel_body, dtype=float).reshape(2)
        if not np.isfinite(obs_vel_body).all():
            return "none", 0.0

        if float(np.linalg.norm(obs_vel_body)) < self.apf_dynamic_speed_threshold_m_s:
            return "none", 0.0

        lateral_speed = float(obs_vel_body[1])
        if abs(lateral_speed) < self.apf_dynamic_speed_threshold_m_s:
            return "none", 0.0

        return "pass_astern", self.apf_pass_astern_side_from_velocity(obs_vel_body)

    def apf_encounter_speed_m_s(self, encounter, nearest_active_level, force_angle):
        base_speed = float(getattr(self, "apf_constant_descent_speed_m_s", self.route_tracking_speed_m_s))
        if encounter == "overtaking":
            return float(
                np.clip(
                    max(base_speed, getattr(self, "apf_overtaking_surge_m_s", base_speed)),
                    0.0,
                    self.v_max,
                )
            )

        if isinstance(encounter, str) and encounter.startswith("crossing"):
            pass_ahead_active = "ahead" in str(getattr(self, "apf_colreg_rule", "")).lower()
            if np.isfinite(nearest_active_level) and nearest_active_level <= 1.0:
                target_speed = max(
                    getattr(self, "apf_crossing_min_forward_speed", base_speed),
                    getattr(self, "apf_crossing_close_quarters_surge_m_s", base_speed),
                )
            else:
                target_speed = max(base_speed, getattr(self, "apf_crossing_min_forward_speed", base_speed))
            if pass_ahead_active:
                pass_ahead_speed = float(getattr(self, "apf_crossing_pass_ahead_surge_m_s", target_speed))
                target_speed = max(target_speed, pass_ahead_speed)
                tcpa_s = float(getattr(self, "apf_colreg_tcpa_s", np.nan))
                dcpa_m = float(getattr(self, "apf_colreg_dcpa_m", np.nan))
                horizon_s = max(float(getattr(self, "apf_collision_horizon_s", 1.0)), 1e-6)
                safe_dcpa_m = max(
                    float(getattr(self, "apf_crossing_pass_ahead_safe_dcpa_m", 0.0)),
                    1e-6,
                )
                tcpa_urgency = 0.0 if not np.isfinite(tcpa_s) else float(
                    np.clip((horizon_s - max(tcpa_s, 0.0)) / horizon_s, 0.0, 1.0)
                )
                dcpa_urgency = 0.0 if not np.isfinite(dcpa_m) else float(
                    np.clip((safe_dcpa_m - max(dcpa_m, 0.0)) / safe_dcpa_m, 0.0, 1.0)
                )
                pass_ahead_urgency = max(tcpa_urgency, dcpa_urgency)
                target_speed = max(
                    target_speed,
                    base_speed + pass_ahead_urgency * max(pass_ahead_speed - base_speed, 0.0),
                )
            return float(np.clip(target_speed, 0.0, self.v_max))

        if encounter == "head_on":
            turn_scale = float(np.clip(1.0 - abs(force_angle) / max(self.apf_heading_step_limit_rad, 1e-6), 0.35, 1.0))
            target_speed = max(
                getattr(self, "apf_min_forward_speed", 0.0),
                getattr(self, "apf_head_on_surge_m_s", base_speed) * turn_scale,
            )
            return float(np.clip(target_speed, 0.0, self.v_max))

        if np.isfinite(nearest_active_level) and nearest_active_level <= 1.0:
            target_speed = max(
                getattr(self, "apf_min_forward_speed", base_speed),
                getattr(self, "apf_close_quarters_surge_m_s", base_speed),
            )
            return float(np.clip(target_speed, 0.0, self.v_max))

        return float(np.clip(base_speed, 0.0, self.v_max))

    def own_prediction_velocity_ne(self):
        current_ne = np.array([float(self.North), float(self.East)], dtype=float)
        to_goal_ne = self.goal_ne - current_ne
        to_goal_distance_m = float(np.linalg.norm(to_goal_ne))
        route_direction_ne = (
            to_goal_ne / to_goal_distance_m
            if self.final_approach_active(to_goal_distance_m) and to_goal_distance_m >= 1e-6
            else self.route_path_unit_ne
        )
        return route_direction_ne * max(
            float(self.route_tracking_speed_m_s),
            self.apf_min_forward_speed,
        )

    def obstacle_track_state_at(self, track, dt_s):
        dt_s = max(float(dt_s), 0.0)
        state = ObstacleEKF.normalise_state(
            track.get("state", [np.nan, np.nan, 0.0, 0.0, 0.0]),
            getattr(self, "obstacle_heading_hold_speed_m_s", 0.03),
        )
        if not np.isfinite(state).all():
            return None, None

        ekf = ObstacleEKF(
            state,
            track.get("covariance", np.eye(5, dtype=float)),
            self.obstacle_ekf_config(),
        )
        predicted_state = ekf._ctrv_step(state, dt_s)
        pos_ne = predicted_state[0:2]
        vel_ne = heading_to_velocity_ne(predicted_state[2], predicted_state[3])
        return pos_ne, vel_ne

    def virtual_collision_visuals(self):
        visuals = []
        for obstacle in self.apf_virtual_obstacles:
            if not bool(obstacle.get("predicted_risk_active", False)):
                continue
            collision_position_ne = np.asarray(
                obstacle.get("collision_position_ne", [np.nan, np.nan]),
                dtype=float,
            ).reshape(2)
            if not np.isfinite(collision_position_ne).all():
                continue
            visuals.append({
                "track_id": int(obstacle.get("track_id", 0)),
                "collision_position_ne": collision_position_ne.copy(),
                "own_prediction_ne": np.asarray(
                    obstacle.get("own_prediction_ne", [np.nan, np.nan]),
                    dtype=float,
                ).reshape(2),
                "tcpa_s": float(obstacle.get("tcpa_s", np.nan)),
                "predicted_separation_m": float(obstacle.get("dcpa_m", np.nan)),
                "collision_level": float(obstacle.get("collision_level", np.nan)),
            })
        return visuals

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
        own_bearing_from_obstacle_deg = np.nan

        if dynamic_obstacle:
            obstacle_heading_rad = float(np.arctan2(obs_vel_body[1], obs_vel_body[0]))
            relative_heading_deg = abs(float(np.rad2deg(wrap_angle(obstacle_heading_rad))))
            own_bearing_from_obstacle_deg = abs(
                float(
                    np.rad2deg(
                        wrap_angle(
                            float(np.arctan2(-obs_pos_body[1], -obs_pos_body[0]))
                            - obstacle_heading_rad
                        )
                    )
                )
            )

        if np.isfinite(relative_heading_deg) and abs(bearing_starboard_deg) <= 22.5 and relative_heading_deg >= 157.5:
            return "head_on", -1.0, "COLREG Rule 14: alter to starboard"

        if (
            np.isfinite(relative_heading_deg)
            and abs(bearing_starboard_deg) <= 67.5
            and relative_heading_deg <= 67.5
            and own_bearing_from_obstacle_deg >= 112.5
            and own_speed > obs_speed + self.apf_dynamic_speed_threshold_m_s
        ):
            return "overtaking", 1.0, "COLREG Rule 13: overtake on port side"

        crossing_strategy, crossing_side = self.apf_crossing_strategy_from_velocity(obs_vel_body)

        if dynamic_obstacle and 0.0 < bearing_starboard_deg <= 112.5:
            requested_side = crossing_side if crossing_side != 0.0 else -1.0
            return "crossing_from_starboard", requested_side, "COLREG Rule 15: give way, pass astern"

        if dynamic_obstacle and -112.5 <= bearing_starboard_deg < 0.0:
            requested_side = crossing_side if crossing_side != 0.0 else 1.0
            return "crossing_from_port", requested_side, "Crossing: pass astern"

        return "static_obstacle", 0.0, "none"

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
        obs_pos_body = np.asarray(obstacle.get("centre_body", [np.nan, np.nan]), dtype=float).reshape(2)
        if not np.isfinite(obs_pos_body).all():
            return False

        body_angle_rad = float(np.arctan2(obs_pos_body[1], obs_pos_body[0]))
        return abs(wrap_angle(body_angle_rad)) <= self.apf_priority_front_half_angle_rad

    def apf_obstacle_in_forward_half_plane(self, obstacle):
        obs_pos_body = np.asarray(obstacle.get("centre_body", [np.nan, np.nan]), dtype=float).reshape(2)
        if not np.isfinite(obs_pos_body).all():
            return False
        return float(obs_pos_body[0]) >= 0.0

    def apf_lock_side(self, requested_side, obstacle_level):
        now_s = float(self.timefromstart) if self.timefromstart is not None else 0.0
        obstacle_close = (
            np.isfinite(obstacle_level)
            and obstacle_level <= self.apf_side_lock_exit_level
        )

        if requested_side == 0.0:
            if self.apf_side_lock_sign != 0.0 and (obstacle_close or now_s < self.apf_side_lock_until_s):
                self.apf_side_lock_active = True
                return self.apf_side_lock_sign

            self.apf_side_lock_sign = 0.0
            self.apf_side_lock_active = False
            return 0.0

        requested_side = float(np.sign(requested_side))
        if (
            self.apf_side_lock_sign != 0.0
            and (obstacle_close or now_s < self.apf_side_lock_until_s)
        ):
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
        obstacle_close = (
            np.isfinite(nearest_forward_level)
            and nearest_forward_level <= self.apf_side_lock_exit_level
        )

        if obstacle_close or now_s < self.apf_side_lock_until_s:
            self.apf_side_lock_active = True
            return True

        self.apf_side_lock_sign = 0.0
        self.apf_side_lock_active = False
        return False

    def route_progress_and_point(self, lookahead_m=0.0):
        current_ne = np.array([float(self.North), float(self.East)], dtype=float)

        if self.route_path_length_m < 1e-9:
            return 0.0, self.goal_ne.copy()

        along_m = float(np.dot(current_ne - self.start_ne, self.route_path_unit_ne))
        closest_along_m = float(np.clip(along_m, 0.0, self.route_path_length_m))
        target_along_m = float(np.clip(along_m + lookahead_m, 0.0, self.route_path_length_m))
        target_ne = self.start_ne + target_along_m * self.route_path_unit_ne
        return closest_along_m, target_ne

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

        if self.route_path_length_m < 1e-9:
            return True

        current_ne = np.array([float(self.North), float(self.East)], dtype=float)
        along_m = float(np.dot(current_ne - self.start_ne, self.route_path_unit_ne))
        return along_m >= self.route_path_length_m - self.final_approach_distance_m

    def apf_path_attraction_body(self):
        path_vec = self.goal_ne - self.start_ne
        path_len_sq = float(np.dot(path_vec, path_vec))
        if path_len_sq < 1e-9:
            return np.zeros(2, dtype=float)

        current_ne = np.array([float(self.North), float(self.East)], dtype=float)
        ratio = float(np.clip(np.dot(current_ne - self.start_ne, path_vec) / path_len_sq, 0.0, 1.0))
        closest_ne = self.start_ne + ratio * path_vec
        to_path_body = self.earth_point_to_body(closest_ne)
        distance_m = float(np.linalg.norm(to_path_body))

        if distance_m < self.apf_path_threshold_m or distance_m < 1e-6:
            return np.zeros(2, dtype=float)

        return self.apf_path_gain * to_path_body

    def apf_goal_attraction_body(self, target_body):
        target_body = np.asarray(target_body, dtype=float).reshape(2)
        distance_m = float(np.linalg.norm(target_body))
        if distance_m < 1e-6:
            return np.zeros(2, dtype=float)

        magnitude = self.apf_goal_gain * min(distance_m, self.apf_attraction_saturation_m)
        return magnitude * target_body / distance_m

    def apf_avoidance_needed(self):
        virtual_obstacles = self.update_apf_virtual_obstacles()
        for obstacle in self.lidar_obstacles + virtual_obstacles:
            obs_pos_body = np.asarray(obstacle.get("centre_body", [np.nan, np.nan]), dtype=float).reshape(2)
            if not np.isfinite(obs_pos_body).all():
                continue
            if (
                not bool(obstacle.get("virtual", False))
                and not self.apf_obstacle_in_forward_half_plane(obstacle)
            ):
                continue

            angle_rad = abs(wrap_angle(float(np.arctan2(obs_pos_body[1], obs_pos_body[0]))))
            if not bool(obstacle.get("virtual", False)) and angle_rad > self.apf_activation_front_half_angle_rad:
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
        avoidance_pc_scale = float(getattr(self, "apf_avoidance_pc_scale", 2.0))
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
                obstacle_encounter = obstacle.get("encounter_mode", self.apf_encounter_mode)
                clearance_offset_m = max(
                    clearance_offset_m,
                    self.apf_direction_clearance_m(obstacle, obstacle_encounter),
                )
                nearest_active_level = min(
                    nearest_active_level,
                    self.apf_obstacle_level_and_away(obstacle, avoidance_pc_scale, encounter=obstacle_encounter)[0],
                )

        if not any_repulsion:
            for obstacle in secondary_obstacles:
                repulsion, active = self.apf_repulsion_for_obstacle(obstacle, target_body, own_vel_body)[:2]
                repulsive_force += repulsion
                force_body += repulsion
                any_repulsion = any_repulsion or active
                if active:
                    obstacle_encounter = obstacle.get("encounter_mode", self.apf_encounter_mode)
                    clearance_offset_m = max(
                        clearance_offset_m,
                        self.apf_direction_clearance_m(obstacle, obstacle_encounter),
                    )
                    nearest_active_level = min(
                        nearest_active_level,
                        self.apf_obstacle_level_and_away(obstacle, avoidance_pc_scale, encounter=obstacle_encounter)[0],
                    )

        if any_repulsion and self.apf_side_lock_sign != 0.0:
            # side_sign uses the COLREG convention used throughout this
            # controller: +1 is port/left and -1 is starboard/right.
            route_normal_left_ne = np.array(
                [self.route_path_unit_ne[1], -self.route_path_unit_ne[0]],
                dtype=float,
            )
            offset_target_ne = (
                target_ne
                + self.apf_side_lock_sign
                * max(clearance_offset_m, float(getattr(self, "obstacle_min_pc2_m", 0.16)))
                * route_normal_left_ne
            )
            offset_target_body = self.earth_point_to_body(offset_target_ne)
            offset_distance = float(np.linalg.norm(offset_target_body))
            if offset_distance > 1e-6:
                offset_force = (
                    self.apf_clearance_gain
                    * min(offset_distance, self.apf_attraction_saturation_m)
                    * offset_target_body
                    / offset_distance
                )
                force_body += offset_force
                attractive_force += offset_force
                self.apf_target_ne = offset_target_ne.copy()

        if any_repulsion and np.isfinite(nearest_active_level):
            close_quarters_scale = float(
                np.clip(
                    nearest_active_level,
                    float(getattr(self, "apf_close_quarters_force_scale", 0.15)),
                    1.0,
                )
            )
            force_body = repulsive_force + close_quarters_scale * attractive_force

        self.apf_attractive_force_body = attractive_force
        force_norm = float(np.linalg.norm(force_body))
        if not np.isfinite(force_norm) or force_norm < 1e-6:
            force_body = np.array([1e-3, 0.0], dtype=float)
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

            lateral_mag = max(abs(float(steering_force[1])), 0.5 * abs(float(force_body[0])), 0.20)
            steering_force[1] = side_sign * lateral_mag
            steering_force[0] = max(0.25 * lateral_mag, 0.05)

        self.apf_steering_force_body = steering_force
        now_s = float(self.timefromstart) if self.timefromstart is not None else 0.0
        self.apf_visual_hold_until_s = now_s + self.apf_visual_hold_s
        force_angle = wrap_angle(float(np.arctan2(steering_force[1], steering_force[0])))
        force_angle = float(np.clip(force_angle, -self.apf_heading_step_limit_rad, self.apf_heading_step_limit_rad))
        if self.apf_encounter_mode == "head_on":
            force_angle = min(force_angle, -abs(np.deg2rad(8.0)))

        u_cmd = Vector(2)
        u_cmd[1, 0] = np.clip(-self.apf_heading_gain * force_angle / max(self.lastdt, 1e-3), -self.w_max, self.w_max)

        # Use constant-speed (fixed-step) descent during APF avoidance.  The
        # gradient magnitude no longer changes surge speed; only its direction
        # changes the commanded heading.
        surge_cmd = self.apf_encounter_speed_m_s(
            self.apf_encounter_mode,
            nearest_active_level,
            force_angle,
        )
        u_cmd[0, 0] = float(np.clip(surge_cmd, 0.0, self.v_max))

        if any_repulsion or self.apf_colreg_active or self.apf_side_lock_active:
            if self.apf_colreg_active:
                self.navigation_mode = "apf_colreg"
            else:
                self.navigation_mode = "apf_avoid"
        else:
            self.navigation_mode = "apf_track"

        return u_cmd

    def compute_route_tracking_control(self, t):
        current_ne = np.array([self.North, self.East], dtype=float)
        final_distance = float(np.linalg.norm(self.goal_ne - current_ne))

        if self.final_approach_active(final_distance):
            ref_ne = self.goal_ne.copy()
            to_goal_ne = ref_ne - current_ne
            if final_distance > 1e-6:
                desired_heading = float(np.arctan2(to_goal_ne[1], to_goal_ne[0]))
            else:
                desired_heading = self.route_heading_rad

            heading_error = wrap_angle(float(self.Yaw) - desired_heading)
            speed_fraction = float(np.clip(final_distance / max(self.final_slowdown_distance_m, 1e-3), 0.0, 1.0))
            if abs(heading_error) >= self.final_heading_slow_angle_rad:
                heading_speed_scale = 0.0
            else:
                heading_speed_scale = max(float(np.cos(heading_error)), 0.15)

            p_ref = Vector(3)
            p_ref[0, 0] = ref_ne[0]
            p_ref[1, 0] = ref_ne[1]
            p_ref[2, 0] = desired_heading

            u_ref = Vector(2)
            u_ref[0, 0] = self.route_tracking_speed_m_s * speed_fraction

            u_track = Vector(2)
            u_track[0, 0] = self.route_tracking_speed_m_s * speed_fraction * heading_speed_scale
            u_track[1, 0] = 1.4 * heading_error
            return p_ref, u_ref, u_track

        # Track the straight start-goal line by projecting the current position onto
        # the route and aiming at a short look-ahead point on that same line.
        _, ref_ne = self.route_progress_and_point(self.route_tracking_lookahead_m)

        # Convert route error into the robot body frame:
        # lateral error and heading error adjust yaw rate while surge stays steady.
        error_body = self.earth_vector_to_body(ref_ne - current_ne)
        heading_error = wrap_angle(float(self.Yaw) - self.route_heading_rad)

        p_ref = Vector(3)
        p_ref[0, 0] = ref_ne[0]
        p_ref[1, 0] = ref_ne[1]
        p_ref[2, 0] = self.route_heading_rad

        u_ref = Vector(2)
        u_ref[0, 0] = self.route_tracking_speed_m_s

        u_track = Vector(2)
        u_track[0, 0] = self.route_tracking_speed_m_s
        u_track[1, 0] = -0.8 * error_body[1] + 1.2 * heading_error
        return p_ref, u_ref, u_track

    def groundtruth_callback(self, msg):
        # generate fake aruco data at a set interval
        self.pseudo_aruco_counter += 1 

        t = time.time()
        pose = msg.pose
        n = pose.position.x              
        e = pose.position.y
        d = pose.position.z
        ox = pose.orientation.x
        oy = pose.orientation.y
        oz = pose.orientation.z
        ow = pose.orientation.w
        q = [ox,oy,oz,ow]                
        r = R.from_quat(q)  # note: [x, y, z, w] order
        roll, pitch, yaw = r.as_euler('xyz', degrees=True)  # radians                
        yaw = np.mod(yaw, 360.0)

        if self.pseudo_aruco_counter== 80: 
            self.pseudo_aruco_counter = 0
            self.sensed_pos_stamp_s = t
            self.sensed_pos_northings_m = n
            self.sensed_pos_eastings_m = e
            self.sensed_pos_yaw_rad = np.deg2rad(yaw)
            broadcast = True
        else:
            broadcast = False
                    
        # log groundtruth if running webots simulation
        with self.groundtruth_log.open('a') as f:
            f.write(f"{t},{t-self.starttime},{n},{e},{d},{roll},{pitch},{yaw},{broadcast}\n")
         
    def feedback_control(self, ds, ks = None, kn = None, kg = None):

        if ks == None: ks = 0.1
        if kn == None: kn = 0.1
        if kg == None: kg = 0.1        
        
        dv = ks*ds[0]
        dw = kn*ds[1]+kg*ds[2]
        
        du = Vector(2)
        
        du[0] = dv
        du[1] = dw
        
        return du
    def motion_model(self, state, control_input, dt):
       """
       EKF motion model:
       state x = [N, E, G, Ndot, Edot, Gdot]^T
       control_input = T = [T_R, T_L]^T (thruster forces in N)

       Returns:
           predicted_state (6x1 Vector)
           F               (6x6 Jacobian)
       """

       # Thruster forces in body frame from allocation matrix
       Fb = self.G @ control_input  # [Fx, Fy, tau_z]^T in body frame

       # Convert thrust from body to earth frame
       H_eb = HomogeneousTransformation(state[N:E+1], state[G])
       Fe = H_eb.H_R @ Fb  # [Fx_e, Fy_e, tau_z_e]^T

       # Dynamics in earth frame
       ve = self.robot.model(
           dynamics_translation_e,
           dynamics_rotation_e,
           Fe,
           state[DOTN:DOTG+1],
           dt,
       )  # ve = [Ndot, Edot, Gdot]^T

       # Body-frame velocity
       vb = Inverse(H_eb.H_R) @ ve

       # Twist used for kinematics
       u = Vector(2)
       u[0, 0] = vb[0, 0]  # surge speed v
       u[1, 0] = vb[2, 0]  # yaw rate w

       # Pose update
       p = rigid_body_kinematics(state[N:G+1], u, dt)
       p[2, 0] = p[2, 0] % (2 * np.pi)

       # Build predicted state vector
       predicted_state = Vector(6)
       predicted_state[N]    = p[0]
       predicted_state[E]    = p[1]
       predicted_state[G]    = p[2]
       predicted_state[DOTN] = ve[0]
       predicted_state[DOTE] = ve[1]
       predicted_state[DOTG] = ve[2]

       # Simple Jacobian: integrate velocity (good enough for EKF here)
       F = Identity(6)
       F[N, DOTN]   = dt
       F[E, DOTE]   = dt
       F[G, DOTG]   = dt

       return predicted_state, F

    def thruster_force_limits(self):
        max_rpm = self.prop_rate_limit_rad_s * 60.0 / (2.0 * np.pi)
        forward_force = float(rpm2N(max_rpm))
        reverse_force = float(rpm2N(-max_rpm))

        if not np.isfinite(forward_force) or forward_force <= 0.0:
            forward_force = 1.0

        if not np.isfinite(reverse_force) or reverse_force >= 0.0:
            reverse_force = -0.5 * forward_force

        return reverse_force, forward_force

    def allocate_propulsion_rates(self, v_cmd, w_cmd):
        v_cmd = float(v_cmd) if np.isfinite(v_cmd) else 0.0
        w_cmd = float(w_cmd) if np.isfinite(w_cmd) else 0.0
        v_cmd = max(v_cmd, 0.0)

        desired_force_x = self.robot.k_drag * v_cmd * abs(v_cmd)
        desired_tau_z = self.robot.B_66 * w_cmd
        reverse_force, forward_force = self.thruster_force_limits()

        yaw_arm = 0.5 * (float(self.G[2, 0]) - float(self.G[2, 1]))
        if abs(yaw_arm) < 1e-6:
            thrust = np.linalg.pinv(self.G) @ l2m([desired_force_x, 0.0, desired_tau_z])
            right_force = float(np.clip(thrust[0, 0], reverse_force, forward_force))
            left_force = float(np.clip(thrust[1, 0], reverse_force, forward_force))
        else:
            # Preserve yaw authority first. If the requested surge and yaw cannot
            # both fit within the prop limits, reduce surge instead of losing turn.
            desired_delta = desired_tau_z / yaw_arm
            max_delta = max(forward_force - reverse_force, 1e-6)
            delta = float(np.clip(desired_delta, -max_delta, max_delta))

            force_lower = max(
                0.0,
                2.0 * reverse_force - delta,
                2.0 * reverse_force + delta,
            )
            force_upper = min(
                2.0 * forward_force - delta,
                2.0 * forward_force + delta,
            )

            if force_upper < force_lower:
                force_x = max(0.0, min(desired_force_x, 2.0 * forward_force))
            else:
                force_x = float(np.clip(desired_force_x, force_lower, force_upper))

            right_force = 0.5 * (force_x + delta)
            left_force = 0.5 * (force_x - delta)
            right_force = float(np.clip(right_force, reverse_force, forward_force))
            left_force = float(np.clip(left_force, reverse_force, forward_force))

        rpm_R = -N2rpm(right_force)
        rpm_L = N2rpm(left_force)

        right_rate = float(np.clip(rpm_R * (2.0 * np.pi / 60.0), -self.prop_rate_limit_rad_s, self.prop_rate_limit_rad_s))
        left_rate = float(np.clip(rpm_L * (2.0 * np.pi / 60.0), -self.prop_rate_limit_rad_s, self.prop_rate_limit_rad_s))
        return right_rate, left_rate

    def empty_measurement(x):
        H = Matrix(5)
        return x, H
    
    ######## MAIN ROBOT LOOP ##################
    def loop(self):
        """This main loop is completed every 0.2 seconds.        
        Once initialised, it repeats until stopped.        
        It runs sequentially so consider how to structure your code.        
        You won't receive data from the IMU or ARUCO in every loop. 
        Don't make the loop rely on new data.
        """
        current_epoch_s = time.time()
        self.timefromstart = current_epoch_s - self.starttime

        ### RECEIVE SENSOR DATA ##############################
        self.sensed_pos_stamp_s = None
        self.sensed_pos_northings_m = None
        self.sensed_pos_eastings_m = None
        self.sensed_pos_yaw_rad = None

        sensed_pos = self.aruco_driver.read()
        if sensed_pos is not None:
            self.sensed_pos_stamp_s = sensed_pos[0]
            self.sensed_pos_northings_m = sensed_pos[1]
            self.sensed_pos_eastings_m = sensed_pos[2]
            self.sensed_pos_yaw_rad = sensed_pos[6]
            print(
                "Received position update from",
                current_epoch_s - self.sensed_pos_stamp_s,
                "seconds ago",
            )

        if self.initialise_pose and self.sensed_pos_northings_m is not None:
            self.mu[N] = self.sensed_pos_northings_m
            self.mu[E] = self.sensed_pos_eastings_m
            self.mu[G] = self.sensed_pos_yaw_rad
            self.mu[DOTN] = 0
            self.mu[DOTE] = 0
            self.mu[DOTG] = 0

            self.p_robot[0] = self.mu[N]
            self.p_robot[1] = self.mu[E]
            self.p_robot[2] = self.mu[G]
            self.v_robot[0] = self.mu[DOTN]
            self.v_robot[1] = self.mu[DOTE]
            self.v_robot[2] = self.mu[DOTG]
            self.integrated_yaw = self.sensed_pos_yaw_rad
            self.initialise_pose = False
            print("Initialised pose")

        imu_fresh = (
            self.sensed_imu_stamp_s is not None
            and current_epoch_s - self.sensed_imu_stamp_s < self.lastdt
        )

        if imu_fresh:
            if self.sensed_imu_prev_stamp_s is not None:
                dt_imu = self.sensed_imu_stamp_s - self.sensed_imu_prev_stamp_s
                if dt_imu <= 0 or dt_imu > 1.0:
                    dt_imu = self.lastdt
            else:
                dt_imu = self.lastdt

            print(
                "Received IMU update from",
                current_epoch_s - self.sensed_imu_stamp_s,
                "seconds ago",
            )

            self.sensed_imu_prev_stamp_s = self.sensed_imu_stamp_s
            self.sensed_yaw_rate = self.sensed_imu_yaw_rate_rad_s
            self.integrated_yaw += self.sensed_yaw_rate * dt_imu
            self.integrated_yaw %= 2 * np.pi

        if (
            self.sensed_bottom_depth_stamp_s is not None
            and current_epoch_s - self.sensed_bottom_depth_stamp_s < self.lastdt
        ):
            print(
                "Received Echosounder update from",
                current_epoch_s - self.sensed_bottom_depth_stamp_s,
                "seconds ago",
            )

        if self.sensed_imu_stamp_s is not None or self.OPERATING_MODE == 2:
            ### EKF PREDICT/UPDATE ##############################
            right_N = rpm2N(-self.right_rate * 60 / (2 * np.pi))
            left_N = rpm2N(self.left_rate * 60 / (2 * np.pi))
            u_thrusters = l2m([right_N, left_N])
            self.mu, self.Sigma = extended_kalman_filter_predict(
                self.mu,
                self.Sigma,
                u_thrusters,
                self.motion_model,
                self.Q,
                self.lastdt,
            )

            if self.sensed_pos_stamp_s is not None:
                z_pose = Vector(6)
                z_pose[N] = self.sensed_pos_northings_m
                z_pose[E] = self.sensed_pos_eastings_m
                z_pose[G] = self.sensed_pos_yaw_rad

                self.mu, self.Sigma = extended_kalman_filter_update(
                    self.mu,
                    self.Sigma,
                    z_pose,
                    h_pose_update,
                    self.R_pose,
                    wrap_index=G,   
                )

            if imu_fresh and self.sensed_yaw_rate is not None:
                z_rate = Vector(6)
                z_rate[DOTG] = self.sensed_yaw_rate

                self.mu, self.Sigma = extended_kalman_filter_update(
                    self.mu,
                    self.Sigma,
                    z_rate,
                    h_grate_update,
                    self.R_grate,
                )

            self.p_robot[0] = self.mu[N]
            self.p_robot[1] = self.mu[E]
            self.p_robot[2] = self.mu[G]
            self.v_robot[0] = self.mu[DOTN]
            self.v_robot[1] = self.mu[DOTE]
            self.v_robot[2] = self.mu[DOTG]
            self.Yaw = self.p_robot[2][0]
            self.North = self.p_robot[0][0]
            self.East = self.p_robot[1][0]

            ### ROUTE TRACKING + MODIFIED APF CONTROL #######
            t = self.timefromstart
            _, _, u_track = self.compute_route_tracking_control(t)
            final_distance = self.goal_distance_m()
            final_approach = self.final_approach_active(final_distance)

            if final_distance <= self.goal_tolerance_m:
                self.goal_reached = True
                if np.isnan(self.s.t_complete):
                    self.s.t_complete = self.timefromstart

            if self.goal_reached:
                self.u = Vector(2)
                self.navigation_mode = "arrived"
                self.reset_apf_diagnostics()
            elif self.apf_avoidance_needed():
                self.u = self.compute_apf_control(t, u_track)
            else:
                self.u = u_track
                self.navigation_mode = "track"
                self.reset_apf_diagnostics(clear_visual=not self.apf_visual_hold_active())

            if self.goal_reached:
                self.u[0, 0] = 0.0
                self.u[1, 0] = 0.0
            else:
                if not final_approach:
                    self.u[1, 0] = self.limit_heading_deviation_command(self.u[1, 0])
                self.u[1, 0] = np.clip(self.u[1, 0], -self.w_max, self.w_max)
                self.u[0, 0] = np.clip(self.u[0, 0], 0.0, self.v_max)
            self.prev_sensed = t
            self.U = self.u.T

            v = float(self.U[0][0])
            w = float(self.U[0][1])
            if self.goal_reached:
                self.right_rate = 0.0
                self.left_rate = 0.0
            else:
                self.right_rate, self.left_rate = self.allocate_propulsion_rates(v, w)

            control_msg = Vector3()
            control_msg.x = int(self.right_rate)
            control_msg.y = int(self.left_rate)
            control_msg.z = float(self.route_tracking_speed_m_s)

            self.control_pub.publish(control_msg)
            print(
                'Navigation mode:', self.navigation_mode,
                'encounter:', self.apf_encounter_mode,
                'side=', int(self.apf_avoidance_side_sign),
                'DCPA=', round(float(self.apf_colreg_dcpa_m), 2) if np.isfinite(self.apf_colreg_dcpa_m) else 'nan',
                'TCPA=', round(float(self.apf_colreg_tcpa_s), 2) if np.isfinite(self.apf_colreg_tcpa_s) else 'nan',
                'v=', round(float(v), 3),
                'w=', round(float(w), 3),
                'F_body=', np.round(self.apf_force_body, 3),
            )
            print('Prop rates: R=',self.right_rate,', L=',self.left_rate,'rad/s')
            
        nearest_obstacle_north = np.nan
        nearest_obstacle_east = np.nan
        nearest_obstacle_distance = np.nan
        if self.nearest_lidar_obstacle is not None:
            centre_ne = np.asarray(
                self.nearest_lidar_obstacle.get("centre_ne", [np.nan, np.nan]),
                dtype=float,
            ).reshape(2)
            nearest_obstacle_north = centre_ne[0]
            nearest_obstacle_east = centre_ne[1]
            nearest_obstacle_distance = float(
                self.nearest_lidar_obstacle.get(
                    "min_distance_m",
                    self.nearest_lidar_obstacle.get("distance_m", np.nan),
                )
            )

        ### LOG DATA ##############################
        with self.filename.open("a") as f:
            f.write(f"{current_epoch_s},{self.timefromstart},{self.right_rate},{self.left_rate},{self.lastdt},{self.Yaw},{self.North},{self.East},{self.sensed_yaw_rate},{self.integrated_yaw},{self.sensed_imu_stamp_s},{self.sensed_pos_northings_m},{self.sensed_pos_eastings_m},{self.sensed_pos_yaw_rad}, {self.sensed_pos_stamp_s}, {self.sensed_bottom_depth_stamp_s},{self.sensed_bottom_depth_m},{self.navigation_mode},{self.apf_encounter_mode},{self.apf_avoidance_side_sign},{self.apf_colreg_dcpa_m},{self.apf_colreg_tcpa_s},{self.apf_force_body[0]},{self.apf_force_body[1]},{nearest_obstacle_north},{nearest_obstacle_east},{nearest_obstacle_distance}\n")
        self.write_obstacle_snapshot()
        
        ### VISUALISE DATA ##############################
        if self.OPERATING_MODE != 0:
            reference_path = getattr(self.s, "P_arc", getattr(self.s, "P", None))
            mission_complete = not np.isnan(getattr(self.s, "t_complete", np.nan))
            return(self.right_rate, self.left_rate, self.lastdt, self.Yaw, self.North, self.East, self.sensed_yaw_rate,self.integrated_yaw, self.sensed_imu_stamp_s, self.sensed_pos_northings_m, self.sensed_pos_eastings_m, self.sensed_pos_yaw_rad, self.sensed_pos_stamp_s, self.waypoints, reference_path, self.sensed_bottom_depth_m, self.sensed_bottom_depth_stamp_s, mission_complete, self.lidar_data)



        ############################# END MAIN LOOP ###########################
        
def main():
    LaptopController(OPERATING_MODE=0)
    
if __name__ == "__main__":
    main()
            
