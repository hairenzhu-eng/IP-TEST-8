"""
Copyright (c) 2023 The uos_sess6072_build Authors.
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import json
import time
import sys
from pathlib import Path
from threading import Thread
import argparse
import numpy as np
from drivers.rpi import Rate

if not hasattr(np, "VisibleDeprecationWarning"):
    np.VisibleDeprecationWarning = DeprecationWarning

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QApplication, QGridLayout
from pglive.sources.data_connector import DataConnector
from pglive.sources.live_plot import LiveLinePlot
from pglive.sources.live_plot import LiveScatterPlot
from pglive.sources.live_plot_widget import LivePlotWidget
import laptop as lt

PLOT_RATE_HZ = 5.0
AUTO_STOP_STATIONARY_SAMPLES = 10


def plot_connector(plot, max_points, auto_range=False):
    return DataConnector(
        plot,
        max_points=max_points,
        plot_rate=PLOT_RATE_HZ,
        ignore_auto_range=not auto_range,
    )


def connector_append_points(connector, x_values, y_values):
    append_many = getattr(connector, "cb_append_data_array", None)
    if callable(append_many):
        append_many(x_values, y_values)
        return

    for x_value, y_value in zip(x_values, y_values):
        connector.cb_append_data_point(x_value, y_value)


class ShowLaptop(QWidget):
    obstacle_summary_signal = pyqtSignal(str)
    shutdown_signal = pyqtSignal(str)
    running = False

    def __init__(self, parent=None):

        rate = 10.0
        self.r = Rate(rate)
        self.lastdt = 1/rate

        super().__init__(parent)
        self.rpmplot = LivePlotWidget()
        self.headingplot = LivePlotWidget()
        self.positionplot = LivePlotWidget()
        self.timeplot = LivePlotWidget()
        self.depthplot = LivePlotWidget()
        layout = QGridLayout(self)
        layout.addWidget(self.rpmplot, 0, 3, 1, 2)
        layout.addWidget(self.headingplot, 1, 3, 1, 2)
        layout.addWidget(self.positionplot, 0, 0, 2, 2)
        layout.addWidget(self.timeplot, 2, 0, 1, 2)
        layout.addWidget(self.depthplot, 2, 3, 1, 2)
        self.loopcounter = 0
        self.Laptop = lt.LaptopController(OPERATING_MODE)
        self.obstacle_summary_signal.connect(self._set_obstacle_summary_title)
        self.shutdown_signal.connect(self._shutdown_application)
        
        # Create one curve pre dataset
        thruster1plot = LiveLinePlot(pen="blue", name = 'Thruster 1')
        thruster2plot = LiveLinePlot(pen="red", name = 'Thruster 2')
        
        EKFheadingplot = LiveLinePlot(pen = 'blue', name = 'Model Heading')
        sensedheadingplot = LiveLinePlot(pen='red', name = 'IMU Integrated Heading')
        ARUCOheadingplot = LiveLinePlot(symbol = 'x', pen = 'green', name = 'ARUCO Sensed Heading')       
        
        positionplot = LiveLinePlot(pen = 'blue', name = 'Model Path')
        ARUCOplot = LiveScatterPlot(symbol = 'x', pen = 'green', name = 'ARUCO Sensed Position')
        WayPoint = LiveScatterPlot(symbol = 'o', pen = 'red', name = 'Waypoints')
        lidarplot = LiveScatterPlot(
            symbol='o',
            size=1,
            pen='w',
            name='LiDAR point cloud (earth frame)',
        )
        plannedPathPlot = LiveLinePlot(pen = 'gray', name = 'Planned Path')
        apfWaypointPathPlot = LiveLinePlot(pen = 'cyan', name = 'APF Avoidance Path')
        apfWaypointPlot = LiveScatterPlot(symbol = 'o', size = 9, pen = 'cyan', name = 'APF Avoidance Waypoints')
        apfForcePlot = LiveLinePlot(pen = 'yellow', name = 'APF Total Force')
        apfRepulsivePlot = LiveLinePlot(pen = 'magenta', name = 'APF Repulsive Force')
        apfSteeringPlot = LiveLinePlot(pen = 'cyan', name = 'APF Steering Force')
        apfTargetPlot = LiveScatterPlot(symbol = 't', size = 10, pen = 'orange', name = 'APF Target')
        obstacleEkfPositionPlot = LiveScatterPlot(symbol = 'o', size = 8, pen = 'orange', name = 'Obstacle EKF Position')
        obstacleVirtualPositionPlot = LiveScatterPlot(symbol = 'x', size = 12, pen = 'magenta', name = 'Obstacle Predicted Position')
        virtualCollisionPositionPlot = LiveScatterPlot(symbol = 'd', size = 14, pen = 'yellow', name = 'Virtual Collision Position')
        obstacleEkfHistoryPlot = LiveLinePlot(pen = 'green', name = 'LiDAR Cluster Centre Track (e-frame)')
        obstacleEkfPredictionPlot = LiveLinePlot(pen = 'orange', name = 'Obstacle EKF Prediction (e-frame)')
        obstacleEkfDirectionPlot = LiveLinePlot(pen = 'red', name = 'Obstacle Velocity (e-frame)')
        
        dtplot = LiveLinePlot(pen='blue', name = 'Laptop Update')
        Idtplot = LiveScatterPlot(symbol = 'x', pen = 'red', name = 'IMU Update')
        Adtplot = LiveScatterPlot(symbol = 'x', pen = 'green', name = 'ARUCO Update')

        depthplot = LiveLinePlot(pen = 'red', name = 'Sensed Depth')

        # Data connectors for each plot with dequeue of 600 points
        self.thruster1plot = plot_connector(thruster1plot, 1500, auto_range=True)
        self.thruster2plot = plot_connector(thruster2plot, 1500)
        
        self.DHP = plot_connector(ARUCOheadingplot, 1500)
        self.SHP = plot_connector(sensedheadingplot, 1500)
        self.EHP = plot_connector(EKFheadingplot, 1500, auto_range=True)
        
        self.pos = plot_connector(positionplot, 1500)
        self.ASP = plot_connector(ARUCOplot, 1500)
        self.WP = plot_connector(WayPoint, 1500)
        self.lidar = plot_connector(lidarplot, 3000)
        self.planned_path = plot_connector(plannedPathPlot, 20, auto_range=True)
        self.apf_waypoint_path = plot_connector(apfWaypointPathPlot, 20)
        self.apf_waypoints = plot_connector(apfWaypointPlot, 20)
        self.apf_force = plot_connector(apfForcePlot, 2)
        self.apf_repulsive = plot_connector(apfRepulsivePlot, 2)
        self.apf_steering = plot_connector(apfSteeringPlot, 2)
        self.apf_target = plot_connector(apfTargetPlot, 1)
        self.obstacle_ekf_position = plot_connector(obstacleEkfPositionPlot, 50)
        self.obstacle_virtual_position = plot_connector(obstacleVirtualPositionPlot, 50)
        self.virtual_collision_position = plot_connector(virtualCollisionPositionPlot, 50)
        self.obstacle_ekf_history = plot_connector(obstacleEkfHistoryPlot, 3000)
        self.obstacle_ekf_prediction = plot_connector(obstacleEkfPredictionPlot, 1000)
        self.obstacle_ekf_direction = plot_connector(obstacleEkfDirectionPlot, 200)
        
        self.dtplot = plot_connector(dtplot, 1500, auto_range=True)
        self.Idtplot = plot_connector(Idtplot, 1500)
        self.Adtplot = plot_connector(Adtplot, 1500)

        self.deplot = plot_connector(depthplot, 1500, auto_range=True)

        # Create plot itself
        #self.rpmplot = LivePlotWidget(title="Line Plot - Time series @ 2Hz", axisItems={'bottom': bottom_axis})
        # Show grid
        self.rpmplot.showGrid(x=True, y=True, alpha=0.3)
        self.headingplot.showGrid(x=True, y=True, alpha=0.3)
        self.positionplot.setAspectLocked()
        self.positionplot.showGrid(x = True, y = True, alpha = 0.3)
        self.timeplot.showGrid(x = True, y = True, alpha = 0.3)
        self.depthplot.showGrid(x = True, y = True, alpha = 0.3)

        # Set labels
        self.rpmplot.setLabel('bottom', 'Time', units="s")
        self.rpmplot.setLabel('left', 'Thruster RPM')
        self.rpmplot.addLegend()
        self.headingplot.setLabel('bottom', 'Time', units="s")
        self.headingplot.setLabel('left', 'Heading', units="degrees")
        self.headingplot.addLegend()
        self.positionplot.setLabel('bottom', 'East', units="m")
        self.positionplot.setLabel('left', 'North', units="m")
        self.positionplot.addLegend()
        self.timeplot.setLabel('bottom', 'Time', units="s")
        self.timeplot.setLabel('left', 'Time from last update', units="s")
        self.timeplot.addLegend()
        self.depthplot.setLabel('bottom', 'Time', units="s")
        self.depthplot.setLabel('left', 'Depth of bottom', units="m")
        self.depthplot.addLegend()
        # Add all three curves
        self.rpmplot.addItem(thruster1plot)
        self.rpmplot.addItem(thruster2plot)
        self.headingplot.addItem(ARUCOheadingplot)
        self.headingplot.addItem(sensedheadingplot)
        self.headingplot.addItem(EKFheadingplot)
        self.positionplot.addItem(positionplot)
        self.positionplot.addItem(ARUCOplot)
        self.positionplot.addItem(WayPoint)
        self.positionplot.addItem(plannedPathPlot)
        self.positionplot.addItem(apfWaypointPathPlot)
        self.positionplot.addItem(apfWaypointPlot)
        self.positionplot.addItem(lidarplot)
        self.positionplot.addItem(apfForcePlot)
        self.positionplot.addItem(apfRepulsivePlot)
        self.positionplot.addItem(apfSteeringPlot)
        self.positionplot.addItem(apfTargetPlot)
        self.positionplot.addItem(obstacleEkfPositionPlot)
        self.positionplot.addItem(obstacleVirtualPositionPlot)
        self.positionplot.addItem(virtualCollisionPositionPlot)
        self.positionplot.addItem(obstacleEkfHistoryPlot)
        self.positionplot.addItem(obstacleEkfPredictionPlot)
        self.positionplot.addItem(obstacleEkfDirectionPlot)
        self.timeplot.addItem(dtplot)
        self.timeplot.addItem(Idtplot)
        self.timeplot.addItem(Adtplot)
        self.depthplot.addItem(depthplot)

        # using -1 to span through all rows available in the window
        #layout.addWidget(self.rpmplot, 2, 0, -1, 3)
        
        self.ST = time.time()
        self.lastimu = 0
        self.lastARUCO = 0
        self._lidar_timestamp_s_prev = None
        self._map_x = []
        self._map_y = []
        self._map_x_store = []
        self._map_y_store = []
        self._stationary_samples = 0
        self._shutdown_requested = False

    def _set_obstacle_summary_title(self, summary):
        self.positionplot.setTitle(summary)

    def _write_run_summary(self, reason, mission_complete, left_rate, right_rate):
        run_dir = getattr(self.Laptop, "run_dir", None)
        if run_dir is None:
            return

        payload = {
            "status": "completed",
            "stop_reason": reason,
            "mission_complete": bool(mission_complete),
            "goal_reached": bool(getattr(self.Laptop, "goal_reached", False)),
            "collision_detected": bool(getattr(self.Laptop, "webots_collision_detected", False)),
            "navigation_mode": getattr(self.Laptop, "navigation_mode", "unknown"),
            "switch_combination": getattr(lt, "SWITCH_COMBINATION", "unknown"),
            "webots_environment": getattr(self.Laptop, "webots_environment", "unknown"),
            "elapsed_s": round(float(self.TFS), 3),
            "north_m": None if self.Laptop.North is None else round(float(self.Laptop.North), 4),
            "east_m": None if self.Laptop.East is None else round(float(self.Laptop.East), 4),
            "goal_distance_m": round(float(self.Laptop.goal_distance_m()), 4),
            "left_rate_rad_s": None if left_rate is None else round(float(left_rate), 6),
            "right_rate_rad_s": None if right_rate is None else round(float(right_rate), 6),
        }
        try:
            summary_path = Path(run_dir) / "run_summary.json"
            summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            return

    def _request_shutdown(self, reason, mission_complete, left_rate, right_rate):
        if self._shutdown_requested:
            return

        self._shutdown_requested = True
        self._write_run_summary(reason, mission_complete, left_rate, right_rate)
        print(
            f"SIMULATION_STOP reason={reason} mission_complete={bool(mission_complete)} "
            f"collision={bool(getattr(self.Laptop, 'webots_collision_detected', False))} "
            f"elapsed_s={self.TFS:.3f}",
            flush=True,
        )
        self.shutdown_signal.emit(reason)

    def _shutdown_application(self, _reason):
        self.running = False
        try:
            self.Laptop.stopcommand()
        finally:
            app = QApplication.instance()
            if app is not None:
                app.quit()

    def _check_auto_shutdown(self, mission_complete, left_rate, right_rate):
        if bool(getattr(self.Laptop, "webots_collision_detected", False)):
            self._request_shutdown("collision_detected", mission_complete, left_rate, right_rate)
            return

        stationary = (
            left_rate is not None
            and right_rate is not None
            and abs(float(left_rate)) <= 1e-3
            and abs(float(right_rate)) <= 1e-3
        )
        if bool(mission_complete) and stationary:
            self._stationary_samples += 1
        else:
            self._stationary_samples = 0

        if self._stationary_samples >= AUTO_STOP_STATIONARY_SAMPLES:
            self._request_shutdown("goal_reached_stopped", mission_complete, left_rate, right_rate)
        
        
    def _force_line_ne(self, force_body, scale=0.8):
        if force_body is None or self.Laptop.North is None or self.Laptop.East is None:
            return None

        force_body = np.asarray(force_body, dtype=float).reshape(2)
        if not np.isfinite(force_body).all():
            return None

        force_norm = float(np.linalg.norm(force_body))
        if force_norm < 1e-6:
            return None

        origin_ne = np.array([float(self.Laptop.North), float(self.Laptop.East)], dtype=float)
        force_ne = self.Laptop.body_vector_to_earth(force_body)
        force_ne_norm = float(np.linalg.norm(force_ne))
        if force_ne_norm < 1e-6 or not np.isfinite(force_ne_norm):
            return None

        line_len_m = min(1.2, max(0.25, scale * force_norm))
        end_ne = origin_ne + force_ne / force_ne_norm * line_len_m
        return [origin_ne[0], end_ne[0]], [origin_ne[1], end_ne[1]]

    def _set_force_line(self, connector, force_body, scale=0.8):
        line = self._force_line_ne(force_body, scale)
        if line is None:
            connector.cb_set_data([], [])
            return

        northings, eastings = line
        connector.cb_set_data(northings, eastings)

    def _update_apf_plot(self):
        self._set_force_line(self.apf_force, getattr(self.Laptop, "apf_force_body", None), scale=0.8)
        self._set_force_line(self.apf_repulsive, getattr(self.Laptop, "apf_repulsive_force_body", None), scale=0.8)
        self._set_force_line(self.apf_steering, getattr(self.Laptop, "apf_steering_force_body", None), scale=0.8)

        target_ne = np.asarray(
            getattr(self.Laptop, "apf_target_ne", [np.nan, np.nan]),
            dtype=float,
        ).reshape(2)
        if np.isfinite(target_ne).all():
            self.apf_target.cb_set_data([target_ne[0]], [target_ne[1]])
        else:
            self.apf_target.cb_set_data([], [])

    def _update_apf_waypoints_plot(self):
        points_ne = np.asarray(
            getattr(self.Laptop, "apf_waypoint_path_ne", []),
            dtype=float,
        )
        if points_ne.ndim != 2 or points_ne.shape[1] < 2:
            self.apf_waypoint_path.cb_set_data([], [])
            self.apf_waypoints.cb_set_data([], [])
            return
        points_ne = points_ne[np.isfinite(points_ne[:, 0]) & np.isfinite(points_ne[:, 1])]
        self.apf_waypoint_path.cb_set_data(points_ne[:, 0], points_ne[:, 1])
        self.apf_waypoints.cb_set_data(points_ne[:, 0], points_ne[:, 1])

    def _append_track_segments(self, northings, eastings, points_ne):
        points_ne = np.asarray(points_ne, dtype=float)
        if points_ne.ndim != 2 or points_ne.shape[0] < 2 or points_ne.shape[1] < 2:
            return

        finite = np.isfinite(points_ne[:, 0]) & np.isfinite(points_ne[:, 1])
        points_ne = points_ne[finite]
        if len(points_ne) < 2:
            return

        northings.extend(points_ne[:, 0].tolist())
        eastings.extend(points_ne[:, 1].tolist())
        northings.append(np.nan)
        eastings.append(np.nan)

    def _update_obstacle_tracks_plot(self):
        tracks = self.Laptop.obstacle_track_visuals()
        position_northings = []
        position_eastings = []
        virtual_northings = []
        virtual_eastings = []
        history_northings = []
        history_eastings = []
        prediction_northings = []
        prediction_eastings = []
        direction_northings = []
        direction_eastings = []
        summaries = []

        for track in tracks:
            position_ne = np.asarray(track.get("position_ne", [np.nan, np.nan]), dtype=float).reshape(2)
            velocity_ne = np.asarray(track.get("velocity_ne", [0.0, 0.0]), dtype=float).reshape(2)

            if not np.isfinite(position_ne).all():
                continue

            position_northings.append(position_ne[0])
            position_eastings.append(position_ne[1])
            track_id = int(track.get("id", 0))
            virtual_position_ne = np.asarray(
                track.get("virtual_position_ne", position_ne),
                dtype=float,
            ).reshape(2)
            if np.isfinite(virtual_position_ne).all():
                virtual_northings.append(virtual_position_ne[0])
                virtual_eastings.append(virtual_position_ne[1])
            self._append_track_segments(
                history_northings,
                history_eastings,
                track.get("lidar_history_ne", []),
            )
            self._append_track_segments(prediction_northings, prediction_eastings, track.get("prediction_ne", []))

            heading_deg = float(track.get("heading_deg", np.nan))
            speed_m_s = float(track.get("speed_m_s", np.linalg.norm(velocity_ne)))
            pc1_m = float(track.get("pc1_m", np.nan))
            pc2_m = float(track.get("pc2_m", np.nan))
            if np.isfinite(heading_deg):
                summaries.append(
                    f"#{track_id} {heading_deg:.0f}deg {speed_m_s:.2f}m/s "
                    f"pc1={pc1_m:.2f} pc2={pc2_m:.2f}"
                )

            if np.isfinite(speed_m_s) and speed_m_s >= 1e-3 and np.isfinite(velocity_ne).all():
                # velocity_ne is already in the e-frame; vector length encodes speed.
                vector_len_m = float(np.clip(2.0 * speed_m_s, 0.15, 1.0))
                direction_ne = velocity_ne / max(float(np.linalg.norm(velocity_ne)), 1e-6)
                end_ne = position_ne + direction_ne * vector_len_m
                direction_northings.extend([position_ne[0], end_ne[0], np.nan])
                direction_eastings.extend([position_ne[1], end_ne[1], np.nan])

        self.obstacle_ekf_position.cb_set_data(position_northings, position_eastings)
        self.obstacle_virtual_position.cb_set_data(virtual_northings, virtual_eastings)
        self.obstacle_ekf_history.cb_set_data(history_northings, history_eastings)
        self.obstacle_ekf_prediction.cb_set_data(prediction_northings, prediction_eastings)
        self.obstacle_ekf_direction.cb_set_data(direction_northings, direction_eastings)
        collision_text = (
            "Webots ShipObstacle collision: YES"
            if getattr(self.Laptop, "webots_collision_detected", False)
            else "Webots ShipObstacle collision: NO"
        )
        track_text = " | Obstacle EKF: " + " | ".join(summaries[:3]) if summaries else ""
        self.obstacle_summary_signal.emit(collision_text + track_text)

    def _update_virtual_collision_position_plot(self):
        northings = []
        eastings = []
        for collision in self.Laptop.virtual_collision_visuals():
            collision_position_ne = np.asarray(
                collision.get("collision_position_ne", [np.nan, np.nan]),
                dtype=float,
            ).reshape(2)
            if not np.isfinite(collision_position_ne).all():
                continue
            northings.append(collision_position_ne[0])
            eastings.append(collision_position_ne[1])

        self.virtual_collision_position.cb_set_data(northings, eastings)

    def _update_lidar_plot(self, lidar_cloud_ne):
        lidar_timestamp_s = self.Laptop.lidar_timestamp_s
        if lidar_cloud_ne is None or lidar_timestamp_s == self._lidar_timestamp_s_prev:
            return

        valid_northings = []
        valid_eastings = []
        for point in lidar_cloud_ne:
            if len(point) < 2:
                continue
            if not np.isnan(point[0]) and not np.isnan(point[1]):
                self._map_x_store.append(point[0])
                self._map_y_store.append(point[1])
                self._map_x.append(point[0])
                self._map_y.append(point[1])
                valid_northings.append(point[0])
                valid_eastings.append(point[1])

        self._map_x_store = self._map_x_store[-3000:]
        self._map_y_store = self._map_y_store[-3000:]
        if valid_northings:
            connector_append_points(self.lidar, valid_northings, valid_eastings)

        if len(self._map_x) > 1500 and len(self._map_x_store) >= 1000:
            ind = np.random.choice(len(self._map_x_store), 1000, replace=False)
            self.lidar.cb_set_data(
                [self._map_x_store[i] for i in ind],
                [self._map_y_store[i] for i in ind],
            )
            self._map_x = []
            self._map_y = []

        self._lidar_timestamp_s_prev = lidar_timestamp_s


    def update(self):
        """Run control and visualization updates at the configured rate."""
        while self.running:

            (
                right_rate,
                left_rate,
                lastdt,
                current_heading,
                North,
                East,
                _sensed_yaw_rate,
                sensed_yaw,
                imu_time1,
                sensed_pos_northings_m,
                sensed_pos_eastings_m,
                sensed_pos_yaw_rad,
                ARUCO_time1,
                _waypoints,
                _reference_path,
                depth,
                depth_time1,
                mission_complete,
                lidar_cloud_ne,
            ) = self.Laptop.loop()
            if self.loopcounter == 0:
                for waypoint in self.Laptop.waypoints:
                    self.WP.cb_append_data_point(waypoint.y, waypoint.x)
                self.planned_path.cb_set_data(
                    [waypoint.y for waypoint in self.Laptop.waypoints],
                    [waypoint.x for waypoint in self.Laptop.waypoints],
                )
            self.loopcounter = self.loopcounter + 1            
            self.TFS = time.time() - self.ST
            
            if left_rate != None and right_rate != None:
                self.thruster1plot.cb_append_data_point(left_rate*60/(2*np.pi), self.TFS)
                self.thruster2plot.cb_append_data_point(right_rate*60/(2*np.pi), self.TFS)
            
            if imu_time1 != None:
                imu_time = imu_time1 - self.ST
            else:
                imu_time = None
                
            if ARUCO_time1 != None:
                ARUCO_time = ARUCO_time1 - self.ST
            else:
                ARUCO_time = None

            if depth_time1 != None:
                depth_time = depth_time1 - self.ST
            else:
                depth_time = None                
            
            if sensed_yaw != None and imu_time >= self.TFS - lastdt:
                self.SHP.cb_append_data_point(np.rad2deg(sensed_yaw), self.TFS)
            if sensed_pos_yaw_rad != None:
                self.DHP.cb_append_data_point(np.rad2deg(sensed_pos_yaw_rad), self.TFS)
            if current_heading != None:
                self.EHP.cb_append_data_point(np.rad2deg(current_heading), self.TFS)
            
            if East != None and North != None:
                self.pos.cb_append_data_point(North,East)
            if sensed_pos_northings_m != None and sensed_pos_eastings_m != None:
                self.ASP.cb_append_data_point(sensed_pos_northings_m, sensed_pos_eastings_m)
            self._update_lidar_plot(lidar_cloud_ne)
            self._update_apf_plot()
            self._update_apf_waypoints_plot()
            self._update_obstacle_tracks_plot()
            self._update_virtual_collision_position_plot()
            
            if lastdt != None:
                self.dtplot.cb_append_data_point(lastdt, self.TFS)
                
            if imu_time != None and imu_time >= self.TFS - lastdt:
                self.Idtplot.cb_append_data_point(imu_time - self.lastimu, imu_time)
            if ARUCO_time != None:
                self.Adtplot.cb_append_data_point(ARUCO_time - self.lastARUCO, ARUCO_time)  

            if depth_time != None and depth_time >= self.TFS - lastdt:
                self.deplot.cb_append_data_point(depth, self.TFS)
                
            if imu_time != None and imu_time >= self.TFS - lastdt:
                self.lastimu = imu_time
            if ARUCO_time != None:
                self.lastARUCO = ARUCO_time

            self._check_auto_shutdown(mission_complete, left_rate, right_rate)
            self.r.sleep()
            
    def breaker(self):
        self.Laptop.stopcommand()
        

    def start_app(self):
        """Start Thread generator"""
        self.running = True
        Thread(target=self.update, daemon=True).start()
        
if __name__ == '__main__':    

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Run in simulation mode. Defaults to False",
    )

    args = parser.parse_args()

    if args.simulation == True: 
        OPERATING_MODE = 2
        print('Running laptop.py in simulation')
    else: 
        OPERATING_MODE = 1
        print('Running laptop.py on robot')
    print(
        "Obstacle EKF tracking/CPA:",
        "enabled" if lt.ENABLE_OBSTACLE_EKF_PREDICTION else "disabled",
    )
    print(
        "Cluster-based obstacle size:",
        "enabled" if lt.ENABLE_CLUSTER_BASED_APF_RANGE else "disabled",
    )



    app = QApplication(sys.argv)
    window = ShowLaptop()
    window.show()
    window.start_app()
    app.exec()
    window.running = False
    window.breaker()
