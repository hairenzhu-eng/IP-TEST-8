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
        apfTargetPlot = LiveScatterPlot(symbol = 't', size = 10, pen = 'orange', name = 'APF Target')
        
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
        self.apf_target = plot_connector(apfTargetPlot, 1)
        
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
        self.positionplot.addItem(apfTargetPlot)
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
        
        
    def _update_apf_plot(self):
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
