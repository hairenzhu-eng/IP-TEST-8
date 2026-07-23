import numpy as np

from colreg_apf import classify_colreg_zone, constrain_velocity_to_axis, merge_collinear_cluster_labels, smooth_undirected_axis, straight_line_cpa


def test_straight_line_cpa_lies_on_both_predictions():
    tcpa_s, dcpa_m, own_cpa, obstacle_cpa = straight_line_cpa(
        [0.0, 0.0], [1.0, 0.0], [5.0, -5.0], [0.0, 1.0], horizon_s=10.0
    )
    assert np.isclose(tcpa_s, 5.0)
    assert np.isclose(dcpa_m, 0.0)
    assert np.allclose(own_cpa, [5.0, 0.0])
    assert np.allclose(obstacle_cpa, [5.0, 0.0])


def test_cpa_horizon_clamps_both_straight_trajectories():
    tcpa_s, dcpa_m, own_cpa, obstacle_cpa = straight_line_cpa(
        [0.0, 0.0], [1.0, 0.0], [5.0, -5.0], [0.0, 1.0], horizon_s=2.0
    )
    assert np.isclose(tcpa_s, 2.0)
    assert np.isclose(dcpa_m, np.sqrt(18.0))
    assert np.allclose(own_cpa, [2.0, 0.0])
    assert np.allclose(obstacle_cpa, [5.0, -3.0])


def test_unified_controller_uses_four_dimensional_ekf_state():
    from laptop import LaptopController, _crossing

    predict = LaptopController.obstacle_track_state_at
    position, velocity = predict(object(), {"state": [1.0, 2.0, 3.0, 4.0]}, 2.0)
    assert np.allclose(position, [7.0, 10.0])
    assert np.allclose(velocity, [3.0, 4.0])

    controller = object.__new__(_crossing.LaptopController)
    controller.obstacle_ekf_accel_std_m_s2 = 0.1
    predicted, covariance = controller.obstacle_ekf_predict(
        [1.0, 2.0, 3.0, 4.0], np.eye(4), 2.0
    )
    assert np.allclose(predicted, [7.0, 10.0, 3.0, 4.0])
    assert covariance.shape == (4, 4)


def test_unified_controller_keeps_measured_motion_when_prediction_disabled():
    from laptop import LaptopController

    controller = object.__new__(LaptopController)
    controller.obstacle_ekf_prediction_enabled = False
    controller.latest_lidar_received_s = 10.0
    controller.apf_track_timeout_s = 1.5
    controller.apf_track_association_m = 0.8
    controller.lidar_dbscan_eps_m = 0.25
    controller.apf_obstacle_tracks = [
        {"pos_ne": np.array([4.0, 1.0]), "last_seen_s": 10.0, "stamp_s": 10.0}
    ]

    assert controller.apf_track_for_obstacle({"centre_ne": [4.1, 1.0]}) is controller.apf_obstacle_tracks[0]
    tcpa_s, dcpa_m = controller.apf_cpa_metrics([4.0, -2.0], [0.0, 1.0], [1.0, 0.0])
    assert np.isclose(tcpa_s, 3.0)
    assert np.isclose(dcpa_m, np.sqrt(2.0))


def test_unified_crossing_side_matches_north_east_coordinates():
    from laptop import LaptopController

    controller = object.__new__(LaptopController)
    controller.route_path_unit_ne = np.array([1.0, 0.0])
    controller.apf_dynamic_speed_threshold_m_s = 0.05
    assert np.allclose(controller._route_normal_left_ne(), [0.0, -1.0])
    assert controller.apf_pass_astern_side_from_velocity([0.0, 1.0]) == 1.0


def test_colreg_uses_obstacle_robot_relative_sector_before_course():
    assert classify_colreg_zone(0.0, 180.0)[0:2] == ("head_on", -1.0)
    assert classify_colreg_zone(67.0, 202.0)[0] == "head_on"
    assert classify_colreg_zone(68.0, 180.0)[0] != "head_on"
    assert classify_colreg_zone(0.0, 157.0)[0] != "head_on"
    assert classify_colreg_zone(90.0, 0.0, own_speed_m_s=1.0, obstacle_speed_m_s=0.2)[0] == "crossing_from_starboard"
    assert classify_colreg_zone(0.0, 0.0, own_speed_m_s=1.0, obstacle_speed_m_s=0.2, own_bearing_from_obstacle_deg=180.0)[0:2] == ("overtaking", -1.0)
    assert classify_colreg_zone(0.0, 0.0, own_speed_m_s=1.0, obstacle_speed_m_s=0.2, own_bearing_from_obstacle_deg=90.0)[0] == "static_obstacle"
    assert classify_colreg_zone(180.0, 0.0, own_speed_m_s=0.2, obstacle_speed_m_s=1.0)[0] == "being_overtaken"


def test_overtaking_requires_a_stable_same_course_stern_approach():
    kwargs = {
        "own_speed_m_s": 0.5,
        "obstacle_speed_m_s": 0.2,
        "own_bearing_from_obstacle_deg": 180.0,
    }
    assert classify_colreg_zone(0.0, 0.0, **kwargs)[0] == "overtaking"
    # A crossing-like bearing, course, or lateral velocity must not activate
    # Rule 13 even if the own vessel is faster.
    assert classify_colreg_zone(35.0, 0.0, **kwargs)[0] != "overtaking"
    assert classify_colreg_zone(0.0, 35.0, **kwargs)[0] != "overtaking"
    assert classify_colreg_zone(0.0, 0.0, own_speed_m_s=0.5, obstacle_speed_m_s=0.2,
                                own_bearing_from_obstacle_deg=145.0)[0] != "overtaking"
    assert classify_colreg_zone(
        0.0, 25.0, own_speed_m_s=0.5, obstacle_speed_m_s=0.3,
        own_bearing_from_obstacle_deg=180.0,
    )[0] != "overtaking"


def test_colreg_uses_only_a_stable_ekf_direction_prediction():
    from laptop import LaptopController

    controller = object.__new__(LaptopController)
    controller.obstacle_ekf_prediction_enabled = True
    controller.apf_colreg_prediction_horizon_s = 1.0
    controller.apf_colreg_max_heading_std_rad = np.deg2rad(30.0)
    controller.apf_dynamic_speed_threshold_m_s = 0.05
    controller._current_ne = lambda: np.array([0.0, 0.0])
    controller.body_vector_to_earth = lambda vector: np.asarray(vector, dtype=float)
    controller.earth_point_to_body = lambda point: np.asarray(point, dtype=float)
    controller.earth_vector_to_body = lambda vector: np.asarray(vector, dtype=float)
    controller.apf_track_for_obstacle = lambda obstacle: {
        "state": [4.0, 0.0, 0.2, 0.0], "heading_var_rad2": np.deg2rad(5.0) ** 2,
    }
    controller.obstacle_track_motion_is_stable = lambda track: True
    controller.obstacle_track_prediction_accel_ne = lambda track: np.array([0.0, 0.0])

    position, velocity, trusted = controller.apf_ekf_colreg_prediction([4.0, 0.0], [0.0, 0.2])
    assert trusted
    assert np.allclose(position, [4.2, 0.0])
    assert np.allclose(velocity, [0.2, 0.0])


def test_overtaking_close_turn_keeps_starboard_force_direction():
    from laptop import LaptopController

    controller = object.__new__(LaptopController)
    controller.webots_environment = "mr_webots_cross_left_to_right_large_ship.wbt"
    controller.apf_overtaking_close_distance_m = 3.0
    controller.apf_overtaking_close_turn_angle_rad = np.deg2rad(55.0)
    controller.apf_overtaking_turn_release_angle_rad = np.deg2rad(40.0)
    controller.apf_overtaking_close_turn_speed_m_s = 0.1
    controller.apf_side_lock_sign = -1.0
    controller.Yaw = 0.0
    controller.route_heading_rad = 0.0

    turn, speed = controller.apf_overtaking_close_turn_command("overtaking", 0.5, -0.1, 0.3)
    assert turn < 0.0 and np.isclose(abs(turn), np.deg2rad(55.0)) and speed == 0.1


def test_unified_controller_keeps_the_selected_avoidance_side():
    from laptop import LaptopController

    controller = object.__new__(LaptopController)
    controller._dynamic_ellipse_repulsion = lambda obstacle, scale: (np.array([1.0, 0.0]), True)
    controller._obstacle_body_position = lambda obstacle: np.array([2.0, 0.0])
    controller._obstacle_body_velocity = lambda obstacle: np.array([0.0, 0.0])
    controller.apf_classify_encounter = lambda position, velocity, own: (
        controller.encounter, 1.0, "rule"
    )
    controller.apf_cpa_metrics = lambda position, velocity, own: (1.0, 0.1)
    controller.apf_lock_side = lambda requested, level: requested
    controller.timefromstart = 0.0
    controller.apf_side_lock_sign = 0.0
    controller.v_robot = np.zeros((3, 1))
    controller.p_robot = np.zeros((3, 1))

    for encounter in ("head_on", "overtaking", "static_obstacle"):
        controller.encounter = encounter
        controller.apf_repulsion_for_obstacle({}, None, None)
        assert controller.apf_avoidance_side_sign == 1.0


def test_head_on_latch_prevents_a_virtual_crossing_from_changing_side():
    from laptop import LaptopController

    controller = object.__new__(LaptopController)
    controller._dynamic_ellipse_repulsion = lambda obstacle, scale: (np.array([1.0, 0.0]), True)
    controller._obstacle_body_position = lambda obstacle: np.array([2.0, 0.0])
    controller._obstacle_body_velocity = lambda obstacle: np.array([0.0, 0.2])
    controller.current_velocity_body = lambda: np.array([0.2, 0.0])
    controller.apf_classify_encounter = lambda position, velocity, own: (
        "crossing_from_starboard", 1.0, "crossing"
    )
    controller.apf_cpa_metrics = lambda position, velocity, own: (1.0, 0.1)
    controller.apf_lock_side = lambda requested, level: requested
    controller._CROSSING_RULES = {"crossing_from_starboard"}
    controller.apf_head_on_latched = True
    controller.timefromstart = 0.0

    controller.apf_repulsion_for_obstacle({}, None, None)
    assert controller.apf_encounter_mode == "head_on"
    assert controller.apf_avoidance_side_sign == -1.0


def test_overtaking_latch_overrides_a_previous_crossing_side_lock():
    from laptop import LaptopController

    controller = object.__new__(LaptopController)
    controller._dynamic_ellipse_repulsion = lambda obstacle, scale: (np.array([1.0, 0.0]), True)
    controller._obstacle_body_position = lambda obstacle: np.array([2.0, 0.0])
    controller._obstacle_body_velocity = lambda obstacle: np.array([0.0, 0.2])
    controller.current_velocity_body = lambda: np.array([0.2, 0.0])
    controller.apf_classify_encounter = lambda position, velocity, own: (
        "crossing_from_port", 1.0, "crossing"
    )
    controller.apf_cpa_metrics = lambda position, velocity, own: (1.0, 0.1)
    controller._CROSSING_RULES = {"crossing_from_port"}
    controller.apf_head_on_latched = False
    controller.apf_overtaking_latched = True
    controller.apf_side_lock_sign = 1.0
    controller.apf_side_lock_s = 5.0
    controller.timefromstart = 2.0

    controller.apf_repulsion_for_obstacle({}, None, None)
    assert controller.apf_encounter_mode == "overtaking"
    assert controller.apf_avoidance_side_sign == -1.0
    assert controller.apf_side_lock_sign == -1.0


def test_ellipse_repulsion_gain_scales_with_detected_pca_area():
    from laptop import LaptopController

    controller = object.__new__(LaptopController)
    controller._obstacle_body_position = lambda obstacle: np.asarray(obstacle["centre"], dtype=float)
    controller.earth_vector_to_body = lambda vector: np.asarray(vector, dtype=float)
    controller.obstacle_length_axis_ne = lambda obstacle: np.array([1.0, 0.0])
    controller.obstacle_pc_dimensions = lambda obstacle: (obstacle["pc1_m"], obstacle["pc2_m"])
    controller.apf_own_equivalent_radius_m = 0.25
    controller.apf_repulsive_gain = 3.0

    large_force, large_active = controller._dynamic_ellipse_repulsion(
        {"pc1_m": 1.0, "pc2_m": 1.0, "centre": [-0.9, 0.0]}, 4.0
    )
    small_force, small_active = controller._dynamic_ellipse_repulsion(
        {"pc1_m": 0.25, "pc2_m": 0.25, "centre": [-0.45, 0.0]}, 4.0
    )

    assert large_active and small_active
    assert np.isclose(np.linalg.norm(large_force), 2.0 * np.linalg.norm(small_force))


def test_virtual_field_uses_twice_tcpa_and_bridges_to_the_real_field():
    from laptop import LaptopController

    controller = object.__new__(LaptopController)
    controller.obstacle_ekf_prediction_enabled = True
    controller.North, controller.East, controller.Yaw = 0.0, 0.0, 0.0
    controller.v_robot = np.array([[1.0], [0.0], [0.0]])
    controller.route_tracking_speed_m_s = 1.0
    controller.apf_collision_horizon_s = 10.0
    controller.apf_dynamic_speed_threshold_m_s = 0.05
    controller.apf_current_field_size_scale = 4.0
    controller.apf_predicted_field_size_scale = 7.0
    controller.apf_own_equivalent_radius_m = 0.25
    controller.apf_obstacle_tracks = [{"id": 1, "state": [7.0, -3.0, 0.0, 1.0]}]
    controller.obstacle_track_motion_is_stable = lambda track: True
    controller.obstacle_pc_dimensions = lambda track: (1.0, 1.0)
    controller.body_vector_to_earth = lambda vector: np.asarray(vector, dtype=float)
    controller.earth_point_to_body = lambda point: np.asarray(point, dtype=float)

    fields = controller.update_apf_virtual_obstacles()
    assert fields and np.allclose(fields[-1]["centre_ne"], [7.0, 7.0])
    assert fields[-1]["virtual_time_s"] == 10.0
    assert fields[-1]["dcpa_m"] > 1.0  # DCPA no longer gates virtual-field creation.
    assert any(field["bridge"] for field in fields[:-1])


def test_snapshot_potential_includes_virtual_field_bridge():
    from plot_apf_snapshots import potential_components

    north_grid, east_grid = np.meshgrid(np.linspace(0.0, 10.0, 11), np.linspace(0.0, 10.0, 11))
    payload = {
        "apf": {"goal_ne": [10.0, 10.0]},
        "apf_settings": {"cluster_range_enabled": True},
        "virtual_obstacles": [
            {"predicted_risk_active": True, "bridge": True, "centre_ne": [5.0, 5.0],
             "length_axis_ne": [1.0, 0.0], "pc1_m": 1.0, "pc2_m": 1.0,
             "field_half_along_m": 1.0, "field_half_lateral_m": 1.0},
            {"predicted_risk_active": True, "centre_ne": [7.0, 5.0],
             "length_axis_ne": [1.0, 0.0], "pc1_m": 1.0, "pc2_m": 1.0,
             "field_half_along_m": 1.0, "field_half_lateral_m": 1.0},
        ],
    }
    _, _, virtual_potential, _ = potential_components(
        north_grid, east_grid, payload, [10.0, 10.0], 1.0, 1.0,
    )
    assert virtual_potential[5, 5] > 0.9 and virtual_potential[5, 7] > 0.9


def test_snapshot_3d_surface_is_written():
    import tempfile
    from pathlib import Path
    from plot_apf_snapshots import plot_snapshot_3d

    with tempfile.TemporaryDirectory() as directory:
        output_path = Path(directory) / "apf_3d.png"
        plot_snapshot_3d(
            {"payload": {"apf": {"goal_ne": [1.0, 1.0]}}},
            (0.0, 1.0, 0.0, 1.0), output_path, 80, 0.0, [1.0, 1.0], 1.0, 1.0,
        )
        assert output_path.exists() and output_path.stat().st_size > 0


def test_ellipse_direction_stays_close_to_lidar_axis():
    velocity = constrain_velocity_to_axis([1.0, 0.0], [0.0, 1.0], np.deg2rad(30.0))
    gap_deg = np.rad2deg(np.arccos(abs(np.dot(velocity / np.linalg.norm(velocity), [0.0, 1.0]))))
    assert np.isclose(gap_deg, 30.0)
    assert np.allclose(smooth_undirected_axis([1.0, 0.0], [-1.0, 0.0], 0.2), [1.0, 0.0])


def test_collinear_lidar_fragments_merge_before_tracking():
    fragments = [
        np.array([[0.00, -0.05], [0.10, 0.05], [0.20, -0.04]]),
        np.array([[0.65, -0.04], [0.75, 0.04], [0.85, 0.00]]),
        np.array([[1.30, -0.03], [1.40, 0.05], [1.50, 0.00]]),
        np.array([[0.45, 1.00], [0.55, 1.05], [0.65, 0.98]]),
    ]
    points = np.vstack(fragments)
    labels = np.repeat(np.arange(4), 3)
    merged = merge_collinear_cluster_labels(points, labels)
    assert len(set(merged[:9])) == 1
    assert merged[9] != merged[0]


if __name__ == "__main__":
    test_straight_line_cpa_lies_on_both_predictions()
    test_cpa_horizon_clamps_both_straight_trajectories()
    test_unified_controller_uses_four_dimensional_ekf_state()
    test_unified_controller_keeps_measured_motion_when_prediction_disabled()
    test_unified_crossing_side_matches_north_east_coordinates()
    test_colreg_uses_obstacle_robot_relative_sector_before_course()
    test_overtaking_close_turn_keeps_starboard_force_direction()
    test_unified_controller_keeps_the_selected_avoidance_side()
    test_ellipse_repulsion_gain_scales_with_detected_pca_area()
    test_virtual_field_uses_twice_tcpa_and_bridges_to_the_real_field()
    test_snapshot_potential_includes_virtual_field_bridge()
    test_snapshot_3d_surface_is_written()
    test_ellipse_direction_stays_close_to_lidar_axis()
    test_collinear_lidar_fragments_merge_before_tracking()
