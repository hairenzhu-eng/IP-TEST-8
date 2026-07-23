import numpy as np

from colreg_apf import constrain_velocity_to_axis, merge_collinear_cluster_labels, smooth_undirected_axis, straight_line_cpa


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
    test_ellipse_direction_stays_close_to_lidar_axis()
    test_collinear_lidar_fragments_merge_before_tracking()
