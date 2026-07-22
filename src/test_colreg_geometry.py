import numpy as np

from colreg_apf import constrain_velocity_to_axis, smooth_undirected_axis, straight_line_cpa


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


def test_ellipse_direction_stays_close_to_lidar_axis():
    velocity = constrain_velocity_to_axis([1.0, 0.0], [0.0, 1.0], np.deg2rad(30.0))
    gap_deg = np.rad2deg(np.arccos(abs(np.dot(velocity / np.linalg.norm(velocity), [0.0, 1.0]))))
    assert np.isclose(gap_deg, 30.0)
    assert np.allclose(smooth_undirected_axis([1.0, 0.0], [-1.0, 0.0], 0.2), [1.0, 0.0])


if __name__ == "__main__":
    test_straight_line_cpa_lies_on_both_predictions()
    test_cpa_horizon_clamps_both_straight_trajectories()
    test_unified_controller_uses_four_dimensional_ekf_state()
    test_ellipse_direction_stays_close_to_lidar_axis()
