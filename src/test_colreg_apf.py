import numpy as np

from colreg_apf import classify_colreg_zone, obstacle_stern_waypoint, smooth_ellipse_repulsion


def test_colreg_four_zone_rules():
    assert classify_colreg_zone(0.0, 180.0)[0:2] == ("head_on", 1.0)
    assert classify_colreg_zone(0.0, 100.0)[0:2] == ("crossing_from_port", 0.0)
    assert classify_colreg_zone(0.0, 250.0)[0:2] == ("crossing_from_starboard", -1.0)
    assert classify_colreg_zone(0.0, 350.0)[0:2] == ("overtaking", 1.0)
    assert classify_colreg_zone(0.0, 10.0)[0:2] == ("overtaking", 1.0)
    assert classify_colreg_zone(300.0, 90.0)[0:2] == ("crossing_from_port", 0.0)
    assert classify_colreg_zone(300.0, 90.0, emergency=True)[0:2] == ("crossing_from_port", -1.0)
    assert classify_colreg_zone(60.0, 270.0)[0:2] == ("crossing_from_starboard", -1.0)
    assert classify_colreg_zone(180.0, 10.0)[0:2] == ("being_overtaken", 0.0)


def test_only_dynamic_ellipse_fade():
    away = np.array([1.0, 0.0])
    at_hull, active = smooth_ellipse_repulsion(1.0, away, 2.0, 2.0)
    halfway, _ = smooth_ellipse_repulsion(1.5, away, 2.0, 2.0)
    outside, active_outside = smooth_ellipse_repulsion(2.0, away, 2.0, 2.0)
    assert active and np.allclose(at_hull, [2.0, 0.0])
    assert np.allclose(halfway, [1.0, 0.0])
    assert not active_outside and np.allclose(outside, 0.0)


def test_crossing_passes_astern():
    from laptop import LaptopController

    controller = LaptopController.__new__(LaptopController)
    controller.apf_dynamic_speed_threshold_m_s = 0.05
    controller.p_robot = np.zeros((3, 1))
    assert controller.apf_pass_astern_side_from_velocity(
        controller.earth_vector_to_body([0.0, 1.0])
    ) == -1.0
    assert controller.apf_pass_astern_side_from_velocity([0.0, -1.0]) == 1.0
    assert np.allclose(obstacle_stern_waypoint([5.0, 2.0], [0.0, 1.0], 3.0), [5.0, -1.0])


def test_repulsion_uses_the_locked_colreg_side():
    from laptop import LaptopController

    controller = LaptopController.__new__(LaptopController)
    controller.__dict__.update(
        timefromstart=0.0,
        apf_last_dynamic_field_s=-np.inf,
        apf_overtaking_params={},
        apf_dynamic_speed_threshold_m_s=0.05,
        apf_current_field_size_scale={
            "head_on": 4.0, "overtaking": 4.0,
            "crossing": 4.0, "static_obstacle": 4.0,
        },
        apf_predicted_field_size_scale={
            "head_on": 7.0, "overtaking": 7.0,
            "crossing": 7.0, "static_obstacle": 7.0,
        },
    )
    controller._dynamic_ellipse_repulsion = lambda *_args: (np.array([2.0, 1.0]), True)
    controller._obstacle_body_position = lambda _obstacle: np.array([1.0, 0.0])
    controller._obstacle_body_velocity = lambda _obstacle: np.array([-1.0, 0.0])
    controller.current_velocity_body = lambda: np.array([1.0, 0.0])
    controller.apf_classify_encounter = lambda *_args: ("head_on", 1.0, "head-on")
    controller.apf_cpa_metrics = lambda *_args: (2.0, 0.5)
    controller.apf_lock_side = lambda requested, _level: requested

    force, active, _ = controller.apf_repulsion_for_obstacle({}, None, None)
    assert active and np.allclose(force, [2.0, 1.0])

    controller.apf_classify_encounter = lambda *_args: (
        "crossing_from_starboard", 1.0, "crossing"
    )
    controller._obstacle_body_velocity = lambda _obstacle: np.array([0.0, -1.0])
    force, active, _ = controller.apf_repulsion_for_obstacle({}, None, None)
    assert active and np.allclose(force, [2.0, 1.0])


def test_field_size_scales_are_independent_by_colreg_profile():
    from laptop import LaptopController

    controller = LaptopController.__new__(LaptopController)
    controller.apf_current_field_size_scale = {
        "head_on": 1.0, "overtaking": 2.0,
        "crossing": 3.0, "static_obstacle": 4.0,
    }
    controller.apf_predicted_field_size_scale = {
        "head_on": 5.0, "overtaking": 6.0,
        "crossing": 7.0, "static_obstacle": 8.0,
    }
    assert controller.apf_field_size_scale({}, "head_on") == 1.0
    assert controller.apf_field_size_scale({}, "crossing_from_starboard") == 3.0
    assert controller.apf_field_size_scale({"virtual": True}, "being_overtaken") == 6.0
    assert controller.apf_field_size_scale({"virtual": True}, "static_obstacle") == 8.0


def test_active_waypoint_survives_failed_periodic_replan():
    from laptop import LaptopController

    controller = LaptopController.__new__(LaptopController)
    target = np.array([2.0, 1.0])
    controller.apf_waypoint_path_ne = [target]
    controller.apf_waypoint_index = 0
    controller.timefromstart = 1.0
    controller.apf_waypoint_planned_at_s = 0.0
    controller.apf_waypoint_replan_interval_s = 0.6
    controller.webots_environment = "mr_webots_overtaking_large_ship.wbt"
    controller._advance_apf_waypoint_progress = lambda: None
    controller._detour_candidate_obstacles = lambda: []
    replanned = []
    controller._plan_apf_waypoint_path = lambda: replanned.append(True) and False
    controller._plan_force_guided_waypoint_path = lambda: (_ for _ in ()).throw(
        AssertionError("unsafe fixed-offset overtaking planner selected")
    )
    assert controller._ensure_apf_waypoint_path()
    assert replanned
    assert np.array_equal(controller.apf_waypoint_path_ne[0], target)


def test_dynamic_replan_replaces_current_target():
    from laptop import LaptopController

    controller = LaptopController.__new__(LaptopController)
    target = np.array([2.0, 1.0])
    controller.apf_waypoint_path_ne = [target]
    controller.apf_waypoint_index = 0
    controller.apf_waypoint_side_sign = 1.0
    controller.timefromstart = 1.0
    controller._project_to_main_route = lambda point: (float(point[0]), 0.0, point)
    controller._advance_apf_waypoint_progress = lambda current: None
    controller._update_display_waypoints = lambda: None
    assert controller._activate_apf_waypoint_path(
        [np.array([3.0, 2.0]), np.array([4.0, 2.0])],
        1.0,
        4.0,
        current_ne=np.zeros(2),
    )
    assert not np.array_equal(controller.apf_waypoint_path_ne[0], target)
    assert [point[0] for point in controller.apf_waypoint_path_ne] == [3.0, 4.0]


def test_waypoint_path_never_targets_behind_robot():
    from laptop import LaptopController

    controller = LaptopController.__new__(LaptopController)
    controller.apf_waypoint_path_ne = []
    controller.apf_waypoint_index = 0
    controller.timefromstart = 0.0
    controller._project_to_main_route = lambda point: (float(point[0]), 0.0, point)
    controller._advance_apf_waypoint_progress = lambda current: None
    controller._update_display_waypoints = lambda: None
    assert controller._activate_apf_waypoint_path(
        [np.array([-1.0, 1.0]), np.array([1.0, 1.0]), np.array([2.0, 1.0])],
        1.0,
        2.0,
        current_ne=np.zeros(2),
    )
    assert [point[0] for point in controller.apf_waypoint_path_ne] == [1.0, 2.0]


def test_terminal_waypoint_stays_active_until_goal_tolerance():
    from laptop import LaptopController

    controller = LaptopController.__new__(LaptopController)
    controller.goal_ne = np.array([15.0, 1.0])
    controller.goal_tolerance_m = 0.40
    controller.apf_waypoint_acceptance_m = 0.45
    controller.apf_waypoint_path_ne = [controller.goal_ne.copy()]
    controller.apf_waypoint_index = 0
    controller._update_display_waypoints = lambda: None
    cleared = []
    controller._clear_apf_waypoint_path = lambda: cleared.append(True)

    controller._advance_apf_waypoint_progress(np.array([14.5, 1.0]))
    assert controller.apf_waypoint_index == 0 and not cleared
    controller._advance_apf_waypoint_progress(np.array([14.8, 1.0]))
    assert controller.apf_waypoint_index == 1 and cleared


def test_planned_detour_rejoins_route_and_ends_at_goal():
    from laptop import LaptopController

    controller = LaptopController.__new__(LaptopController)
    controller.__dict__.update(
        apf_waypoint_detour_enabled=True,
        route_path_length_m=15.0,
        route_path_unit_ne=np.array([1.0, 0.0]),
        start_ne=np.array([0.0, 1.0]),
        goal_ne=np.array([15.0, 1.0]),
        apf_repulsive_force_body=np.array([1.0, 0.0]),
        apf_min_detour_offset_m=0.5,
        apf_waypoint_lateral_margin_m=0.45,
        apf_own_equivalent_radius_m=0.3,
        obstacle_min_pc2_m=0.2,
        apf_waypoint_entry_margin_m=1.2,
        route_tracking_lookahead_m=2.1,
        apf_waypoint_merge_margin_m=1.8,
        apf_waypoint_force_blend=0.65,
        apf_waypoint_preview_points=2,
        apf_waypoint_step_m=0.9,
        apf_waypoint_acceptance_m=0.45,
        apf_encounter_mode="none",
    )
    controller._current_ne = lambda: np.array([0.0, 1.0])
    controller._waypoint_detour_side = lambda: 1.0
    controller._detour_candidate_obstacles = lambda: []
    controller._force_guidance_direction_ne = lambda: np.array([1.0, 0.0])
    controller._keep_waypoints_outside_apf_fields = lambda points, *_args: points
    planned = []
    controller._activate_apf_waypoint_path = (
        lambda path, *_args, **_kwargs: planned.extend(path) or True
    )

    assert controller._plan_apf_waypoint_path()
    assert any(np.isclose(point[1], controller.start_ne[1]) for point in planned[:-1])
    assert np.array_equal(planned[-1], controller.goal_ne)


def test_route_left_normal_matches_body_left():
    from laptop import LaptopController

    controller = LaptopController.__new__(LaptopController)
    controller.route_path_unit_ne = np.array([1.0, 0.0])
    assert np.allclose(controller._route_normal_left_ne(), [0.0, 1.0])
    controller.route_path_unit_ne = np.array([0.0, 1.0])
    assert np.allclose(controller._route_normal_left_ne(), [-1.0, 0.0])


def test_feature_switches_and_cluster_size_fallback():
    from laptop import LaptopController, _switches_from_combination

    assert _switches_from_combination("ekf_off_cluster_on") == (
        "ekf_off_cluster_on", False, True
    )
    try:
        _switches_from_combination("bad")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid switch combination accepted")
    controller = LaptopController.__new__(LaptopController)
    controller.apf_cluster_range_enabled = False
    controller.obstacle_min_pc1_m = 0.36
    controller.obstacle_min_pc2_m = 0.20
    obstacle = {"pc1_m": 2.0, "pc2_m": 1.0, "length_axis_ne": [0.0, 1.0]}
    assert controller.obstacle_pc_dimensions(obstacle) == (0.36, 0.20)
    assert np.array_equal(controller.obstacle_length_axis_ne(obstacle), [1.0, 0.0])


if __name__ == "__main__":
    test_colreg_four_zone_rules()
    test_only_dynamic_ellipse_fade()
    test_crossing_passes_astern()
    test_repulsion_uses_the_locked_colreg_side()
    test_active_waypoint_survives_failed_periodic_replan()
    test_dynamic_replan_replaces_current_target()
    test_waypoint_path_never_targets_behind_robot()
    test_terminal_waypoint_stays_active_until_goal_tolerance()
    test_planned_detour_rejoins_route_and_ends_at_goal()
    test_route_left_normal_matches_body_left()
    test_feature_switches_and_cluster_size_fallback()
    print("COLREG/APF checks passed")
