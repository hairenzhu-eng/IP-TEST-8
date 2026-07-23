"""Pure COLREG zone classification and the single APF repulsion law."""

import numpy as np


def merge_collinear_cluster_labels(
    points,
    labels,
    max_gap_m=0.50,
    max_width_m=0.65,
    max_length_m=4.0,
    min_aspect_ratio=1.5,
):
    """Merge nearby DBSCAN fragments that form one elongated obstacle hull."""
    points = np.asarray(points, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) != len(labels):
        raise ValueError("points must be Nx2 and labels must have length N")

    groups = [[int(label)] for label in sorted(set(labels) - {-1})]

    def group_points(group):
        return points[np.isin(labels, group)]

    def can_merge(left, right):
        left_points = group_points(left)
        right_points = group_points(right)
        gap_m = float(np.min(np.linalg.norm(
            left_points[:, None, :] - right_points[None, :, :], axis=2
        )))
        if gap_m > float(max_gap_m):
            return False

        combined = np.vstack((left_points, right_points))
        centred = combined - np.mean(combined, axis=0)
        covariance = centred.T @ centred / max(len(combined) - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        normal = np.array([-axis[1], axis[0]])
        length_m = float(np.ptp(centred @ axis))
        width_m = float(np.ptp(centred @ normal))
        aspect_ratio = length_m / max(width_m, 0.05)
        return (
            length_m <= float(max_length_m)
            and width_m <= float(max_width_m)
            and aspect_ratio >= float(min_aspect_ratio)
        )

    merged = True
    while merged:
        merged = False
        for left_index in range(len(groups)):
            for right_index in range(left_index + 1, len(groups)):
                if not can_merge(groups[left_index], groups[right_index]):
                    continue
                groups[left_index].extend(groups.pop(right_index))
                merged = True
                break
            if merged:
                break

    result = labels.copy()
    for merged_label, group in enumerate(groups):
        result[np.isin(labels, group)] = merged_label
    return result


def straight_line_cpa(own_position, own_velocity, obstacle_position, obstacle_velocity, horizon_s=None):
    """Continuous-time CPA for two constant-velocity trajectories."""
    own_position = np.asarray(own_position, dtype=float).reshape(2)
    own_velocity = np.asarray(own_velocity, dtype=float).reshape(2)
    obstacle_position = np.asarray(obstacle_position, dtype=float).reshape(2)
    obstacle_velocity = np.asarray(obstacle_velocity, dtype=float).reshape(2)
    relative_position = obstacle_position - own_position
    relative_velocity = obstacle_velocity - own_velocity
    relative_speed_sq = float(np.dot(relative_velocity, relative_velocity))
    tcpa_s = 0.0 if relative_speed_sq < 1e-12 else max(
        -float(np.dot(relative_position, relative_velocity)) / relative_speed_sq,
        0.0,
    )
    if horizon_s is not None:
        tcpa_s = min(tcpa_s, max(float(horizon_s), 0.0))
    own_cpa = own_position + own_velocity * tcpa_s
    obstacle_cpa = obstacle_position + obstacle_velocity * tcpa_s
    return tcpa_s, float(np.linalg.norm(obstacle_cpa - own_cpa)), own_cpa, obstacle_cpa


def smooth_undirected_axis(previous_axis, measured_axis, alpha):
    """Smooth a PCA axis after resolving its arbitrary 180-degree sign."""
    previous = np.asarray(previous_axis, dtype=float).reshape(2)
    measured = np.asarray(measured_axis, dtype=float).reshape(2)
    previous /= max(float(np.linalg.norm(previous)), 1e-12)
    measured /= max(float(np.linalg.norm(measured)), 1e-12)
    if np.dot(previous, measured) < 0.0:
        measured = -measured
    blended = (1.0 - float(alpha)) * previous + float(alpha) * measured
    return blended / max(float(np.linalg.norm(blended)), 1e-12)


def constrain_velocity_to_axis(velocity, axis, max_gap_rad):
    """Keep motion direction within max_gap_rad of an undirected hull axis."""
    velocity = np.asarray(velocity, dtype=float).reshape(2)
    axis = np.asarray(axis, dtype=float).reshape(2)
    speed = float(np.linalg.norm(velocity))
    axis_norm = float(np.linalg.norm(axis))
    if speed < 1e-12 or axis_norm < 1e-12:
        return velocity.copy()
    direction = velocity / speed
    axis = axis / axis_norm
    if np.dot(direction, axis) < 0.0:
        axis = -axis
    gap = float(np.arctan2(direction[0] * axis[1] - direction[1] * axis[0], np.dot(direction, axis)))
    correction = np.sign(gap) * max(abs(gap) - max(float(max_gap_rad), 0.0), 0.0)
    c, s = np.cos(correction), np.sin(correction)
    return speed * np.array([c * direction[0] - s * direction[1], s * direction[0] + c * direction[1]])


def obstacle_stern_waypoint(centre_ne, velocity_ne, clearance_m):
    """Point behind the obstacle along its EKF motion direction."""
    centre = np.asarray(centre_ne, dtype=float).reshape(2)
    velocity = np.asarray(velocity_ne, dtype=float).reshape(2)
    speed = float(np.linalg.norm(velocity))
    if not np.isfinite(centre).all() or not np.isfinite(velocity).all() or speed < 1e-9:
        return None
    return centre - max(float(clearance_m), 0.0) * velocity / speed


def classify_colreg_zone(
    bearing_deg,
    relative_heading_deg,
    emergency=False,
    own_speed_m_s=np.nan,
    obstacle_speed_m_s=np.nan,
    own_bearing_from_obstacle_deg=np.nan,
):
    """Classify by the obstacle's robot-relative sector, then its course."""
    bearing = float(bearing_deg) % 360.0
    heading = float(relative_heading_deg) % 360.0
    ahead = bearing >= 292.5 or bearing < 67.5
    astern = 112.5 <= bearing < 247.5
    on_port = 247.5 <= bearing < 337.5
    on_starboard = 22.5 <= bearing < 112.5
    same_course = heading < 67.5 or heading >= 292.5
    # Rule 13 must be deliberately stricter than the general ahead/same-course
    # sectors.  Otherwise a crossing vessel can briefly look like a vessel
    # ahead when its measured bearing or heading jitters near a sector edge.
    # Rule 14 uses the original COLREG capture sector.
    head_on_ahead = bearing >= 292.5 or bearing < 67.5
    head_on_reciprocal = 157.5 <= heading < 202.5
    overtaking_ahead = bearing >= 340.0 or bearing < 20.0
    overtaking_same_course = heading < 20.0 or heading >= 340.0
    own_bearing_from_obstacle_deg = float(own_bearing_from_obstacle_deg) % 360.0
    own_in_obstacle_stern_sector = (
        np.isfinite(own_bearing_from_obstacle_deg)
        and 150.0 <= own_bearing_from_obstacle_deg < 210.0
    )
    own_speed_m_s = float(own_speed_m_s)
    obstacle_speed_m_s = float(obstacle_speed_m_s)
    speed_known = np.isfinite(own_speed_m_s) and np.isfinite(obstacle_speed_m_s)
    obstacle_lateral_speed_m_s = abs(
        obstacle_speed_m_s * np.sin(np.deg2rad(heading))
    ) if np.isfinite(obstacle_speed_m_s) else np.inf

    if head_on_ahead and head_on_reciprocal:
        # Planner side -1 is route-right/starboard.  Keep this aligned
        # with Rule 13 so both dynamic manoeuvres always pass right.
        return "head_on", -1.0, "COLREG Rule 14: both alter to starboard"

    if ahead:
        if (
            speed_known
            and overtaking_ahead
            and overtaking_same_course
            and own_in_obstacle_stern_sector
            and own_speed_m_s > obstacle_speed_m_s + 0.08
            and obstacle_lateral_speed_m_s <= 0.06
        ):
            return "overtaking", -1.0, "COLREG Rule 13: overtake on starboard side"
        if 67.5 <= heading < 157.5:
            return "crossing_from_port", 0.0, "COLREG Rule 15: stand on"
        if 202.5 <= heading < 292.5:
            return "crossing_from_starboard", -1.0, "COLREG Rule 15: give way, pass astern"

    if astern and same_course and (not speed_known or own_speed_m_s < obstacle_speed_m_s):
        return "being_overtaken", 0.0, "COLREG Rule 13: keep course and speed"

    if on_port:
        return (
            "crossing_from_port",
            -1.0 if emergency else 0.0,
            "COLREG Rule 17: emergency starboard action" if emergency else "COLREG Rule 17: stand on",
        )

    if on_starboard:
        return "crossing_from_starboard", -1.0, "COLREG Rule 15: give way, pass astern"

    return "static_obstacle", 0.0, "none"


def smooth_ellipse_repulsion(level, away, gain, outer_scale):
    """Maximum at/inside the hull, C1-smooth to zero at the size multiple."""
    level = float(level)
    outer_scale = float(outer_scale)
    if not np.isfinite(level) or outer_scale <= 1.0 or level >= outer_scale:
        return np.zeros(2, dtype=float), False
    u = np.clip((level - 1.0) / (outer_scale - 1.0), 0.0, 1.0)
    fade = 1.0 - (3.0 * u * u - 2.0 * u * u * u)
    return float(gain) * fade * np.asarray(away, dtype=float).reshape(2), True
