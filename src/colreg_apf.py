"""Pure COLREG zone classification and the single APF repulsion law."""

import numpy as np


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


def classify_colreg_zone(bearing_deg, relative_heading_deg, emergency=False):
    """Classify the four Fig. 3 zones; side is +1 port, -1 starboard."""
    bearing = float(bearing_deg) % 360.0
    heading = float(relative_heading_deg) % 360.0
    zone_a = bearing >= 337.5 or bearing < 22.5
    zone_b = 247.5 <= bearing < 337.5  # obstacle on port
    zone_c = 22.5 <= bearing < 112.5   # obstacle on starboard
    zone_d = 112.5 <= bearing < 247.5

    if zone_a:
        if 157.5 <= heading < 202.5:
            return "head_on", 1.0, "COLREG Rule 14: both alter to starboard"
        if 67.5 <= heading < 157.5:
            return "crossing_from_port", 0.0, "COLREG Rule 15: stand on"
        if 202.5 <= heading < 292.5:
            return "crossing_from_starboard", -1.0, "COLREG Rule 15: give way, pass astern"
        return "overtaking", 1.0, "COLREG Rule 13: overtake to starboard"

    if zone_b and heading < 180.0:
        return (
            "crossing_from_port",
            -1.0 if emergency else 0.0,
            "COLREG Rule 17: emergency starboard action" if emergency else "COLREG Rule 17: stand on",
        )

    if zone_c and heading >= 180.0:
        return "crossing_from_starboard", -1.0, "COLREG Rule 15: give way, pass astern"

    if zone_d and (heading >= 292.5 or heading < 67.5):
        return "being_overtaken", 0.0, "COLREG Rule 13: keep course and speed"

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
