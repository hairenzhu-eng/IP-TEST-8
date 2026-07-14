"""Pure COLREG zone classification and the single APF repulsion law."""

import numpy as np


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
            return "head_on", -1.0, "COLREG Rule 14: both alter to starboard"
        if 67.5 <= heading < 157.5:
            return "crossing_from_port", 0.0, "COLREG Rule 15: stand on"
        if 202.5 <= heading < 292.5:
            return "crossing_from_starboard", -1.0, "COLREG Rule 15: give way, pass astern"
        side = -1.0 if heading >= 292.5 else 1.0
        return "overtaking", side, "COLREG Rule 13: overtake on the selected side"

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
