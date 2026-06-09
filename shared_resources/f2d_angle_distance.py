"""f2d_angle_distance.py — fovea-to-disc angle and distance utilities.

Computes the Euclidean distance and polar angle between the fovea and the optic
disc centre, expressed in physical millimetres (corrected for Littmann
magnification).  Used by imageAlignment.py and alignment_schemes.py to compute
the f2d_angle that drives the pre-alignment rotation step.

Author: Marco Miranda
Date: 28 May 2026
"""

import math

def f2d_angle_distance(disc_center, fovea, scan_size, eye, littmanMagnification = 1):
    """Compute the fovea-to-disc distance (mm) and angle (degrees).

    Parameters
    ----------
    disc_center : list[float, float]
        [x, y] optic disc centre as fractional image coordinates in [0, 1].
        Values are read only; no in-place mutation is performed.
    fovea : list[float, float]
        [x, y] fovea centre as fractional image coordinates in [0, 1].
        Values are read only; no in-place mutation is performed.
    scan_size : tuple[float, float]
        (width_mm, height_mm) physical size of the scan area in millimetres,
        *before* Littmann magnification correction.
    eye : str
        ``'R'`` for right eye / ``'L'`` for left eye.
        Right eye angle uses ``atan2(dy, dx)``.
        Left eye angle uses ``atan2(dy, -dx)``.
        This mirrors the x-axis convention used in grid_diameter and avoids
        explicit coordinate mutation while keeping OD-convention angle output.
    littmanMagnification : float, optional
        Littmann correction factor (default ``1``, i.e. no correction).
        Divides each physical dimension so that distances are in corrected mm.

    Returns
    -------
    f2d_distance : float
        Euclidean disc-to-fovea distance in corrected millimetres.
    f2d_angle : float
        Angle from fovea to disc in degrees, in the range [0, 360).
        Computed with ``atan2(dy, dx)`` — correctly handles ``dx == 0`` and
        all four quadrants (unlike the former ``atan(dy/dx)`` which raised
        ``ZeroDivisionError`` and misreported quadrant for negative ``dx``).

    Side-effects
    ------------
    None. Input coordinates are not modified.
    """

    # needs to compensate for this inversion on reference position
    #disc_center[1] = 1.0 - disc_center[1]
    #fovea[1] = 1.0 - fovea[1]

    # Translate to fovea-centered physical coordinates.
    dy = (disc_center[1] - fovea[1]) * (scan_size[1] / littmanMagnification)
    dx = (disc_center[0] - fovea[0]) * (scan_size[0] / littmanMagnification)
    f2d_distance = math.sqrt(dy ** 2 + dx ** 2)

    # Laterality-aware angle convention:
    # - Right eyes: atan2(dy,  dx)
    # - Left eyes : atan2(dy, -dx)
    # This is algebraically equivalent to mirroring x first, but avoids in-place
    # coordinate mutation and is consistent with grid_diameter's OS handling.
    if eye == 'L':
        f2d_angle = math.atan2(dy, -dx) * 180.0 / math.pi
    else:
        f2d_angle = math.atan2(dy, dx) * 180.0 / math.pi

    if f2d_angle < 0:
        f2d_angle = (f2d_angle % 360 + 360) % 360
    else:
        f2d_angle = f2d_angle % 360

    return f2d_distance, f2d_angle