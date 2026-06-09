"""Ocular magnification and axial length formulae.

Provides:
- ``estimateAL``            : Estimate axial length from FDA scan parameters.
- ``littmann``              : Littmann-Bennett linear ocular magnification factor.
- ``littmann_AsInMaestro``  : Littmann magnification as implemented in Topcon
                              Maestro C++ code (Gullstrand paraxial ray-tracing).

All functions accept scalar inputs and return scalar floats.

Author: Marco Miranda
Date: 28 May 2026
"""
def estimateAL(scan_info):
    """Estimate axial length (mm) from FDA scan metadata.

    Uses the empirical linear model derived from mirror position and mean
    A-scan depth to approximate the axial length of the eye.

    Parameters
    ----------
    scan_info : object
        Object with attributes ``mirror_pos`` (int), ``scan_axial_res`` (float,
        µm/pixel), and ``z_mean`` (float, mean A-scan depth index).

    Returns
    -------
    float
        Estimated axial length in millimetres.
    """
    ALest = (5.0786 /1000) * (scan_info.mirror_pos - 2000) + (scan_info.scan_axial_res/1000) * ( scan_info.z_mean - 45) + 22.6072

    return ALest

def littmann(Alest, ALass = 24.38539, meanRest = 7.70, meanRass = 7.70):
    """Compute the Littmann-Bennett linear ocular magnification factor.

    Returns the ratio of retinal image sizes between the study eye and a
    reference eye, used to correct fundus photo or OCT scan measurements
    for individual ocular magnification.

    The linear scale ratio applies isotropically to X and Y; areas scale
    with the square of this factor.

    Parameters
    ----------
    Alest : float
        Estimated axial length of the study eye (mm).
    ALass : float, default 24.38539
        Reference axial length (mm). Topcon Maestro default.
    meanRest : float, default 7.70
        Mean corneal radius of the study eye (mm).
    meanRass : float, default 7.70
        Mean corneal radius of the reference eye (mm).

    Returns
    -------
    float
        Magnification scale factor S (dimensionless). Values > 1 indicate
        a larger-than-reference eye; < 1 indicates a smaller eye.
    """
    # Littmann-Bennett ocular magnification factor (linear)
    # Linear scale ratio (applies isotropically to X and Y; areas scale with square)
    S = (0.01306 * (float(Alest) - 1.82) * (-0.033 * meanRest + 1.274)) / (0.01306 * (float(ALass) - 1.82) * (-0.033 * meanRass + 1.274))

    littmannMag = S
    
    return littmannMag


# ---------------------------------------------------------------------------
# littmann3 – Gullstrand ray-tracing method (matches Topcon Maestro C++ code)
# Transliterated from LittmannDlg.cpp (calcu_m_Maestro / GetEyeMagniGullstrandMaestro)
# ---------------------------------------------------------------------------

def _para_ray(r, d, n, obj_dis):
    """Paraxial ray trace through 6 refracting surfaces (indices 1-6).
    Returns (r6, p, beta_f) where p is the back image distance from surface 6."""
    import math
    h = 1.0
    a = 0.0 if abs(obj_dis) >= 100000.0 else (n[0] * h / obj_dis if obj_dis != 0 else 0.0)

    for i in range(1, 6):
        if r[i] != 0:
            a += (n[i] - n[i - 1]) * h / r[i]
        if n[i] != 0:
            h -= d[i] * a / n[i]

    if r[6] != 0:
        a += (n[6] - n[5]) * h / r[6]

    p = 0.0
    beta_f = 0.0
    if a != 0 and obj_dis != 0:
        p = h * n[6] / a
        beta_f = (n[6] / a) if abs(obj_dis) >= 100000.0 else (n[0] / obj_dis / a)

    return r[6], p, beta_f


def _inv_para_ray(r, d, n, im_dis):
    """Reverse (conjugate) paraxial ray trace. Returns (r1_new, p, beta_f)."""
    obj_dis = -im_dis
    rr = [0.0] * 10
    dd = [0.0] * 10
    nn = [0.0] * 10
    rr[0] = 100000.0
    dd[0] = 0.0
    nn[0] = n[6]
    for i in range(6):
        rr[i + 1] = -r[6 - i]
        dd[i + 1] = d[5 - i]      # d[6-1-i]
        nn[i + 1] = n[6 - 1 - i]

    _, p, beta_f = _para_ray(rr, dd, nn, obj_dis)
    r1_new = -rr[6]               # iresult[0] = -result[0] = -rr[6]
    return r1_new, p, beta_f


def _ray_trace(r, d, n, obj_dis, ray_ang, Im):
    """Finite (non-paraxial) ray trace through 7 surfaces. Returns retinal image height."""
    import math
    k = 7
    u = ray_ang
    s = obj_dis
    height = 0.0

    for i in range(1, k + 1):
        udash = u * math.pi / 180.0
        isin  = (s - r[i]) * math.sin(udash) / r[i]
        rsin  = n[i - 1] * isin / n[i]
        ang_inci = math.degrees(math.asin(isin))
        ang_ref  = math.degrees(math.asin(rsin))
        udash  = (u + ang_inci) * math.pi / 180.0
        height = r[i] * math.sin(udash)
        u      = ang_inci - ang_ref + u
        udash  = u * math.pi / 180.0
        s      = rsin * r[i] / math.sin(udash) + r[i] - d[i]

    ang    = math.degrees(math.asin(height / r[7]))
    height = Im * math.pi * ang / 360.0
    return height


def _calcu_m_maestro(Dg, R1, Im):
    """Compute Gullstrand ray-tracing magnification factor for the given eye.

    Parameters
    ----------
    Dg  : refractive error (diopters). 0 for the reference Gullstrand eye.
    R1  : anterior corneal radius (mm).
    Im  : axial length (mm).

    Returns
    -------
    result_m : eye magnification scalar (|0.1 / y0|)
    Ic       : calculated axial length from paraxial trace
    """
    import math

    # Gullstrand 6-surface schematic eye (indices 0-10, surfaces at 1-6)
    r = [100000.0, 7.7,   6.8,   10.0,  7.911, -5.76, -6.0,  100000.0, 100000.0, 0.0, 0.0]
    d = [0.0,      0.5,   3.1,   0.546, 2.419,  0.635, 0.0,   0.0,      0.0,      0.0, 0.0]
    n = [1.0,      1.376, 1.336, 1.386, 1.406,  1.386, 1.336, 1.336,    1.336,    0.0, 0.0]

    if abs(Dg) <= 0.001:
        Dg = 0.0001

    r[1] = R1
    r[2] = 6.8 / 7.7 * R1

    s1 = (1000.0 / Dg) - 12.0 if Dg != 0 else 0.0

    _, p, _ = _para_ray(r, d, n, s1)
    Ic = p + 7.2
    d[6] = p

    if abs(Im) < 0.0001:
        Im = Ic

    u = 0.370844  # ray angle (degrees) for the Maestro instrument

    if abs((Im - Ic) / Im) <= 0.003:
        # Eye close to reference: only shift the retinal mirror surface
        r[7] = -(Ic / 2.0)
    else:
        # type=1 path: scale all internal lens radii to match Dg and Im simultaneously
        s1r = Im - 7.2
        threshold = 0.26
        best_rd = [r[3], r[4], r[5], r[6]]

        rdash = [100000.0, 7.7, 6.8, 10.0, 7.911, -5.76, -6.0, 100000.0, 100000.0]
        dd_step = 0.00001
        dd = 0.5
        while dd <= 1.5:
            rdash[3] = 10.0   * dd
            rdash[4] = 7.911  * dd
            rdash[5] = -5.76  * dd
            rdash[6] = -6.0   * dd

            r1_new, p_inv, _ = _inv_para_ray(rdash, d, n, s1r)
            Dg_calc = -1000.0 / (p_inv - 12.0) if (p_inv - 12.0) != 0 else 0.0
            dx = abs(Dg_calc - Dg)

            if dx < threshold or dx == 0.0:
                threshold = dx
                # reverse signs to get the forward-direction radii
                best_rd = [-rdash[6], -rdash[5], -rdash[4], -rdash[3]]
                if dx == 0.0:
                    break
            dd += dd_step

        r[3] = best_rd[0]
        r[4] = best_rd[1]
        r[5] = best_rd[2]
        r[6] = best_rd[3]
        r[7] = -(Im / 2.0)
        d[6] = Im - 7.2

    _para_ray(r, d, n, 100000.0)  # update internal state (not needed in Python but mirrors C++)

    y0 = _ray_trace(r, d, n, 3.05, u, Im)
    result_m = abs(0.1 / y0)
    return result_m, Ic


def littmann_AsInMaestro(Dg, R1, Im, Dg_ref=0.0, R1_ref=7.70, Im_ref=24.38539):
    """Littmann ocular magnification factor using Gullstrand ray-tracing (Topcon method).

    Matches the algorithm in the Topcon Maestro C++ application (LittmannDlg.cpp).
    The correction factor is the ratio of the patient eye magnification to the
    reference (Gullstrand) eye magnification, both computed with finite ray tracing
    through the 6-surface Gullstrand schematic eye model. This only works if Dg, R1
    and Im are known, otherwise formula will provide wrong results. If one of these
    is not known, use the simpler Bennett formula (littmann).

    Parameters
    ----------
    Dg     : patient refractive error (diopters, typically spectacle plane).
    R1     : patient anterior corneal radius (mm).
    Im     : patient axial length (mm).
    Dg_ref : refractive error of reference eye (default 0.0 – emmetropic Gullstrand).
    R1_ref : corneal radius of reference eye (default 7.70 mm – Gullstrand).
    Im_ref : axial length of reference eye (default 24.38539 mm – Gullstrand).

    Returns
    -------
    littmannMag : linear scale correction factor (patient / reference).
                  Multiply measured image distances by this value to obtain
                  true retinal distances.
    """
    m_patient, _ = _calcu_m_maestro(Dg, R1, Im)
    m_reference, _ = _calcu_m_maestro(Dg_ref, R1_ref, Im_ref)
    return m_patient / m_reference