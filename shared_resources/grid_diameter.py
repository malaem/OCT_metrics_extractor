"""grid_diameter — Annular pixel selection around an OCT scan centre.

Builds an annulus (ring) grid around a scan centre and joins it to a pixel
table.  Used by the pipeline to select pixels at a specified radial distance
(e.g. 3.4 mm for cpRNFL) from the optic disc or fovea before sector averaging.

Coordinate conventions
----------------------
- Input pixel coordinates use image convention: origin at top-left;
  width increases left→right, height increases top→bottom.
- ``centre_x`` and ``centre_y`` are expected as proportions in [0, 1].
- ``centre_y`` is internally flipped (1 − centre_y) to match the IM6 geometry
  convention used in the original R implementation.
- Returned ``angle`` column is in degrees in [0, 360).
  - Right eyes (OD / R / RIGHT / 0): angle = atan2(dy,  dx)
  - Left eyes  (OS / L / LEFT  / 1): angle = atan2(dy, −dx)

Author: Marco Miranda
Date: 28 May 2026
"""
import math
from datetime import datetime

import numpy as np
import pandas as pd


_RIGHT_TOKENS = {"OD", "R", "RIGHT", "0"}
_LEFT_TOKENS = {"OS", "L", "LEFT", "1"}


def _as_1d_array(value):
    """Coerce *value* to a flat NumPy array regardless of input type."""
    if isinstance(value, pd.Series):
        return value.to_numpy()
    if isinstance(value, np.ndarray):
        return value.ravel()
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    return np.asarray([value])


def _expand_to_n(value, n, name):
    """Broadcast a scalar or length-1 array to length *n*, or validate that it already has length *n*."""
    arr = _as_1d_array(value)
    if arr.size == 1:
        return np.repeat(arr, n)
    if arr.size != n:
        raise ValueError(
            f"{name} must be scalar or length n (number of unique scans = {n}), got {arr.size}."
        )
    return arr


def _normalize_laterality(values):
    """Map laterality tokens (OD/R/OS/L etc.) to numeric codes: 0 = right, 1 = left."""
    vals = _as_1d_array(values)
    out = np.full(vals.size, np.nan)
    strs = np.char.upper(np.asarray(vals, dtype=str))
    right_mask = np.isin(strs, list(_RIGHT_TOKENS))
    left_mask = np.isin(strs, list(_LEFT_TOKENS))
    out[right_mask] = 0
    out[left_mask] = 1
    return out.astype(float)


def _compute_geometry(swpx, shpx, swmm, shmm, img_h, z_res, litman):
    """Compute IM6 grid-space pixel-to-mm ratios (x_ratio, y_ratio) and rescaled dimensions.

    Replicates the Topcon IM6 geometry rescaling from the reference R implementation.
    For 512×128 or 512×256 scans the scan dimensions are divided by the Littmann
    magnification factor and then transformed to IM6 grid space.  All other scan
    resolutions return ratio = 1.0 (no rescaling).

    Parameters
    ----------
    swpx, shpx : int   Scan width and height in pixels.
    swmm, shmm : float Scan width and height in millimetres.
    img_h      : float B-scan JPEG pixel height.
    z_res      : float Axial resolution in µm/pixel.
    litman     : float Littmann magnification factor (dimensionless).

    Returns
    -------
    dict
        Keys: scan_width_px, scan_height_px, scan_height_px_orig,
              scan_width_mm, scan_height_mm,
              gridSampling_xRatio, gridSampling_yRatio.
    """
    swpx = int(swpx)
    shpx = int(shpx)
    swmm = float(swmm)
    shmm = float(shmm)
    img_h = float(img_h)
    z_res = float(z_res)
    litman = float(litman)

    if swpx == 512 and shpx in (128, 256):
        swmm = swmm / litman
        shmm = shmm / litman
        scan_depth = math.floor((img_h * z_res / 1000.0) * 1e8) / 1e8
        f_scale_w = math.floor((img_h - 1.0) * (swmm / scan_depth) * 1e8) / 1e8
        f_size_r = math.floor((shmm / swmm) * 1e8) / 1e8
        x_ratio = math.floor((f_scale_w / 4.0 / (swpx - 1.0)) * 1e8) / 1e8
        y_ratio = math.floor((((swpx - 1.0) / (shpx - 1.0)) * f_size_r * x_ratio) * 1e8) / 1e8
        shpx_orig = shpx
        if x_ratio == y_ratio:
            swpx = swpx - 1
            shpx = shpx - 1
        else:
            swpx = math.floor(math.floor(((swpx - 1.0) * x_ratio) * 1e8) / 1e8 + 0.0000005)
            shpx = math.floor(math.floor(((shpx - 1.0) * y_ratio) * 1e8) / 1e8 + 0.0000005)
    else:
        x_ratio = 1.0
        y_ratio = 1.0
        shpx_orig = shpx

    return {
        "scan_width_px": float(swpx),
        "scan_height_px": float(shpx),
        "scan_height_px_orig": float(shpx_orig),
        "scan_width_mm": float(swmm),
        "scan_height_mm": float(shmm),
        "gridSampling_xRatio": float(x_ratio),
        "gridSampling_yRatio": float(y_ratio),
    }


def grid_diameter(
    df,
    diameter=3.4,
    tolerance=1,
    tolerance_unit="px",
    scan_width_px=512,
    scan_height_px=128,
    scan_width_mm=12,
    scan_height_mm=9,
    centre_x=0.68,
    centre_y=0.48,
    laterality="OD",
    imgHeight=885,
    zResolution=2.6,
    littmanMagnification=1,
):
    """
    Select annulus pixels and compute polar angle per pixel.

    Parameters
    ----------
    df : pandas.DataFrame or compatible table
        Input pixel table. Must contain columns:
        - width: x index in pixels
        - height: y index in pixels
        Optional metadata columns are used to infer per-scan identity and geometry.

    diameter : float, default 3.4
        Annulus nominal diameter in mm.

    tolerance : float or length-2 sequence, default 1
        Ring half-thickness around the nominal radius.
        - scalar: same tolerance on x and y
        - length-2: [tolerance_x, tolerance_y]

    tolerance_unit : {"px", "mm"}, default "px"
        Unit for tolerance. If "mm", tolerance is converted to pixel-metric space
        using scan_width_px/scan_width_mm and scan_height_px/scan_height_mm.

    scan_width_px, scan_height_px : scalar or length-n
        Scan raster size in pixels for each scan.

    scan_width_mm, scan_height_mm : scalar or length-n
        Physical scan size in mm for each scan.

    centre_x, centre_y : scalar or length-n, default 0.68 and 0.48
        Disc/fovea centre as proportions in [0, 1].
        Start-point convention:
        - input is top-left origin image space
        - centre_y is internally flipped to IM6 geometry (bottom-left style)
        centre_x/centre_y can be None/empty only when df contains:
        - new_disc_centre_position_x
        - new_disc_centre_position_y

    laterality : scalar or length-n, default "OD"
        Eye side tokens accepted (case-insensitive):
        - right: OD, R, RIGHT, 0
        - left:  OS, L, LEFT, 1
        If df has an eye column, it takes precedence over this argument.

    imgHeight : scalar or length-n, default 885
        JPEG image height used by the IM6-derived geometry transform.

    zResolution : scalar or length-n, default 2.6
        Axial pixel resolution used by the IM6-derived geometry transform.

    littmanMagnification : scalar or length-n, default 1
        Littman magnification factor. Applied before geometry scaling.

    Per-scan geometry columns in df (override function arguments when present)
    -------------------------------------------------------------------------
    - scan_res_width  -> scan_width_px
    - scan_res_height -> scan_height_px
    - scan_width      -> scan_width_mm
    - scan_height     -> scan_height_mm
    - jpeg_height     -> imgHeight
    - axial_resolution -> zResolution

    Scan identity (n)
    -----------------
    Scans are inferred by dense-ranking concatenated ID parts from available
    columns: patient_id, capture_datetime (or capture_date + capture_time),
    serial_no, data_no, eye.

    Returns
    -------
    pandas.DataFrame
        Input columns left-joined onto annulus pixels with an added angle column.
        Temporary columns width_m, height_m, ID_rank are removed before return.
        Sorted by [patient_id, height, width] when patient_id exists; otherwise
        sorted by [height, width].

    Notes
    -----
    - Uses grouped geometry execution for speed: scans with identical geometry
      share one computed candidate grid.
    - Uses a bulk path when n_group * grid_size <= 5e6, otherwise per-scan path.
    """
    df = pd.DataFrame(df).copy()

    if "width" not in df.columns or "height" not in df.columns:
        raise ValueError("df must contain width and height columns.")

    has_pid = "patient_id" in df.columns
    has_cdt = "capture_datetime" in df.columns
    has_cd = "capture_date" in df.columns
    has_ct = "capture_time" in df.columns
    has_sno = "serial_no" in df.columns
    has_dno = "data_no" in df.columns
    has_eye = "eye" in df.columns

    if not has_pid:
        print(
            "No column with a patient_id identifier. Assuming there is only one patient in the df dataset."
        )
    if not has_cdt:
        if not has_cd:
            print(
                "No column with a capture_date identifier. Assuming there is only one time point per patient."
            )
        if not has_ct:
            print(
                "No column with a capture_time identifier. Assuming there is only one time point per patient."
            )
    if not has_sno:
        print("No column with a serial_no identifier. Assuming there is only one device per patient.")
    if not has_dno:
        print("No column with a data_no identifier. Assuming there is only one fda file per patient.")

    now = datetime.now()
    pid_part = df["patient_id"].astype(str) if has_pid else pd.Series("1", index=df.index)
    if has_cdt:
        dt_part = df["capture_datetime"].astype(str)
    else:
        date_part = (
            df["capture_date"].astype(str)
            if has_cd
            else pd.Series(now.strftime("%Y-%m-%d"), index=df.index)
        )
        time_part = (
            df["capture_time"].astype(str)
            if has_ct
            else pd.Series(now.strftime("%H%M%S"), index=df.index)
        )
        dt_part = date_part + time_part
    sno_part = df["serial_no"].astype(str) if has_sno else pd.Series("1", index=df.index)
    dno_part = df["data_no"].astype(str) if has_dno else pd.Series("1", index=df.index)
    eye_part = df["eye"].astype(str) if has_eye else pd.Series("1", index=df.index)

    unique_id = pid_part + dt_part + sno_part + dno_part + eye_part
    df["ID_rank"] = pd.Series(pd.factorize(unique_id, sort=True)[0] + 1, index=df.index, dtype=np.int64)

    id_ranks = np.sort(df["ID_rank"].unique())
    n = id_ranks.size

    if n == 0:
        out = df.copy()
        out["angle"] = np.nan
        if "ID_rank" in out.columns:
            out = out.drop(columns=["ID_rank"])
        return out

    has_scan_res_w = "scan_res_width" in df.columns
    has_scan_res_h = "scan_res_height" in df.columns
    has_scan_mm_w = "scan_width" in df.columns
    has_scan_mm_h = "scan_height" in df.columns
    has_jpeg_h = "jpeg_height" in df.columns
    has_axial_res = "axial_resolution" in df.columns
    has_cx_col = "new_disc_centre_position_x" in df.columns
    has_cy_col = "new_disc_centre_position_y" in df.columns

    if has_eye:
        eye_per_scan = df.groupby("ID_rank", sort=True)["eye"].first().reindex(id_ranks)
        lat_vals = _normalize_laterality(eye_per_scan.to_numpy())
    else:
        lat_vals = _normalize_laterality(_expand_to_n(laterality, n, "laterality"))

    if not has_eye:
        bad_lat = np.isnan(lat_vals)
        if np.any(bad_lat):
            raw_lat = _expand_to_n(laterality, n, "laterality")
            bad_values = np.unique(raw_lat[bad_lat])
            print(
                "grid_diameter: "
                f"{int(np.sum(bad_lat))} scan(s) have unrecognised laterality value(s): "
                f"{', '.join(map(str, bad_values))}. Angle will be NA for those scans."
            )

    if centre_x is None or _as_1d_array(centre_x).size == 0:
        if not has_cx_col:
            raise ValueError(
                "centre_x is NULL/empty and df does not contain new_disc_centre_position_x."
            )
        cx_vals = (
            df.groupby("ID_rank", sort=True)["new_disc_centre_position_x"]
            .first()
            .reindex(id_ranks)
            .to_numpy(dtype=float)
        )
    else:
        cx_vals = _expand_to_n(centre_x, n, "centre_x").astype(float)

    if centre_y is None or _as_1d_array(centre_y).size == 0:
        if not has_cy_col:
            raise ValueError(
                "centre_y is NULL/empty and df does not contain new_disc_centre_position_y."
            )
        cy_vals = (
            df.groupby("ID_rank", sort=True)["new_disc_centre_position_y"]
            .first()
            .reindex(id_ranks)
            .to_numpy(dtype=float)
        )
        cy_vals = 1.0 - cy_vals
    else:
        cy_vals = _expand_to_n(centre_y, n, "centre_y").astype(float)
        cy_vals = 1.0 - cy_vals

    disc_laterality = pd.DataFrame(
        {
            "ID_rank": id_ranks,
            "centre_x": cx_vals,
            "centre_y": cy_vals,
            "laterality": lat_vals,
        }
    )

    geom_cols = []
    if has_scan_res_w:
        geom_cols.append("scan_res_width")
    if has_scan_res_h:
        geom_cols.append("scan_res_height")
    if has_scan_mm_w:
        geom_cols.append("scan_width")
    if has_scan_mm_h:
        geom_cols.append("scan_height")
    if has_jpeg_h:
        geom_cols.append("jpeg_height")
    if has_axial_res:
        geom_cols.append("axial_resolution")

    if geom_cols:
        nunique = df.groupby("ID_rank", sort=True)[geom_cols].nunique(dropna=False)
        bad_mask = (nunique > 1).any(axis=1)
        if bad_mask.any():
            bad_ranks = nunique.index[bad_mask].tolist()
            bad_cols = nunique.columns[(nunique[bad_mask] > 1).any(axis=0)].tolist()
            preview = ", ".join(map(str, bad_ranks[:5]))
            if len(bad_ranks) > 5:
                preview = f"{preview}, ... [{len(bad_ranks)} total]"
            raise ValueError(
                "Geometry columns are not constant within scan for: "
                f"{', '.join(bad_cols)}. Problem ID_rank(s): {preview}."
            )

        geom_scan = (
            df.groupby("ID_rank", sort=True)[geom_cols]
            .first()
            .reindex(id_ranks)
            .reset_index()
        )
    else:
        geom_scan = pd.DataFrame({"ID_rank": id_ranks})

    geom_scan["_swpx"] = (
        geom_scan["scan_res_width"].astype(float)
        if has_scan_res_w
        else _expand_to_n(scan_width_px, n, "scan_width_px").astype(float)
    )
    geom_scan["_shpx"] = (
        geom_scan["scan_res_height"].astype(float)
        if has_scan_res_h
        else _expand_to_n(scan_height_px, n, "scan_height_px").astype(float)
    )
    geom_scan["_swmm"] = (
        geom_scan["scan_width"].astype(float)
        if has_scan_mm_w
        else _expand_to_n(scan_width_mm, n, "scan_width_mm").astype(float)
    )
    geom_scan["_shmm"] = (
        geom_scan["scan_height"].astype(float)
        if has_scan_mm_h
        else _expand_to_n(scan_height_mm, n, "scan_height_mm").astype(float)
    )
    geom_scan["_imgh"] = (
        geom_scan["jpeg_height"].astype(float)
        if has_jpeg_h
        else _expand_to_n(imgHeight, n, "imgHeight").astype(float)
    )
    geom_scan["_zres"] = (
        geom_scan["axial_resolution"].astype(float)
        if has_axial_res
        else _expand_to_n(zResolution, n, "zResolution").astype(float)
    )
    geom_scan["_litm"] = _expand_to_n(littmanMagnification, n, "littmanMagnification").astype(float)

    if np.isscalar(tolerance) or _as_1d_array(tolerance).size == 1:
        tol_x = float(_as_1d_array(tolerance)[0])
        tol_y = float(_as_1d_array(tolerance)[0])
    else:
        tol_arr = _as_1d_array(tolerance)
        if tol_arr.size != 2:
            raise ValueError("tolerance must be scalar or length-2.")
        tol_x = float(tol_arr[0])
        tol_y = float(tol_arr[1])

    df = df.drop_duplicates()

    geom_scan["_geom_group"] = (
        geom_scan[["_swpx", "_shpx", "_swmm", "_shmm", "_imgh", "_zres", "_litm"]]
        .astype(str)
        .agg("\x01".join, axis=1)
    )

    results = []
    by_rank = df.groupby("ID_rank", sort=False)
    dl_idx = disc_laterality.set_index("ID_rank")

    for _, grp in geom_scan.groupby("_geom_group", sort=False):
        grp_ids = grp["ID_rank"].to_numpy(dtype=np.int64)
        g0 = grp.iloc[0]
        g = _compute_geometry(
            g0["_swpx"],
            g0["_shpx"],
            g0["_swmm"],
            g0["_shmm"],
            g0["_imgh"],
            g0["_zres"],
            g0["_litm"],
        )

        tol_x_px = tol_x
        tol_y_px = tol_y
        if tolerance_unit == "mm":
            tol_x_px = tol_x_px * (g["scan_width_px"] / g["scan_width_mm"])
            tol_y_px = tol_y_px * (g["scan_height_px"] / g["scan_height_mm"])

        r_x = diameter / 2.0 * (g["scan_width_px"] / g["scan_width_mm"])
        r_y = diameter / 2.0 * (g["scan_height_px"] / g["scan_height_mm"])

        cx_prop = dl_idx.loc[grp_ids, "centre_x"].to_numpy(dtype=float)
        cy_prop = dl_idx.loc[grp_ids, "centre_y"].to_numpy(dtype=float)
        cx_px = cx_prop * g["scan_width_px"]
        cy_px = cy_prop * g["scan_height_px"]

        wm_from = int(math.floor(np.min(cx_px - r_x - tol_x_px) - 5))
        wm_to = int(math.ceil(np.max(cx_px + r_x + tol_x_px) + 5))
        hm_from = int(math.floor(np.min(cy_px - r_y - tol_y_px) - 5))
        hm_to = int(math.ceil(np.max(cy_px + r_y + tol_y_px) + 5))

        width_m_vals = np.arange(wm_from, wm_to + 1, dtype=np.int32)
        height_m_vals = np.arange(hm_from, hm_to + 1, dtype=np.int32)
        wm, hm = np.meshgrid(width_m_vals, height_m_vals, indexing="xy")
        wm = wm.ravel()
        hm = hm.ravel()

        width = np.floor((wm / g["gridSampling_xRatio"]) + 0.5).astype(np.int32)
        height = (
            g["scan_height_px_orig"]
            - 1.0
            - np.floor((hm / g["gridSampling_yRatio"]) + 0.5)
        ).astype(np.int32)

        in_bounds = (
            (wm >= 0)
            & (wm <= g["scan_width_px"])
            & (hm >= 0)
            & (hm <= g["scan_height_px"])
        )

        cg = pd.DataFrame(
            {
                "width_m": wm[in_bounds],
                "height_m": hm[in_bounds],
                "width": width[in_bounds],
                "height": height[in_bounds],
            }
        )

        if cg.empty:
            continue

        df_grp = df[df["ID_rank"].isin(grp_ids)]
        df_grp = df_grp[
            (df_grp["width"] >= cg["width"].min())
            & (df_grp["width"] <= cg["width"].max())
            & (df_grp["height"] >= cg["height"].min())
            & (df_grp["height"] <= cg["height"].max())
        ].drop_duplicates()

        meta = dl_idx.loc[grp_ids, ["centre_x", "centre_y", "laterality"]].copy()
        meta["centre_x"] = meta["centre_x"].to_numpy(dtype=float) * g["scan_width_px"]
        meta["centre_y"] = meta["centre_y"].to_numpy(dtype=float) * g["scan_height_px"]

        n_grp = len(grp_ids)
        ngrid = len(cg)
        use_bulk = (float(n_grp) * float(ngrid)) <= 5e6

        if use_bulk:
            rep = pd.DataFrame(np.repeat(cg.to_numpy(), n_grp, axis=0), columns=cg.columns)
            rep["ID_rank"] = np.tile(grp_ids, ngrid)
            rep = rep.merge(meta.reset_index().rename(columns={"index": "ID_rank"}), on="ID_rank", how="left")

            # Safe denominators: prevent division by zero for near-zero annulus inner boundary
            r_x_inner = max(r_x - tol_x_px, 1e-10)
            r_y_inner = max(r_y - tol_y_px, 1e-10)
            
            inner = np.sqrt(
                ((rep["width_m"].to_numpy(dtype=float) - rep["centre_x"].to_numpy(dtype=float)) / r_x_inner) ** 2
                + ((rep["height_m"].to_numpy(dtype=float) - rep["centre_y"].to_numpy(dtype=float)) / r_y_inner) ** 2
            )
            outer = np.sqrt(
                ((rep["width_m"].to_numpy(dtype=float) - rep["centre_x"].to_numpy(dtype=float)) / (r_x + tol_x_px)) ** 2
                + ((rep["height_m"].to_numpy(dtype=float) - rep["centre_y"].to_numpy(dtype=float)) / (r_y + tol_y_px)) ** 2
            )
            inner = np.floor(inner * 1e8) / 1e8
            outer = np.floor(outer * 1e8) / 1e8
            keep = (inner > 1.0) & (outer <= 1.0)
            rep = rep.loc[keep].copy()

            rep["angle"] = np.nan
            lat = rep["laterality"].to_numpy(dtype=float)
            right = lat == 0
            left = lat == 1
            rep.loc[right, "angle"] = (
                np.arctan2(
                    rep.loc[right, "height_m"].to_numpy(dtype=float) - rep.loc[right, "centre_y"].to_numpy(dtype=float),
                    rep.loc[right, "width_m"].to_numpy(dtype=float) - rep.loc[right, "centre_x"].to_numpy(dtype=float),
                )
                * 180.0
                / math.pi
            ) % 360.0
            rep.loc[left, "angle"] = (
                np.arctan2(
                    rep.loc[left, "height_m"].to_numpy(dtype=float) - rep.loc[left, "centre_y"].to_numpy(dtype=float),
                    rep.loc[left, "centre_x"].to_numpy(dtype=float) - rep.loc[left, "width_m"].to_numpy(dtype=float),
                )
                * 180.0
                / math.pi
            ) % 360.0
            rep["angle"] = np.floor(rep["angle"].to_numpy(dtype=float) * 1e8) / 1e8

            rep = rep.drop(columns=["centre_x", "centre_y", "laterality"]).drop_duplicates()
            merged = rep.merge(df_grp, on=["width", "height", "ID_rank"], how="left")
            results.append(merged)
        else:
            # Safe denominators for non-bulk case
            r_x_inner = max(r_x - tol_x_px, 1e-10)
            r_y_inner = max(r_y - tol_y_px, 1e-10)
            
            for scan_id in np.sort(grp_ids):
                gi = cg.copy()
                gi["ID_rank"] = scan_id

                m = meta.loc[scan_id]
                cx_i = float(m["centre_x"])
                cy_i = float(m["centre_y"])
                lat_i = m["laterality"]

                inner = np.sqrt(
                    ((gi["width_m"].to_numpy(dtype=float) - cx_i) / r_x_inner) ** 2
                    + ((gi["height_m"].to_numpy(dtype=float) - cy_i) / r_y_inner) ** 2
                )
                outer = np.sqrt(
                    ((gi["width_m"].to_numpy(dtype=float) - cx_i) / (r_x + tol_x_px)) ** 2
                    + ((gi["height_m"].to_numpy(dtype=float) - cy_i) / (r_y + tol_y_px)) ** 2
                )
                inner = np.floor(inner * 1e8) / 1e8
                outer = np.floor(outer * 1e8) / 1e8
                keep = (inner > 1.0) & (outer <= 1.0)
                gi = gi.loc[keep].copy()

                if lat_i == 0:
                    angle = (
                        np.arctan2(
                            gi["height_m"].to_numpy(dtype=float) - cy_i,
                            gi["width_m"].to_numpy(dtype=float) - cx_i,
                        )
                        * 180.0
                        / math.pi
                    ) % 360.0
                    gi["angle"] = np.floor(angle * 1e8) / 1e8
                elif lat_i == 1:
                    angle = (
                        np.arctan2(
                            gi["height_m"].to_numpy(dtype=float) - cy_i,
                            cx_i - gi["width_m"].to_numpy(dtype=float),
                        )
                        * 180.0
                        / math.pi
                    ) % 360.0
                    gi["angle"] = np.floor(angle * 1e8) / 1e8
                else:
                    gi["angle"] = np.nan

                gi = gi.drop_duplicates()
                try:
                    df_i = by_rank.get_group(scan_id)
                except KeyError:
                    df_i = df_grp.iloc[0:0]
                merged = gi.merge(df_i, on=["width", "height", "ID_rank"], how="left")
                results.append(merged)

    if not results:
        out = df.iloc[0:0].copy()
        out["angle"] = np.nan
    else:
        out = pd.concat(results, ignore_index=True)

    drop_cols = [c for c in ["width_m", "height_m", "ID_rank"] if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    if "patient_id" in out.columns:
        out = out.sort_values(["patient_id", "height", "width"], kind="mergesort")
    else:
        out = out.sort_values(["height", "width"], kind="mergesort")

    out = out.reset_index(drop=True)
    return out