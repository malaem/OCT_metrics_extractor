"""build_enface.py — OCT en face image construction from 3-D volumes.

Provides ``compute_enface_slab``, which projects an (Z, X, B) OCT volume
between two segmentation layer boundaries into a 2-D (X, B) en face image.
Six projection modes are supported: ``mean``, ``sum``, ``max``, ``median``,
``registration`` (a low-percentile shadow-enhanced mode for feature matching),
and ``topcon_plane`` (single-depth plane sampled at layer-1 z-positions).

All projections operate on float32 arrays and propagate NaN for samples
outside the segmentation slab.  ``_contrast_stretch`` is a helper for the
``registration`` and ``topcon_plane`` modes.
"""

import numpy as np
from typing import Literal

def _contrast_stretch(img: np.ndarray, p_low: float = 2, p_high: float = 98) -> np.ndarray:
    """Robust contrast stretch to [0,1] using percentiles."""
    img = img.astype(np.float32, copy=False)
    finite = np.isfinite(img)
    if not np.any(finite):
        return np.zeros_like(img, dtype=np.float32)

    lo, hi = np.percentile(img[finite], (p_low, p_high))
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)

    out = (img - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def compute_enface_slab(
    oct3d: np.ndarray,
    seg_data: np.ndarray,
    layer1: int,
    layer2: int,
    layer1_offset: int = 0,   # pixels; can be +/- 
    layer2_offset: int = 0,   # pixels; can be +/- 
    projection: Literal["mean", "sum", "max", "median", "registration"] = "sum",
) -> np.ndarray:
    """
    Compute an en face OCT image by projecting the OCT volume between two segmentation layers.

    Parameters
    ----------
    oct3d : np.ndarray, shape (Z, X, B)
        Raw OCT voxel intensities (any numeric dtype; cast to float32 internally).
    seg_data : np.ndarray, shape (X, B, n_layers)
        Segmentation layer depths in voxel units (float or int).  Layer indices
        ``layer1`` and ``layer2`` select the slab boundaries.
    layer1, layer2 : int
        Layer indices into the last axis of ``seg_data``.  ``layer1`` also
        serves as the plane depth for the ``topcon_plane`` projection.
    layer1_offset, layer2_offset : int, optional
        Pixel offsets applied to ``z1`` and ``z2`` respectively (positive =
        deeper).  The slab is re-ordered after offsets so that ``z_lo <= z_hi``
        regardless of sign.
    projection : str, optional
        One of the following modes:

        ``'mean'``
            Arithmetic mean of voxels within the slab, scaled by slab thickness
            to avoid bias from thin slabs.
        ``'sum'``
            Sum of voxels within the slab (default; fastest; preserves
            absolute intensity).
        ``'max'``
            Maximum voxel within the slab; -inf outside slab replaced by 0.
        ``'median'``
            Spatial median of voxels within the slab; NaN-safe.
        ``'registration'``
            Low-percentile (10th) projection tuned to emphasise vessel shadows
            and structural edges for KAZE feature matching.  Applies a 2–98
            percentile contrast stretch.  When ``layer1 == 0`` the image is
            contrast-inverted (bright vessels on dark background).
        ``'topcon_plane'``
            Samples the OCT volume at a single depth per (x, b) position given
            by ``round(z1)`` (the ``layer1`` surface offset).  Emulates the
            IMAGEnet ``'cut with a plane'`` display mode.  Note: ``z1`` has
            shape ``(X, B)``; using ``z1[0]`` (a bug in earlier versions) would
            sample the depth of the *first scan line* for every column, which
            incorrectly flattens spatial variation across B-scans.

    Returns
    -------
    enface : np.ndarray, shape (X, B), dtype float32
        The projected en face image.  Values are in the native
        intensity scale of *oct3d* for all modes except ``'registration'``
        (which is contrast-stretched to [0, 1]).

    Raises
    ------
    ValueError
        If ``oct3d`` or ``seg_data`` have the wrong number of dimensions, if
        their spatial dimensions are inconsistent, or if *projection* is not
        one of the supported values.
    """

    if oct3d.ndim != 3:
        raise ValueError("oct3d must have shape (Z, X, B)")
    if seg_data.ndim != 3:
        raise ValueError("seg_data must have shape (X, B, n_layers)")

    Z, X, B = oct3d.shape
    if seg_data.shape[0] != X or seg_data.shape[1] != B:
        raise ValueError("seg_data and oct3d spatial dimensions do not match")

    # Depth index
    z_idx = np.arange(Z)[:, None, None]  # (Z,1,1)

    # Layer boundaries (keep as float for offsets)
    z1 = seg_data[:, :, layer1].astype(np.float32) + float(layer1_offset)  # (X,B)
    z2 = seg_data[:, :, layer2].astype(np.float32) + float(layer2_offset)  # (X,B)

    # Enforce ordering (important if offsets can invert the slab)
    z_lo = np.minimum(z1, z2)
    z_hi = np.maximum(z1, z2)

    # Clip to volume bounds
    finite = np.isfinite(z_lo) & np.isfinite(z_hi)
    z_lo = np.clip(z_lo, 0, Z - 1)
    z_hi = np.clip(z_hi, 0, Z)

    # Build slab mask (Z,X,B)
    mask = finite[None, :, :] & (z_idx >= z_lo[None, :, :]) & (z_idx < z_hi[None, :, :])

    if projection == "mean":
        masked = np.where(mask, oct3d.astype(np.float32), 0.0)
        thickness = np.maximum((z_hi - z_lo), 1.0)  # (X,B)
        enface = masked.sum(axis=0) / thickness

    elif projection == "sum":
        enface = np.where(mask, oct3d.astype(np.float32), 0.0).sum(axis=0)

    elif projection == "max":
        # Use -inf outside slab; then replace -inf with 0
        m = np.where(mask, oct3d.astype(np.float32), -np.inf)
        enface = np.max(m, axis=0)
        enface[~np.isfinite(enface)] = 0.0

    elif projection == "median":
        m = np.where(mask, oct3d.astype(np.float32), np.nan)
        # Avoid warnings by only computing where slab exists
        enface = np.zeros((X, B), dtype=np.float32)
        valid = np.any(mask, axis=0)
        if np.any(valid):
            m2 = m.reshape(Z, -1)
            valid_flat = valid.reshape(-1)
            mv = m2[:, valid_flat]
            good = ~np.all(np.isnan(mv), axis=0)
            if np.any(good):
                vals = np.nanmedian(mv[:, good], axis=0).astype(np.float32)
                tmp = np.zeros((valid_flat.sum(),), dtype=np.float32)
                tmp[good] = np.nan_to_num(vals, nan=0.0)
                out = np.zeros((X * B,), dtype=np.float32)
                out[valid_flat] = tmp
                enface = out.reshape(X, B)

    elif projection == "registration":
        # Registration-friendly enface:
        # - thin slab is achieved by caller using layer1==layer2 (ILM) and layer2_offset=+20
        # - use low percentile to emphasize vessel shadows/edges
        oct_f = oct3d.astype(np.float32)
        m = np.where(mask, oct_f, np.nan)

        enface = np.zeros((X, B), dtype=np.float32)
        valid = np.any(mask, axis=0)  # columns where slab exists

        if np.any(valid):
            m2 = m.reshape(Z, -1)
            valid_flat = valid.reshape(-1)
            mv = m2[:, valid_flat]                # (Z, Nvalid)
            good = ~np.all(np.isnan(mv), axis=0)  # extra guard

            if np.any(good):
                vals = np.nanpercentile(mv[:, good], q=10, axis=0).astype(np.float32)
                tmp = np.zeros((valid_flat.sum(),), dtype=np.float32)
                tmp[good] = np.nan_to_num(vals, nan=0.0)

                out = np.zeros((X * B,), dtype=np.float32)
                out[valid_flat] = tmp
                enface = out.reshape(X, B)

        # Mild contrast stretch for registration robustness
        enface = _contrast_stretch(enface, p_low=2, p_high=98)

        if layer1 == 0:
            enface = 1.0 - enface

    elif projection == "topcon_plane":
        # emulate IMAGEnet "cut with a plane" using layer1 as the plane height
        # z1 has shape (X, B); z1[0] would be (B,) — only the first scan line's
        # depths — causing every x to use x=0's z-coordinate.  Use z1 directly.
        z0 = np.round(z1).astype(np.int32)  # (X, B)
        z0 = np.clip(z0, 0, Z-1)
        # sample oct3d at z0 for each (x,b): z0 (X,B), arange(X)[:,None] (X,1),
        # arange(B)[None,:] (1,B) all broadcast to (X,B) as required.
        enface = oct3d[z0, np.arange(X)[:, None], np.arange(B)[None, :]].astype(np.float32)
        
        # Apply contrast stretch to reduce speckle noise
        #enface = _contrast_stretch(enface, p_low=2, p_high=98)

    else:
        raise ValueError(f"Unknown projection type: {projection}")

    return enface