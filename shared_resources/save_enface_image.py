"""Save OCT en face images as PNG files with correct proportions.

Provides ``save_enface_image``, which accepts a 2-D NumPy array, normalises
intensity, optionally applies a matplotlib colormap, scales the height axis
to a target resolution, and writes the result as a PNG.
"""
import numpy as np
import matplotlib.cm as cm
from PIL import Image


def save_enface_image(enface, out_path, target_height_px: int = 384, cmap: str = "gray", normalize: bool = True):
    """
    Save an en face OCT image as a PNG with correct proportions.

    Input arrays are (128, 512) after transposing — 128 rows (depth), 512 cols (width).
    Axis 0 is scaled up to target_height_px (never shrunk); axis 1 is kept unchanged.
    Output PNG: 512 px wide × target_height_px px tall.

    Parameters
    ----------
    enface : np.ndarray
        En face image, shape (128, 512).
    out_path : Path
        Full output path including filename (e.g. .png).
    target_height_px : int
        Minimum output height in pixels. If the input row dimension is already
        larger, the native size is kept (the image is never shrunk).
    cmap : str
        Colormap name (default: 'gray'). Any matplotlib colormap name is accepted.
    normalize : bool
        Normalize image to [0, 1] before saving.
    """
    img = np.nan_to_num(enface.astype(np.float32), nan=0.0)

    if normalize:
        mn, mx = img.min(), img.max()
        if mx > mn:
            img = (img - mn) / (mx - mn)
        else:
            img = np.zeros_like(img)

    # Scale axis 0 up to target_height_px; never shrink if already larger.
    out_h = max(img.shape[0], target_height_px)
    out_w = img.shape[1]

    if cmap == "gray":
        pil_img = Image.fromarray((img * 255).clip(0, 255).astype(np.uint8), mode="L")
    else:
        # Apply matplotlib colormap → RGBA float [0,1] → RGB uint8
        rgba = cm.get_cmap(cmap)(img)          # shape (H, W, 4)
        rgb = (rgba[..., :3] * 255).astype(np.uint8)
        pil_img = Image.fromarray(rgb, mode="RGB")

    # Resize using BILINEAR (PIL uses (width, height) order)
    if (out_h, out_w) != img.shape:
        pil_img = pil_img.resize((out_w, out_h), Image.BILINEAR)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(out_path)