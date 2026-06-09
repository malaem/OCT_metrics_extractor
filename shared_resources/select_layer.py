
"""Retinal layer boundary selector for OCT segmentation data.

Provides the ``SEG_LAYER`` enum (which enumerates the segmented boundary
surfaces stored in Topcon FDA files) and ``selectLayer``, which maps a
human-readable layer name (e.g. ``'RNFL'``, ``'Retina'``, ``'GCL+'``) to the
corresponding pair of boundary indices used to compute thickness.
Author: Marco Miranda
Date: 28 May 2026"""
from enum import Enum

class SEG_LAYER(Enum):
    """Indices of segmented boundary surfaces in the Topcon FDA format.

    Values correspond to the surface ordering stored in the FDA
    segmentation block.  Layer thickness is computed as the pixel
    distance between two surfaces: ``surface2.value - surface1.value``.
    """
    ILM = 0
    NFL = 1
    GCL = 2
    IPL = 3
    ISOS = 4
    RPE = 5
    BM = 6
    INL = 7
    ELM = 8

def selectLayer(layer):
    """Return the pair of ``SEG_LAYER`` boundaries that define *layer*.

    Parameters
    ----------
    layer : str
        Layer name.  Supported values:
        ``'RNFL'``, ``'Retina'``, ``'GCL'``, ``'IPL'``, ``'GCL+'``, ``'GCL++'``.

    Returns
    -------
    (SEG_LAYER, SEG_LAYER)
        ``(inner_boundary, outer_boundary)`` — the two surfaces between which
        thickness is measured (outer minus inner pixel row index).
    """

    if layer == "RNFL":
        ## RNFL
        seg_layer1 = SEG_LAYER.ILM
        seg_layer2 = SEG_LAYER.NFL
    elif layer == "Retina":
        ##Retina
        seg_layer1 = SEG_LAYER.ILM
        seg_layer2 = SEG_LAYER.RPE
    elif layer == "GCL":
        ## GCL
        seg_layer1 = SEG_LAYER.NFL
        seg_layer2 = SEG_LAYER.GCL
    elif layer == "IPL":
        ## IPL
        seg_layer1 = SEG_LAYER.GCL
        seg_layer2 = SEG_LAYER.IPL
    elif layer == "GCL+":
        ## GCL+
        seg_layer1 = SEG_LAYER.NFL
        seg_layer2 = SEG_LAYER.IPL
    elif layer == "GCL++":
        ## GCL++
        seg_layer1 = SEG_LAYER.ILM
        seg_layer2 = SEG_LAYER.IPL
    else:
        raise ValueError(
            f"Unknown layer: '{layer}'. "
            f"Valid options: 'RNFL', 'Retina', 'GCL', 'IPL', 'GCL+', 'GCL++'"
        )
    # end of modification

    return seg_layer1, seg_layer2