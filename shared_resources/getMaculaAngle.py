"""Macula sector angle definitions for ETDRS and radial grid analysis.

Provides ``get_macula_angle``, which returns a DataFrame of sector boundaries
(angle1, angle2, sector label) used by ``sectorAverage`` to assign each pixel
to its corresponding macula sector.  Supports total (0), 4-, 6-, and N-sector
grid types.

Laterality must be compensated by the caller before passing ``angle_offset``.

Author: Marco Miranda
Date: 28 May 2026
"""
import pandas as pd
import math

def get_macula_angle(gridtype, angle_offset=0, unit="deg"):
    """
    This function assumes that laterality has already been compensated for.
    It calculates macula (ETDRS for retina and sector grid for GCL+) angles and sectors.

    Args:
        gridtype (int): The type of grid (0, 4, 6).
        angle_offset (float): The amount to rotate the angle by. Defaults to 0.
        unit (str): The unit of the angle, either "deg" (degrees) or "rad" (radians).
                    Defaults to "deg".

    Returns:
        pd.DataFrame: A DataFrame with angle1, angle2, and sector columns.
    """
    if not isinstance(angle_offset, (int, float)):
        raise ValueError("angle_offset can only be a single number.")
    
    # Initialize angles and sector
    angles = []
    sector = []

    if gridtype == 6:
        angles = [0, 60, 120, 180, 240, 300, 360]  # NSTIN
        sector = ["NS", "S", "TS", "TI", "I", "NI"]
    elif gridtype == 0:
        angles = [0, 360]
        sector = ["Total"]
    else:
        section = 360 / gridtype
        partitions = range(gridtype + 1)
        angles = [(p * section) - (section / 2) for p in partitions]
        
        # Compensate for negative angles by adding 360
        angles = [angle + 360 if angle < 0 else angle for angle in angles]
    
    if gridtype == 4:
        sector = ["N", "S", "T", "I"]
    elif gridtype != 6 and gridtype != 0:
        sector = list(range(1, gridtype + 1))
        
    if unit == "rad":
        angles = [math.pi * a / 180 for a in angles]
        
    angles_df = pd.DataFrame({
        "angle1": [a + angle_offset for a in angles[:-1]],
        "angle2": [a + angle_offset for a in angles[1:]],
        "sector": sector
    })
    
    return angles_df

# Example Usage
# angles_6 = get_macula_angle(gridtype=6)
# print(angles_6)

# angles_12 = getcp_macula_angle(gridtype=12, angle_offset=5)
# print(angles_12)