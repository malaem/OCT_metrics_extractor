
"""Sector-wise thickness averaging for OCT retinal layer analysis.

Provides ``sectorAverage``, which assigns each pixel (A-scan) to a
angular sector based on its polar angle, then computes the mean thickness
per sector.  Mirrors the behaviour of the R ``sectorAverage()`` function.

Supports disc (cpRNFL) and macula area types, scalar or per-row angle
offsets, and an optional polar diagnostic plot.  For the special
1024-sector case a TSNIT smoothing pass is applied after averaging.

Author: Marco Miranda
Date: 28 May 2026
"""
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from typing import Union
from shared_resources.getcpRNFLAngle import get_cprnfl_angle
from shared_resources.getMaculaAngle import get_macula_angle

def sectorAverage(
    df: pd.DataFrame,
    gridtype,
    angleOffset: Union[int, float, np.ndarray, pd.Series, list] = 0,
    area: str = "disc",
    unit: str = "deg",
    plot: str = "no",
    behaviour: str = "data_extractor",
) -> pd.DataFrame:
    """
    Adapted to mirror the R sectorAverage() behavior.

    Input expectations:
    - df must contain thickness and angle columns.
    - If multiple patients are present, include patient_id.

    Parameters:
    - gridtype: number of sectors (for example 4, 6, 12 for cpRNFL; 4 for ETDRS).
    - angleOffset: angle offset in degrees. Scalar or per-row vector.
    - area: 'disc' (default) for cpRNFL, or 'macula'.
    - unit: 'deg' (default) or 'rad'.
    - plot: 'yes' or 'no' (default).
        - behaviour: 'data_extractor' (default) or 'ImageNET'.
            data_extractor ignores negative thickness values.
            ImageNET sets negative thickness values to zero.

    Output:
    - DataFrame with columns: patient_id (unless injected), sector,
      avg_thickness, count.

    Implementation notes:
    - Negative thickness values are clamped to zero before averaging.
    - If patient_id is missing, a temporary patient_id = 1 is injected and
      removed before return.
    - If gridtype == 1024, tsnit_smooth is applied per patient to avg_thickness,
      matching the R special case.
    """

    if area not in {"disc", "macula"}:
        raise ValueError("area can only be 'disc' or 'macula'")

    behaviour_key = str(behaviour).lower()
    if behaviour_key not in {"data_extractor", "imagenet"}:
        raise ValueError("behaviour can only be 'data_extractor' or 'ImageNET'")

    df = pd.DataFrame(df).copy()

    # Basic checks
    lower_cols = {c.lower(): c for c in df.columns}
    if 'thickness' not in lower_cols or 'angle' not in lower_cols:
        raise ValueError("df must contain 'thickness' and 'angle' columns (case-insensitive).")

    thickness_col = lower_cols['thickness']
    angle_col = lower_cols['angle']

    # Compute angles table and/or apply per-row angleOffset
    # Scalar vs per-row offset logic (as in R)
    if np.isscalar(angleOffset) and area == "disc":
        # Scalar: angles are computed with the scalar offset
        angles = get_cprnfl_angle(gridtype, float(angleOffset), unit)
    elif np.isscalar(angleOffset) and area == "macula":
        angles = get_macula_angle(gridtype, float(angleOffset), unit)
    else:
        # Vector-like: compute angles with 0 offset and shift df['angle'] per-row
        if area == "disc":
            angles = get_cprnfl_angle(gridtype, 0.0, unit)
        elif area == "macula":
            angles = get_macula_angle(gridtype, 0.0, unit)
        off = np.asarray(angleOffset)
        if off.shape[0] != len(df):
            raise ValueError("When angleOffset is array-like, its length must match df.")
        # Wrap a + offset into [0, 360)
        shifted = (df[angle_col].to_numpy(dtype=float) + off) % 360.0
        shifted = np.where(shifted < 0, shifted + 360.0, shifted)
        df[angle_col] = shifted
        # angleOffset variable 'removed' in spirit; nothing to do in Python.


    # Validate angles table
    required_angle_cols = {'sector', 'angle1', 'angle2'}
    missing = required_angle_cols - set(c.lower() for c in angles.columns)
    if missing:
        raise ValueError(
            f"angles DataFrame returned by getcpRNFLAngle must contain columns {required_angle_cols}."
        )
    # Normalize angle columns to the canonical names/case
    angles = angles.rename(
        columns={c: c.lower() for c in angles.columns}
    )[['sector', 'angle1', 'angle2']].copy()
    # Enforce numeric angles and wrap to [0, 360)
    for col in ['angle1', 'angle2']:
        angles[col] = np.asarray(angles[col], dtype=float) % 360.0

    # Check for patient_id (case-insensitive); if missing, add and remember to drop later
    if 'patient_id' in (c.lower() for c in df.columns):
        print("Will perform analysis per patient, but will not plot the results, even if asked to")
        plot = "no"
        rm_patient_id = "no"
        patient_col = [c for c in df.columns if c.lower() == 'patient_id'][0]
    else:
        print("No column with a patient_id identifier. Assuming there is only one patient in df.")
        df['patient_id'] = 1
        rm_patient_id = "yes"
        patient_col = 'patient_id'

    # Assign sector per-row using angle intervals with wrap-around
    # Vectorized assignment: for each sector interval, fill in matches
    def assign_sector(df_angles: pd.DataFrame, angle_values: np.ndarray) -> pd.Series:
        """Map each pixel angle to its sector label using the pre-computed angle table.

        Supports wrap-around intervals (e.g. 305°–55° for the nasal sector).
        Each pixel is assigned to the first matching sector; unmatched pixels
        remain ``None``.
        """
        n = len(angle_values)
        result = np.array([None] * n, dtype=object)

        angle_values = angle_values.astype(float) % 360.0

        for _, row in df_angles.iterrows():
            a1, a2, sec = float(row['angle1']), float(row['angle2']), row['sector']
            if a1 < a2:
                mask = (angle_values >= a1) & (angle_values < a2)
            else:
                # Wrap-around interval
                mask = (angle_values >= a1) | (angle_values < a2)
            # Only assign where not yet assigned
            unassigned = pd.isna(result)
            result[np.where(mask & unassigned)] = sec

        return pd.Series(result, index=df.index)

    df['sector'] = assign_sector(angles, df[angle_col].to_numpy())
    if behaviour_key == "data_extractor":
        df = df[(df[thickness_col].isna()) | (df[thickness_col] >= 0)].copy()
    else:
        df.loc[df[thickness_col] < 0, thickness_col] = 0

    # Group and summarize
    grouped = (
        df.groupby([patient_col, 'sector'], dropna=False, as_index=False)
          .agg(avg_thickness=(thickness_col, lambda x: np.nanmean(x.astype(float))),
               count=(thickness_col, 'size'))
    )

    averageGrid = grouped.copy()

    if gridtype == 1024:
        # NOTE: gridtype=1024 (TSNIT smoothing) is NOT fully implemented.
        # The tsnit_smooth module does not exist in this repository and the
        # algorithm has never been tested end-to-end here. Agreement with the
        # reference C++ implementation was never validated. Do not use this
        # gridtype in production until the module is written and validated.
        # The ImportError below gives a clear message rather than a raw crash.
        try:
            from shared_resources.tsnit_smooth import tsnit_smooth
        except ImportError as exc:
            raise ImportError(
                "gridtype=1024 requires the 'tsnit_smooth' module, which is not available. "
                "Use gridtype 4, 6, 12, or 36 instead."
            ) from exc
        averageGrid["avg_thickness"] = (
            averageGrid.groupby(patient_col, sort=False)["avg_thickness"]
            .transform(lambda s: tsnit_smooth(s.to_numpy(dtype=float)))
        )

    # Optional plot (disabled if multiple patients present)
    #if str(plot).lower() == "yes":
    if isinstance(plot, str) and plot.lower() == "yes":
        # Lazy optional import so non-plot workflows do not require matplotlib.
        import importlib
        plt = importlib.import_module("matplotlib.pyplot")

        # Join angles to averageGrid to get angle1 and angle2 per sector
        avg_plot = averageGrid.merge(angles, on='sector', how='left')

        # Expand sectors to per-degree rows (integer degrees)
        def expand_sector(angle1, angle2, sector, avg_thickness):
            """Return a per-degree DataFrame row for each integer degree in the sector arc.

            Used only for the diagnostic polar plot.  Handles wrap-around arcs
            (e.g. 305°–55°) by splitting into two ranges before concatenating.
            """
            angle1 = float(angle1) % 360.0
            angle2 = float(angle2) % 360.0

            if angle1 < angle2:
                degrees = np.arange(np.ceil(angle1), np.floor(angle2), 1.0)
            else:
                part1 = np.arange(np.ceil(angle1), 360.0, 1.0)
                part2 = np.arange(0.0, np.floor(angle2), 1.0)
                degrees = np.concatenate([part1, part2])

            df_out = pd.DataFrame({
                'angle': degrees.astype(float),
                'sector': sector,
                'avg_thickness': avg_thickness
            })
            return df_out

        expanded_frames = []
        for _, r in avg_plot.iterrows():
            if pd.isna(r['angle1']) or pd.isna(r['angle2']):
                continue
            expanded_frames.append(
                expand_sector(r['angle1'], r['angle2'], r['sector'], r['avg_thickness'])
            )

        if len(expanded_frames) > 0:
            plot_df = pd.concat(expanded_frames, ignore_index=True)
            # Flip: angle_flipped = (180 - angle) % 360
            plot_df['angle_flipped'] = (180.0 - plot_df['angle']) % 360.0

            # Compute label positions: mid-angle per sector
            label_df = avg_plot.dropna(subset=['angle1', 'angle2']).copy()
            def mid_angle(a1, a2):
                """Compute the mid-point angle of a sector arc, handling wrap-around."""
                a1 = float(a1) % 360.0
                a2 = float(a2) % 360.0
                if a1 < a2:
                    m = (a1 + a2) / 2.0
                else:
                    m = ((a1 + a2 + 360.0) / 2.0) % 360.0
                return m

            label_df['mid_angle'] = label_df.apply(lambda r: mid_angle(r['angle1'], r['angle2']), axis=1)
            label_df['x_deg'] = (180.0 - label_df['mid_angle']) % 360.0
            label_df['y'] = 1.2

            # --- Plot (polar) ---
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(7, 7))

            # For each sector, draw bars across its degrees
            # Convert degrees to radians; width = 1 degree in radians
            width = np.deg2rad(1.0)
            sectors = plot_df['sector'].unique()
            # Basic color cycle by sector
            color_map = {sec: plt.cm.tab20(i % 20) for i, sec in enumerate(sectors)}

            for sec in sectors:
                sub = plot_df[plot_df['sector'] == sec]
                theta = np.deg2rad(sub['angle_flipped'].to_numpy())
                ax.bar(theta, np.ones_like(theta), width=width, bottom=0.0,
                       color=color_map.get(sec, 'C0'), edgecolor='none', label=str(sec))

            # Add labels (avg_thickness rounded) at the sector mid angles
            for _, r in label_df.iterrows():
                theta_label = np.deg2rad(r['x_deg'])
                ax.text(theta_label, r['y'], f"{np.round(r['avg_thickness'], 0):.0f}",
                        ha='center', va='center', fontsize=10)

            # Match ggplot-like polar orientation (start=-pi/2): 0° at top
            ax.set_theta_zero_location('N')  # 0° at North (top)
            ax.set_theta_direction(-1)       # clockwise

            ax.set_yticklabels([])  # theme_void-like
            ax.set_xticklabels([])
            ax.set_title("Average Thickness by Sector (Flipped)")
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
            plt.tight_layout()
            plt.show()

            # Remove angle columns from return, as in R after plotting
            # (only if we actually merged them)
            averageGrid = averageGrid  # no need to keep angle1/angle2 here

    # Order sectors as per angles.sector and sort rows
    sector_order = list(angles['sector'])
    averageGrid['sector'] = pd.Categorical(averageGrid['sector'], categories=sector_order, ordered=True)
    averageGrid = averageGrid.sort_values(by=[patient_col, 'sector']).reset_index(drop=True)

    # Drop patient_id if we created it
    if rm_patient_id == "yes":
        averageGrid = averageGrid.drop(columns=[patient_col])

    return averageGrid