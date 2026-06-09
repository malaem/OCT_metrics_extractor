"""Sector-based retinal layer thickness metrics computation.

Supports multiple retinal layers and grid types:
- cpRNFL   : disc-centred 3.4 mm annulus; 4/6/12/36-sector and total-average grids
- GCL+ / GCL++ : fovea-centred 6-sector radial macula grid
- Retina   : ETDRS 9-zone macula grid (centre + 4 inner + 4 outer zones)

Internally uses grid_diameter for annular pixel selection and sectorAverage for
sector-wise thickness averaging.

Author: Marco Miranda
Date: 28 May 2026
"""

import pandas as pd
import numpy as np
import os
import contextlib
from typing import Dict, Any, List

from shared_resources.grid_diameter import grid_diameter
from shared_resources.sectorAverage import sectorAverage
from shared_resources.maryfdaQ import maryfdaQ


def _compute_sector_averages_for_gridtype(
    layerthick: pd.DataFrame,
    gridtype: int,
    behaviour: str = 'data_extractor'
) -> pd.DataFrame:
    """Compute sector-averaged thickness for a single gridtype.
    
    Wrapper around sectorAverage for disc-centered RNFL analysis.
    
    Parameters
    ----------
    layerthick : pd.DataFrame
        DataFrame with columns: thickness, angle (polar coordinates).
        Must already have pixels selected by grid_diameter.
    gridtype : int
        Number of sectors: 4, 6, 12, or 36.
    behaviour : str, default 'data_extractor'
        How to handle negative thickness values:
        - 'data_extractor': exclude negatives (set to np.nan)
        - 'imageNET': include negatives in averages
    
    Returns
    -------
    pd.DataFrame
        Sector-averaged results with columns: sector, avg_thickness.
    
    Notes
    -----
    - Uses angleOffset=0 (no rotation)
    - area="disc" (disc-centered analysis)
    - unit="deg" (angles in degrees)
    - Suppresses plot output
    """
    return sectorAverage(
        df=layerthick,
        gridtype=gridtype,
        angleOffset=0,
        area="disc",
        unit="deg",
        plot="no",
        behaviour=behaviour
    )


def compute_all_sector_metrics(
    scan_data: Dict[str, Any],
    layer_name: str = 'cpRNFL',
    behaviour: str = 'data_extractor'
) -> pd.DataFrame:
    """Compute layer-appropriate sector-averaged metrics for a single scan.
    
    Dispatches to appropriate grid function based on layer type:
    - cpRNFL: Disc-centered 3.4mm annulus with 4/6/12/36 sectors
    - GCL+/GCL++: Macula-centered 6-sector radial grid
    - Retina: Macula-centered ETDRS 9-zone grid
    
    Parameters
    ----------
    scan_data : dict
        Dictionary from read_fda_scan_for_wide_rnfl() containing layer data.
    layer_name : str, default 'cpRNFL'
        Layer to analyze: 'cpRNFL', 'GCL+', 'GCL++', or 'Retina'.
    behaviour : str, default 'data_extractor'
        How to handle negative thickness values:
        - 'data_extractor': exclude negatives (set to np.nan)
        - 'imageNET': include negatives in averages
    
    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with layer-specific columns.
        
        cpRNFL: Total, 4_T..4_I, 6_T..6_TI, 12_T..12_TI, 36_01..36_36
        GCL+/GCL++: Total, TS, S, NS, NI, I, TI
        Retina: ETDRS_Center, ETDRS_In_T/S/N/I, ETDRS_Out_T/S/N/I, 
                average_thick, center_thick, total_vol
        
        All values in micrometers (µm), except total_vol in mm³.
    
    Raises
    ------
    ValueError
        If required keys are missing from scan_data or layer_name is invalid.
    
    Notes
    -----
    - For backward compatibility, default layer_name='cpRNFL' preserves 
      original behavior for single-layer cpRNFL analysis.
    """
    # Get grid type for layer
    grid_type = _get_grid_type_for_layer(layer_name)
    
    # Dispatch to appropriate grid function
    if grid_type == 'disc':
        return _compute_disc_sectors(scan_data, layer_name, behaviour)
    elif grid_type == 'macula_6':
        return _compute_macula_6_sectors(scan_data, layer_name, behaviour)
    elif grid_type == 'macula_etdrs':
        return _compute_macula_etdrs(scan_data, behaviour)
    else:
        raise ValueError(f"Unknown grid type: {grid_type}")


def _compute_disc_sectors(scan_data: Dict[str, Any], layer_name: str = 'cpRNFL', behaviour: str = 'data_extractor') -> pd.DataFrame:
    """Compute disc-centered cpRNFL sector metrics (internal function).
    
    Applies grid_diameter to select 3.4mm annulus around optic disc, then
    computes sector averages for gridtypes 4, 6, 12, and 36, plus total average.
    
    Parameters
    ----------
    scan_data : dict
        Dictionary from read_fda_scan_for_wide_rnfl() containing layer data.
    layer_name : str, default 'cpRNFL'
        Layer name (for accessing correct thickness data from layers dict).
    behaviour : str, default 'data_extractor'
        How to handle negative thickness values:
        - 'data_extractor': exclude negatives (set to np.nan)
        - 'imageNET': include negatives in averages
    
    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with columns:
        - Total : float, overall average thickness in annulus
        - 4_T, 4_S, 4_N, 4_I : sector averages for 4-sector grid
        - 6_T, 6_TS, 6_NS, 6_N, 6_NI, 6_TI : sector averages for 6-sector grid
        - 12_T, 12_TS, ..., 12_TI : sector averages for 12-sector grid
        - 36_01, 36_02, ..., 36_36 : sector averages for 36-sector grid
        
        All values in micrometers (µm).
    
    Raises
    ------
    ValueError
        If required keys are missing from scan_data.
    
    Notes
    -----
    - Uses 3.4mm diameter annulus (standard cpRNFL measurement)
    - Tolerance: 1 pixel
    - Laterality: native eye orientation from scan metadata
    - Suppresses console output from grid_diameter and sectorAverage
    
    Processing steps:
    1. Create DataFrame from 1D thickness array with width/height columns
    2. Apply grid_diameter to select annulus pixels and compute polar angles
    3. Compute total average within annulus
    4. For each gridtype (4, 6, 12, 36):
       - Compute sector averages
       - Rename sectors with gridtype prefix (e.g., "4_T", "12_NS")
    5. Combine all metrics into single-row DataFrame
    """
    # Validate required keys
    required_base_keys = {
        'scan_size_set', 'disc_center_x', 'disc_center_y', 'scan_axial_res',
        'jpeg_height', 'littmann_mag'
    }
    missing_keys = required_base_keys - set(scan_data.keys())
    if missing_keys:
        raise ValueError(f"Missing required keys in scan_data: {missing_keys}")
    
    # Extract thickness data
    if 'layers' not in scan_data or layer_name not in scan_data['layers']:
        raise ValueError(f"Layer '{layer_name}' not found in scan_data['layers']")
    thickness = scan_data['layers'][layer_name]['thickness']
    scan_res_width = scan_data['layers'][layer_name]['thickness_width']
    scan_res_height = scan_data['layers'][layer_name]['thickness_height']
    
    # Extract common parameters
    scan_size = scan_data['scan_size_set']  # (width_mm, height_mm) tuple for grid calculations
    disc_center = [scan_data['disc_center_x'], scan_data['disc_center_y']]
    img_height = scan_data['jpeg_height']
    axial_res = scan_data['scan_axial_res']
    littman_mag = scan_data['littmann_mag']
    # Use native-eye laterality from scan metadata.
    laterality = scan_data.get('eye', 'R')
    
    # Create DataFrame with thickness, width, height columns
    layerthick = pd.DataFrame({
        'thickness': thickness,
        'width': np.tile(np.arange(scan_res_width), scan_res_height),
        'height': np.repeat(np.arange(scan_res_height), scan_res_width)
    })
    
    # Suppress console output for grid operations
    with open(os.devnull, 'w') as _devnull, contextlib.redirect_stdout(_devnull):
        # Apply grid_diameter to select 3.4mm annulus and compute angles
        layerthick = grid_diameter(
            df=layerthick,
            diameter=3.4,
            tolerance=1,
            tolerance_unit="px",
            scan_width_px=scan_res_width,
            scan_height_px=scan_res_height,
            scan_width_mm=scan_size[0],
            scan_height_mm=scan_size[1],
            centre_x=disc_center[0],
            centre_y=disc_center[1],
            laterality=laterality,
            imgHeight=img_height,
            zResolution=axial_res,
            littmanMagnification=littman_mag
        )
        
        # Compute total average within annulus
        thickness_values = layerthick['thickness'].copy()
        if behaviour == 'data_extractor':
            # Negatives should be IGNORED (converted to NaN, not counted)
            thickness_values[thickness_values < 0] = np.nan
            total_avg = np.nanmean(thickness_values)
        else:  # imageNET
            # Negatives should be treated as ZERO (counted in average)
            thickness_values[thickness_values < 0] = 0
            total_avg = np.nanmean(thickness_values)
        
        # Initialize results dictionary
        metrics = {'Total': total_avg}
        
        # Compute sector averages for each gridtype
        grids = {
            4: _compute_sector_averages_for_gridtype(layerthick, 4, behaviour),
            6: _compute_sector_averages_for_gridtype(layerthick, 6, behaviour),
            12: _compute_sector_averages_for_gridtype(layerthick, 12, behaviour),
            36: _compute_sector_averages_for_gridtype(layerthick, 36, behaviour)
        }
        
        # Reformat sector labels and add to metrics dictionary
        for gridtype, sector_df in grids.items():
            for _, row in sector_df.iterrows():
                # Get sector identifier
                sector_id = row['sector']
                
                # Format numeric sectors with zero-padding (e.g., "01", "02")
                try:
                    sector_id = int(sector_id)
                    sector_id = f"{sector_id:02d}"
                except (ValueError, TypeError):
                    # Keep string sectors as-is (e.g., "T", "NS")
                    pass
                
                # Create column name: <gridtype>_<sector>
                col_name = f"{gridtype}_{sector_id}"
                metrics[col_name] = row['avg_thickness']
    
    # Convert to single-row DataFrame
    df_result = pd.DataFrame([metrics])
    
    # Reorder columns to match CSV header expectations (T, S, N, I order, not alphabetical)
    # This prevents column/value mismatch when writing to pre-initialized CSV files
    expected_order = ['Total']
    # 4-sector: T, S, N, I (not alphabetical)
    expected_order.extend([f'4_{s}' for s in ['T', 'S', 'N', 'I']])
    # 6-sector: T, TS, NS, N, NI, TI
    expected_order.extend([f'6_{s}' for s in ['T', 'TS', 'NS', 'N', 'NI', 'TI']])
    # 12-sector: T, TS, ST, S, SN, NS, N, NI, IN, I, IT, TI
    expected_order.extend([f'12_{s}' for s in ['T', 'TS', 'ST', 'S', 'SN', 'NS', 'N', 'NI', 'IN', 'I', 'IT', 'TI']])
    # 36-sector: 01-36
    expected_order.extend([f'36_{i:02d}' for i in range(1, 37)])
    
    # Reorder only the columns that exist in the DataFrame
    cols_present = [col for col in expected_order if col in df_result.columns]
    other_cols = [col for col in df_result.columns if col not in expected_order]
    df_result = df_result[cols_present + other_cols]
    
    return df_result


def compute_quality_score(
    scan_data: Dict[str, Any],
    all_sector_metrics: Dict[str, pd.DataFrame],
    layers_to_extract: List[str]
) -> str:
    """Compute Mary's Quality (MaryQ) score for scan based on available layers.
    
    Quality scoring logic:
    - For Wide scans: Checks ALL selected layers (cpRNFL, GCL+, GCL++, Retina)
    - For Disc scans: Checks cpRNFL only (if selected)
    - For Macula scans: Checks Retina only (if selected)
    - Combines all layer metrics into single DataFrame for maryQ
    
    Parameters
    ----------
    scan_data : dict
        Dictionary from read_fda_scan_for_wide_rnfl() containing:
        - fixation, q_mean, scan_size, disc_center_x/y, fovea_x/y,
          disc_area, mirror_pos, f2d_distance, f2d_angle
    all_sector_metrics : Dict[str, pd.DataFrame]
        Dictionary of sector metrics keyed by layer name.
        All metrics combined into single DataFrame for maryQ.
    layers_to_extract : List[str]
        List of layers being extracted: ['cpRNFL', 'GCL+', 'GCL++', 'Retina'].
    
    Returns
    -------
    str or None
        Quality score: "Pass", "Fail", or None (if not computed).
    
    Notes
    -----
    - Returns None if maryfdaQ returns 999999999999999 (not computed).
    - For Wide scans, maryQ checks only the columns from selected layers:
      - cpRNFL → disc sectors
      - GCL+/GCL++ → macula 6 sectors  
      - Retina → ETDRS zones
    - Quality check performed once per scan using combined metrics from all layers.
    """
    # Check if we should run quality scoring
    has_cprnfl = 'cpRNFL' in layers_to_extract
    has_retina = 'Retina' in layers_to_extract
    has_gcl = 'GCL+' in layers_to_extract or 'GCL++' in layers_to_extract
    
    # Run quality check if any layer selected (all layers now support maryQ for Wide scans)
    # For non-Wide scans (Disc/Macula), maryQ will check appropriate columns
    if not (has_cprnfl or has_retina or has_gcl):
        return None  # No layers selected that support quality checks
    
    # Combine all available layer metrics into one single-row DataFrame.
    # Both cpRNFL and macula layers have a 'Total' column. To ensure maryfdaQ
    # sees every value (so a NA in any layer's Total causes a Fail), columns that
    # appear in more than one layer are suffixed with the layer name on second and
    # subsequent occurrences. The first occurrence keeps its original name so that
    # maryfdaQ's column-presence checks (has_cprnfl, has_macula6, etc.) still work.
    combined_dict = {}
    for layer_name in layers_to_extract:
        if layer_name in all_sector_metrics:
            layer_df = all_sector_metrics[layer_name]
            for col in layer_df.columns:
                if col not in combined_dict:
                    combined_dict[col] = layer_df[col].values[0]
                else:
                    # Duplicate: store under a layer-qualified name so the value
                    # is present in the DataFrame and included in the notna() check.
                    combined_dict[f"{col}__{layer_name}"] = layer_df[col].values[0]
    
    # Create single-row DataFrame from combined dict
    combined_metrics = pd.DataFrame([combined_dict])
    
    # Run maryQ once with all combined metrics
    try:
        index_to_keep = maryfdaQ(
            scan_data['fixation'],
            scan_data['q_mean'],
            scan_data['scan_size_set'],   # (width_mm, height_mm) tuple expected by maryfdaQ
            [scan_data['disc_center_x'], scan_data['disc_center_y']],
            [scan_data['fovea_x'], scan_data['fovea_y']],
            scan_data.get('disc_area'),
            combined_metrics,
            scan_data['mirror_pos'],
            scan_data.get('f2d_distance'),
            scan_data.get('f2d_angle')
        )
        
        if index_to_keep == 999999999999999:
            return None  # Quality not computed
        elif isinstance(index_to_keep, list):
            # maryfdaQ returns list of indices that pass
            # We passed a single-row DataFrame (index 0), so check if 0 is in the list
            return "Pass" if 0 in index_to_keep else "Fail"
        else:
            # Fallback for other return types (shouldn't happen, but be safe)
            return "Pass" if bool(index_to_keep) else "Fail"
    
    except (KeyError, ValueError, TypeError) as e:
        # If maryQ fails, return None
        return None


def _get_grid_type_for_layer(layer_name: str) -> str:
    """Return grid type for specified retinal layer.
    
    Parameters
    ----------
    layer_name : str
        Layer name: 'cpRNFL', 'GCL+', 'GCL++', or 'Retina'.
    
    Returns
    -------
    str
        Grid type: 'disc' (cpRNFL), 'macula_6' (GCL+/GCL++), or 'macula_etdrs' (Retina).
    
    Notes
    -----
    - cpRNFL: Disc-centered 3.4mm annulus with 4/6/12/36 sectors
    - GCL+/GCL++: Macula-centered 6-sector radial grid
    - Retina: Macula-centered ETDRS 9-zone grid
    """
    grid_map = {
        'cpRNFL': 'disc',
        'GCL+': 'macula_6',
        'GCL++': 'macula_6',
        'Retina': 'macula_etdrs'
    }
    
    if layer_name not in grid_map:
        raise ValueError(
            f"Unknown layer name: '{layer_name}'. "
            f"Valid options: {list(grid_map.keys())}"
        )
    
    return grid_map[layer_name]


def _compute_macula_6_sectors(scan_data: Dict[str, Any], layer_name: str, behaviour: str = 'data_extractor') -> pd.DataFrame:
    """Compute 6-sector macula metrics for GCL+ or GCL++ layers.
    
    Uses fovea-centered 6-sector radial grid (TS, S, NS, NI, I, TI).
    
    Parameters
    ----------
    scan_data : dict
        Dictionary from read_fda_scan_for_wide_rnfl() containing layer data.
    layer_name : str
        Layer name: 'GCL+' or 'GCL++'.
    behaviour : str, default 'data_extractor'
        How to handle negative thickness values:
        - 'data_extractor': exclude negatives (set to np.nan)
        - 'imageNET': include negatives in averages
    
    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with columns: Total, TS, S, NS, NI, I, TI.
        All values in micrometers (µm).
    
    Notes
    -----
    - Uses sectorAverage with gridtype=6, area="macula"
    - Centered on fovea coordinates
    - Laterality follows native eye orientation
    - Suppresses console output
    """
    # Get layer-specific thickness
    thickness = scan_data['layers'][layer_name]['thickness']
    scan_res_width = scan_data['layers'][layer_name]['thickness_width']
    scan_res_height = scan_data['layers'][layer_name]['thickness_height']
    
    # Create DataFrame with thickness, width, height columns
    layerthick = pd.DataFrame({
        'thickness': thickness,
        'width': np.tile(np.arange(scan_res_width), scan_res_height),
        'height': np.repeat(np.arange(scan_res_height), scan_res_width)
    })
    

    # Extract parameters
    scan_size = scan_data['scan_size_set']  # (width_mm, height_mm) tuple for grid calculations
    fovea_center = [scan_data['fovea_x'], scan_data['fovea_y']]
    img_height = scan_data['jpeg_height']
    axial_res = scan_data['scan_axial_res']
    littman_mag = scan_data['littmann_mag']
    # Use native-eye laterality from scan metadata.
    laterality = scan_data.get('eye', 'R')
    
    # Suppress console output for grid operations
    with open(os.devnull, 'w') as _devnull, contextlib.redirect_stdout(_devnull):
        # GCL+/GCL++ use 3.5mm diameter centered on fovea
        layerthick = grid_diameter(
            df=layerthick,
            diameter=3.5,
            tolerance=1.25,
            tolerance_unit="mm",
            scan_width_px=scan_res_width,
            scan_height_px=scan_res_height,
            scan_width_mm=scan_size[0],
            scan_height_mm=scan_size[1],
            centre_x=fovea_center[0],
            centre_y=fovea_center[1],
            laterality=laterality,
            imgHeight=img_height,
            zResolution=axial_res,
            littmanMagnification=littman_mag
        )
        
        # Compute total average
        thickness_values = layerthick['thickness'].copy()
        if behaviour == 'data_extractor':
            # Negatives should be IGNORED (converted to NaN, not counted)
            thickness_values[thickness_values < 0] = np.nan
            total = np.nanmean(thickness_values)
        else:  # imageNET
            # Negatives should be treated as ZERO (counted in average)
            thickness_values[thickness_values < 0] = 0
            total = np.nanmean(thickness_values)
        
        # Compute 6-sector averages
        sector6 = sectorAverage(
            df=layerthick,
            gridtype=6,
            angleOffset=0,
            area="macula",
            unit="deg",
            plot="no",
            behaviour=behaviour
        )
    
    # Build metrics dictionary
    metrics = {'Total': total}
    
    # Match sectors by name (not by index) to avoid order mismatch
    # Expected CSV column order: TS, S, NS, NI, I, TI
    for _, row in sector6.iterrows():
        sector_name = str(row['sector'])
        metrics[sector_name] = row['avg_thickness']
    
    # Convert to DataFrame and ensure column order matches CSV header
    df_result = pd.DataFrame([metrics])
    expected_order = ['Total', 'TS', 'S', 'NS', 'NI', 'I', 'TI']
    cols_present = [col for col in expected_order if col in df_result.columns]
    other_cols = [col for col in df_result.columns if col not in expected_order]
    df_result = df_result[cols_present + other_cols]
    
    return df_result


def _get_sector_value(sector_df: pd.DataFrame, label) -> float:
    """Look up avg_thickness for a sector by its label (string) or positional index (int).

    sectorAverage returns a DataFrame with a 'sector' column containing string labels
    (e.g. 'N', 'S', 'T', 'I') and a RangeIndex. Looking up by positional integer is
    fragile when a sector has no pixels (groupby produces fewer rows, shifting the index).
    This helper looks up by label when a string is passed, falling back to NaN when the
    sector is absent rather than silently returning a neighbour's value.
    For the 1mm center zone (label=0, single-sector), positional lookup is still used.
    """
    if isinstance(label, str):
        rows = sector_df[sector_df['sector'] == label]
        if len(rows) == 0:
            return np.nan
        return float(rows['avg_thickness'].iloc[0])
    else:
        # Integer label: positional lookup (used for single-sector center zone)
        return sector_df['avg_thickness'].get(label, np.nan)


def _compute_macula_etdrs(scan_data: Dict[str, Any], behaviour: str = 'data_extractor') -> pd.DataFrame:
    """Compute ETDRS 9-zone metrics for Retina (full thickness) layer.
    
    ETDRS grid: Center (1mm), Inner ring (4 sectors, 1-3mm), Outer ring (4 sectors, 3-6mm).
    Plus global averages and total volume.
    
    Parameters
    ----------
    scan_data : dict
        Dictionary from read_fda_scan_for_wide_rnfl() containing Retina layer data.
    behaviour : str, default 'data_extractor'
        How to handle negative thickness values:
        - 'data_extractor': exclude negatives (set to np.nan)
        - 'imageNET': include negatives in averages
    
    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with columns:
        - ETDRS_Center, ETDRS_In_T/S/N/I, ETDRS_Out_T/S/N/I
        - average_thick, center_thick, total_vol
        All thickness values in micrometers (µm), volume in mm³.
    
    Notes
    -----
    - Center: 0.5mm diameter (~1mm diameter zone)
    - Inner ring: 2mm diameter (1-3mm zone), 4 sectors (T, S, N, I)
    - Outer ring: 4.5mm diameter (3-6mm zone), 4 sectors (T, S, N, I)
    - Centered on fovea coordinates
    - Laterality follows native eye orientation
    """
    # Get Retina layer thickness
    thickness = scan_data['layers']['Retina']['thickness']
    scan_res_width = scan_data['layers']['Retina']['thickness_width']
    scan_res_height = scan_data['layers']['Retina']['thickness_height']
    
    # Create DataFrame
    layerthick = pd.DataFrame({
        'thickness': thickness,
        'width': np.tile(np.arange(scan_res_width), scan_res_height),
        'height': np.repeat(np.arange(scan_res_height), scan_res_width)
    })
    
    # Extract parameters
    scan_size = scan_data['scan_size_set']  # (width_mm, height_mm) tuple for grid calculations
    fovea_center = [scan_data['fovea_x'], scan_data['fovea_y']]
    img_height = scan_data['jpeg_height']
    axial_res = scan_data['scan_axial_res']
    littman_mag = scan_data['littmann_mag']
    # Use native-eye laterality from scan metadata.
    laterality = scan_data.get('eye', 'R')
    
    with open(os.devnull, 'w') as _devnull, contextlib.redirect_stdout(_devnull):
        # ---------------------------------------------------------------------------
        # CENTER POINT THICKNESS — NEEDS FURTHER INVESTIGATION
        # Best agreement achieved vs C++ reference (within 1 µm):
        #   Triton: 62%  |  Maestro: 20%
        # Configs tried (all tolerance_unit="mm" unless noted):
        #   d=0.05, tol=0.05  → Triton 62%, Maestro 20%  (best)
        #   d=0.01, tol=0.05  → Triton 40%, Maestro 20%
        #   d=0, tol=1, px    → Triton 24%, Maestro 16%
        #   d=0, tol=3, px    → Triton 62%, Maestro 20%  (same as d=0.05/0.05 mm)
        #   d=0.2, tol=0.1    → Triton 9%,  Maestro 12%
        #   d=0.5, tol=0.25   → Triton 2%,  Maestro 2%
        #   single nearest A-scan (pixel lookup) → worse than all area configs
        # Maestro discrepancy (~20% ceiling) appears device-specific, likely
        # different C++ foveal center convention or layer definition for Maestro.
        # tolerance_unit="mm" is isotropic in physical space (x_ratio cancels).
        # Output set to None until resolved.
        # ---------------------------------------------------------------------------
        # layerCPT = grid_diameter(
        #     df=layerthick,
        #     diameter=0.05,
        #     tolerance=0.05,
        #     tolerance_unit="mm",
        #     scan_width_px=scan_res_width,
        #     scan_height_px=scan_res_height,
        #     scan_width_mm=scan_size[0],
        #     scan_height_mm=scan_size[1],
        #     centre_x=fovea_center[0],
        #     centre_y=fovea_center[1],
        #     laterality='R',
        #     imgHeight=img_height,
        #     zResolution=axial_res,
        #     littmanMagnification=littman_mag
        # )
        # center_point_thickness_values = layerCPT['thickness'].copy()
        # if behaviour == 'data_extractor':
        #     center_point_thickness_values[center_point_thickness_values < 0] = np.nan
        #     center_point_thickness = np.nanmean(center_point_thickness_values)
        # else:
        #     center_point_thickness_values[center_point_thickness_values < 0] = 0
        #     center_point_thickness = np.nanmean(center_point_thickness_values)
        center_point_thickness = None
        
        # Center 0-1mm diameter (0-0.5mm radius): nominal radius=0.25mm, tolerance=0.25mm (centre subfield thickness)
        layer0_1mm = grid_diameter(
            df=layerthick,
            diameter=0.5,
            tolerance=0.25,
            tolerance_unit="mm",
            scan_width_px=scan_res_width,
            scan_height_px=scan_res_height,
            scan_width_mm=scan_size[0],
            scan_height_mm=scan_size[1],
            centre_x=fovea_center[0],
            centre_y=fovea_center[1],
            laterality=laterality,
            imgHeight=img_height,
            zResolution=axial_res,
            littmanMagnification=littman_mag
        )
        center_1mm = sectorAverage(
            df=layer0_1mm,
            gridtype=0,
            angleOffset=0,
            area="macula",
            unit="deg",
            plot="no",
            behaviour=behaviour
        )
        
        # Inner ring 1-3mm diameter (0.5-1.5mm radius): nominal radius=1.0mm, tolerance=0.5mm
        layer1_3mm = grid_diameter(
            df=layerthick,
            diameter=2.0,
            tolerance=0.5,
            tolerance_unit="mm",
            scan_width_px=scan_res_width,
            scan_height_px=scan_res_height,
            scan_width_mm=scan_size[0],
            scan_height_mm=scan_size[1],
            centre_x=fovea_center[0],
            centre_y=fovea_center[1],
            laterality=laterality,
            imgHeight=img_height,
            zResolution=axial_res,
            littmanMagnification=littman_mag
        )
        sector_inner = sectorAverage(
            df=layer1_3mm,
            gridtype=4,
            area="macula",
            angleOffset=0,
            unit="deg",
            plot="no",
            behaviour=behaviour
        )
        
        # Outer ring 3-6mm diameter (1.5-3mm radius): nominal radius=2.25mm, tolerance=0.75mm
        layer3_6mm = grid_diameter(
            df=layerthick,
            diameter=4.5,
            tolerance=0.75,
            tolerance_unit="mm",
            scan_width_px=scan_res_width,
            scan_height_px=scan_res_height,
            scan_width_mm=scan_size[0],
            scan_height_mm=scan_size[1],
            centre_x=fovea_center[0],
            centre_y=fovea_center[1],
            laterality=laterality,
            imgHeight=img_height,
            zResolution=axial_res,
            littmanMagnification=littman_mag
        )
        sector_outer = sectorAverage(
            df=layer3_6mm,
            gridtype=4,
            angleOffset=0,
            area="macula",
            unit="deg",
            plot="no",
            behaviour=behaviour
        )
    
    # Compute global average
    sector_thickness = np.concatenate([
        layer0_1mm['thickness'],
        layer1_3mm['thickness'],
        layer3_6mm['thickness']
    ])
    if behaviour == 'data_extractor':
        # Negatives IGNORED
        sector_thickness_filtered = sector_thickness.copy()
        sector_thickness_filtered[sector_thickness_filtered < 0] = np.nan
        global_avg_thickness = np.nanmean(sector_thickness_filtered)
    else:
        # Negatives treated as ZERO
        sector_thickness_filtered = sector_thickness.copy()
        sector_thickness_filtered[sector_thickness_filtered < 0] = 0
        global_avg_thickness = np.nanmean(sector_thickness_filtered)
    
    # Total volume within ETDRS 0-6mm grid (sum of all three zones)
    # NOTE: Currently using 0-6mm diameter. May change to 0-3mm if requested.
    # TODO (BUG-08): total_vol is computed here but intentionally not output (set to None below).
    # Verify the formula and decide on zone extent (0-6mm vs 0-3mm) before enabling.
    pixel_area = (scan_size[0] / scan_res_width) * (scan_size[1] / scan_res_height)
    total_vol = np.sum(sector_thickness / 1000) * pixel_area
    
    # Build metrics dictionary
    metrics = {
        'ETDRS_Center': _get_sector_value(center_1mm, 0),
        'ETDRS_In_T':   _get_sector_value(sector_inner, 'T'),
        'ETDRS_In_S':   _get_sector_value(sector_inner, 'S'),
        'ETDRS_In_N':   _get_sector_value(sector_inner, 'N'),
        'ETDRS_In_I':   _get_sector_value(sector_inner, 'I'),
        'ETDRS_Out_T':  _get_sector_value(sector_outer, 'T'),
        'ETDRS_Out_S':  _get_sector_value(sector_outer, 'S'),
        'ETDRS_Out_N':  _get_sector_value(sector_outer, 'N'),
        'ETDRS_Out_I':  _get_sector_value(sector_outer, 'I'),
        'average_thick': global_avg_thickness,
        'center_thick': center_point_thickness,  # None — see investigation notes above
        'total_vol': None  # Excluded from output
    }
    
    return pd.DataFrame([metrics])


def prepare_output_row(
    scan_data: Dict[str, Any],
    sector_metrics: pd.DataFrame,
    quality_score: str,
    layer_name: str = 'cpRNFL',
    suffix: str = ""
) -> pd.DataFrame:
    """Prepare single output row combining metadata, sector metrics, and quality.
    
    Combines scan metadata with computed sector metrics into a single-row
    DataFrame ready for CSV output. Optionally adds suffix to column names
    for paired-scan outputs (e.g., "_ref", "_fu").
    
    Parameters
    ----------
    scan_data : dict
        Dictionary from read_fda_scan_for_wide_rnfl().
    sector_metrics : pd.DataFrame
        Single-row DataFrame from compute_all_sector_metrics().
    quality_score : str or None
        Quality score from compute_quality_score().
    layer_name : str, default 'cpRNFL'
        Layer name for Contents column: 'cpRNFL', 'GCL+', 'GCL++', 'Retina'.
    suffix : str, optional
        Suffix to append to all column names (e.g., "_ref", "_fu").
        Default "" (no suffix).
    
    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with layer-specific columns plus placeholders.
        
        If suffix provided, all columns renamed with suffix (e.g., patient_id_ref).
    
    Notes
    -----
    - All coordinate values are fractional [0,1] in OD orientation.
    - Thickness values in micrometers (µm).
    - F2D distance in millimeters (mm), F2D angle in degrees.
    - Placeholder fields (TopQ, landmarks, disc metrics) set to np.nan/None.
    """
    # Map layer name to Contents display value (must match reference CSV format)
    layer_display_map = {
        'cpRNFL': 'RNFL',
        'GCL+': 'GCL+',
        'GCL++': 'GCL++',
        'Retina': 'Retina'
    }
    
    # Metadata dictionary
    metadata = {
        'patient_id': scan_data['patient_id'],
        'gender': scan_data['gender'],
        'dob': scan_data['dob'],
        'age': scan_data['age'],
        'model_name': scan_data['model_name'],
        'data_no': scan_data['data_no'],
        'eye': scan_data['eye'],
        'capture_date': scan_data['capture_date'],
        'capture_time': scan_data['capture_time'],
        'capture_mode': scan_data['capture_mode'],
        'fixation': scan_data['fixation'],
        'focus_mode': scan_data['focus_mode'],
        'mirror_pos': scan_data['mirror_pos'],
        'F2D_distance': scan_data['f2d_distance'],
        'F2D_angle': scan_data['f2d_angle'],
        'est_axial_length': scan_data['estimated_AL'],
        'littmann_mag': scan_data['littmann_mag'],
        'littmann_mag_bennett': scan_data['littmann_mag_bennett'],
        'littmann_mag_maestro': scan_data['littmann_mag_maestro'],
        'scan_axial_px_res': scan_data['scan_axial_res'],
        'scan_mode': scan_data['scan_mode'],
        'scan_protocol': scan_data['scan_protocol'],
        'scan_resolution': scan_data['scan_resolution'],
        'scan_size': scan_data['scan_size'],
        'MarysQ': quality_score,
        'TopQ': scan_data.get('top_q'),
        # Landmarks stored in CSV: raw FDA values (native image orientation),
        # matching Topcon reference exports for direct metadata comparison.
        'manual_disc_center_x': scan_data.get('disc_center_x_raw', scan_data.get('disc_center_x')),
        'manual_disc_center_y': scan_data.get('disc_center_y_raw', scan_data.get('disc_center_y')),
        'manual_fovea_center_x': scan_data.get('fovea_x_raw', scan_data.get('fovea_x')),
        'manual_fovea_center_y': scan_data.get('fovea_y_raw', scan_data.get('fovea_y')),
        'auto_disc_center_x': scan_data.get('disc_center_auto_x_raw', scan_data.get('disc_center_auto_x')),
        'auto_disc_center_y': scan_data.get('disc_center_auto_y_raw', scan_data.get('disc_center_auto_y')),
        'auto_fovea_center_x': scan_data.get('fovea_auto_x_raw', scan_data.get('fovea_auto_x')),
        'auto_fovea_center_y': scan_data.get('fovea_auto_y_raw', scan_data.get('fovea_auto_y')),
        'Contents': layer_display_map.get(layer_name, layer_name)
        # NOTE: filepath added at end, AFTER sector metrics
    }
    
    # Combine metadata with sector metrics
    output_row = pd.DataFrame([metadata])
    for col in sector_metrics.columns:
        output_row[col] = sector_metrics[col].values[0]
    
    # Add disc metrics for cpRNFL only
    if layer_name == 'cpRNFL':
        disc_area = scan_data.get('disc_area')
        cup_area = scan_data.get('cup_area')
        rim_area = scan_data.get('rim_area')
        cup_volume = scan_data.get('cup_volume')
        disc_volume = scan_data.get('disc_volume')
        vertical_diameter = scan_data.get('vertical_disc_diameter')
        horizontal_diameter = scan_data.get('horizontal_disc_diameter')
        
        # Calculate C/D ratios if data available
        cd_area_ratio = (cup_area / disc_area) if (cup_area is not None and disc_area is not None and disc_area > 0) else np.nan
        linear_cd = np.sqrt(cd_area_ratio) if not np.isnan(cd_area_ratio) else np.nan
        
        output_row['disc_area'] = disc_area
        output_row['cup_area'] = cup_area
        output_row['rim_area'] = None  # Excluded from output
        output_row['cup_volume'] = cup_volume
        output_row['rim_volume'] = np.nan  # Not available in FDA files
        output_row['C/D_area_ratio'] = cd_area_ratio
        output_row['linear_C/D_ratio'] = linear_cd
        output_row['vertical_C/D_ratio'] = np.nan  # Would need cup diameter (not available)
        output_row['disc_Dia.(V)'] = vertical_diameter
        output_row['disc_dia.(H)'] = horizontal_diameter
    
    # Add filepath as LAST column (after sector metrics and disc metrics)
    output_row['filepath'] = scan_data['filepath']
    
    # Add suffix if provided
    if suffix:
        output_row = output_row.rename(columns=lambda c: f"{c}{suffix}")
    
    return output_row
