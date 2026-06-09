"""CSV output utilities for batch processing results.

This module provides functions for atomic CSV writing with proper header
management and error handling. Supports append mode for batch processing.

Author: Marco Miranda
Date: 28 May 2026
"""

import pandas as pd
import csv
import numpy as np
from pathlib import Path
from typing import List, Optional
import time


def initialize_csv_file(csv_path: Path, column_names: List[str]):
    """Initialize CSV file with header row.
    
    Creates new file and writes header. If file exists, truncates it.
    
    Parameters
    ----------
    csv_path : Path
        Path to CSV file to create/truncate.
    column_names : List[str]
        List of column names for header row.
    
    Notes
    -----
    - Parent directory must exist.
    - Overwrites existing file without warning.
    - Uses csv.writer for consistent quoting behavior.
    """
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(column_names)


def append_dataframe_to_csv(df: pd.DataFrame, csv_path: Path, max_retries: int = 3):
    """Append DataFrame to CSV file, with retry on PermissionError.
    
    Atomic append operation: either entire DataFrame is written or none of it.
    Automatically handles header (writes only if file doesn't exist).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to append. Must have same columns as existing CSV.
    csv_path : Path
        Path to CSV file. Created if doesn't exist.
    max_retries : int, default 3
        Number of retry attempts on PermissionError (file locked).
        Waits 1 second between retries.
    
    Raises
    ------
    PermissionError
        If file is locked after all retries exhausted.
    Exception
        Any other unexpected error during write.
    
    Notes
    -----
    - Uses mode='a' (append).
    - Writes header only if file doesn't exist.
    - Retries help with transient file locks from antivirus, backup, etc.
    - Does NOT validate column matching (pandas will raise ValueError).
    """
    if df.empty:
        return
    
    file_exists = csv_path.exists()
    
    for attempt in range(max_retries):
        try:
            df.to_csv(csv_path, mode='a', index=False, header=not file_exists)
            return  # Success
        
        except PermissionError as e:
            if attempt < max_retries - 1:
                print(f"Permission error on {csv_path}. Retrying in 1 second... "
                      f"(attempt {attempt + 1}/{max_retries})", flush=True)
                time.sleep(1)
            else:
                raise PermissionError(
                    f"Failed to write to {csv_path} after {max_retries} attempts: {e}"
                ) from e
        
        except Exception as e:
            print(f"Unexpected error writing to {csv_path}: {e}", flush=True)
            raise


def get_column_names_for_layer(layer_name: str, include_registration: bool = False) -> List[str]:
    """Get column names for specific layer output.
    
    Parameters
    ----------
    layer_name : str
        Layer name: 'cpRNFL', 'GCL+', 'GCL++', or 'Retina'.
    include_registration : bool, default False
        Whether to include registration columns (for aligned paired output).
    
    Returns
    -------
    List[str]
        Column names in output order for specified layer.
    
    Notes
    -----
    Common metadata (30 columns):
    - patient_id, gender, dob, age, model_name, data_no, eye
    - capture_date, capture_time, capture_mode
    - fixation, focus_mode, mirror_pos
    - F2D_distance, F2D_angle, est_axial_length
    - littmann_mag, littmann_mag_bennett, littmann_mag_maestro
    - scan_axial_px_res, scan_mode, scan_resolution, scan_size
    - MarysQ, TopQ (placeholder)
    - manual/auto landmarks (8 placeholders)
    - Contents (layer identifier)
    
    Layer-specific columns:
    - cpRNFL (71 total): + 36 RNFL sectors + 10 disc metrics + filepath
    - GCL+/GCL++ (38 total): + 7 macula sectors + filepath
    - Retina (42 total): + 12 ETDRS zones/metrics + filepath
    
    Registration columns (if include_registration=True):
    - reg_tx, reg_ty, reg_rotation_deg, reg_scale_x, reg_scale_y
    - num_good_matches, num_inliers, reg_processing_time
    """
    # Common metadata (30 columns)
    base_cols = [
        'patient_id', 'gender', 'dob', 'age',
        'model_name', 'data_no', 'eye',
        'capture_date', 'capture_time', 'capture_mode',
        'fixation', 'focus_mode', 'mirror_pos',
        'F2D_distance', 'F2D_angle',
        'est_axial_length',
        'littmann_mag', 'littmann_mag_bennett', 'littmann_mag_maestro',
        'scan_axial_px_res',
        'scan_mode', 'scan_protocol', 'scan_resolution', 'scan_size',
        'MarysQ',
        'TopQ',  # Placeholder
        'manual_disc_center_x', 'manual_disc_center_y',  # Placeholders
        'manual_fovea_center_x', 'manual_fovea_center_y',
        'auto_disc_center_x', 'auto_disc_center_y',
        'auto_fovea_center_x', 'auto_fovea_center_y',
        'Contents'  # Layer identifier
    ]
    
    # Layer-specific columns
    if layer_name == 'cpRNFL':
        # RNFL sector columns (36 columns)
        sector_cols = ['Total']
        sector_cols.extend([f'4_{s}' for s in ['T', 'S', 'N', 'I']])
        sector_cols.extend([f'6_{s}' for s in ['T', 'TS', 'NS', 'N', 'NI', 'TI']])
        sectors_12 = ['T', 'TS', 'ST', 'S', 'SN', 'NS', 'N', 'NI', 'IN', 'I', 'IT', 'TI']
        sector_cols.extend([f'12_{s}' for s in sectors_12])
        sector_cols.extend([f'36_{i:02d}' for i in range(1, 37)])
        
        # Disc metrics (10 placeholders)
        disc_cols = [
            'disc_area', 'cup_area', 'rim_area',
            'cup_volume', 'rim_volume',
            'C/D_area_ratio', 'linear_C/D_ratio', 'vertical_C/D_ratio',
            'disc_Dia.(V)', 'disc_dia.(H)'
        ]
        
        layer_cols = sector_cols + disc_cols
        
    elif layer_name in ['GCL+', 'GCL++']:
        # Macula 6-sector columns (7 columns)
        layer_cols = ['Total', 'TS', 'S', 'NS', 'NI', 'I', 'TI']
        
    elif layer_name == 'Retina':
        # ETDRS 9-zone + metrics (13 columns)
        layer_cols = [
            'ETDRS_Center',
            'ETDRS_In_T', 'ETDRS_In_S', 'ETDRS_In_N', 'ETDRS_In_I',
            'ETDRS_Out_T', 'ETDRS_Out_S', 'ETDRS_Out_N', 'ETDRS_Out_I',
            'average_thick', 'center_thick', 'total_vol'
        ]
    else:
        raise ValueError(
            f"Unknown layer name: '{layer_name}'. "
            f"Valid options: 'cpRNFL', 'GCL+', 'GCL++', 'Retina'"
        )
    
    # Combine: base + layer + filepath
    all_cols = base_cols + layer_cols + ['filepath']
    
    # Add registration columns if needed (for paired/aligned output)
    if include_registration:
        reg_cols = [
            'baseline_scan_id',  # data_no of the reference scan (None for ref row itself)
            'reg_tx', 'reg_ty', 'reg_rotation_deg',
            'reg_scale_x', 'reg_scale_y',
            'num_good_matches', 'num_inliers', 'reg_processing_time'
        ]
        # Insert registration columns before filepath
        all_cols = base_cols + layer_cols + reg_cols + ['filepath']
    
    return all_cols


def write_error_log(
    error_log_path: Path,
    filepath: str,
    error_message: str,
    status: str = 'failed'
):
    """Append error entry to error log CSV.
    
    Parameters
    ----------
    error_log_path : Path
        Path to error log CSV file.
    filepath : str
        File path that caused the error.
    error_message : str
        Error message or exception string.
    status : str, default 'failed'
        Status code. Options: 'failed', 'timeout', 'corrupted'.
    
    Notes
    -----
    - Creates file with header if doesn't exist.
    - Header: filepath, timestamp, status, error_message
    - Timestamp automatically added (current time).
    """
    import datetime
    
    # Create header if file doesn't exist
    if not error_log_path.exists():
        initialize_csv_file(error_log_path, ['filepath', 'timestamp', 'status', 'error_message'])
    
    # Append error row
    error_row = {
        'filepath': filepath,
        'timestamp': datetime.datetime.now(),
        'status': status,
        'error_message': error_message
    }
    
    df_error = pd.DataFrame([error_row])
    append_dataframe_to_csv(df_error, error_log_path)
