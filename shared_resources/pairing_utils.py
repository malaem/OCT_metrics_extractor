"""Pairing utilities for longitudinal OCT scan analysis.

This module provides functions to generate scan pairs from metadata for
longitudinal analysis. Supports multiple pairing strategies:
- all_pairs: All C(n,2) chronological combinations within each patient-eye group
- first_vs_all: Baseline scan paired with all follow-up scans
- first_vs_second: Only baseline vs first follow-up

All pairs are guaranteed to be chronologically ordered (earlier scan as reference).

Author: Marco Miranda
Date: 28 May 2026
"""

import pandas as pd
from typing import List, Tuple
from pathlib import Path


def generate_pairs(
    metadata: pd.DataFrame,
    pairing_mode: str = 'first_vs_all'
) -> Tuple[List[Tuple[str, str]], List[Tuple[List[str], List[Tuple[str, str]]]]]:
    """Generate scan pairs from metadata according to specified pairing strategy.
    
    Groups scans by (patient_id, eye), ensures chronological order within each group,
    then generates pairs according to the selected mode. Returns both a flat list of
    all pairs and a structured list preserving group information.
    
    Parameters
    ----------
    metadata : pd.DataFrame
        Metadata with columns: patient_id, eye, filepath, capture_date, capture_time.
        Optional: full_timestamp (pd.Timestamp) for more reliable sorting.
    pairing_mode : str, default 'first_vs_all'
        Pairing strategy. Options:
        - 'all_pairs': All C(n,2) chronological combinations (e.g., (1,2), (1,3), (2,3))
        - 'first_vs_all': Earliest scan paired with every later scan
        - 'first_vs_second': Only first two chronological scans paired
    
    Returns
    -------
    all_pairs : List[Tuple[str, str]]
        Flat list of (reference_filepath, followup_filepath) tuples.
        Reference is always chronologically earlier than follow-up.
    raw_groups : List[Tuple[List[str], List[Tuple[str, str]]]]
        List of (filepaths_in_group, pairs_in_group) tuples, one per patient-eye group.
        Preserves group structure for efficient batch processing.
    
    Raises
    ------
    ValueError
        If pairing_mode is not one of the recognized options.
    KeyError
        If required columns are missing from metadata.
    
    Examples
    --------
    >>> metadata = pd.DataFrame({
    ...     'patient_id': ['P1', 'P1', 'P1', 'P2', 'P2'],
    ...     'eye': ['OD', 'OD', 'OD', 'OS', 'OS'],
    ...     'filepath': ['f1', 'f2', 'f3', 'f4', 'f5'],
    ...     'capture_date': ['2023-01-01', '2023-06-01', '2024-01-01', '2023-02-01', '2023-08-01'],
    ...     'capture_time': ['10:00', '14:00', '09:00', '11:00', '15:00']
    ... })
    >>> all_pairs, groups = generate_pairs(metadata, pairing_mode='all_pairs')
    >>> len(all_pairs)
    4
    >>> all_pairs[0]
    ('f1', 'f2')
    
    Notes
    -----
    - Assumes metadata is already filtered to include only eligible scans.
    - If full_timestamp column exists, uses it for sorting (more reliable).
    - Otherwise falls back to sorting by capture_date then capture_time strings.
    - Empty groups (n < 2 for all_pairs/first_vs_all, n < 2 for first_vs_second)
      contribute no pairs but are still included in raw_groups for consistency.
    """
    # Validate required columns
    required_cols = {'patient_id', 'eye', 'filepath', 'capture_date', 'capture_time'}
    missing_cols = required_cols - set(metadata.columns)
    if missing_cols:
        raise KeyError(f"Missing required columns in metadata: {missing_cols}")
    
    # Validate pairing_mode
    valid_modes = {'all_pairs', 'first_vs_all', 'first_vs_second'}
    if pairing_mode not in valid_modes:
        raise ValueError(
            f"Unknown pairing_mode '{pairing_mode}'. "
            f"Choose from: {', '.join(valid_modes)}"
        )
    
    all_pairs = []
    raw_groups = []
    
    # Group by patient_id and eye
    for (patient_id, eye), group in metadata.groupby(['patient_id', 'eye']):
        
        # Ensure chronological order within group
        # Prefer full_timestamp if available (more reliable), otherwise use string columns
        if 'full_timestamp' in group.columns:
            group = group.sort_values('full_timestamp')
        else:
            group = group.sort_values(['capture_date', 'capture_time'])
        
        filepaths = group['filepath'].tolist()
        n_scans = len(filepaths)
        
        # Generate pairs according to mode
        if pairing_mode == 'all_pairs':
            # All C(n,2) combinations where i < j
            # Example: [1,2,3] → [(1,2), (1,3), (2,3)]
            pairs = [
                (filepaths[i], filepaths[j])
                for i in range(n_scans)
                for j in range(i + 1, n_scans)
            ]
        
        elif pairing_mode == 'first_vs_all':
            # Earliest scan (baseline) paired with every later scan
            # Example: [1,2,3,4] → [(1,2), (1,3), (1,4)]
            if n_scans < 2:
                pairs = []
            else:
                ref_path = filepaths[0]
                pairs = [(ref_path, filepaths[j]) for j in range(1, n_scans)]
        
        elif pairing_mode == 'first_vs_second':
            # Only first two chronological scans
            # Example: [1,2,3,4] → [(1,2)]
            if n_scans < 2:
                pairs = []
            else:
                pairs = [(filepaths[0], filepaths[1])]
        
        # Store pairs and group structure
        all_pairs.extend(pairs)
        raw_groups.append((filepaths, pairs))
    
    return all_pairs, raw_groups


def filter_metadata_for_wide_scans(metadata: pd.DataFrame) -> pd.DataFrame:
    """Filter metadata to include only 3D Wide scans from Maestro models.
    
    Retains only scans that meet ALL of the following criteria:
    - Model: '3D OCT-1' or '3DOCT-1Maestro2'
    - Fixation: 'Wide'
    - Scan mode: '3D(H)' or '3D(V)'
    
    Parameters
    ----------
    metadata : pd.DataFrame
        Metadata with columns: model_name, fixation, scan_mode, plus standard
        metadata columns (patient_id, eye, filepath, etc.)
    
    Returns
    -------
    pd.DataFrame
        Filtered metadata containing only eligible scans. Original index preserved.
        Empty DataFrame if no scans meet criteria.
    
    Examples
    --------
    >>> metadata = pd.DataFrame({
    ...     'model_name': ['3D OCT-1', '3DOCT-1Maestro2', 'Triton plus', '3D OCT-1'],
    ...     'fixation': ['Wide', 'Wide', 'Wide', 'Macula'],
    ...     'scan_mode': ['3D(H)', '3D(V)', '3D(H)', '3D(H)'],
    ...     'patient_id': ['P1', 'P2', 'P3', 'P4']
    ... })
    >>> filtered = filter_metadata_for_wide_scans(metadata)
    >>> len(filtered)
    2
    >>> filtered['patient_id'].tolist()
    ['P1', 'P2']
    
    Notes
    -----
    - Does not modify the input DataFrame.
    - Returns a view/copy depending on pandas version; safe to modify result.
    """
    # Define eligible models and scan modes
    eligible_models = {'3D OCT-1', '3DOCT-1Maestro2'}
    eligible_scan_modes = {'3D(H)', '3D(V)'}
    
    # Apply filters
    mask = (
        metadata['model_name'].isin(eligible_models) &
        (metadata['fixation'] == 'Wide') &
        metadata['scan_mode'].isin(eligible_scan_modes)
    )
    
    return metadata[mask].copy()


def count_scans_per_group(metadata: pd.DataFrame) -> pd.DataFrame:
    """Count number of scans per patient-eye group.
    
    Useful for determining how many pairs will be generated under different
    pairing modes before actually generating them.
    
    Parameters
    ----------
    metadata : pd.DataFrame
        Metadata with columns: patient_id, eye
    
    Returns
    -------
    pd.DataFrame
        Summary with columns: patient_id, eye, n_scans, n_all_pairs, n_first_vs_all.
        Sorted by (patient_id, eye).
        - n_all_pairs: C(n,2) = n*(n-1)/2
        - n_first_vs_all: n-1
    
    Examples
    --------
    >>> metadata = pd.DataFrame({
    ...     'patient_id': ['P1', 'P1', 'P1', 'P2', 'P2'],
    ...     'eye': ['OD', 'OD', 'OD', 'OS', 'OS']
    ... })
    >>> summary = count_scans_per_group(metadata)
    >>> summary
      patient_id eye  n_scans  n_all_pairs  n_first_vs_all
    0         P1  OD        3            3               2
    1         P2  OS        2            1               1
    """
    summary = metadata.groupby(['patient_id', 'eye']).size().reset_index(name='n_scans')
    
    # Calculate potential pair counts
    summary['n_all_pairs'] = summary['n_scans'] * (summary['n_scans'] - 1) // 2
    summary['n_first_vs_all'] = summary['n_scans'] - 1
    
    return summary.sort_values(['patient_id', 'eye']).reset_index(drop=True)


def should_include_scan(
    fixation: str,
    model_name: str,
    fixation_filter: str = 'All',
    instrument_filter: str = 'Both'
) -> bool:
    """Check if scan passes fixation and instrument filters.
    
    Parameters
    ----------
    fixation : str
        Scan fixation type from metadata (e.g., 'Wide', 'Macula', 'Disc').
    model_name : str
        Instrument model name from metadata.
    fixation_filter : str, default 'All'
        User-selected filter: 'All', '3D Wide', 'Macula', 'Disc'.
    instrument_filter : str, default 'Both'
        User-selected filter: 'Both', 'Maestro', 'Triton'.
    
    Returns
    -------
    bool
        True if scan should be included, False otherwise.
    
    Notes
    -----
    Fixation filter behavior:
    - 'All': Include Wide, Macula, Disc (exclude all other types)
    - '3D Wide': Only Wide fixation
    - 'Macula': Only Macula fixation
    - 'Disc': Only Disc fixation
    
    Instrument filter behavior:
    - 'Both': All instruments (Maestro + Triton)
    - 'Maestro': '3D OCT-1' or '3DOCT-1Maestro2'
    - 'Triton': 'Triton plus'
    
    Examples
    --------
    >>> should_include_scan('Wide', '3D OCT-1', 'All', 'Both')
    True
    >>> should_include_scan('Wide', 'Triton plus', '3D Wide', 'Maestro')
    False  # Wrong instrument
    >>> should_include_scan('Macula', '3D OCT-1', 'All', 'Maestro')
    True
    >>> should_include_scan('Line', '3D OCT-1', 'All', 'Both')
    False  # 'Line' not in allowed fixations when filter='All'
    """
    # Fixation filter
    if fixation_filter != 'All':
        # Map GUI filter name to expected metadata value
        fixation_map = {
            '3D Wide': 'Wide',
            'Macula': 'Macula',
            'Disc': 'Disc'
        }
        expected_fixation = fixation_map.get(fixation_filter)
        if expected_fixation is None:
            raise ValueError(
                f"Unknown fixation_filter: '{fixation_filter}'. "
                f"Valid options: 'All', '3D Wide', 'Macula', 'Disc'"
            )
        if fixation != expected_fixation:
            return False
    else:
        # INTENTIONAL ALLOWLIST — do not change this to pass-all:
        # 'All' in the GUI means "all supported scan types", not literally every
        # fixation value that exists in the FDA files. Only Wide, Macula, and Disc
        # have been tested end-to-end. External has layer extraction logic in
        # LAYER_FIXATION_COMPAT and maryfdaQ support, but has not been validated
        # in this pipeline yet. Re-add 'External' here once tested.
        # Other types (Line, Radial, Cross, 5-Line Raster, etc.) are silently
        # unsupported and must be excluded here to prevent undefined behaviour
        # downstream. If a new fixation type needs support, add it to
        # LAYER_FIXATION_COMPAT in data_extractor_paired.py AND add it here.
        if fixation not in ['Wide', 'Macula', 'Disc']:
            return False
    
    # Instrument filter
    if instrument_filter != 'Both':
        maestro_models = {'3D OCT-1', '3DOCT-1Maestro2'}
        triton_models = {'Triton plus'}
        
        if instrument_filter == 'Maestro':
            if model_name not in maestro_models:
                return False
        elif instrument_filter == 'Triton':
            if model_name not in triton_models:
                return False
        else:
            raise ValueError(
                f"Unknown instrument_filter: '{instrument_filter}'. "
                f"Valid options: 'Both', 'Maestro', 'Triton'"
            )
    
    return True
