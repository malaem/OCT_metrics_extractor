"""Longitudinal retinal layer analysis with optional image alignment.

Main script for batch processing of paired OCT Wide scans. Extracts metrics for
multiple retinal layers (cpRNFL, GCL+, GCL++, Retina), optionally registers
follow-up scans to baseline using KAZE-based image alignment, and writes
results to layer-specific CSVs.

Features:
- Automatic pair generation (all_pairs, first_vs_all, first_vs_second modes)
- Checkpoint/resume support for long-running jobs
- Parallel processing with ProcessPoolExecutor
- Three output CSVs: unpaired scans, paired scans, errors
- Optional KAZE en-face image registration (Wide scans only)
- Graceful fallback to no-aligned for non-Wide scans

Usage:
    python data_extractor_paired.py --input metadata.csv --output output_dir/

Author: Marco Miranda
Date: 28 May 2026
"""

import pandas as pd
import numpy as np
import copy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait as futures_wait, FIRST_COMPLETED
from typing import List, Tuple, Dict, Any, Optional
import datetime
import time
import argparse
import sys
import warnings
from multiprocessing import cpu_count

# Layers compatible with each scan fixation type.
# Used to silently skip incompatible layer/scan combinations when fixation_filter='All'.
# None / unknown fixation values are treated as unrestricted (attempt all requested layers).
LAYER_FIXATION_COMPAT: Dict[str, List[str]] = {
    'Wide':     ['cpRNFL', 'GCL+', 'GCL++', 'Retina'],
    'Macula':   ['GCL+', 'GCL++', 'Retina'],
    'Disc':     ['cpRNFL'],
    'External': ['cpRNFL', 'GCL+', 'GCL++', 'Retina'],  # treat as unrestricted
}

# Local modules
from shared_resources.pairing_utils import (
    generate_pairs,
    filter_metadata_for_wide_scans,
    count_scans_per_group
)
from shared_resources.scan_processor import read_fda_scan, build_enface_image_for_registration
from shared_resources.sector_metrics import (
    compute_all_sector_metrics,
    compute_quality_score,
    prepare_output_row
)
from shared_resources.checkpoint_manager import (
    load_checkpoint,
    save_checkpoint,
    run_config_hash,
    write_checkpoint_header,
    read_checkpoint_config_hash,
    separate_self_and_pair_checkpoints,
    filter_remaining_pairs,
    get_pending_self_scans,
    checkpoint_key,
    print_resume_summary
)
from shared_resources.csv_writer import (
    initialize_csv_file,
    append_dataframe_to_csv,
    write_error_log
)
from shared_resources.OCTenfaceWideRegistration import compute_registration, apply_transform
from shared_resources.system_awake import keep_system_awake


# Global constants
DEFAULT_N_WORKERS = max(1, cpu_count() - 1)
PER_TASK_TIMEOUT = 300  # seconds per FDA file read
NO_PROGRESS_TIMEOUT = 600  # seconds without any completed/timed-out task before aborting


def _process_single_scan(
    filepath: str,
    layers_to_extract: List[str] = None,
    behaviour: str = 'data_extractor',
    scan_fixation: Optional[str] = None
) -> Tuple[str, Dict[str, pd.DataFrame], Optional[str]]:
    """Process single unpaired scan and return layer-specific metrics.
    
    Worker function for parallel processing. Reads FDA file, computes sector
    metrics for each requested layer, and prepares output rows.
    
    Parameters
    ----------
    filepath : str
        Path to FDA file.
    layers_to_extract : List[str], optional
        List of layers to extract: 'cpRNFL', 'GCL+', 'GCL++', 'Retina'.
        Default ['cpRNFL'] if None.
    behaviour : str, default 'data_extractor'
        How to handle negative thickness values in sector/total calculations:
        - 'data_extractor': exclude negatives (set to np.nan)
        - 'imageNET': include negatives in averages
    
    Returns
    -------
    filepath : str
        Original file path (for tracking).
    results_dict : Dict[str, pd.DataFrame]
        Dictionary keyed by layer name, values are single-row DataFrames with metrics.
        Empty dict if processing failed.
    error_msg : str or None
        Error message if processing failed, None if successful.
    
    Notes
    -----
    - Returns (filepath, {}, error_msg) on any exception.
    - Exceptions are caught and converted to error messages.
    - Worker-safe: no shared state, all I/O through return value.
    - FDA file read once, all layers extracted together.
    """
    if layers_to_extract is None:
        layers_to_extract = ['cpRNFL']
    
    try:
        # Apply fixation-layer compatibility filtering (same logic as paired path) so
        # e.g. a Macula scan does not attempt cpRNFL extraction. Unknown/None fixation
        # → attempt all requested layers (FDA file read will determine eligibility).
        if scan_fixation is not None and scan_fixation in LAYER_FIXATION_COMPAT:
            compat_layers = [l for l in layers_to_extract if l in LAYER_FIXATION_COMPAT[scan_fixation]]
        else:
            compat_layers = list(layers_to_extract)
        if not compat_layers:
            return filepath, {}, None  # No compatible layers — silent skip
        scan_data = read_fda_scan(filepath, compat_layers, read_oct3d=False)
        
        # Skip if not eligible (returns None)
        if scan_data is None:
            return filepath, {}, "Not a 3D Wide scan from eligible device"
        
        # Compute sector metrics for all compatible layers first
        all_sector_metrics = {}
        for layer_name in compat_layers:
            sector_metrics = compute_all_sector_metrics(scan_data, layer_name, behaviour)
            all_sector_metrics[layer_name] = sector_metrics
        
        # Compute quality score once for the scan (using all available layer metrics)
        quality_score = compute_quality_score(scan_data, all_sector_metrics, compat_layers)
        
        # Prepare output rows for each compatible layer
        results_dict = {}
        for layer_name in compat_layers:
            result_df = prepare_output_row(scan_data, all_sector_metrics[layer_name], quality_score, layer_name)
            results_dict[layer_name] = result_df
        
        return filepath, results_dict, None
    
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        return filepath, {}, error_msg


def process_scan_pair(
    ref_path: str,
    fu_path: str,
    alignment_mode: str = 'no-aligned',
    layers_to_extract: List[str] = None,
    behaviour: str = 'data_extractor',
    ref_fixation: Optional[str] = None,
    fu_fixation: Optional[str] = None,
) -> Tuple[str, str, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Optional[str]]:
    """Process pair of scans and return layer-specific combined metrics.
    
    Worker function for parallel processing of scan pairs. Reads both FDA files,
    optionally applies image alignment, computes sector metrics for each layer, and combines
    into single output row per layer.
    
    Parameters
    ----------
    ref_path : str
        Path to reference (earlier) scan FDA file.
    fu_path : str
        Path to follow-up (later) scan FDA file.
    alignment_mode : str, default 'no-aligned'
        Alignment mode: 'aligned', 'no-aligned', or 'both'.
        - 'aligned': Only compute aligned metrics (with registration)
        - 'no-aligned': Only compute unaligned metrics (original data)
        - 'both': Compute both aligned and unaligned metrics (separate outputs)
    layers_to_extract : List[str], optional
        List of layers to extract: 'cpRNFL', 'GCL+', 'GCL++', 'Retina'.
        Default ['cpRNFL'] if None.
    behaviour : str, default 'data_extractor'
        How to handle negative thickness values in sector/total calculations:
        - 'data_extractor': exclude negatives (set to np.nan)
        - 'imageNET': include negatives in averages
    
    Returns
    -------
    ref_path : str
        Reference file path (for tracking).
    fu_path : str
        Follow-up file path (for tracking).
    results_aligned : Dict[str, pd.DataFrame]
        Dictionary keyed by layer name, values are aligned metrics DataFrames
        (if alignment_mode in ['aligned', 'both']). Empty dict otherwise.
    results_unaligned : Dict[str, pd.DataFrame]
        Dictionary keyed by layer name, values are unaligned metrics DataFrames
        (if alignment_mode in ['no-aligned', 'both']). Empty dict otherwise.
    error_msg : str or None
        Error message if processing failed, None if successful.
    
    Notes
    -----
    - Returns (ref_path, fu_path, {}, {}, error_msg) on any exception.
    - When alignment is applied, follow-up metrics are computed on aligned data.
    - Registration metrics (translation, rotation, scale, inliers) added to aligned output.
    - Transform computed once from en face images, applied to all layers.
    - Output rows contain all _ref columns, all _fu columns, then registration columns (if aligned).
    """
    if layers_to_extract is None:
        layers_to_extract = ['cpRNFL']

    # If alignment requested but either scan is known to be non-Wide, fall back gracefully.
    # Image registration (KAZE en-face) is only valid for 3D Wide scans.
    #
    # INTENTIONAL FALLBACK BEHAVIOUR — do not treat this as a silent data-loss bug:
    #   effective_alignment_mode is set to 'no-aligned' for this pair only.
    #   The pair is still processed and sector metrics are computed.
    #   Results land in results_unaligned (not results_aligned).
    #   Back in the orchestrator, results_unaligned is written to the 'unaligned'
    #   file if one exists ('both' mode), or silently discarded if there is no
    #   'unaligned' file ('aligned' mode). In 'aligned' mode this is correct: the
    #   user asked only for aligned output, so non-Wide pairs produce no rows.
    #   The pair is counted in n_alignment_skipped and reported in the end-of-run
    #   advisory so the user is informed.
    #   Do NOT add a fallback write to the aligned file — mixing unaligned rows
    #   (no registration columns) into the aligned CSV corrupts its schema.
    effective_alignment_mode = alignment_mode
    if alignment_mode in ['aligned', 'both']:
        non_wide = [
            f"{label}: {fix!r}"
            for label, fix in (('ref', ref_fixation), ('fu', fu_fixation))
            if fix is not None and fix != 'Wide'
        ]
        if non_wide:
            effective_alignment_mode = 'no-aligned'
            print(
                f"  [INFO] Alignment skipped — not a 3D Wide scan "
                f"({', '.join(non_wide)}). Processing without alignment."
            )

    # Compute per-scan compatible layer subsets.
    # Layers incompatible with a scan's fixation type are silently skipped for that scan.
    # Unknown / None fixation → attempt all requested layers.
    def _compat(fix, requested):
        if fix is None or fix not in LAYER_FIXATION_COMPAT:
            return list(requested)
        return [l for l in requested if l in LAYER_FIXATION_COMPAT[fix]]

    ref_compat_layers = _compat(ref_fixation, layers_to_extract)
    fu_compat_layers  = _compat(fu_fixation,  layers_to_extract)

    # Union (order-preserving): layers at least one scan can provide.
    active_layers = list(dict.fromkeys(ref_compat_layers + fu_compat_layers))
    if not active_layers:
        # Neither scan can provide any of the requested layers → silent skip.
        return ref_path, fu_path, {}, {}, None

    try:
        need_oct3d = effective_alignment_mode in ['aligned', 'both']

        # Read each scan with only the layers it can provide.
        ref_data = (
            read_fda_scan(ref_path, ref_compat_layers, read_oct3d=need_oct3d)
            if ref_compat_layers else None
        )
        if ref_data is None and ref_compat_layers:
            if effective_alignment_mode == 'aligned':
                return ref_path, fu_path, {}, {}, "Reference scan not eligible"
            # INTENTIONAL: for no-aligned / both mode, a failed read on one side is not fatal.
            # The readable scan still appears in the no-aligned CSV. Only aligned output
            # requires both sides (alignment is impossible with a missing scan).

        fu_data = (
            read_fda_scan(fu_path, fu_compat_layers, read_oct3d=need_oct3d)
            if fu_compat_layers else None
        )
        if fu_data is None and fu_compat_layers:
            if effective_alignment_mode == 'aligned':
                return ref_path, fu_path, {}, {}, "Follow-up scan not eligible"
            # INTENTIONAL: same reasoning as above — ref data (if readable) will still be written.

        # If both scans returned None, there is nothing to write and nothing to checkpoint.
        # Return an error so the pair is logged rather than silently marked done.
        if ref_data is None and fu_data is None:
            return ref_path, fu_path, {}, {}, "Both scans ineligible (not 3D OCT or unsupported mode)"

        results_aligned = {}
        results_unaligned = {}

        # Compute registration transform once (if needed and both sides are present).
        # TODO (FUTURE): Cross-device resolution normalisation before registration.
        #   When ref and fu are from different devices (e.g. Triton 512×256 vs Maestro 512×128)
        #   the en-face images cover the same physical area but have different pixel densities
        #   (e.g. 2× difference in the y-axis). The current pipeline passes raw pixel images to
        #   KAZE, so the estimated affine matrix is expressed in *moving-image pixel* coordinates.
        #   The correct approach is:
        #     1. Resize both en-face images to a common physical resolution (e.g. ref pixel/mm)
        #        before calling compute_registration().
        #     2. Scale the resulting affine matrix back to moving-scan pixel coordinates before
        #        calling apply_transform() on the thickness layer (which lives in moving-scan
        #        pixel space).
        #   Until this is implemented, cross-device pairs with mismatched pixel densities will
        #   produce a degenerate scale estimate and fall back to no-aligned via the try/except
        #   below — which is safe but loses the alignment benefit.
        transform_info = None
        if effective_alignment_mode in ['aligned', 'both']:
            if ref_data is not None and fu_data is not None:
                # Guard: skip registration when devices or pixel resolutions differ.
                # Different devices can produce the same physical FOV at different pixel
                # densities (e.g. Triton 512×256 vs Maestro 512×128), making the affine
                # matrix meaningless without prior normalisation.  Until cross-device
                # normalisation is implemented (see TODO above), bail out early here.
                ref_model = ref_data.get('model_name')
                fu_model  = fu_data.get('model_name')
                ref_res   = ref_data.get('scan_resolution_set')
                fu_res    = fu_data.get('scan_resolution_set')
                model_mismatch = (ref_model is not None and fu_model is not None
                                  and ref_model != fu_model)
                res_mismatch   = (ref_res is not None and fu_res is not None
                                  and tuple(ref_res) != tuple(fu_res))
                if model_mismatch or res_mismatch:
                    reason = (f"model mismatch ({ref_model} vs {fu_model})" if model_mismatch
                              else f"resolution mismatch ({ref_res} vs {fu_res})")
                    print(f"  [INFO] Skipping registration for {ref_path} → {fu_path}: "
                          f"{reason}. Falling back to no-aligned.", flush=True)
                    effective_alignment_mode = 'no-aligned'
                else:
                    enface_ref = build_enface_image_for_registration(ref_data)
                    enface_fu  = build_enface_image_for_registration(fu_data)
                    try:
                        transform_info = compute_registration(enface_ref, enface_fu)
                    except Exception as reg_err:
                        # Registration failed (degenerate transform, too few matches, etc.).
                        # Fall back to no-aligned for this pair so both scans are still
                        # written to the unaligned output rather than being dropped entirely.
                        print(f"  [WARN] Registration failed for {ref_path} → {fu_path}: {reg_err}. "
                              f"Falling back to no-aligned.", flush=True)
                        effective_alignment_mode = 'no-aligned'
            else:
                effective_alignment_mode = 'no-aligned'

        # Step 1: Compute sector metrics for available layers on each scan.
        ref_all_metrics       = {}
        fu_all_metrics        = {}
        fu_all_metrics_aligned = {}

        for layer_name in active_layers:
            has_ref = ref_data is not None and layer_name in ref_compat_layers
            has_fu  = fu_data  is not None and layer_name in fu_compat_layers

            if has_ref:
                ref_layer = ref_data['layers'][layer_name]
                ref_data_current = ref_data.copy()
                ref_data_current['thickness']        = ref_layer['thickness']
                ref_data_current['thickness_height'] = ref_layer['thickness_height']
                ref_data_current['thickness_width']  = ref_layer['thickness_width']
                ref_all_metrics[layer_name] = compute_all_sector_metrics(ref_data_current, layer_name, behaviour)

            if has_fu:
                fu_layer = fu_data['layers'][layer_name]
                fu_data_current = fu_data.copy()
                fu_data_current['thickness']        = fu_layer['thickness']
                fu_data_current['thickness_height'] = fu_layer['thickness_height']
                fu_data_current['thickness_width']  = fu_layer['thickness_width']
                fu_all_metrics[layer_name] = compute_all_sector_metrics(fu_data_current, layer_name, behaviour)

                if effective_alignment_mode in ['aligned', 'both'] and transform_info is not None:
                    thickness_fu_2d = fu_layer['thickness'].reshape(
                        fu_layer['thickness_height'], fu_layer['thickness_width']
                    )
                    thickness_fu_aligned_2d = apply_transform(transform_info, thickness_fu_2d)
                    # compute_all_sector_metrics reads thickness from
                    # scan_data['layers'][layer_name]['thickness'], NOT from
                    # scan_data['thickness'].  We must update the layers entry with the
                    # warped values; use a deep copy so the original fu_data is untouched.
                    fu_data_aligned = fu_data_current.copy()
                    fu_data_aligned['layers'] = copy.copy(fu_data_aligned['layers'])
                    fu_data_aligned['layers'][layer_name] = dict(fu_layer)
                    fu_data_aligned['layers'][layer_name]['thickness']        = thickness_fu_aligned_2d.reshape(-1)
                    fu_data_aligned['layers'][layer_name]['thickness_height'] = thickness_fu_aligned_2d.shape[0]
                    fu_data_aligned['layers'][layer_name]['thickness_width']  = thickness_fu_aligned_2d.shape[1]
                    fu_all_metrics_aligned[layer_name] = compute_all_sector_metrics(fu_data_aligned, layer_name, behaviour)

        # Step 2: Compute quality scores once per scan (not per layer).
        ref_quality       = compute_quality_score(ref_data, ref_all_metrics, ref_compat_layers) if ref_data is not None else None
        fu_quality        = compute_quality_score(fu_data,  fu_all_metrics,  fu_compat_layers)  if fu_data  is not None else None
        fu_quality_aligned = compute_quality_score(fu_data, fu_all_metrics_aligned, fu_compat_layers) if fu_all_metrics_aligned else None

        # Step 3: Build output rows for each layer using pre-computed quality scores.
        for layer_name in active_layers:
            has_ref = layer_name in ref_all_metrics
            has_fu  = layer_name in fu_all_metrics

            # Build reference row if available.
            ref_row = None
            if has_ref:
                ref_layer = ref_data['layers'][layer_name]
                ref_data_current = ref_data.copy()
                ref_data_current['thickness']        = ref_layer['thickness']
                ref_data_current['thickness_height'] = ref_layer['thickness_height']
                ref_data_current['thickness_width']  = ref_layer['thickness_width']
                ref_row = prepare_output_row(ref_data_current, ref_all_metrics[layer_name], ref_quality, layer_name, suffix="")

            # Aligned output (only when both sides are present).
            if effective_alignment_mode in ['aligned', 'both'] and transform_info is not None:
                if has_ref and has_fu:
                    # prepare_output_row uses only metadata from scan_data, not thickness keys,
                    # so fu_data can be passed directly — the aligned sector metrics are already
                    # in fu_all_metrics_aligned (computed in Step 1, no second apply_transform needed).
                    fu_row_aligned = prepare_output_row(fu_data, fu_all_metrics_aligned[layer_name], fu_quality_aligned, layer_name, suffix="")

                    ref_row_with_baseline = ref_row.copy()
                    ref_row_with_baseline['baseline_scan_id'] = None

                    fu_row_with_baseline = fu_row_aligned.copy()
                    fu_row_with_baseline['baseline_scan_id'] = ref_data['data_no']

                    fu_row_with_baseline['reg_tx']               = transform_info.translation[0]
                    fu_row_with_baseline['reg_ty']               = transform_info.translation[1]
                    fu_row_with_baseline['reg_rotation_deg']     = transform_info.rotation_deg
                    fu_row_with_baseline['reg_scale_x']          = transform_info.scale[0]
                    fu_row_with_baseline['reg_scale_y']          = transform_info.scale[1]
                    fu_row_with_baseline['num_good_matches']     = transform_info.num_good_matches
                    fu_row_with_baseline['num_inliers']          = transform_info.num_inliers
                    fu_row_with_baseline['reg_processing_time']  = transform_info.time_processing

                    for col in ('reg_tx', 'reg_ty', 'reg_rotation_deg', 'reg_scale_x',
                                'reg_scale_y', 'num_good_matches', 'num_inliers', 'reg_processing_time'):
                        ref_row_with_baseline[col] = None

                    results_aligned[layer_name] = pd.concat(
                        [ref_row_with_baseline, fu_row_with_baseline], axis=0, ignore_index=True
                    )

            # Unaligned output: include whichever rows exist.
            if effective_alignment_mode in ['no-aligned', 'both']:
                rows = []
                if ref_row is not None:
                    rows.append(ref_row)
                if has_fu:
                    fu_layer = fu_data['layers'][layer_name]
                    fu_data_current = fu_data.copy()
                    fu_data_current['thickness']        = fu_layer['thickness']
                    fu_data_current['thickness_height'] = fu_layer['thickness_height']
                    fu_data_current['thickness_width']  = fu_layer['thickness_width']
                    fu_row = prepare_output_row(fu_data_current, fu_all_metrics[layer_name], fu_quality, layer_name, suffix="")
                    rows.append(fu_row)
                if rows:
                    results_unaligned[layer_name] = pd.concat(rows, axis=0, ignore_index=True)

        return ref_path, fu_path, results_aligned, results_unaligned, None
    
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        return ref_path, fu_path, {}, {}, error_msg


def _write_metadata_mismatches(output_files: dict, metadata_filtered: pd.DataFrame, mismatch_path: Path):
    """Compare FDA-extracted metadata in output CSVs against metadata.csv and write mismatches.

    Checks patient_id, eye, and capture_date for each filepath that appears in any
    output CSV. Writes metadata_mismatches.csv only if at least one mismatch is found.
    """
    available_meta_cols = [c for c in metadata_filtered.columns if c not in ('filepath', 'fixation', 'full_timestamp')]
    if not available_meta_cols:
        return

    def _normalize_for_compare(series: pd.Series, col_name: str) -> pd.Series:
        """Normalize values for tolerant string/date comparisons."""
        norm = series.astype(str).str.strip().str.lower()
        if 'date' not in col_name.lower():
            return norm

        # Two-pass parse to avoid noisy dayfirst warnings on ISO dates.
        parsed = pd.to_datetime(series, errors='coerce', format='%Y-%m-%d')
        missing_mask = parsed.isna()
        if missing_mask.any():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                parsed_fallback = pd.to_datetime(series[missing_mask], errors='coerce', dayfirst=True)
            parsed.loc[missing_mask] = parsed_fallback

        parsed_norm = parsed.dt.strftime('%Y-%m-%d')
        return parsed_norm.where(parsed.notna(), norm)

    meta_df = metadata_filtered[['filepath'] + available_meta_cols].copy()
    # Normalise for comparison (patient_id and eye were already lowercased at load time).
    # Date-like fields are canonicalized to YYYY-MM-DD when parsable so semantically
    # equal values (e.g., 2017-03-14 vs 14/03/2017) do not produce false mismatches.
    for col in available_meta_cols:
        meta_df[col] = _normalize_for_compare(meta_df[col], col)

    # Collect all output CSVs
    csv_paths = set()
    for files in output_files.values():
        for key in ('main', 'aligned', 'unaligned'):
            if key in files:
                csv_paths.add(files[key])

    all_mismatch_rows = []

    for csv_path in csv_paths:
        if not Path(csv_path).exists():
            continue
        try:
            output_df = pd.read_csv(csv_path, usecols=lambda c: c in (['filepath'] + available_meta_cols))
        except Exception:
            continue
        if 'filepath' not in output_df.columns:
            continue

        fields_to_check = [f for f in available_meta_cols if f in output_df.columns]
        if not fields_to_check:
            continue

        # Normalise output values the same way
        norm_output = output_df[['filepath'] + fields_to_check].copy()
        for col in fields_to_check:
            norm_output[col] = _normalize_for_compare(norm_output[col], col)

        # Merge on filepath
        merged = norm_output.merge(
            meta_df[['filepath'] + fields_to_check],
            on='filepath',
            how='inner',
            suffixes=('_fda', '_meta')
        )

        for col in fields_to_check:
            fda_col = f'{col}_fda'
            meta_col = f'{col}_meta'
            if fda_col not in merged.columns or meta_col not in merged.columns:
                continue
            mismatch_mask = (
                (merged[fda_col] != '') &
                (merged[meta_col] != '') &
                (merged[fda_col] != 'nan') &
                (merged[meta_col] != 'nan') &
                (merged[fda_col] != merged[meta_col])
            )
            bad = merged.loc[mismatch_mask, ['filepath', fda_col, meta_col]]
            if not bad.empty:
                bad = bad.rename(columns={fda_col: 'fda_value', meta_col: 'metadata_csv_value'})
                bad.insert(1, 'field', col)
                all_mismatch_rows.append(bad)

    if all_mismatch_rows:
        result = pd.concat(all_mismatch_rows, ignore_index=True).drop_duplicates(subset=['filepath', 'field'])
        result.to_csv(mismatch_path, index=False)
        print(f"  WARNING: {len(result)} metadata mismatch(es) found — see {mismatch_path.name}")
    else:
        print("  Metadata audit: no mismatches between FDA and metadata.csv")


def run_paired_analysis(
    metadata_csv: str,
    output_dir: str,
    output_base_name: str = 'scan_metrics',
    pairing_mode: str = 'first_vs_all',
    alignment_mode: str = 'no-aligned',
    layers_to_extract: List[str] = None,
    fixation_filter: str = 'All',
    instrument_filter: str = 'Both',
    n_workers: int = DEFAULT_N_WORKERS,
    resume: bool = True,
    behaviour: str = 'data_extractor',
    stop_event=None,
):
    """Main orchestration function for paired retinal layer analysis with multi-layer support.
    
    Workflow:
    1. Load metadata and apply fixation/instrument filters
    2. Generate pairs according to pairing_mode (skipped in no-aligned mode)
    3. Load checkpoint (if resume=True)
    4. Initialize output CSV files only when data is expected
    5. Process unpaired scans and, when enabled, process pairs in parallel
    6. Write results to layer-specific CSVs
    7. Generate summary statistics
    
    Parameters
    ----------
    metadata_csv : str
        Path to CSV file with columns:
        - patient_id, eye, filepath, capture_date, capture_time
        - model_name, fixation, scan_mode
        - Optional: full_timestamp (pd.Timestamp)
    output_dir : str
        Directory for output files. Created if doesn't exist.
    output_base_name : str, default 'scan_metrics'
        Base name for output files. Layer suffixes added automatically.
    pairing_mode : str, default 'first_vs_all'
        Pair generation mode: 'all_pairs', 'first_vs_all', or 'first_vs_second'.
    alignment_mode : str, default 'no-aligned'
        Alignment/output mode:
        - 'aligned': pair Wide scans and write aligned paired output only
        - 'no-aligned': skip pair generation and process all eligible scans as unpaired
        - 'both': pair Wide scans, write aligned/unaligned paired output, and route
          non-Wide scans to unpaired output
    layers_to_extract : List[str], default ['cpRNFL']
        List of layers to extract: 'cpRNFL', 'GCL+', 'GCL++', 'Retina'.
    fixation_filter : str, default 'All'
        Fixation filter: 'All', '3D Wide', 'Macula', 'Disc'.
    instrument_filter : str, default 'Both'
        Instrument filter: 'Both', 'Maestro', 'Triton'.
    n_workers : int, default cpu_count()-1
        Number of parallel worker processes.
    resume : bool, default True
        Whether to resume from checkpoint if available.
    behaviour : str, default 'data_extractor'
        How to handle negative thickness values in sector/total calculations:
        - 'data_extractor': exclude negatives (set to np.nan)
        - 'imageNET': include negatives in averages
    
    Outputs
    -------
    Creates files in output_dir with layer-specific suffixes (only when data is
    written; empty placeholder files are not created):
    - {base}_TSNIT[_aligned].csv : cpRNFL metrics
    - {base}_Macula6[_aligned].csv : GCL+/GCL++ metrics
    - {base}_ETDRS[_aligned].csv : Retina metrics
    - {base}_TSNIT_unpaired.csv : Unpaired cpRNFL scans
    - {base}_Macula6_unpaired.csv : Unpaired GCL+/GCL++ scans
    - {base}_ETDRS_unpaired.csv : Unpaired Retina scans
    - error_log.csv : Failed files with error messages
    - checkpoint.done : Checkpoint file for resume support
    
    Raises
    ------
    FileNotFoundError
        If metadata_csv doesn't exist.
    ValueError
        If metadata_csv is missing required columns or invalid parameters.
    KeyboardInterrupt
        User interruption is caught and handled gracefully.
    
    Notes
    -----
    - Filters applied before pairing (fixation and instrument)
    - Each FDA file read once, all requested layers extracted
    - Layer-specific CSVs have appropriate column schemas
    - GCL+ and GCL++ share same _Macula6 output (differentiated by Contents column)
    - Checkpoint allows safe interruption and resume
    """
    # Default layers if none specified
    if layers_to_extract is None:
        layers_to_extract = ['cpRNFL']
    
    print("="*80)
    print("Longitudinal Multi-Layer Retinal Analysis")
    print("="*80)
    print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Layers: {', '.join(layers_to_extract)}")
    print(f"Fixation filter: {fixation_filter}")
    print(f"Instrument filter: {instrument_filter}")
    print()
    
    # Setup paths
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Map layers to output suffixes
    layer_suffix_map = {
        'cpRNFL': 'TSNIT',
        'GCL+': 'Macula6',
        'GCL++': 'Macula6',
        'Retina': 'ETDRS'
    }
    
    # Build output file paths for each layer
    # OUTPUT FILE DESIGN — three keys, mode-dependent:
    #
    # 'main'      X.csv          — all non-aligned data (exists in no-aligned + both)
    # 'aligned'   X_aligned.csv  — aligned pairs         (exists in aligned  + both)
    # 'unaligned' X_unaligned.csv— scans that could not be aligned, WITH full sector
    #                              metrics (exists ONLY in aligned mode, so the user
    #                              can see exactly what didn't align and why)
    #
    # alignment_mode='no-aligned'
    #   All supported scans (Wide + Macula + Disc) → 'main'. No pairing attempted.
    #
    # alignment_mode='both'
    #   Aligned Wide pairs → 'aligned'.
    #   Everything else (unaligned pairs, lone Wide, Macula, Disc) → 'main'.
    #
    # alignment_mode='aligned'
    #   Successfully aligned Wide pairs → 'aligned'.
    #   Everything that could not be aligned (failed pairs, lone Wide, Macula, Disc)
    #   → 'unaligned', WITH full sector metrics, so no data disappears silently.
    #
    # Files are created on first write (append_dataframe_to_csv); they are never
    # pre-created as empty files.  'aligned' is pre-initialized with a header row
    # when pairs exist so the schema is visible even before processing completes.
    output_files = {}
    for layer_name in layers_to_extract:
        suffix = layer_suffix_map[layer_name]

        if alignment_mode == 'aligned':
            output_files[layer_name] = {
                'aligned':   output_path / f"{output_base_name}_{suffix}_aligned.csv",
                'unaligned': output_path / f"{output_base_name}_{suffix}_unaligned.csv",
            }
        elif alignment_mode == 'no-aligned':
            output_files[layer_name] = {
                'main': output_path / f"{output_base_name}_{suffix}.csv",
            }
        else:  # 'both'
            output_files[layer_name] = {
                'aligned': output_path / f"{output_base_name}_{suffix}_aligned.csv",
                'main':    output_path / f"{output_base_name}_{suffix}.csv",
            }
    
    error_log = output_path / "error_log.csv"
    checkpoint_file = output_path / "checkpoint.done"
    
    # Load metadata
    print(f"Loading metadata from: {metadata_csv}")
    try:
        metadata = pd.read_csv(metadata_csv)
    except FileNotFoundError:
        print(f"ERROR: Metadata file not found: {metadata_csv}")
        sys.exit(1)
    
    print(f"  Total scans in metadata: {len(metadata)}")

    # Normalise key string columns to lowercase so grouping/pairing is case-insensitive.
    # Cast to str first: pandas infers digit-only patient_id values as int64, and
    # calling .str on a non-object Series raises AttributeError.
    for col in ('patient_id', 'eye'):
        if col in metadata.columns:
            metadata[col] = metadata[col].astype(str).str.strip().str.lower()

    # Apply fixation and instrument filters
    print(f"\nApplying filters (fixation={fixation_filter}, instrument={instrument_filter})...")
    from shared_resources.pairing_utils import should_include_scan
    
    mask = metadata.apply(
        lambda row: should_include_scan(
            row['fixation'],
            row['model_name'],
            fixation_filter,
            instrument_filter
        ),
        axis=1
    )
    metadata_filtered = metadata[mask].reset_index(drop=True)
    
    # Report any fixation types that were excluded — important when fixation_filter='All'
    # because 'All' still only covers Wide, Macula and Disc (the three supported scan
    # types). Any other fixation value (Line, Radial, Cross, etc.) is excluded here.
    excluded = metadata[~mask]
    unsupported_fixations = (
        excluded[~excluded['fixation'].isin(['Wide', 'Macula', 'Disc'])]['fixation']
        .value_counts()
    )
    if not unsupported_fixations.empty:
        print(f"  [NOTE] {len(excluded)} scan(s) excluded — unsupported fixation type(s):")
        for fix, count in unsupported_fixations.items():
            print(f"    {fix!r}: {count} scan(s) (not supported — only Wide, Macula, Disc are processed)")
    
    print(f"  Scans after filters: {len(metadata_filtered)}")
    
    if len(metadata_filtered) == 0:
        print("  ERROR: No scans remaining after filters. Exiting.")
        return

    # Alignment is only meaningful for Wide scans. The unpaired source depends
    # on alignment mode:
    # - aligned: only Wide scans are kept anywhere in the output
    # - both: non-Wide scans stay in the unpaired/no-aligned path
    # - no-aligned: all supported scans stay in the unpaired path and no pairing
    #   work is attempted
    metadata_for_pairing = metadata_filtered[metadata_filtered['fixation'] == 'Wide'].reset_index(drop=True)
    if alignment_mode == 'no-aligned':
        metadata_for_pairing = metadata_filtered.iloc[0:0].copy()
    # In all modes, every supported scan that is not covered by a pair is processed
    # via the unpaired path and written to 'main' (no-aligned/both) or 'unaligned'
    # (aligned).  This includes Macula, Disc, lone Wide, and cross-device scans.
    metadata_for_unpaired = metadata_filtered

    non_wide_scans = len(metadata_filtered) - len(metadata_filtered[metadata_filtered['fixation'] == 'Wide'])
    print(f"  Wide scans eligible for pairing: {len(metadata_for_pairing)}")
    if alignment_mode == 'no-aligned':
        print("  Alignment mode no-aligned: all supported scans processed without pairing.")
    elif non_wide_scans > 0:
        dest = "_unaligned.csv" if alignment_mode == 'aligned' else "X.csv"
        print(f"  Non-Wide scans ({non_wide_scans}) will be written to {dest} (no alignment possible).")
    
    # Show scan count per patient-eye group
    scan_summary = count_scans_per_group(metadata_for_pairing)
    if len(scan_summary) > 0:
        print(f"\n  Unique patient-eye groups eligible for pairing: {len(scan_summary)}")
        print(f"  Pairing-group scans per group (mean ± std): {scan_summary['n_scans'].mean():.1f} ± {scan_summary['n_scans'].std():.1f}")
    else:
        print("\n  No Wide scans eligible for pairing.")
    
    # Generate pairs only when aligned output is requested.
    if alignment_mode == 'no-aligned':
        print("\nSkipping pair generation (alignment mode: no-aligned).")
        all_pairs, raw_groups = [], []
    else:
        print(f"\nGenerating pairs from Wide scans only (mode: {pairing_mode})...")
        all_pairs, raw_groups = generate_pairs(metadata_for_pairing, pairing_mode)
        print(f"  Total pairs to process: {len(all_pairs)}")
    
    # Identify truly unpaired scans (scans with no pairs at all)
    scans_in_pairs = set()
    for ref_fp, fu_fp in all_pairs:
        scans_in_pairs.add(ref_fp)
        scans_in_pairs.add(fu_fp)
    
    all_scans = set(metadata_for_unpaired['filepath'].tolist())
    truly_unpaired_scans = all_scans - scans_in_pairs
    
    print(f"  Scans in pairs: {len(scans_in_pairs)}")
    print(f"  Standalone scans (processed without registration): {len(truly_unpaired_scans)}")
    
    if len(all_pairs) == 0 and len(truly_unpaired_scans) == 0:
        print("  No pairs and no standalone scans. Exiting.")
        return
    
    # Compute config hash for this run — used to detect parameter changes on resume.
    config_hash = run_config_hash(
        layers=layers_to_extract,
        alignment_mode=alignment_mode,
        pairing_mode=pairing_mode,
        fixation_filter=fixation_filter,
        instrument_filter=instrument_filter,
        behaviour=behaviour,
    )

    # Load checkpoint
    done_pairs = set()
    done_unpaired = set()
    if resume and checkpoint_file.exists():
        stored_hash = read_checkpoint_config_hash(checkpoint_file)
        if stored_hash is None:
            # Legacy checkpoint (no header) — warn and resume anyway; the
            # parameters cannot be verified but the pair-path deduplication
            # is still valid as a best-effort resume.
            print(
                f"\n  [WARNING] Checkpoint has no config header (written by an older"
                f" version). Cannot verify that run parameters match."
                f" Resuming anyway — if parameters changed, delete {checkpoint_file.name} and re-run."
            )
        elif stored_hash != config_hash:
            print(
                f"\n  [ERROR] Run parameters have changed since the checkpoint was written."
                f"\n    Checkpoint hash : {stored_hash}"
                f"\n    Current hash    : {config_hash}"
                f"\n  Resuming would silently skip pairs that must be reprocessed with"
                f" the new parameters, producing incomplete output."
                f"\n"
                f"\n  To proceed, choose one of:"
                f"\n    1. Restore the original parameters (layers, alignment_mode,"
                f" pairing_mode, fixation_filter, instrument_filter, behaviour) to resume."
                f"\n    2. Delete {checkpoint_file} and re-run to start fresh."
            )
            sys.exit(1)

        print(f"\nLoading checkpoint from: {checkpoint_file}")
        done_pairs = load_checkpoint(checkpoint_file)
        done_self_fps, done_real_pairs = separate_self_and_pair_checkpoints(done_pairs)
        
        # Separate completed unpaired scans from pairs
        done_unpaired = done_self_fps
        
        # Filter remaining work
        remaining_pairs = filter_remaining_pairs(all_pairs, done_real_pairs)
        # Normalize unpaired scan paths before set subtraction: checkpoint stores
        # normalized paths (via save_checkpoint → normalize_checkpoint_path) but
        # truly_unpaired_scans contains raw metadata paths. Without normalization,
        # path-format differences (e.g. /Volumes vs network mount) cause already-done
        # unpaired scans to be reprocessed on resume.
        from shared_resources.checkpoint_manager import normalize_checkpoint_path
        done_unpaired_norm = done_self_fps  # already normalized by load_checkpoint
        truly_unpaired_norm = {normalize_checkpoint_path(p) for p in truly_unpaired_scans}
        remaining_unpaired_norm = truly_unpaired_norm - done_unpaired_norm
        # Map back to original paths for the actual processing
        norm_to_raw = {normalize_checkpoint_path(p): p for p in truly_unpaired_scans}
        remaining_unpaired = {norm_to_raw[n] for n in remaining_unpaired_norm if n in norm_to_raw}
        
        # Print summary
        n_done_pairs = len(all_pairs) - len(remaining_pairs)
        n_done_unpaired = len(truly_unpaired_scans) - len(remaining_unpaired)
        print(f"  Resume: {n_done_pairs}/{len(all_pairs)} pairs done, {n_done_unpaired}/{len(truly_unpaired_scans)} unpaired done")
        
        # Check if all work done
        if len(remaining_pairs) == 0 and len(remaining_unpaired) == 0:
            print("\nAll work complete. Nothing to do.")
            return
        
        # Update work lists
        all_pairs = remaining_pairs
        truly_unpaired_scans = remaining_unpaired
        is_resume = True
    else:
        is_resume = False
    
    # Always import here so get_column_names_for_layer is in scope for both fresh runs
    # and resumed runs (the import was previously inside 'if not is_resume:' which caused
    # a NameError on resume because line ~831 uses it unconditionally).
    from shared_resources.csv_writer import get_column_names_for_layer

    # Initialize output files (if fresh run)
    if not is_resume:
        print("\nInitializing output files...")
        
        # Pre-initialize the aligned CSV with a header row when pairs exist so the
        # schema is visible before processing completes.  All other output files
        # (main, unaligned) are created on first write by append_dataframe_to_csv.
        if len(all_pairs) > 0:
            for layer_name, files in output_files.items():
                if 'aligned' in files:
                    cols = get_column_names_for_layer(layer_name, include_registration=True)
                    initialize_csv_file(files['aligned'], cols)
                    print(f"  {layer_name} aligned pairs: {files['aligned']}")
        else:
            print("  No pairs to process — skipping aligned output file initialization.")
        
        print(f"  Error log: {error_log}")

        # Write config hash as first line of a fresh checkpoint file so a later
        # resume attempt with different parameters is detected immediately.
        write_checkpoint_header(checkpoint_file, config_hash)
    
    # Build fixation lookup once — used by both unpaired and paired workers
    # so each worker can filter layers to those compatible with its scan type.
    fp_to_fixation = dict(zip(metadata_filtered['filepath'], metadata_filtered['fixation']))

    # Process standalone scans first (Macula, Disc, lone Wide, all scans in no-aligned mode)
    if truly_unpaired_scans:
        print(f"\nProcessing standalone scans ({len(truly_unpaired_scans)} files)...")
        print(f"  Workers: {n_workers}")
        
        n_success = 0
        n_errors = 0
        
        executor = ProcessPoolExecutor(max_workers=n_workers)
        try:
            # Submit tasks
            future_to_fp = {
                executor.submit(_process_single_scan, fp, layers_to_extract, behaviour, fp_to_fixation.get(fp)): fp
                for fp in truly_unpaired_scans
            }
            
            # Collect results using wait() polling so a hung worker does not block
            # the main thread indefinitely. as_completed() suspends at __next__()
            # until a future resolves — if a worker hangs, the finally-block cleanup
            # is never reached. wait(timeout=1.0) returns within 1 second regardless
            # of worker state, guaranteeing per-task timeouts and stop_event are
            # always checked.
            total_unpaired = len(future_to_fp)
            pending_fp = dict(future_to_fp)          # future → fp, mutable
            task_start_fp = {f: time.monotonic() for f in pending_fp}  # submission time
            exec_start_fp: dict = {}   # populated when future.running() first becomes True
            i = 0
            stopped_early = False
            last_progress_at = time.monotonic()
            while pending_fp:
                done_set, _ = futures_wait(list(pending_fp), timeout=1.0, return_when=FIRST_COMPLETED)
                for future in done_set:
                    fp = pending_fp.pop(future)
                    exec_start_fp.pop(future, None)
                    i += 1
                    last_progress_at = time.monotonic()
                    try:
                        filepath, results_dict, error_msg = future.result()
                        if error_msg is None and results_dict:
                            # Route to 'main' (no-aligned/both) or 'unaligned' (aligned).
                            dest_key = 'unaligned' if alignment_mode == 'aligned' else 'main'
                            for layer_name, result_df in results_dict.items():
                                dest_file = output_files[layer_name][dest_key]
                                append_dataframe_to_csv(result_df, dest_file)
                            save_checkpoint(checkpoint_file, filepath, filepath)
                            n_success += 1
                        else:
                            # Error or ineligible
                            if error_msg:
                                write_error_log(error_log, filepath, error_msg)
                                n_errors += 1
                    except Exception as e:
                        write_error_log(error_log, fp, f"Unexpected error: {e}")
                        n_errors += 1
                    if i % 10 == 0:
                        print(f"    Progress: {i}/{total_unpaired} "
                              f"(success: {n_success}, errors: {n_errors})", flush=True)
                # Per-task timeout: only applied once a future is actually running
                # (future.running() == True). Queued futures waiting for a free worker
                # are not timed out — submission time ≠ execution start time.
                # Execution-start time is recorded the first time running() is True.
                now = time.monotonic()
                for future in list(pending_fp):
                    if future.running() and future not in exec_start_fp:
                        exec_start_fp[future] = now  # record when execution began
                    start = exec_start_fp.get(future)
                    if start is not None and now - start > PER_TASK_TIMEOUT:
                        fp = pending_fp.pop(future)
                        exec_start_fp.pop(future, None)
                        write_error_log(error_log, fp, f"Timeout after {PER_TASK_TIMEOUT}s", status='timeout')
                        n_errors += 1
                        i += 1
                        last_progress_at = now
                        future.cancel()

                # Global no-progress watchdog for queued futures that never start.
                if pending_fp and (time.monotonic() - last_progress_at) > NO_PROGRESS_TIMEOUT:
                    print(
                        f"  WARNING: No task progress for {NO_PROGRESS_TIMEOUT}s. "
                        f"Marking {len(pending_fp)} pending unpaired scan(s) as stalled.",
                        flush=True,
                    )
                    for future, fp in list(pending_fp.items()):
                        write_error_log(error_log, fp, f"Stalled: no progress for {NO_PROGRESS_TIMEOUT}s", status='timeout')
                        n_errors += 1
                        i += 1
                        future.cancel()
                    pending_fp.clear()
                    stopped_early = True
                    break
                # Check for stop request
                if stop_event is not None and stop_event.is_set():
                    print("  Stop requested — cancelling pending tasks...", flush=True)
                    stopped_early = True
                    break
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        
        if stopped_early:
            print(f"  Stopped early: {n_success} success, {n_errors} errors (partial)")
            return
        print(f"  Completed: {n_success} success, {n_errors} errors")

    # Process pairs
    if all_pairs:
        print(f"\nProcessing scan pairs ({len(all_pairs)} pairs)...")
        print(f"  Workers: {n_workers}")
        print(f"  Alignment mode: {alignment_mode}")
        
        n_success = 0
        n_errors = 0
        n_alignment_skipped = 0  # pairs that fell back to no-aligned in 'aligned' mode (non-Wide, etc.)
        
        # Deduplication strategy depends on pairing mode:
        # - first_vs_all: the baseline scan is always the reference in every pair and its
        #   metrics are identical each time, so write it once (dedup by filepath).
        #   Follow-up scans each appear in exactly one pair, so dedup is harmless there too.
        # - all_pairs: a middle scan (B) appears as follow-up in (A,B) AND as reference in
        #   (B,C). Its sector metrics differ in each context (warped vs reference space) and
        #   its baseline_scan_id / registration columns are different. Both rows are
        #   legitimate and must both be written — deduplication would silently drop one.
        deduplicate_writes = (pairing_mode == 'first_vs_all')
        written_scans_aligned   = {layer: set() for layer in layers_to_extract} if deduplicate_writes else None
        written_scans_unaligned = {layer: set() for layer in layers_to_extract} if deduplicate_writes else None
        
        executor = ProcessPoolExecutor(max_workers=n_workers)
        try:
            # Submit tasks (fp_to_fixation already built above, before unpaired processing)
            future_to_pair = {
                executor.submit(
                    process_scan_pair,
                    ref_fp, fu_fp,
                    alignment_mode, layers_to_extract, behaviour,
                    fp_to_fixation.get(ref_fp),
                    fp_to_fixation.get(fu_fp),
                ): (ref_fp, fu_fp)
                for ref_fp, fu_fp in all_pairs
            }
            
            # Collect results using wait() polling — see unpaired block above for rationale.
            total_pairs_count = len(future_to_pair)
            pending_pair = dict(future_to_pair)      # future → (ref_fp, fu_fp), mutable
            task_start_pair = {f: time.monotonic() for f in pending_pair}  # submission time
            exec_start_pair: dict = {}   # populated when future.running() first becomes True
            i = 0
            stopped_early = False
            last_progress_at = time.monotonic()
            while pending_pair:
                done_set, _ = futures_wait(list(pending_pair), timeout=1.0, return_when=FIRST_COMPLETED)
                for future in done_set:
                    ref_fp, fu_fp = pending_pair.pop(future)
                    exec_start_pair.pop(future, None)
                    i += 1
                    last_progress_at = time.monotonic()
                    try:
                        ref_path, fu_path, results_aligned, results_unaligned, error_msg = future.result()
                        if error_msg is None:
                            # Success: write each layer to its respective paired CSV
                            for layer_name in layers_to_extract:
                                if layer_name in results_aligned:
                                    aligned_file = output_files[layer_name].get('aligned')
                                    if aligned_file:
                                        df = results_aligned[layer_name]
                                        expected_cols = get_column_names_for_layer(layer_name, include_registration=True)
                                        if deduplicate_writes:
                                            rows_to_write = []
                                            if ref_path not in written_scans_aligned[layer_name]:
                                                ref_rows = df[df['filepath'] == ref_path]
                                                if len(ref_rows) > 0:
                                                    rows_to_write.append(ref_rows)
                                                written_scans_aligned[layer_name].add(ref_path)
                                            if fu_path not in written_scans_aligned[layer_name]:
                                                fu_rows = df[df['filepath'] == fu_path]
                                                if len(fu_rows) > 0:
                                                    rows_to_write.append(fu_rows)
                                                written_scans_aligned[layer_name].add(fu_path)
                                            if rows_to_write:
                                                append_dataframe_to_csv(pd.concat(rows_to_write, ignore_index=True).reindex(columns=expected_cols), aligned_file)
                                        else:
                                            # all_pairs: write all rows unconditionally
                                            append_dataframe_to_csv(df.reindex(columns=expected_cols), aligned_file)

                                if layer_name in results_unaligned:
                                    # 'both' mode  → write to 'main' (X.csv)
                                    # 'aligned' mode → write to 'unaligned' (X_unaligned.csv)
                                    # Neither is None under the new design, so no rows are
                                    # silently discarded.  DO NOT write to 'aligned' here:
                                    # rows without registration columns corrupt the aligned schema.
                                    unaligned_dest_key = 'unaligned' if alignment_mode == 'aligned' else 'main'
                                    unaligned_file = output_files[layer_name].get(unaligned_dest_key)
                                    if unaligned_file is None:
                                        if layer_name == layers_to_extract[0]:
                                            n_alignment_skipped += 1
                                    else:
                                        df = results_unaligned[layer_name]
                                        expected_cols_unaligned = get_column_names_for_layer(layer_name, include_registration=False)
                                        if deduplicate_writes:
                                            rows_to_write = []
                                            if ref_path not in written_scans_unaligned[layer_name]:
                                                ref_rows = df[df['filepath'] == ref_path]
                                                if len(ref_rows) > 0:
                                                    rows_to_write.append(ref_rows)
                                                written_scans_unaligned[layer_name].add(ref_path)
                                            if fu_path not in written_scans_unaligned[layer_name]:
                                                fu_rows = df[df['filepath'] == fu_path]
                                                if len(fu_rows) > 0:
                                                    rows_to_write.append(fu_rows)
                                                written_scans_unaligned[layer_name].add(fu_path)
                                            if rows_to_write:
                                                append_dataframe_to_csv(pd.concat(rows_to_write, ignore_index=True).reindex(columns=expected_cols_unaligned), unaligned_file)
                                        else:
                                            append_dataframe_to_csv(df.reindex(columns=expected_cols_unaligned), unaligned_file)
                            save_checkpoint(checkpoint_file, ref_path, fu_path)
                            n_success += 1
                        else:
                            # Error
                            pair_id = f"{ref_path}|||{fu_path}"
                            write_error_log(error_log, pair_id, error_msg)
                            n_errors += 1
                    except Exception as e:
                        # Use ref_fp/fu_fp (from submission dict) — these are always
                        # available even if future.result() was never reached.
                        pair_id = f"{ref_fp}|||{fu_fp}"
                        write_error_log(error_log, pair_id, f"Unexpected error: {e}")
                        n_errors += 1
                    if i % 10 == 0:
                        print(f"    Progress: {i}/{total_pairs_count} "
                              f"(success: {n_success}, errors: {n_errors})", flush=True)
                # Per-task timeout check — only once a future is actually running.
                # See unpaired block for full rationale.
                now = time.monotonic()
                for future in list(pending_pair):
                    if future.running() and future not in exec_start_pair:
                        exec_start_pair[future] = now
                    start = exec_start_pair.get(future)
                    if start is not None and now - start > PER_TASK_TIMEOUT * 2:
                        ref_fp, fu_fp = pending_pair.pop(future)
                        exec_start_pair.pop(future, None)
                        pair_id = f"{ref_fp}|||{fu_fp}"
                        write_error_log(error_log, pair_id, f"Timeout after {PER_TASK_TIMEOUT * 2}s", status='timeout')
                        n_errors += 1
                        i += 1
                        last_progress_at = now
                        future.cancel()

                # Global no-progress watchdog for queued pairs that never start.
                if pending_pair and (time.monotonic() - last_progress_at) > NO_PROGRESS_TIMEOUT:
                    print(
                        f"  WARNING: No task progress for {NO_PROGRESS_TIMEOUT}s. "
                        f"Marking {len(pending_pair)} pending pair(s) as stalled.",
                        flush=True,
                    )
                    for future, (ref_fp, fu_fp) in list(pending_pair.items()):
                        pair_id = f"{ref_fp}|||{fu_fp}"
                        write_error_log(error_log, pair_id, f"Stalled: no progress for {NO_PROGRESS_TIMEOUT}s", status='timeout')
                        n_errors += 1
                        i += 1
                        future.cancel()
                    pending_pair.clear()
                    stopped_early = True
                    break
                # Check for stop request
                if stop_event is not None and stop_event.is_set():
                    print("  Stop requested — cancelling pending tasks...", flush=True)
                    stopped_early = True
                    break
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        
        if stopped_early:
            print(f"  Stopped early: {n_success} success, {n_errors} errors (partial)")
            return
        print(f"  Completed: {n_success} success, {n_errors} errors")

        # Advisory for alignment modes: inform the user if any pairs were excluded
        # from the aligned output (non-Wide fixation, read failure, etc.).
        # Note: scans that could not be paired at all by metadata are already written
        # to the unpaired output file above — they are NOT counted here.
        # For 'no-aligned' this advisory is not applicable.
        if alignment_mode in ['aligned', 'both'] and (n_alignment_skipped > 0 or n_errors > 0):
            print(
                f"\n  [NOTE] Some pairs could not be written to the aligned output:"
                f"\n    - {n_alignment_skipped} pair(s) excluded because alignment is not supported "
                f"(non-Wide fixation type, e.g. macula or disc scans)."
                f"\n    - {n_errors} pair(s) failed due to FDA file read errors or other processing issues."
                f"\n  Check the error log for details: {error_log}"
            )
    
    # --- Metadata mismatch audit ---
    # Compare FDA-extracted values in the output CSVs against metadata.csv.
    # Fields checked: patient_id, eye, capture_date.
    # Mismatches are written to metadata_mismatches.csv (separate from error_log).
    mismatch_path = output_path / "metadata_mismatches.csv"
    _write_metadata_mismatches(output_files, metadata_filtered, mismatch_path)

    # Final summary
    print("\n" + "="*80)
    print("Processing Complete")
    print("="*80)
    print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutput files:")
    
    # Show layer-specific output files
    for layer_name, files in output_files.items():
        print(f"\n  Layer: {layer_name}")
        if 'main' in files and Path(files['main']).exists():
            print(f"    Data: {files['main']}")
        if 'aligned' in files and Path(files['aligned']).exists():
            print(f"    Aligned: {files['aligned']}")
        if 'unaligned' in files and Path(files['unaligned']).exists():
            print(f"    Unaligned (could not align): {files['unaligned']}")
    
    print(f"\n  Error log: {error_log}")
    print(f"  Metadata mismatches: {mismatch_path}")
    print(f"  Checkpoint: {checkpoint_file}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Longitudinal RNFL analysis with optional image alignment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all eligible scans without pairing or alignment (default)
  python data_extractor_paired.py --input metadata.csv --output results/
  
  # First-vs-all pairing mode
  python data_extractor_paired.py --input metadata.csv --output results/ --mode first_vs_all
  
  # With KAZE-based image alignment only
  python data_extractor_paired.py --input metadata.csv --output results/ --alignment aligned
  
    # Export both aligned and unaligned paired outputs (Wide scans), with
    # non-Wide scans routed to unpaired output
  python data_extractor_paired.py --input metadata.csv --output results/ --alignment both
  
  # Custom worker count
  python data_extractor_paired.py --input metadata.csv --output results/ --workers 8
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input metadata CSV file'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['all_pairs', 'first_vs_all', 'first_vs_second'],
        default='first_vs_all',
        help='Pairing mode (default: first_vs_all)'
    )
    
    parser.add_argument(
        '--alignment', '-a',
        choices=['aligned', 'no-aligned', 'both'],
        default='no-aligned',
        help='Alignment/output mode: aligned (paired Wide scans, aligned output only), no-aligned (no pairing; unpaired-only processing), both (paired Wide scans + aligned/unaligned outputs)'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=DEFAULT_N_WORKERS,
        help=f'Number of parallel workers (default: {DEFAULT_N_WORKERS})'
    )
    
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Ignore checkpoint and start from scratch'
    )
    
    parser.add_argument(
        '--output-base-name',
        default='scan_metrics',
        help='Base name for output files (default: scan_metrics). Suffixes (_TSNIT, _Macula6, _ETDRS) added automatically.'
    )
    
    parser.add_argument(
        '--layers',
        default='cpRNFL',
        help='Comma-separated list of layers to extract: cpRNFL, GCL+, GCL++, Retina (default: cpRNFL)'
    )
    
    parser.add_argument(
        '--fixation',
        default='All',
        choices=['All', '3D Wide', 'Macula', 'Disc'],
        help='Filter by fixation type (default: All). "All" includes Wide, Macula, and Disc; other fixation types ignored.'
    )
    
    parser.add_argument(
        '--instrument',
        default='Both',
        choices=['Both', 'Maestro', 'Triton'],
        help='Filter by instrument (default: Both). Maestro = 3D OCT-1 + Maestro2; Triton = Triton plus.'
    )
    
    parser.add_argument(
        '--behaviour',
        default='data_extractor',
        choices=['data_extractor', 'imageNET'],
        help='How to handle negative thickness values: data_extractor (exclude negatives) or imageNET (include negatives). Default: data_extractor.'
    )
    
    args = parser.parse_args()
    
    # Parse layers
    layers_to_extract = [layer.strip() for layer in args.layers.split(',')]
    valid_layers = {'cpRNFL', 'GCL+', 'GCL++', 'Retina'}
    if not set(layers_to_extract).issubset(valid_layers):
        print(f"ERROR: Invalid layers. Choose from: {valid_layers}")
        sys.exit(1)
    
    try:
        with keep_system_awake(True, reason="paired analysis"):
            run_paired_analysis(
                metadata_csv=args.input,
                output_dir=args.output,
                output_base_name=args.output_base_name,
                pairing_mode=args.mode,
                alignment_mode=args.alignment,
                layers_to_extract=layers_to_extract,
                fixation_filter=args.fixation,
                instrument_filter=args.instrument,
                n_workers=args.workers,
                resume=not args.no_resume,
                behaviour=args.behaviour
            )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress saved to checkpoint.")
        sys.exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
