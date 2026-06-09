"""FDA file metadata extraction for OCT analysis preprocessing.

Stage 1 of the two-stage pipeline:
1. metadata_extractor.py: Scan folder → extract metadata → metadata.csv
2. data_extractor_paired.py: metadata.csv → paired analysis → results

Recursively finds all FDA files in a directory, extracts metadata from headers,
and outputs a CSV file ready for input to data_extractor_paired.py.

Features:
- Recursive directory scanning
- Parallel metadata extraction
- Progress reporting
- Error handling (skips corrupted files)
- Combines capture_date + capture_time into full_timestamp for reliable sorting

Usage:
    python metadata_extractor.py --input /path/to/fda/files --output metadata.csv

Author: Marco Miranda
Date: 28 May 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor, wait as futures_wait, FIRST_COMPLETED
from typing import Optional, Dict, Any, List
import argparse
import sys
import datetime
import signal
from multiprocessing import cpu_count

from shared_resources.read_fda_file import read_fda_common_info
from shared_resources.system_awake import keep_system_awake


# Global constants
DEFAULT_N_WORKERS = max(1, cpu_count() - 1)  # Parallel processing with timeout protection
PER_FILE_TIMEOUT = 10  # seconds per FDA file metadata read (timeout to catch corrupted files)
NO_PROGRESS_TIMEOUT = 180  # seconds without any completed/timed-out task before aborting

# Blacklist of problematic files that cause hangs (just filename, no path)
# Note: With timeout handler, this should rarely be needed
SKIP_FILES = set()


class TimeoutError(Exception):
    """Custom exception for file read timeouts."""
    pass


def _timeout_handler(signum, frame):
    """Signal handler that raises TimeoutError."""
    raise TimeoutError("File read exceeded timeout")


def _extract_metadata_from_fda(filepath: str) -> Optional[Dict[str, Any]]:
    """Extract metadata from a single FDA file.
    
    Reads FDA file header to extract all metadata needed for paired analysis.
    Does NOT read 3D scan arrays (fast, header-only read).
    
    Parameters
    ----------
    filepath : str
        Path to FDA file.
    
    Returns
    -------
    dict or None
        Dictionary with metadata fields, or None if read fails.
        Keys: filepath, patient_id, eye, capture_date, capture_time,
              full_timestamp, model_name, fixation, scan_mode, data_no
    
    Notes
    -----
    - Returns None on any error (corrupted file, missing data, etc.)
    - Only reads header info (fast, no 3D arrays)
    - Creates full_timestamp from capture_date + capture_time for sorting
    """
    # Check if file is in skip list
    from pathlib import Path
    if Path(filepath).name in SKIP_FILES:
        return None
    
    try:
        # Set up timeout alarm to catch corrupted files that hang.
        # SIGALRM is POSIX-only (macOS/Linux); skip on Windows where it does
        # not exist (workers use the future.result(timeout=...) guard instead).
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(PER_FILE_TIMEOUT)
        
        # Read FDA file metadata only (do NOT read oct3d for speed)
        general_info, patient_info, capture_info, scan_info, seg_info = read_fda_common_info(
            filepath, 
            read_oct3d=False,  # Skip 3D data - only need metadata
            read_images=False  # Skip images too
        )
        
        # Cancel the alarm - read completed successfully
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
        
        # Extract required fields
        patient_id = patient_info.patient_id
        eye = capture_info.eye  # 'L' or 'R' - in capture_info, not scan_info
        capture_date = capture_info.capture_date
        capture_time = capture_info.capture_time
        model_name = general_info.model_name
        fixation = scan_info.fixation
        scan_mode = scan_info.scan_mode
        data_no = general_info.data_no
        
        # Validate required fields exist
        if not all([patient_id, eye, capture_date, capture_time, model_name, fixation, scan_mode]):
            return None
        
        # Create full timestamp for reliable sorting
        # Combine date and time strings
        try:
            # Parse date (format: 'YYYY-MM-DD' or similar)
            date_parts = str(capture_date).split('-')
            time_parts = str(capture_time).split(':')
            
            if len(date_parts) >= 3 and len(time_parts) >= 2:
                year = int(date_parts[0])
                month = int(date_parts[1])
                day = int(date_parts[2])
                hour = int(time_parts[0])
                minute = int(time_parts[1])
                second = int(time_parts[2]) if len(time_parts) > 2 else 0
                
                full_timestamp = datetime.datetime(year, month, day, hour, minute, second)
            else:
                full_timestamp = None
        except (ValueError, IndexError):
            full_timestamp = None
        
        # Return metadata dictionary
        return {
            'filepath': filepath,
            'patient_id': patient_id,
            'eye': eye,
            'capture_date': capture_date,
            'capture_time': capture_time,
            'full_timestamp': full_timestamp,
            'model_name': model_name,
            'fixation': fixation,
            'scan_mode': scan_mode,
            'data_no': data_no
        }
    
    except TimeoutError:
        # Corrupted file that hangs during read
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)  # Cancel alarm
        return None
    
    except Exception as e:
        # Return None on any error
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)  # Cancel alarm in case it's still active
        return None


def _find_fda_files(root_dir: Path) -> List[Path]:
    """Recursively find all FDA files in directory.
    
    Parameters
    ----------
    root_dir : Path
        Root directory to search.
    
    Returns
    -------
    List[Path]
        List of Path objects for all .fda files found.
    
    Notes
    -----
    - Case-insensitive (.fda or .FDA)
    - Recursive search (all subdirectories)
    - Returns empty list if root_dir doesn't exist
    """
    if not root_dir.exists():
        return []
    
    # Find all .fda files (case-insensitive)
    fda_files = list(root_dir.rglob('*.[fF][dD][aA]'))
    
    return fda_files


def extract_metadata_batch(
    input_dir: str,
    output_csv: str,
    n_workers: int = DEFAULT_N_WORKERS,
    stop_event=None,
):
    """Extract metadata from all FDA files in directory.
    
    Main orchestration function. Finds all FDA files, extracts metadata in
    parallel, and writes results to CSV.
    
    Parameters
    ----------
    input_dir : str
        Root directory containing FDA files (searched recursively).
    output_csv : str
        Output path for metadata CSV file.
    n_workers : int, default cpu_count()-1
        Number of parallel worker processes.
    
    Outputs
    -------
    Creates CSV file with columns:
    - filepath, patient_id, eye, capture_date, capture_time, full_timestamp,
      model_name, fixation, scan_mode, data_no
    
    Notes
    -----
    - Skips files that fail to read (logs warning)
    - Progress updates every 10 files
    - Final summary shows success/error counts
    """
    print("="*80, flush=True)
    print("FDA Metadata Extraction", flush=True)
    print("="*80, flush=True)
    print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print()
    
    # Find all FDA files
    print(f"Scanning directory: {input_dir}", flush=True)
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}", flush=True)
        sys.exit(1)
    
    fda_files = _find_fda_files(input_path)
    print(f"  Found {len(fda_files)} FDA files", flush=True)
    
    if len(fda_files) == 0:
        print("  No FDA files found. Exiting.", flush=True)
        return
    
    # Extract metadata in parallel
    print(f"\nExtracting metadata from {len(fda_files)} files...", flush=True)
    print(f"  Workers: {n_workers}", flush=True)
    print(f"  Timeout: {PER_FILE_TIMEOUT}s per file (catches corrupted files)", flush=True)
    if SKIP_FILES:
        print(f"  Note: {len(SKIP_FILES)} file(s) in skip list (known problematic files)", flush=True)
    
    metadata_list = []
    n_success = 0
    n_errors = 0
    n_skipped = 0
    error_files = []
    stopped_early = False
    
    # Parallel processing with timeout protection for corrupted files.
    # NOTE: We do NOT use the context manager (`with` block) because its __exit__
    # calls shutdown(wait=True), which blocks until every worker process finishes.
    # Worker processes running C-extension code (e.g. PyCryptodome decryption) cannot
    # be interrupted by SIGALRM, so a stuck worker would hang the whole program.
    # Instead we call shutdown(wait=False, cancel_futures=True) after the loop to
    # return immediately and let the OS clean up any lingering worker processes.
    print(f"  Processing files in parallel...", flush=True)
    executor = ProcessPoolExecutor(max_workers=n_workers)
    try:
        # Submit tasks
        future_to_file = {
            executor.submit(_extract_metadata_from_fda, str(fda_file)): fda_file
            for fda_file in fda_files
        }

        # Collect results using wait() polling so a hung worker does not block the
        # main thread indefinitely. as_completed() suspends at __next__() until a
        # future resolves — if a worker hangs, the finally-block cleanup is never
        # reached. wait(timeout=1.0) returns within 1 second regardless of worker
        # state, guaranteeing per-file timeouts and stop_event are always checked.
        total_files = len(future_to_file)
        pending_file = dict(future_to_file)      # future → fda_file, mutable
        task_start_file = {f: time.monotonic() for f in pending_file}  # submission time (unused after refactor)
        exec_start_file: dict = {}   # populated when future.running() first becomes True
        i = 0
        last_progress_at = time.monotonic()
        while pending_file:
            done_set, _ = futures_wait(list(pending_file), timeout=1.0, return_when=FIRST_COMPLETED)
            for future in done_set:
                fda_file = pending_file.pop(future)
                exec_start_file.pop(future, None)
                i += 1
                last_progress_at = time.monotonic()
                try:
                    metadata = future.result()
                    if metadata is not None:
                        metadata_list.append(metadata)
                        n_success += 1
                    else:
                        # Check if it was skipped vs error
                        if fda_file.name in SKIP_FILES:
                            n_skipped += 1
                        else:
                            n_errors += 1
                            error_files.append(str(fda_file))
                except Exception as e:
                    n_errors += 1
                    error_files.append(str(fda_file))
                    print(f"  WARNING: Error reading {fda_file.name}: {e}", flush=True)
                if i % 10 == 0:
                    status_parts = [f"success: {n_success}"]
                    if n_skipped > 0:
                        status_parts.append(f"skipped: {n_skipped}")
                    if n_errors > 0:
                        status_parts.append(f"errors: {n_errors}")
                    print(f"    Progress: {i}/{total_files} ({', '.join(status_parts)})", flush=True)
            # Per-file timeout: only applied once a future is actually running.
            # Queued futures waiting for a free worker are not timed out —
            # submission time ≠ execution start time.
            now = time.monotonic()
            for future in list(pending_file):
                if future.running() and future not in exec_start_file:
                    exec_start_file[future] = now
                start = exec_start_file.get(future)
                if start is not None and now - start > PER_FILE_TIMEOUT:
                    fda_file = pending_file.pop(future)
                    exec_start_file.pop(future, None)
                    n_errors += 1
                    error_files.append(str(fda_file))
                    i += 1
                    last_progress_at = now
                    print(f"  WARNING: Timeout after {PER_FILE_TIMEOUT}s: {fda_file.name}", flush=True)
                    future.cancel()

            # Global no-progress watchdog: if no task has completed or timed out for
            # too long, assume workers are wedged and mark all pending tasks as stalled.
            if pending_file and (time.monotonic() - last_progress_at) > NO_PROGRESS_TIMEOUT:
                print(
                    f"  WARNING: No task progress for {NO_PROGRESS_TIMEOUT}s. "
                    f"Marking {len(pending_file)} pending file(s) as stalled and aborting loop.",
                    flush=True,
                )
                for future, fda_file in list(pending_file.items()):
                    error_files.append(str(fda_file))
                    n_errors += 1
                    i += 1
                    future.cancel()
                pending_file.clear()
                stopped_early = True
                break
            # Check for stop request
            if stop_event is not None and stop_event.is_set():
                print("  Stop requested — cancelling pending tasks...", flush=True)
                stopped_early = True
                break
    finally:
        # Shut down without waiting — prevents hang if any worker is stuck in C-extension code.
        executor.shutdown(wait=False, cancel_futures=True)
    
    # Print completion summary
    summary_parts = [f"{n_success} success"]
    if n_skipped > 0:
        summary_parts.append(f"{n_skipped} skipped")
    if n_errors > 0:
        summary_parts.append(f"{n_errors} errors")
    if stopped_early:
        summary_parts.append("PARTIAL — stopped early")
    print(f"  Completed: {', '.join(summary_parts)}", flush=True)

    if stopped_early:
        print(
            f"\n  [WARNING] Stage 1 was stopped before all files were processed."
            f" The metadata CSV was NOT written to avoid creating a partial output"
            f" file that could be mistaken for a complete run."
            f" Re-run Stage 1 to produce a full metadata CSV.",
            flush=True,
        )
        return
    
    # Create DataFrame
    if len(metadata_list) == 0:
        print("\nERROR: No metadata extracted successfully. Cannot create output file.", flush=True)
        return
    
    df = pd.DataFrame(metadata_list)
    
    # Sort by patient_id, eye, full_timestamp
    if 'full_timestamp' in df.columns:
        df = df.sort_values(['patient_id', 'eye', 'full_timestamp'])
    else:
        df = df.sort_values(['patient_id', 'eye', 'capture_date', 'capture_time'])
    
    # Write to CSV
    print(f"\nWriting metadata to: {output_csv}")
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_csv, index=False)
    
    # Summary statistics
    print("\n" + "="*80)
    print("Extraction Complete")
    print("="*80)
    print(f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nMetadata CSV: {output_csv}")
    print(f"  Total records: {len(df)}")
    print(f"  Unique patients: {df['patient_id'].nunique()}")
    print(f"  Unique patient-eye groups: {df.groupby(['patient_id', 'eye']).ngroups}")
    print(f"  Date range: {df['capture_date'].min()} to {df['capture_date'].max()}")
    
    if n_errors > 0:
        print(f"\n  {n_errors} files failed to read")
        print(f"  First 5 error files:")
        for error_file in error_files[:5]:
            print(f"    - {error_file}")
    
    print("\nNext step:")
    print(f"  python data_extractor_paired.py --input {output_csv} --output results/")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Extract metadata from FDA files for OCT paired analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract metadata from all FDA files in directory
  python metadata_extractor.py --input /data/fda_files/ --output metadata.csv
  
  # Custom worker count
  python metadata_extractor.py --input /data/fda_files/ --output metadata.csv --workers 8
  
  # Then run paired analysis
  python data_extractor_paired.py --input metadata.csv --output results/
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input directory containing FDA files (searched recursively)'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output metadata CSV file'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=DEFAULT_N_WORKERS,
        help=f'Number of parallel workers (default: {DEFAULT_N_WORKERS})'
    )
    
    args = parser.parse_args()
    
    try:
        with keep_system_awake(True, reason="metadata extraction"):
            extract_metadata_batch(
                input_dir=args.input,
                output_csv=args.output,
                n_workers=args.workers
            )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
