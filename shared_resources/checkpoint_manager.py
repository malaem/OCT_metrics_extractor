"""Checkpoint management for resumable batch processing.

This module provides checkpoint/resume functionality for long-running batch
processing jobs. Maintains a list of completed file pairs in a .done file,
allowing jobs to be safely interrupted and resumed.

Author: Marco Miranda
Date: 28 May 2026
"""

import hashlib
import json
import re
from urllib.parse import unquote, urlparse
from pathlib import Path
from typing import List, Optional, Set, Tuple


CONFIG_HEADER_PREFIX = "#config:"


def run_config_hash(layers: List[str], alignment_mode: str, pairing_mode: str,
                    fixation_filter: str, instrument_filter: str,
                    behaviour: str = 'data_extractor') -> str:
    """Return a short SHA-256 hex digest of the run configuration.

    The digest is stored as the first line of the checkpoint file so that a
    resume attempt with different parameters can be detected immediately.

    Only parameters that affect which pairs are processed or what output is
    produced are included.  Worker count, output directory, and output base
    name are intentionally excluded.
    """
    config = {
        "layers": sorted(layers),
        "alignment_mode": alignment_mode,
        "pairing_mode": pairing_mode,
        "fixation_filter": fixation_filter,
        "instrument_filter": instrument_filter,
        "behaviour": behaviour,
    }
    payload = json.dumps(config, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def write_checkpoint_header(checkpoint_file: Path, config_hash: str) -> None:
    """Write (or overwrite) the checkpoint file with just the config header line.

    Must be called once before the first ``save_checkpoint`` call on a fresh run.
    """
    with open(checkpoint_file, 'w') as f:
        f.write(f"{CONFIG_HEADER_PREFIX}{config_hash}\n")


def read_checkpoint_config_hash(checkpoint_file: Path) -> Optional[str]:
    """Read the config hash from the first line of an existing checkpoint file.

    Returns ``None`` if the file does not exist, is empty, or has no header line
    (legacy checkpoint written before config-hash support was added).
    """
    if not checkpoint_file.exists():
        return None
    with open(checkpoint_file, 'r') as f:
        first_line = f.readline().strip()
    if first_line.startswith(CONFIG_HEADER_PREFIX):
        return first_line[len(CONFIG_HEADER_PREFIX):]
    return None  # legacy file — no header


def normalize_checkpoint_path(filepath: str) -> str:
    """Normalize file path for cross-platform checkpoint matching.
    
    Ensures consistent path representation in checkpoint files regardless of
    OS or mount point. Converts backslashes to forward slashes and handles
    macOS volume mount points.
    
    Parameters
    ----------
    filepath : str
        File path to normalize.
    
    Returns
    -------
    str
        Normalized path with:
        - Forward slashes only
        - macOS /Volumes/SecureData/ → //laj-fs2/SecureData/
        - Consecutive slashes collapsed to //
    
    Examples
    --------
    >>> normalize_checkpoint_path('C:\\\\Data\\\\file.fda')
    'C:/Data/file.fda'
    >>> normalize_checkpoint_path('/Volumes/SecureData/RWD/file.fda')
    '//laj-fs2/SecureData/RWD/file.fda'
    
    Notes
    -----
    This normalization is critical for checkpoint matching across:
    - Windows vs macOS workstations
    - Different network mount points
    - Workspace syncing (OneDrive, etc.)
    """
    # Convert to text, strip wrappers, and normalize separators.
    p = str(filepath).strip().strip('"').strip("'")
    p = p.replace('\\', '/')

    # Accept file:// URIs from mixed tooling/logs.
    if p.lower().startswith('file://'):
        parsed = urlparse(p)
        p = unquote(parsed.path or '')
        # Handle Windows-style file:///C:/... URI paths.
        if re.match(r'^/[a-zA-Z]:/', p):
            p = p[1:]

    # Collapse duplicate slashes while preserving UNC prefix.
    if p.startswith('//'):
        p = '//' + re.sub(r'/+', '/', p[2:])
    else:
        p = re.sub(r'/+', '/', p)
    
    # Handle known cross-OS mount aliases for the same network share.
    low = p.lower()
    vol_prefix = '/volumes/securedata/'
    unc_prefix = '//laj-fs2/securedata/'
    s_drive_prefix = 's:/'
    if low.startswith(vol_prefix):
        p = '//laj-fs2/SecureData/' + p[len(vol_prefix):]
    elif low.startswith(unc_prefix):
        p = '//laj-fs2/SecureData/' + p[len(unc_prefix):]
    elif low.startswith(s_drive_prefix):
        p = '//laj-fs2/SecureData/' + p[len(s_drive_prefix):]
    
    # Normalize drive-letter case for Windows local paths.
    if re.match(r'^[a-zA-Z]:/', p):
        p = p[0].upper() + p[1:]
    
    return p


def checkpoint_key(ref_path: str, fu_path: str) -> Tuple[str, str]:
    """Generate normalized checkpoint key for a file pair.
    
    Parameters
    ----------
    ref_path : str
        Reference (earlier) scan file path.
    fu_path : str
        Follow-up (later) scan file path.
    
    Returns
    -------
    tuple of (str, str)
        Normalized (ref_path, fu_path) tuple for use as dict key or set element.
    
    Examples
    --------
    >>> checkpoint_key('C:\\\\ref.fda', 'C:\\\\fu.fda')
    ('C:/ref.fda', 'C:/fu.fda')
    """
    return (normalize_checkpoint_path(ref_path), normalize_checkpoint_path(fu_path))


def load_checkpoint(checkpoint_file: Path) -> Set[Tuple[str, str]]:
    """Load completed pairs from checkpoint file.
    
    Reads .done file containing one completed pair per line in format:
        ref_path|||fu_path
    
    For self-referential rows (unpaired scans), both paths are identical:
        filepath|||filepath
    
    Parameters
    ----------
    checkpoint_file : Path
        Path to .done checkpoint file.
    
    Returns
    -------
    Set[Tuple[str, str]]
        Set of (ref_path, fu_path) tuples for all completed pairs.
        Both real pairs and self-referential entries included.
        Empty set if file doesn't exist.
    
    Notes
    -----
    - Skips blank lines and lines without ||| separator.
    - Paths in file are assumed already normalized.
    - File corruption (missing |||) logs warning but doesn't crash.
    """
    done_pairs = set()
    
    if not checkpoint_file.exists():
        return done_pairs
    
    with open(checkpoint_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip blank lines
            if not line:
                continue

            # Skip the config-hash header line
            if line.startswith(CONFIG_HEADER_PREFIX):
                continue
            
            # Parse line
            if '|||' not in line:
                print(f"Warning: Malformed checkpoint line {line_num}: {line}")
                continue
            
            ref_fp, fu_fp = line.split('|||', 1)
            done_pairs.add(checkpoint_key(ref_fp, fu_fp))
    
    return done_pairs


def save_checkpoint(checkpoint_file: Path, ref_path: str, fu_path: str):
    """Append completed pair to checkpoint file.
    
    Atomically appends one line to .done file in format:
        normalized_ref|||normalized_fu
    
    Parameters
    ----------
    checkpoint_file : Path
        Path to .done checkpoint file. Created if doesn't exist.
    ref_path : str
        Reference scan file path (will be normalized).
    fu_path : str
        Follow-up scan file path (will be normalized).
    
    Notes
    -----
    - Normalizes paths before writing.
    - Thread-safe: single line write is atomic on most filesystems.
    - Parent directory must exist.
    - For self-referential rows, ref_path == fu_path.
    """
    ref_norm = normalize_checkpoint_path(ref_path)
    fu_norm = normalize_checkpoint_path(fu_path)
    
    with open(checkpoint_file, 'a') as f:
        f.write(f"{ref_norm}|||{fu_norm}\n")


def separate_self_and_pair_checkpoints(
    done_pairs: Set[Tuple[str, str]]
) -> Tuple[Set[str], Set[Tuple[str, str]]]:
    """Separate self-referential checkpoints from real pair checkpoints.
    
    Self-referential checkpoints have identical ref and fu paths (fp|||fp).
    These represent unpaired scans that have been written to output.
    
    Parameters
    ----------
    done_pairs : Set[Tuple[str, str]]
        Set of all completed checkpoints from load_checkpoint().
    
    Returns
    -------
    done_self_fps : Set[str]
        Set of file paths for completed self-referential rows.
    done_real_pairs : Set[Tuple[str, str]]
        Set of (ref, fu) tuples for completed real pairs (ref != fu).
    
    Examples
    --------
    >>> done = {('a.fda', 'b.fda'), ('c.fda', 'c.fda'), ('d.fda', 'e.fda')}
    >>> self_fps, real_pairs = separate_self_and_pair_checkpoints(done)
    >>> self_fps
    {'c.fda'}
    >>> real_pairs
    {('a.fda', 'b.fda'), ('d.fda', 'e.fda')}
    """
    done_self_fps = {ref for ref, fu in done_pairs if ref == fu}
    done_real_pairs = {(ref, fu) for ref, fu in done_pairs if ref != fu}
    return done_self_fps, done_real_pairs


def filter_remaining_pairs(
    all_pairs: list,
    done_real_pairs: Set[Tuple[str, str]]
) -> list:
    """Filter out already-completed pairs from work list.
    
    Parameters
    ----------
    all_pairs : list of (str, str)
        All pairs to process (ref_path, fu_path).
    done_real_pairs : Set[Tuple[str, str]]
        Set of completed real pairs from separate_self_and_pair_checkpoints().
    
    Returns
    -------
    list of (str, str)
        Pairs that still need processing (not in done_real_pairs).
    
    Notes
    -----
    - Normalizes paths in all_pairs before comparison.
    - Preserves original order of remaining pairs.
    """
    return [
        pair for pair in all_pairs
        if checkpoint_key(pair[0], pair[1]) not in done_real_pairs
    ]


def get_pending_self_scans(
    all_ref_paths: Set[str],
    done_self_fps: Set[str]
) -> Set[str]:
    """Identify reference scans whose self-referential rows haven't been written.
    
    Used to handle edge case where pair processing succeeded but self-row
    writing failed in a previous run.
    
    Parameters
    ----------
    all_ref_paths : Set[str]
        Set of all normalized reference file paths from pairs.
    done_self_fps : Set[str]
        Set of normalized paths for completed self-rows.
    
    Returns
    -------
    Set[str]
        Normalized paths for reference scans with pending self-rows.
    
    Examples
    --------
    >>> all_refs = {'a.fda', 'b.fda', 'c.fda'}
    >>> done_self = {'a.fda'}
    >>> get_pending_self_scans(all_refs, done_self)
    {'b.fda', 'c.fda'}
    """
    return all_ref_paths - done_self_fps


def print_resume_summary(
    n_done_pairs: int,
    n_remaining_pairs: int,
    n_done_self: int,
    n_pending_self: int
):
    """Print checkpoint resume summary to console.
    
    Parameters
    ----------
    n_done_pairs : int
        Number of real pairs already completed.
    n_remaining_pairs : int
        Number of real pairs still to process.
    n_done_self : int
        Number of self-referential rows already written.
    n_pending_self : int
        Number of self-referential rows pending.
    
    Notes
    -----
    - Only prints if there's checkpoint data to report.
    - Helps user understand resume state when restarting interrupted jobs.
    """
    if n_done_pairs + n_done_self == 0:
        return
    
    print(f"Checkpoint found: {n_done_pairs} pair(s) and {n_done_self} self-row(s) already written.")
    
    if n_remaining_pairs > 0 or n_pending_self > 0:
        print(f"Resuming: {n_done_pairs} pairs done, {n_remaining_pairs} remaining; "
              f"{n_done_self} self-rows done, {n_pending_self} pending.")
    else:
        print("All work complete. Nothing to do.")
