# Longitudinal RNFL Analysis System

Modular Python system for batch processing of paired OCT scans with sector-averaged RNFL thickness metrics computation.

**Date:** 28 May 2026  
**Author:** Generated for OCT analysis pipeline

---

## Overview

This system processes FDA files containing 3D Wide RNFL scans from Topcon Maestro OCT devices. It:

1. **Filters** metadata to eligible scans (3D Wide, Maestro models only)
2. **Generates pairs** of longitudinal scans per patient-eye
3. **Computes sector metrics** (4/6/12/36-sector grids, total average)
4. **Outputs three CSVs**: unpaired scans, paired scans, errors
5. **Supports checkpointing** for safe interruption and resume

Designed for robust parallel processing over network-mounted data with comprehensive error handling.

---

## System Architecture

### Core Modules

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| **pairing_utils.py** | Pair generation and metadata filtering | `generate_pairs()`, `filter_metadata_for_wide_scans()` |
| **scan_processor.py** | FDA file reading and RNFL extraction | `read_fda_scan_for_wide_rnfl()` |
| **sector_metrics.py** | Sector-averaged thickness computation | `compute_all_sector_metrics()`, `compute_quality_score()` |
| **checkpoint_manager.py** | Checkpoint/resume support | `load_checkpoint()`, `save_checkpoint()` |
| **csv_writer.py** | Atomic CSV writing utilities | `append_dataframe_to_csv()`, `initialize_csv_file()` |
| **data_extractor_paired.py** | Main orchestration script | `run_paired_analysis()`, CLI |

### Data Flow

```
Metadata CSV
    ↓
Filter eligible scans (Wide, Maestro, 3D(H/V))
    ↓
Generate pairs (all_pairs / first_vs_all / first_vs_second)
    ↓
Load checkpoint (if resuming)
    ↓
┌────────────────┬──────────────────┐
│ Process        │  Process         │
│ Unpaired Scans │  Paired Scans    │
│ (parallel)     │  (parallel)      │
└───────┬────────┴────────┬─────────┘
        ↓                 ↓
   Write results + checkpoint atomically
        ↓
┌───────────────┬──────────────┬─────────────┐
│ unpaired_     │ paired_      │ error_log   │
│ scans.csv     │ scans.csv    │ .csv        │
└───────────────┴──────────────┴─────────────┘
```

---

## Installation & Dependencies

### Required Packages

```bash
# Core dependencies
pip install pandas numpy scipy cv2 wakepy

# Project-specific modules (must be in Python path)
# - shared_resources.read_fda_file
# - shared_resources.f2d_angle_distance
# - shared_resources.grid_diameter
# - shared_resources.sectorAverage
# - shared_resources.maryfdaQ
```

### File Structure

```
Python_Scripts/
├── data_extractor_paired.py      # Main script (CLI entry point)
├── pairing_utils.py               # Pair generation
├── scan_processor.py              # FDA file reading
├── sector_metrics.py              # Sector analysis
├── checkpoint_manager.py          # Checkpoint logic
├── csv_writer.py                  # CSV utilities
└── shared_resources/              # Existing utilities (not included here)
    ├── read_fda_file.py
    ├── grid_diameter.py
    ├── sectorAverage.py
    ├── maryfdaQ.py
    └── ...
```

---

## Usage

### Basic Usage

```bash
# Process all eligible scans without pairing/alignment (unpaired-only)
python data_extractor_paired.py \
    --input metadata.csv \
    --output results/
```

### Pairing Modes

```bash
# All C(n,2) chronological pairs within each patient-eye
python data_extractor_paired.py -i metadata.csv -o results/ --mode all_pairs

# Baseline (first) scan paired with all follow-ups
python data_extractor_paired.py -i metadata.csv -o results/ --mode first_vs_all

# Only first two chronological scans
python data_extractor_paired.py -i metadata.csv -o results/ --mode first_vs_second
```

### Performance Tuning

```bash
# Custom worker count (default: cpu_count - 1)
python data_extractor_paired.py -i metadata.csv -o results/ --workers 16

# Ignore checkpoint and restart from scratch
python data_extractor_paired.py -i metadata.csv -o results/ --no-resume
```

### Command-Line Options

```
Required:
  --input, -i    Input metadata CSV file
  --output, -o   Output directory for results

Optional:
  --mode, -m            Pairing mode: all_pairs, first_vs_all, first_vs_second
                                    (default: first_vs_all; used only when pairing is enabled)
  --workers, -w         Number of parallel workers (default: cpu_count - 1)
   --alignment, -a       Alignment/output mode:
                                    aligned    = pair Wide scans, output aligned paired CSV only
                                    no-aligned = skip pairing, process scans as unpaired only
                                    both       = pair Wide scans, output aligned + unaligned paired CSVs
  --no-resume           Ignore checkpoint and restart from scratch
  --help, -h            Show help message
```

---

## Input Format

### Metadata CSV Requirements

Required columns:
- `patient_id` : str, patient identifier
- `eye` : str, 'L' or 'R'
- `filepath` : str, absolute path to FDA file
- `capture_date` : str or datetime, scan date
- `capture_time` : str or datetime, scan time
- `model_name` : str, OCT device model
- `fixation` : str, fixation type (Wide, Macula, Disc)
- `scan_mode` : str, scan mode (3D(H), 3D(V), etc.)

Optional but recommended:
- `full_timestamp` : pd.Timestamp, combined date+time for reliable sorting

### Example Metadata

```csv
patient_id,eye,filepath,capture_date,capture_time,model_name,fixation,scan_mode
P001,OD,/data/P001_baseline.fda,2023-01-15,10:30,3D OCT-1,Wide,3D(H)
P001,OD,/data/P001_followup.fda,2023-07-20,14:15,3D OCT-1,Wide,3D(H)
P002,OS,/data/P002_baseline.fda,2023-02-10,09:00,3DOCT-1Maestro2,Wide,3D(V)
```

---

## Output Files

### 1. unpaired_scans.csv

Single-scan metrics (no pairing). One row per eligible scan.

**Columns:**
- Patient metadata: `patient_id`, `gender`, `dob`, `age`
- Device metadata: `model_name`, `data_no`, `eye`
- Capture info: `capture_date`, `capture_time`, `capture_mode`
- Scan parameters: `fixation`, `focus_mode`, `mirror_pos`, `scan_mode`, `scan_resolution`, `scan_size`, etc.
- Anatomical landmarks: `disc_center_x/y`, `fovea_x/y`, `F2D_distance`, `F2D_angle`, `est_axial_length`
- Quality: `MarysQ` (Pass/Fail/None)
- **RNFL metrics:** `Total`, `4_T`, `4_S`, `4_N`, `4_I`, `6_T`, ..., `36_36`
- File: `filepath`

### 2. paired_scans.csv

Combined metrics for scan pairs. One row per pair.

**Columns:**
- All columns from `unpaired_scans.csv` with `_ref` suffix (reference scan)
- All columns from `unpaired_scans.csv` with `_fu` suffix (follow-up scan)

Example: `patient_id_ref`, `Total_ref`, `4_T_ref`, ..., `patient_id_fu`, `Total_fu`, `4_T_fu`, ...

### 3. error_log.csv

Failed files with error messages.

**Columns:**
- `filepath` : str, failed file path (or pair: "ref|||fu")
- `timestamp` : datetime, when error occurred
- `status` : str, 'failed', 'timeout', or 'corrupted'
- `error_message` : str, exception message

### 4. checkpoint.done

Resume support file. One line per completed item:
```
/path/to/ref.fda|||/path/to/fu.fda
/path/to/self.fda|||/path/to/self.fda
```

---

## RNFL Sector Metrics

All thickness values in **micrometers (µm)**.

### Sector Grids

| Grid | Sectors | Column Names |
|------|---------|--------------|
| **Total** | 1 | `Total` (overall average in 3.4mm annulus) |
| **4-sector** | 4 | `4_T`, `4_S`, `4_N`, `4_I` |
| **6-sector** | 6 | `6_T`, `6_TS`, `6_NS`, `6_N`, `6_NI`, `6_TI` |
| **12-sector** | 12 | `12_T`, `12_TS`, `12_ST`, `12_S`, `12_SN`, `12_NS`, `12_N`, `12_NI`, `12_IN`, `12_I`, `12_IT`, `12_TI` |
| **36-sector** | 36 | `36_01`, `36_02`, ..., `36_36` |

### Sector Naming Convention

- **T** = Temporal
- **S** = Superior
- **N** = Nasal
- **I** = Inferior
- **TS** = Temporal-Superior (transitional sector)
- Numeric sectors (`36_01`): numbered 1-36 with rotation transformation applied

### Processing Pipeline

1. **Annulus selection**: 3.4mm diameter circle centered on optic disc
2. **Polar coordinates**: Convert (x,y) → (radius, angle) relative to disc center
3. **Sector assignment**: Group pixels by angular ranges
4. **Averaging**: Mean thickness within each sector
5. **Quality check**: MaryQ score based on Q-score, disc position, F2D metrics

---

## Coordinate System

### Conventions

- **Origin:** Top-left corner (standard image convention)
- **Fractional coordinates:** [0, 1] for x and y
- **Laterality:** All scans normalized to **OD (right eye) orientation**
  - Right eyes: no change
  - Left eyes: horizontally flipped, x → 1-x
- **Y-axis flip:** FDA files use bottom-left origin; we flip to top-left (y → 1-y)

### Anatomical Landmarks

```
      0.0 ────────────── 1.0 (x)
0.0    ┌─────────────────┐
  │    │                 │
  y    │    Fovea   Disc │  (OD orientation)
  │    │       ●    ●    │
1.0    └─────────────────┘
```

---

## Eligibility Criteria

Scans must meet ALL of the following:
- **Model:** `3D OCT-1` or `3DOCT-1Maestro2`
- **Fixation:** `Wide`
- **Scan mode:** `3D(H)` or `3D(V)`
- **Capture mode:** NOT `Fundus only` or `Fundus Photo only`

Scans failing any criterion are skipped (not counted as errors).

---

## Checkpoint & Resume

### How It Works

1. After each successful scan/pair, an entry is written to `checkpoint.done`
2. On restart, the script:
   - Loads `checkpoint.done`
   - Filters out already-completed work
   - Processes only remaining items
3. Safe to interrupt at any time (Ctrl+C); progress is never lost

### Resume Behavior

```bash
# First run: processes all pairs
python data_extractor_paired.py -i metadata.csv -o results/
# ... interrupted after 50 pairs ...

# Resume: processes only remaining pairs
python data_extractor_paired.py -i metadata.csv -o results/
# Checkpoint found: 50 pair(s) already written.
# Resuming: 50 pairs done, 150 remaining.
```

### Force Fresh Start

```bash
# Ignore checkpoint and restart from scratch
python data_extractor_paired.py -i metadata.csv -o results/ --no-resume

# Or manually delete checkpoint
rm results/checkpoint.done
```

---

## Error Handling

### Three-Tier Approach

1. **Ineligible scans:** Skipped silently (not errors)
   - Non-Wide fixation
   - Non-Maestro models
   - Non-3D scan modes

2. **Recoverable errors:** Logged to `error_log.csv`, processing continues
   - Corrupted FDA files
   - Missing segmentation data
   - Timeout (file too slow to read)

3. **Fatal errors:** Stop entire job
   - Metadata file not found
   - Output directory not writable
   - Unexpected exceptions in orchestration code

### Error Log Format

```csv
filepath,timestamp,status,error_message
/data/P001.fda,2026-05-28 10:30:15,failed,ValueError: Missing segmentation data
/data/P002.fda,2026-05-28 10:31:00,timeout,Timeout after 300s
```

---

## Performance Considerations

### Parallel Processing

- Default: `cpu_count() - 1` workers
- Increase for I/O-bound tasks (network storage): `--workers 16`
- Decrease for memory-constrained systems: `--workers 4`

### Timeouts

- Per FDA file: 300 seconds (5 minutes)
- Per pair: 600 seconds (10 minutes)
- Configurable via `PER_TASK_TIMEOUT` constant in script

### Memory Usage

- Each worker loads 1-2 FDA files (~50-100 MB each)
- Peak memory: `n_workers × 200 MB + overhead`
- Example: 8 workers ≈ 1.6 GB + 500 MB overhead = ~2 GB

### Network I/O

- Designed for VPN-mounted network storage
- Retry logic for transient PermissionErrors
- Checkpoint-based resume minimizes wasted work

---

## Troubleshooting

### "No eligible scans found"

- Check metadata CSV has required columns
- Verify scans match eligibility criteria (Wide, Maestro, 3D)
- Use `filter_metadata_for_wide_scans()` to test filtering logic

### "PermissionError" during write

- Output directory not writable → check permissions
- Files locked by another process → close Excel, antivirus, etc.
- Network mount issue → retry or use local storage

### "TimeoutError" for specific files

- File too large or corrupted → check FDA file integrity
- Network too slow → increase `PER_TASK_TIMEOUT` or use faster mount
- Logged to `error_log.csv`; safe to retry after fixing issue

### Checkpoint not resuming correctly

- Checkpoint file corrupted → delete and restart
- Paths changed (moved files) → checkpoint uses absolute paths, won't match
- Use `--no-resume` to force fresh start

---

## Future Enhancements

### Planned Features

1. **Additional devices:** Support Triton Plus, 3DOCT-2000FA
   - Requires Littmann factor validation
   - May need device-specific coordinate transforms

2. **Macula/Disc scan reading:** `read_fda_scan()` currently returns None for non-Wide fixation scans
   - Layer-fixation compatibility logic already in place in `process_scan_pair()`
   - Requires extending `read_fda_scan()` to handle Macula and Disc scan types

3. **Quality filtering:** Pre-filter by Q-score, F2D metrics
   - Currently computed but not used for filtering
   - Option to skip low-quality scans before processing

### Code TODOs

- [ ] Extend `read_fda_scan()` to read Macula and Disc scan types
- [ ] Guard against degenerate RANSAC registration in `OCTenfaceWideRegistration.py` (scale=0 silently produces garbage rows — see deferred fix in notes)
- [ ] Add unit tests for all modules
- [ ] Add logging module for structured logs

### Completed

- [x] Image alignment in `process_scan_pair()` (SIFT/ORB + RANSAC via `OCTenfaceWideRegistration`)
- [x] Layer-fixation compatibility guards (GUI + backend)
- [x] Multi-layer support: cpRNFL, GCL+, GCL++, Retina
- [x] Standalone macOS `.app` (`dist/OCT Data Extractor.app`, built with Homebrew Python + Tk 9.0)
- [x] Checkpoint/resume for long-running batches
- [x] Parallel processing via `ProcessPoolExecutor`

---

## Module Documentation

### pairing_utils.py

**Purpose:** Pair generation and metadata filtering (moved to `shared_resources/`)

**Key Functions:**
- `generate_pairs()`: Generate scan pairs from metadata
- `filter_metadata_for_wide_scans()`: Filter to eligible scans only
- `count_scans_per_group()`: Summarize scans per patient-eye

### scan_processor.py

**Purpose:** FDA file reading and retinal layer thickness extraction (moved to `shared_resources/`)

**Key Functions:**
- `read_fda_scan()`: Main entry point for FDA reading
- `extract_rnfl_thickness()`: Compute RNFL thickness from segmentation
- `get_littmann_magnification()`: Device-specific correction factor

### sector_metrics.py

**Purpose:** Sector-averaged thickness computation (moved to `shared_resources/`)

**Key Functions:**
- `compute_all_sector_metrics()`: Compute all sector grids (4/6/12/36)
- `compute_quality_score()`: MaryQ quality assessment
- `prepare_output_row()`: Format results for CSV output

### checkpoint_manager.py

**Purpose:** Checkpoint/resume support (moved to `shared_resources/`)

**Key Functions:**
- `load_checkpoint()`: Read completed pairs from .done file
- `save_checkpoint()`: Append completed pair to .done file
- `normalize_checkpoint_path()`: Cross-platform path normalization

### csv_writer.py

**Purpose:** Atomic CSV writing (moved to `shared_resources/`)

**Key Functions:**
- `initialize_csv_file()`: Create CSV with header
- `append_dataframe_to_csv()`: Atomic append with retry
- `write_error_log()`: Log errors to error_log.csv

---

## Contact & Support

For questions or issues:
1. Check this README for solutions
2. Review module docstrings for detailed API docs
3. Examine error_log.csv for specific error messages
4. Contact pipeline maintainer with error logs and metadata sample

---

## License

Internal research tool. Not for distribution outside organization.

---

**End of README**
