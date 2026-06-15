# OCT Metrics Extractor

Primary entry point: [oct_analysis_gui.py](oct_analysis_gui.py)

Two-stage pipeline for extracting OCT metadata and retinal layer metrics from Topcon FDA files.

This project supports:
- Batch metadata extraction from FDA files
- Longitudinal analysis with configurable pairing modes
- Optional Wide-scan image registration (KAZE-based)
- Multi-layer metric export (cpRNFL, GCL+, GCL++, Retina)
- Resume/restart-safe processing via checkpoint files
- GUI and CLI workflows

## Pipeline at a Glance

1. Stage 1 (`metadata_extractor.py`)
   - Recursively scan a folder for `.fda` files
   - Read header metadata only (fast)
   - Build `metadata.csv`

2. Stage 2 (`data_extractor_paired.py`)
   - Load metadata
   - Filter scans by fixation/instrument
   - Process as unpaired or paired (depending on alignment mode)
   - Export layer-specific CSV outputs

## Requirements

- Python 3.10+
- macOS/Windows/Linux (development primarily on macOS)

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `tkinter` is missing on your system, install your platform's Tk package (for GUI use).

## Quick Start (GUI First)

Launch the application:

```bash
python oct_analysis_gui.py
```

Recommended GUI flow:
1. Stage 1: choose FDA folder and create metadata CSV
2. Stage 2: choose results directory and analysis options
3. Run full pipeline and monitor live logs

## Quick Start (CLI)

### 1) Build metadata.csv from FDA files

```bash
python metadata_extractor.py \
  --input /path/to/fda_root \
  --output metadata.csv
```

### 2) Run analysis

```bash
python data_extractor_paired.py \
  --input metadata.csv \
  --output results
```

Default behavior for Stage 2:
- Pairing mode: `first_vs_all`
- Alignment mode: `no-aligned` (no pairing/registration; process supported scans as unpaired)
- Layers: `cpRNFL`

## GUI Workflow

Launch:

```bash
python oct_analysis_gui.py
```

The GUI allows:
- Creating new metadata or using an existing metadata CSV
- Selecting layers, fixation, and instrument filters
- Choosing alignment mode (`no-aligned`, `aligned`, `both`)
- Running the full pipeline with live logs

## Stage 1 CLI Reference (`metadata_extractor.py`)

Required:
- `--input`, `-i`: Root folder containing FDA files (recursive scan)
- `--output`, `-o`: Metadata CSV path to write

Optional:
- `--workers`, `-w`: Number of worker processes

Example:

```bash
python metadata_extractor.py -i /data/fda -o metadata.csv -w 8
```

## Stage 2 CLI Reference (`data_extractor_paired.py`)

Required:
- `--input`, `-i`: Metadata CSV file
- `--output`, `-o`: Output directory

Core optional flags:
- `--mode`, `-m`: `all_pairs` | `first_vs_all` | `first_vs_second`
- `--alignment`, `-a`: `no-aligned` | `aligned` | `both`
- `--layers`: Comma-separated layers from `cpRNFL,GCL+,GCL++,Retina`
- `--fixation`: `All` | `3D Wide` | `Macula` | `Disc`
- `--instrument`: `Both` | `Maestro` | `Triton`
- `--workers`, `-w`: Worker count
- `--output-base-name`: Base output name (default `scan_metrics`)
- `--behaviour`: `data_extractor` | `imageNET`
- `--no-resume`: Ignore checkpoint and start fresh

Examples:

```bash
# First vs all pairs, aligned output only
python data_extractor_paired.py -i metadata.csv -o results --mode first_vs_all --alignment aligned

# Export both aligned and unaligned paired outputs
python data_extractor_paired.py -i metadata.csv -o results --alignment both

# Multi-layer extraction
python data_extractor_paired.py -i metadata.csv -o results --layers cpRNFL,GCL+,GCL++,Retina
```

## Output Files

For each selected layer, output naming is based on:
- cpRNFL -> `TSNIT`
- GCL+ / GCL++ -> `Macula6`
- Retina -> `ETDRS`

Common outputs in `results/`:
- `<base>_TSNIT.csv`, `<base>_Macula6.csv`, `<base>_ETDRS.csv`
- `<base>_TSNIT_aligned.csv` / `_unaligned.csv` (depending on alignment mode)
- `error_log.csv`
- `checkpoint.done`
- `metadata_mismatches.csv` (only created if mismatches are detected)

Alignment mode behavior:
- `no-aligned`: Writes only non-aligned outputs
- `aligned`: Writes aligned outputs; unalignable scans/pairs are routed to `_unaligned.csv`
- `both`: Writes aligned output plus non-aligned output

## Input Metadata Schema

Expected columns in metadata CSV:
- `filepath`
- `patient_id`
- `eye`
- `capture_date`
- `capture_time`
- `model_name`
- `fixation`
- `scan_mode`
- `data_no`

Also supported:
- `full_timestamp` (recommended for reliable chronological sorting)

## Tracked Project Files

- `metadata_extractor.py`: Stage 1 metadata extraction
- `data_extractor_paired.py`: Stage 2 analysis pipeline
- `oct_analysis_gui.py`: Desktop GUI
- `shared_resources/`: Core processing modules (FDA read, sector metrics, registration, CSV/checkpoint utilities)
- `example_usage.py`: Programmatic usage examples
- `README_paired_analysis.md`: Extended technical reference
- `TODO.md`: Pending improvements

## Known Limitations

- Cross-device registration normalization is not fully implemented yet.
  - If paired Wide scans have model/resolution mismatch, registration is intentionally skipped and processing falls back to non-aligned output for safety.
- Disc workflow is backend-capable but GUI exposure is intentionally restricted until full validation is complete.

## Notes

- Processing is checkpointed; reruns resume automatically unless `--no-resume` is used.
- Failed reads and processing errors are appended to `error_log.csv`.
- The default negative-thickness behavior is `data_extractor`.

## License

Internal research tool.
