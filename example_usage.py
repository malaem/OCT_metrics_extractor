"""Example usage script for longitudinal RNFL analysis system.

This script demonstrates how to use the paired analysis system with sample data.
Includes examples for different pairing modes and usage patterns.

Author: Marco Miranda
Date: 28 May 2026
"""

import pandas as pd
from pathlib import Path

# Import main function
from data_extractor_paired import run_paired_analysis


def create_sample_metadata():
    """Create sample metadata CSV for testing.
    
    Creates a minimal metadata file with fictional data that meets
    the required format for the paired analysis system.
    
    Returns
    -------
    str
        Path to created sample metadata file.
    """
    # Sample data: 2 patients, each with 3 scans
    data = {
        'patient_id': ['P001', 'P001', 'P001', 'P002', 'P002', 'P002'],
        'eye': ['OD', 'OD', 'OD', 'OS', 'OS', 'OS'],
        'filepath': [
            '/data/P001_baseline.fda',
            '/data/P001_6mo.fda',
            '/data/P001_12mo.fda',
            '/data/P002_baseline.fda',
            '/data/P002_6mo.fda',
            '/data/P002_12mo.fda'
        ],
        'capture_date': [
            '2023-01-15', '2023-07-15', '2024-01-15',
            '2023-02-10', '2023-08-10', '2024-02-10'
        ],
        'capture_time': [
            '10:30', '14:15', '09:00',
            '11:00', '15:30', '10:15'
        ],
        'model_name': ['3D OCT-1'] * 6,
        'fixation': ['Wide'] * 6,
        'scan_mode': ['3D(H)'] * 6
    }
    
    df = pd.DataFrame(data)
    
    # Create sample_metadata.csv in current directory
    output_path = 'sample_metadata.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Created sample metadata: {output_path}")
    print(f"  Patients: 2")
    print(f"  Scans per patient: 3")
    print(f"  Total scans: 6")
    
    return output_path


def example_all_pairs():
    """Example 1: Process all C(n,2) pairs within each patient-eye."""
    print("\n" + "="*80)
    print("EXAMPLE 1: All Pairs Mode")
    print("="*80)
    print("Generates all chronological combinations: (1,2), (1,3), (2,3)")
    print("For 3 scans: 3 pairs per patient-eye")
    print()
    
    metadata_file = create_sample_metadata()
    
    # Run analysis
    run_paired_analysis(
        metadata_csv=metadata_file,
        output_dir='results_all_pairs/',
        output_base_name='scan_metrics',
        pairing_mode='all_pairs',
        alignment_mode='no-aligned',
        layers_to_extract=['cpRNFL'],
        fixation_filter='All',
        instrument_filter='Both',
        n_workers=2,  # Use fewer workers for testing
        resume=True,
        behaviour='data_extractor'
    )
    
    print("\nExample 1 complete. Check results_all_pairs/ directory.")


def example_first_vs_all():
    """Example 2: Baseline vs all follow-ups."""
    print("\n" + "="*80)
    print("EXAMPLE 2: First vs All Mode")
    print("="*80)
    print("Pairs baseline scan with each follow-up: (1,2), (1,3)")
    print("For 3 scans: 2 pairs per patient-eye")
    print()
    
    metadata_file = 'sample_metadata.csv'
    
    # Run analysis
    run_paired_analysis(
        metadata_csv=metadata_file,
        output_dir='results_first_vs_all/',
        output_base_name='scan_metrics',
        pairing_mode='first_vs_all',
        alignment_mode='no-aligned',
        layers_to_extract=['cpRNFL'],
        fixation_filter='All',
        instrument_filter='Both',
        n_workers=2,
        resume=True,
        behaviour='data_extractor'
    )
    
    print("\nExample 2 complete. Check results_first_vs_all/ directory.")


def example_first_vs_second():
    """Example 3: Only baseline vs first follow-up."""
    print("\n" + "="*80)
    print("EXAMPLE 3: First vs Second Mode")
    print("="*80)
    print("Pairs only first two chronological scans: (1,2)")
    print("For 3 scans: 1 pair per patient-eye")
    print()
    
    metadata_file = 'sample_metadata.csv'
    
    # Run analysis
    run_paired_analysis(
        metadata_csv=metadata_file,
        output_dir='results_first_vs_second/',
        output_base_name='scan_metrics',
        pairing_mode='first_vs_second',
        alignment_mode='no-aligned',
        layers_to_extract=['cpRNFL'],
        fixation_filter='All',
        instrument_filter='Both',
        n_workers=2,
        resume=True,
        behaviour='data_extractor'
    )
    
    print("\nExample 3 complete. Check results_first_vs_second/ directory.")


def example_inspect_outputs():
    """Example 4: Inspect output files after processing."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Inspecting Output Files")
    print("="*80)
    
    output_dir = Path('results_all_pairs')
    
    if not output_dir.exists():
        print(f"Output directory {output_dir} not found.")
        print("Run example_all_pairs() first.")
        return
    
    # Check unpaired scans
    unpaired_file = output_dir / 'unpaired_scans.csv'
    if unpaired_file.exists():
        df_unpaired = pd.read_csv(unpaired_file)
        print(f"\nUnpaired scans: {len(df_unpaired)} rows")
        print(f"Columns: {list(df_unpaired.columns[:10])}... ({len(df_unpaired.columns)} total)")
        print("\nFirst row sample:")
        print(df_unpaired.iloc[0][['patient_id', 'eye', 'capture_date', 'Total', '4_T', '4_S']].to_dict())
    
    # Check paired scans
    paired_file = output_dir / 'paired_scans.csv'
    if paired_file.exists():
        df_paired = pd.read_csv(paired_file)
        print(f"\n\nPaired scans: {len(df_paired)} rows")
        print(f"Columns: {list(df_paired.columns[:10])}... ({len(df_paired.columns)} total)")
        print("\nFirst row sample (reference scan):")
        ref_cols = [c for c in df_paired.columns if c.endswith('_ref')][:6]
        print(df_paired.iloc[0][ref_cols].to_dict())
    
    # Check errors
    error_file = output_dir / 'error_log.csv'
    if error_file.exists():
        df_errors = pd.read_csv(error_file)
        print(f"\n\nErrors: {len(df_errors)} files failed")
        if len(df_errors) > 0:
            print("\nError summary:")
            print(df_errors['status'].value_counts())
    else:
        print("\n\nNo error log (all files processed successfully)")
    
    print("\n" + "="*80)


def run_all_examples():
    """Run all examples in sequence."""
    print("="*80)
    print("RUNNING ALL EXAMPLES")
    print("="*80)
    print("\nNote: These examples use SAMPLE DATA (not real FDA files).")
    print("They will fail at the FDA reading stage, but demonstrate the workflow.")
    print("\nPress Ctrl+C to interrupt at any time.")
    print()
    
    try:
        example_all_pairs()
        example_first_vs_all()
        example_first_vs_second()
        example_inspect_outputs()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETE")
        print("="*80)
        print("\nTo use with real data:")
        print("  1. Prepare metadata CSV with actual FDA file paths")
        print("  2. Run: python data_extractor_paired.py -i metadata.csv -o results/")
        print("  3. Check results/ directory for outputs")
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n\nExample failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # Choose which example to run
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '1':
            example_all_pairs()
        elif sys.argv[1] == '2':
            example_first_vs_all()
        elif sys.argv[1] == '3':
            example_first_vs_second()
        elif sys.argv[1] == '4':
            example_inspect_outputs()
        elif sys.argv[1] == 'all':
            run_all_examples()
        else:
            print("Usage: python example_usage.py [1|2|3|4|all]")
            print("  1 = All pairs mode")
            print("  2 = First vs all mode")
            print("  3 = First vs second mode")
            print("  4 = Inspect outputs")
            print("  all = Run all examples")
    else:
        # Default: run all examples
        run_all_examples()
