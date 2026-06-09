"""Shared utilities for the OCT metrics extractor pipeline.

This package contains helper modules used across the pipeline:

- alignment_schemes          : Registration preset configurations
- benchmark_registration     : Tools for evaluating registration quality
- build_enface               : Construct en face projection images from OCT volumes
- checkpoint_manager         : Checkpoint/resume support for long-running batch jobs
- convert_to_OD_orientation  : Flip scans to a consistent OD (right-eye) orientation
- csv_writer                 : Atomic CSV writing and header management
- f2d_angle_distance         : Fovea-to-disc distance and angle calculations
- getcpRNFLAngle             : cpRNFL sector angle definitions
- getMaculaAngle             : Macula sector angle definitions
- grid_diameter              : Annular pixel selection around a scan centre
- maryfdaQ                   : Scan quality scoring (Mary's FDA quality check)
- OCTenfaceWideRegistration  : KAZE-based en face image registration
- pairing_utils              : Longitudinal scan pair generation from metadata
- read_fda_file              : Low-level Topcon FDA binary file parser
- root_folder                : Workspace root path resolution
- save_enface_image          : Save en face images as PNGs with correct proportions
- scan_processor             : FDA scan reading and retinal layer thickness extraction
- sector_metrics             : Sector-wise thickness metrics for cpRNFL, GCL+, GCL++, Retina
- sectorAverage              : Sector-wise thickness averaging
- select_layer               : Retinal layer boundary selector
- useful_formulas            : Ocular magnification and axial length formulae

Author: Marco Miranda
Date: 28 May 2026
"""

