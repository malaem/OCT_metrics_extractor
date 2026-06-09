"""FDA file processing for OCT retinal layer thickness extraction.

This module handles reading FDA files, extracting retinal layer thickness maps
(cpRNFL, GCL+, GCL++, Retina), and preparing data for sector analysis with
native eye laterality and coordinate transformations.

Author: Marco Miranda
Date: 28 May 2026
"""

import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
from dateutil.relativedelta import relativedelta

from shared_resources.read_fda_file import read_fda_common_info
from shared_resources.f2d_angle_distance import f2d_angle_distance
from shared_resources.build_enface import compute_enface_slab
from shared_resources.select_layer import selectLayer
from shared_resources.useful_formulas import estimateAL, littmann, littmann_AsInMaestro


def _get_littmann_magnification(model_name: str) -> float:
    """Get Littmann magnification factor for device model.
    
    The Littmann magnification corrects for individual eye length variations
    in retinal imaging. Different OCT models have slightly different optical
    geometries requiring model-specific corrections.
    
    Parameters
    ----------
    model_name : str
        OCT device model name from FDA file general_info.
    
    Returns
    -------
    float
        Littmann magnification factor. Typically close to 1.0.
    
    Raises
    ------
    ValueError
        If model_name is not recognized.
    
    Notes
    -----
    Known models and their factors:
    - '3D OCT-1', '3DOCT-1Maestro2': 1.0 (no correction needed)
    - 'Triton plus': 0.9999999997039188 (effectively 1.0, legacy value)
    - '3DOCT-2000FA': 1.0 (placeholder; actual value unknown)
    """
    littmann_factors = {
        '3D OCT-1': 1.0,
        '3DOCT-1Maestro2': 1.0,
        'Triton plus': 0.9999999997039188,
        '3DOCT-2000FA': 1.0  # Placeholder; fovea-to-disc measurements may be inaccurate
    }
    
    if model_name not in littmann_factors:
        raise ValueError(
            f"Unknown model name: '{model_name}'. "
            f"Known models: {list(littmann_factors.keys())}"
        )
    
    return littmann_factors[model_name]


def extract_rnfl_thickness(
    seg_data: np.ndarray,
    scan_axial_res: float,
    eye: str,
) -> Tuple[np.ndarray, int, int]:
    """Extract RNFL thickness map from segmentation data.
    
    Computes thickness as (ILM - RNFL) * axial_resolution, transposes to standard
    orientation, and flips left eyes horizontally to match OD (right eye) convention.
    
    Parameters
    ----------
    seg_data : np.ndarray
        3D segmentation data array from FDA file, shape (H, W, n_layers).
        Layer 0 = RNFL (inner boundary)
        Layer 1 = ILM (outer boundary)
    scan_axial_res : float
        Axial resolution in micrometers per pixel (µm/pixel).
    eye : str
        Eye laterality, 'L' (left) or 'R' (right).
    
    Returns
    -------
    thickness_1d : np.ndarray
        Flattened RNFL thickness array in micrometers, shape (H*W,).
    original_height : int
        Original height before flattening.
    original_width : int
        Original width before flattening.
    
    Notes
    -----
    Processing steps:
    1. Compute thickness: (seg_data[:,:,1] - seg_data[:,:,0]) * scan_axial_res
    2. Transpose to swap height/width (FDA convention → standard image convention)
    3. If left eye: flip horizontally to match OD orientation
    4. Flatten to 1D for downstream processing
    
    All thickness values are in micrometers (µm).
    """
    # Compute RNFL thickness: (ILM - RNFL) * axial_resolution
    thickness_2d = (seg_data[:, :, 1] - seg_data[:, :, 0]).astype(float) * scan_axial_res
    
    # Transpose to standard image orientation
    thickness_2d = np.transpose(thickness_2d)
    
    # Store original dimensions
    original_height, original_width = thickness_2d.shape
    
    # Flatten to 1D
    thickness_1d = thickness_2d.reshape(original_height * original_width)
    
    return thickness_1d, original_height, original_width


def _extract_layer_thickness(
    seg_data: np.ndarray,
    scan_axial_res: float,
    layer_name: str,
    eye: str,
) -> Tuple[np.ndarray, int, int]:
    """Extract specific layer thickness from segmentation data.
    
    Computes thickness for requested layer using selectLayer() to get boundaries,
    transposes to standard orientation, and flips left eyes horizontally.
    
    Parameters
    ----------
    seg_data : np.ndarray
        3D segmentation data array from FDA file, shape (H, W, n_layers).
    scan_axial_res : float
        Axial resolution in micrometers per pixel (µm/pixel).
    layer_name : str
        Layer to extract: 'cpRNFL', 'GCL+', 'GCL++', 'Retina'.
    eye : str
        Eye laterality, 'L' (left) or 'R' (right).
    
    Returns
    -------
    thickness_1d : np.ndarray
        Flattened thickness array in micrometers, shape (H*W,).
    original_height : int
        Original height before flattening.
    original_width : int
        Original width before flattening.
    
    Notes
    -----
    Layer definitions (from selectLayer):
    - cpRNFL: ILM to RNFL  
    - GCL+: RNFL to INL (GCL+IPL)
    - GCL++: RNFL to OPL (GCL+IPL+INL)
    - Retina: ILM to RPE (full thickness)
    
    Processing steps:
    1. Get layer boundaries from selectLayer()
    2. Compute thickness: (boundary2 - boundary1) * scan_axial_res
    3. Transpose to standard image orientation
    4. If left eye: flip horizontally to match OD orientation
    5. Flatten to 1D
    
    All thickness values are in micrometers (µm).
    """
    # Map layer name to selectLayer argument
    layer_map = {
        'cpRNFL': 'RNFL',
        'GCL+': 'GCL+',
        'GCL++': 'GCL++',
        'Retina': 'Retina'
    }
    
    if layer_name not in layer_map:
        raise ValueError(
            f"Unknown layer name: '{layer_name}'. "
            f"Valid options: {list(layer_map.keys())}"
        )
    
    # Get layer boundaries
    seg_layer1, seg_layer2 = selectLayer(layer_map[layer_name])
    
    # Compute thickness and convert to micrometers
    thickness_2d = (
        seg_data[:, :, seg_layer2.value] - seg_data[:, :, seg_layer1.value]
    ).astype(float) * scan_axial_res
    
    # Transpose to standard image orientation
    thickness_2d = np.transpose(thickness_2d)
    
    # Store original dimensions
    original_height, original_width = thickness_2d.shape
    
    # Flatten to 1D
    thickness_1d = thickness_2d.reshape(original_height * original_width)
    
    return thickness_1d, original_height, original_width


def build_enface_image_for_registration(scan_data: Dict[str, Any]) -> np.ndarray:
    """Build en face image from scan data for registration.
    
    Creates an en face projection from the 3D OCT volume optimized for
    KAZE feature detection. Uses 'registration' projection mode which
    emphasizes vessel shadows and structural edges.
    
    Parameters
    ----------
    scan_data : dict
        Dictionary from read_fda_scan() containing:
        - oct3d : np.ndarray, 3D OCT volume
        - seg_data : np.ndarray, segmentation layers
        - thickness_width : int
        - thickness_height : int
    
    Returns
    -------
    np.ndarray
        2D en face image as float32, optimized for registration.
    
    Notes
    -----
    - Uses layer 5 with -50 pixel offset for en face projection (matches imageAlignment.py)
    - 'sum' projection mode for registration
    - Image is transposed to match thickness map orientation
    """
    # Extract from scan_data
    oct3d = scan_data['oct3d']
    seg_data = scan_data['seg_data']
    
    # Build en face using registration-optimized projection
    # Layer 5 with -50 offset (matches imageAlignment.py)
    enface = compute_enface_slab(
        oct3d=oct3d,
        seg_data=seg_data,
        layer1=5,
        layer2=5,
        layer1_offset=-50,
        projection='sum'
    )
    
    # Transpose to match thickness orientation (scan_processor transposes thickness)
    enface = np.transpose(enface)

    return enface.astype(np.float32)


def read_fda_scan(
    filepath: str,
    layers_to_extract: List[str] = ['cpRNFL'],
    read_oct3d: bool = False,
) -> Optional[Dict[str, Any]]:
    """Read FDA file and extract multiple retinal layers from 3D OCT scans.
    
    Main entry point for reading FDA files. Extracts metadata and thickness maps
    for requested layers. Returns None for scans that don't meet inclusion
    criteria (non-OCT scans, non-3D scan modes, etc.).
    
    Parameters
    ----------
    filepath : str
        Path to FDA file.
    layers_to_extract : List[str], default ['cpRNFL']
        List of layer names to extract. Options: 'cpRNFL', 'GCL+', 'GCL++', 'Retina'.
    read_oct3d : bool, default False
        Whether to read the 3D OCT volume (oct3d). Set to True only when needed for 
        en face image generation (e.g., for image alignment). Reading oct3d is slow.
    
    Returns
    -------
    dict or None
        Dictionary with extracted data if scan is eligible, None otherwise.
        
        Dictionary keys:
        - Common metadata fields (patient_id, gender, dob, age, model_name, etc.)
        - layers : dict, keyed by layer name, each containing:
            - thickness : np.ndarray, shape (H*W,)
            - thickness_height : int
            - thickness_width : int
            - layer_name : str
        - For backward compatibility:
            - thickness : np.ndarray (first layer in layers_to_extract)
            - thickness_height : int
            - thickness_width : int
    
    Raises
    ------
    Exception
        Any exception during file reading is caught and re-raised with context.
    
    Notes
    -----
    - Returns None for:
        * Fundus-only scans (no OCT data)
        * Non-3D(H) and non-3D(V) scan modes
    - Supports all fixation types: Wide, Macula, Disc, External.
      Layer compatibility per fixation is enforced by the caller via LAYER_FIXATION_COMPAT.
        - Landmark coordinates are stored both as transformed (for internal geometry)
            and raw FDA values (for CSV export and reference comparisons).
    - Reads FDA file once and extracts all requested layers for efficiency.
    """
    try:
        # Read FDA file (conditionally read oct3d for alignment)
        # Note: Must set read_images=True to actually populate oct3d
        general_info, patient_info, capture_info, scan_info, disc_info = (
            read_fda_common_info(filepath, read_oct3d=read_oct3d, read_images=read_oct3d)
        )
        
        # Skip fundus-only scans
        if capture_info.capture_mode in ('Fundus only', 'Fundus Photo only'):
            return None
        
        # Skip non-3D scans
        if scan_info.scan_mode not in ('3D(H)', '3D(V)'):
            return None
        
        # Get device-specific Littmann magnification (used for calculations)
        try:
            littmann_mag = _get_littmann_magnification(general_info.model_name)
        except ValueError as e:
            # Re-raise with file context
            raise ValueError(f"File {filepath}: {e}")
        
        # Extract requested layers
        layers_data = {}
        for layer_name in layers_to_extract:
            thickness_1d, thickness_height, thickness_width = _extract_layer_thickness(
                scan_info.seg_data,
                scan_info.scan_axial_res,
                layer_name,
                capture_info.eye,
            )
            layers_data[layer_name] = {
                'thickness': thickness_1d,
                'thickness_height': thickness_height,
                'thickness_width': thickness_width,
                'layer_name': layer_name
            }
        
        # Get anatomical landmarks (fractional coordinates [0,1])
        # Manual landmarks (raw values from FDA file)
        disc_center_x_raw = scan_info.regist_info.disc_center_manual_x
        disc_center_y_raw = scan_info.regist_info.disc_center_manual_y
        fovea_x_raw = scan_info.regist_info.fovea_manual_x
        fovea_y_raw = scan_info.regist_info.fovea_manual_y
        
        # Auto landmarks (raw values from FDA file)
        disc_center_auto_x_raw = scan_info.regist_info.disc_center_auto_x
        disc_center_auto_y_raw = scan_info.regist_info.disc_center_auto_y
        fovea_auto_x_raw = scan_info.regist_info.fovea_auto_x
        fovea_auto_y_raw = scan_info.regist_info.fovea_auto_y
        
        # Native x coordinates are preserved (no OD mirroring).
        disc_center_x_transformed = disc_center_x_raw
        fovea_x_transformed = fovea_x_raw
        disc_center_auto_x_transformed = disc_center_auto_x_raw
        fovea_auto_x_transformed = fovea_auto_x_raw
        
        # COORDINATE CONVENTION FLIP (y): FDA uses bottom-left origin while
        # our processing stack uses top-left image coordinates.
        disc_center_y_transformed = 1.0 - disc_center_y_raw
        fovea_y_transformed = 1.0 - fovea_y_raw
        disc_center_auto_y_transformed = 1.0 - disc_center_auto_y_raw
        fovea_auto_y_transformed = 1.0 - fovea_auto_y_raw
        
        # Calculate fovea-to-disc distance and angle using transformed coordinates.
        # Only meaningful for Wide scans (both disc and fovea are within the scan area).
        # For Macula scans the disc is outside the FOV; for Disc scans the fovea is.
        if scan_info.fixation == 'Wide':
            f2d_distance, f2d_angle = f2d_angle_distance(
                [disc_center_x_transformed, disc_center_y_transformed],
                [fovea_x_transformed, fovea_y_transformed],
                scan_info.scan_size_set,
                capture_info.eye,
                littmann_mag
            )
        else:
            f2d_distance, f2d_angle = None, None
        
        # Estimate axial length
        estimated_AL = estimateAL(scan_info)
        
        # Calculate comparison magnifications (for validation/analysis only)
        littmann_mag_bennett = littmann(estimated_AL)
        spherical_equivalent = patient_info.spherical_power + (patient_info.astimatism_deg / 2) if patient_info.spherical_power is not None and patient_info.astimatism_deg is not None else None
        # Guard: littmann_AsInMaestro requires all three parameters; fall back to None
        # when biometry (@GLA_LITTMANN_01 block) is absent from the FDA file.
        # This is normal — the block is only present when the device measured biometry.
        if spherical_equivalent is not None and patient_info.horizontal_corneal_radius is not None and patient_info.axial_length is not None:
            littmann_mag_maestro = littmann_AsInMaestro(Dg=spherical_equivalent, R1=patient_info.horizontal_corneal_radius, Im=patient_info.axial_length)
        else:
            littmann_mag_maestro = None
        
        # Calculate age if DOB available
        if patient_info.birth_date is not None:
            age = relativedelta(capture_info.capture_date, patient_info.birth_date).years
        else:
            age = None
        
        # Assemble return dictionary
        return {
            'oct3d': scan_info.oct3d,  # Store for en face generation if alignment requested
            'seg_data': scan_info.seg_data,  # Store for en face generation
            'patient_id': patient_info.patient_id,
            'gender': patient_info.gender,
            'dob': patient_info.birth_date,
            'age': age,
            'model_name': general_info.model_name,
            'data_no': general_info.data_no,
            'eye': capture_info.eye,
            'capture_date': capture_info.capture_date,
            'capture_time': capture_info.capture_time,
            'capture_mode': capture_info.capture_mode,
            'fixation': scan_info.fixation,
            'focus_mode': scan_info.focus_mode,
            'mirror_pos': scan_info.mirror_pos,
            'scan_mode': scan_info.scan_mode,
            'scan_protocol': scan_info.scan_protocol,
            'scan_resolution': scan_info.scan_resolution,      # formatted string e.g. "512x128"
            'scan_size': scan_info.scan_size,                  # formatted string e.g. "12.0x9.0"
            'scan_resolution_set': scan_info.scan_resolution_set,  # tuple (width_px, height_px)
            'scan_size_set': scan_info.scan_size_set,              # tuple (width_mm, height_mm)
            'scan_axial_res': scan_info.scan_axial_res,
            'jpeg_height': scan_info.scan_jpeg_height,
            'z_mean': scan_info.z_mean,
            'q_mean': scan_info.q_mean,
            'top_q': scan_info.top_q,
            'estimated_AL': estimated_AL,
            'littmann_mag': littmann_mag,
            'littmann_mag_bennett': littmann_mag_bennett,
            'littmann_mag_maestro': littmann_mag_maestro,
            # Manual landmarks used by internal geometry calculations
            # (native-eye x, Y-flipped to top-left origin)
            'disc_center_x': disc_center_x_transformed,
            'disc_center_y': disc_center_y_transformed,
            'fovea_x': fovea_x_transformed,
            'fovea_y': fovea_y_transformed,
            # Auto landmarks used by internal geometry calculations
            # (native-eye x, Y-flipped to top-left origin)
            'disc_center_auto_x': disc_center_auto_x_transformed,
            'disc_center_auto_y': disc_center_auto_y_transformed,
            'fovea_auto_x': fovea_auto_x_transformed,
            'fovea_auto_y': fovea_auto_y_transformed,
            # Raw landmarks from FDA file (native image orientation).
            # Export these values in output CSVs for direct comparison with
            # Topcon reference exports.
            'disc_center_x_raw': disc_center_x_raw,
            'disc_center_y_raw': disc_center_y_raw,
            'fovea_x_raw': fovea_x_raw,
            'fovea_y_raw': fovea_y_raw,
            'disc_center_auto_x_raw': disc_center_auto_x_raw,
            'disc_center_auto_y_raw': disc_center_auto_y_raw,
            'fovea_auto_x_raw': fovea_auto_x_raw,
            'fovea_auto_y_raw': fovea_auto_y_raw,
            'f2d_distance': f2d_distance,
            'f2d_angle': f2d_angle,
            # Disc metrics (for cpRNFL)
            'disc_area': disc_info.actual_disc_area if disc_info else None,
            'cup_area': disc_info.cup_area if disc_info else None,
            'rim_area': disc_info.rim_area if disc_info else None,
            'cup_volume': disc_info.cup_volume if disc_info else None,
            'disc_volume': disc_info.disc_volume if (disc_info and hasattr(disc_info, 'disc_volume')) else None,
            'vertical_disc_diameter': disc_info.vertical_disc_diameter if disc_info else None,
            'horizontal_disc_diameter': disc_info.horizontal_disc_diameter if disc_info else None,
            # Multi-layer data
            'layers': layers_data,
            'filepath': filepath
        }
    
    except Exception as e:
        # Re-raise with file context
        raise Exception(f"Error reading FDA file {filepath}: {type(e).__name__}: {e}") from e
