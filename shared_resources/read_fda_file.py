"""read_fda_file.py — parse Topcon FDA binary files.

Reads a .fda file and extracts patient information, capture metadata, scan
parameters, and (optionally) the full OCT volume with segmentation contours.

Key public function: ``read_fda_common_info``.

Notes on missing data
---------------------
* ``patient_info.birth_date`` is ``None`` when the FDA block contains all-zero
  date components (year == 0 or month == 0 or day == 0).  Callers that compute
  patient age via ``relativedelta`` must guard against this, e.g.::

      age = (
          relativedelta(capture_date, birth_date).years
          if birth_date is not None else None
      )

FDA block reference
-------------------
# read_fda_file.py reads in a .fda file and extracts the patient information, capture information, and scan information.
# Export Date
# Export Time
# Serial No.: @HW_INFO_03.Instrument Serial No.
# Data No.: Fda filename
# Patient ID: @PATIENT_INFO_02.Patient ID
# Last Name: @PATIENT_INFO_02.Last Name
# First Name: @PATIENT_INFO_02.First Name
# Gender: @PATIENT_INFO_02.Gender
# DOB: @PATIENT_INFO_02.DOB
# Ethnicity: @PATIENTEXT_INFO.Ethnic Group
# (skip) Diagnosis1, Diagnosis2, Diagnosis3, Diagnosis4
# Eye: @CAPTURE_INFO_02.Eye
# Capture Mode (new): @CAPTURE_INFO_02.Capture Mode (OCT only, OCT+Fundus, Fundus only)
# Capture Date: @CAPTURE_INFO_02.Capture Date
# Capture Time: @CAPTURE_INFO_02.Capture Time
# Scan Size: @PARAM_SCAN_04.Real Scan Length (1)(2)
# Scan Resolution: (@IMG_JPEG:Image width)x(@IMG_JPEG:Number of images)
# Fixation: @PARAM_SCAN_04.Fixation position
# TopQ Image Quality: @FAST_Q2_INFO.Qmean (.2f)
# Mirror position: @PARAM_SCAN_04.Mirror Position
# Z-mean: @FAST_Q2_INFO.Zmean (.2f)
# OCT Focus Mode: @PARAM_SCAN_04.Reference position of a reference mirror 
#                 (0, negative -> "Deep pos" or "Choroidal") based on data file type CFDSIO::FAA_DATA_FILE, else (Scan protocol?)
#                 (1, positive -> "Cornea" or "Vitreous")
# Model Name: @HW_INFO_03.Instrument Model Name
# Capture Software Ver.: @HW_INFO_03.Main body application software version
# Analysis Software Ver.: @FDA_FILE_INFO.Executed file name
# Analysis Mode: @REPORT_INFO.Thinned-out analysis (0: "Fine"; 5: "Basic"; else: "n/a")
# Manual Disc Center Position X: @REGIST_INFO.Disc Manual grid position X
# Manual Disc Center Position Y: @REGIST_INFO.Disc Manual grid position Y
# Manual Fovea Position X: @REGIST_INFO.Fovea Manual grid position X
# Manual Fovea Position Y: @REGIST_INFO.Fovea Manual grid position Y
# Auto Disc Center Position X: @REGIST_INFO.Disc Auto grid position X
# Auto Disc Center Position Y: @REGIST_INFO.Disc Auto grid position Y
# Auto Fovea Position X: @REGIST_INFO.Fovea Auto grid position X
# Auto Fovea Position Y: @REGIST_INFO.Fovea Auto grid position Y
# Segmentation Version: [IGNORE for now]
"""

import sys
import struct
from datetime import date, datetime, time
import os
import numpy as np
from array import array
from PIL import Image
from io import BytesIO

if sys.platform == "win32":
    from wincrypto import CryptCreateHash, CryptHashData, CryptDeriveKey, CryptDecrypt
    from wincrypto.constants import CALG_RC4, CALG_SHA1
    IS_WINDOWS = True
else:
    from Crypto.Hash import SHA1
    from Crypto.Cipher import ARC4
    IS_WINDOWS = False
    # Stub out Windows-only names so static analysers don't flag references in
    # the guarded `if IS_WINDOWS:` branch as undefined. These stubs are never
    # called at runtime because IS_WINDOWS is False on this path.
    CryptCreateHash = CryptHashData = CryptDeriveKey = CryptDecrypt = None  # type: ignore[assignment]
    CALG_SHA1 = CALG_RC4 = None  # type: ignore[assignment]
    
scan_modes = [
        'Line', 'Circle', '3D(H)', 'Radial', 'Cross', 'Web', '3D(V)', 
        'Raster', 'Raster(V)', 'Concentric1', 'Concentric2', '5LineCross'
    ]

class GeneralInfo:
    def __init__(self):
        self.export_date = None
        self.export_time = None
        self.serial_no = None
        self.data_no = None
        self.model_name = None
        self.capture_software_ver = None
        self.analysis_software_ver = None
        self.analysis_mode = None
        self.reanalysis_datetime = None

class PatientInfo:
    def __init__(self):
        self.patient_id = None
        self.first_name = None
        self.last_name = None
        self.gender = None
        self.birth_date = None
        self.ethnicity = None
        self.axial_length = None
        self.horizontal_corneal_radius = None
        self.astimatism_deg = None
        self.astigmatic_axis = None
        self.spherical_power = None
        self.IOL_information = None
        self.correction_lens_info = None
        self.correction_method = None

class CaptureInfo():
    def __init__(self):
        self.eye = None
        self.capture_label = None
        self.capture_mode = None
        self.capture_date = None
        self.capture_time = None

class RegistInfo():
    def __init__(self):
        self.disc_center_manual_x = None
        self.disc_center_manual_y = None
        self.disc_center_auto_x = None
        self.disc_center_auto_y = None
        self.fovea_manual_x = None
        self.fovea_manual_y = None
        self.fovea_auto_x = None
        self.fovea_auto_y = None

class ScanInfo():
    def __init__(self):
        self.fixation = None
        self.mirror_pos = None
        self.regist_info = RegistInfo()
        self.z_mean = None
        self.q_mean = None
        self.scan_mode = None
        self.scan_size = None
        self.scan_size_set = None
        self.scan_axial_res = None
        self.focus_mode = None
        self.scan_protocol = None
        self.scan_resolution = None
        self.scan_resolution_set = None
        self.seg_data = None
        self.oct3d = None
        self.scan_jpeg_height = None
        self.top_q = None

class FdaDiskSeg():
    def __init__(self):
        self.regist_info = RegistInfo()
        self.disc_seg_version = None
        self.disc_left_x = None
        self.disc_left_y = None
        self.disc_bottom_x = None
        self.disc_bottom_y = None
        self.disc_right_x = None
        self.disc_right_y = None
        self.disc_top_x = None
        self.disc_top_y = None
        self.horizontal_disc_diameter = None
        self.vertical_disc_diameter = None
        self.actual_disc_area = None
        self.disc_projected_area = None
        self.disc_volume = None
        self.cup_area = None
        self.cup_volume = None
        self.rim_area = None
        self.disc_seg_data = None
        self.cup_seg_data = None
        self.reference_surface_offset = None

def read_fda_common_info(filename, read_oct3d=False, read_images=False, debug=False):
    """Parse a .fda file and return all available metadata and (optionally) the OCT volume.

    Parameters
    ----------
    filename : str or Path
        Path to the .fda binary file.
    read_oct3d : bool, optional
        If ``True``, also decode the compressed OCT volume (``@IMG_JPEG`` block)
        and all contour layers (``@CONTOUR_INFO`` blocks).  The parsed data are
        stored in ``scan_info.oct3d`` and ``scan_info.seg_data`` respectively.
        Default ``False`` (faster; metadata-only).
    debug : bool, optional
        If ``True``, print the 15-byte file header (file code, type,
        major/minor version) and each block label as it is parsed.

    Returns
    -------
    general_info : GeneralInfo
        Export timestamp, instrument serial number, model name, software
        versions, analysis mode.  ``export_date`` / ``export_time`` are set
        to the current clock time at parse time (not stored in the file).
    patient_info : PatientInfo
        Patient demographics.  ``birth_date`` may be ``None`` when the FDA
        file records all-zero date components; callers must guard against this
        before computing age with ``relativedelta``.
    capture_info : CaptureInfo
        Eye laterality, capture mode, and capture date/time.
    scan_info : ScanInfo
        Scan geometry, quality metrics, landmark positions, and (if
        ``read_oct3d=True``) the OCT volume and segmentation arrays.
    disc_info : FdaDiskSeg
        Optic disc segmentation measurements (horizontal/vertical diameters,
        areas, volumes, cup measurements, pixel mask).

    Notes
    -----
    ``nBlockBytes`` is initialised to ``(0,)`` before the parsing loop so that
    the error-recovery seek (used when a corrupted block label forces a re-seek)
    is always referencing a defined value, even if the very first block read
    raises an exception.
    """
    patient_info = PatientInfo()
    capture_info = CaptureInfo()
    scan_info = ScanInfo()
    general_info = GeneralInfo()
    disc_info = FdaDiskSeg()

    with open(filename,'rb') as f:        
        if debug: 
            filecode = f.read(4)
            filetype = f.read(3)        
            major_ver = struct.unpack('i', f.read(4))
            minor_ver = struct.unpack('i', f.read(4))
            print(filecode.decode("utf-8"), filetype.decode("utf-8"))
            print(major_ver, minor_ver)
        else:
            f.read(15)
        
        error_flag = False
        nBlockBytes = (0,)   # sentinel so the error-recovery seek below is always defined
        image_width = None
        num_images = None
        jpeg_height = None
        scan_w = None
        scan_h = None
        scan_axial_res = None
        # Always initialize contour arrays for OCT scans (needed for thickness calculations)
        n_contour = 0
        contour_all = np.array([], dtype = np.int16)
        
        while True:
            curpos = f.tell()
            try:
                lab_length = struct.unpack('B', f.read(1))
            except:
                print(f'corrupted file detected: {filename}')
                break # some files seem to be corrupted in the end of the file (as viewed by FDSViewer)
            if lab_length[0] == 0:
                break
            try:
                labname = f.read(lab_length[0]).decode("utf-8")
            except: # some files seem to have a different encoding at the end of the file
                    # this is a very special corner case, and further investigation may be needed
                error_flag = True
                labname = '@PARAM_SCAN_04'
                f.seek(-lab_length[0]-1-nBlockBytes[0]-4, 1)
                lab_length = (14,) # '@PARAM_SCAN_04'
                curpos = f.tell()-1-lab_length[0]
            if debug:
                print(lab_length, labname)

            nBlockBytes = struct.unpack('I', f.read(4))
            if labname == '@PARAM_SCAN_04' and error_flag:
                nBlockBytes = (86,)

            # Add this debug line to find how many bytes this parameter has:
            #if labname == '@GLA_LITTMANN_01':
            #    print(f"@GLA_LITTMANN_01 block size: {nBlockBytes[0]} bytes")


            match labname:
                case '@FDA_FILE_INFO':
                    general_info.analysis_software_ver = read_fda_file_info(f)
                case '@HW_INFO_03':
                    general_info.model_name, general_info.serial_no, general_info.capture_software_ver = read_hw_info(f)
                case '@PATIENT_INFO_02':
                    patient_info.patient_id, patient_info.first_name, patient_info.last_name, patient_info.gender, patient_info.birth_date = read_patient_info(f)
                case '@PATIENT_INFO_03':
                    patient_info.patient_id, patient_info.first_name, patient_info.last_name, patient_info.gender, patient_info.birth_date = read_patient_info_03(f)
                case '@PATIENTEXT_INFO':
                    patient_info.ethnicity = read_patientext_info(f)
                case '@CAPTURE_INFO_02':
                    capture_info.eye, capture_info.capture_mode, capture_info.capture_label, capture_info.capture_date, capture_info.capture_time = read_capture_info(f)
                case '@PARAM_SCAN_04':
                    scan_info.fixation, scan_info.mirror_pos, scan_w, scan_h, scan_axial_res, scan_info.focus_mode, scan_info.scan_protocol = read_scan_info(f, labname)
                case '@IMG_JPEG':
                    if read_oct3d and read_images:
                        scan_info.scan_mode, image_width, num_images, oct3d, jpeg_height = read_oct_jpeg_octdata(f)
                    else:
                        scan_info.scan_mode, image_width, num_images, jpeg_height = read_oct_jpeg(f)
                case '@REGIST_INFO':
                    if capture_info.capture_mode != 'Fundus only': # although fundus only scans have this info (OCT scan image paramters), it has different specs.
                        scan_info.regist_info = read_regist_info(f)                        
                case '@FAST_Q2_INFO':
                    scan_info.q_mean, scan_info.z_mean = read_fast_q(f)
                case '@REPORT_INFO':
                    general_info.analysis_mode = read_report_info(f)
                case '@REANALYSIS_INFO':
                    general_info.reanalysis_datetime = read_reanalysis_info(f)
                case '@TOPQEXT_INFO':
                    scan_info.top_q = read_top_q(f)
                case '@CONTOUR_INFO':
                    # Always read contour data for OCT scans (needed for thickness calculations)
                    contour = read_contour(f)
                    contour_all = np.append(contour_all, contour)
                    n_contour = n_contour + 1
                case '@FDA_DISC_SEGMENTATION':
                    disc_info.disc_seg_version, disc_info.disc_left_x, disc_info.disc_left_y, disc_info.disc_bottom_x, disc_info.disc_bottom_y, disc_info.disc_right_x, disc_info.disc_right_y, disc_info.disc_top_x, disc_info.disc_top_y, disc_info.horizontal_disc_diameter, disc_info.vertical_disc_diameter, disc_info.actual_disc_area, disc_info.disc_projected_area, disc_info.disc_volume, disc_info.cup_area, disc_info.cup_volume, disc_info.rim_area, disc_info.disc_seg_data, disc_info.cup_seg_data, disc_info.reference_surface_offset = read_disc_info(f)
                # case '@CONTOUR_MASK_INFO':
                #     if read_oct3d:
                #         contour_mask = read_mask(f)
                #         contour_all_mask = np.append(contour_all_mask, contour_mask)
                #         n_contour_mask = n_contour_mask + 1
                case '@GLA_LITTMANN_01':
                    patient_info.axial_length, patient_info.horizontal_corneal_radius, patient_info.astimatism_deg, patient_info.astigmatic_axis, patient_info.spherical_power, patient_info.IOL_information, patient_info.correction_lens_info, patient_info.correction_method = read_littmann_info(f)
            f.seek(curpos + nBlockBytes[0] + lab_length[0] + 1 + 4, 0)
    
    if capture_info.capture_mode != 'Fundus only' and image_width is not None and scan_w is not None:
        if scan_info.scan_mode in ['3D(H)', '3D(V)', 'Raster', 'Raster(V)']:
            scan_info.scan_size = f'{scan_w:.1f}x{scan_h:.1f}'
            scan_info.scan_size_set = (scan_w, scan_h)
            scan_info.scan_axial_res = scan_axial_res
            scan_info.scan_resolution = f'{image_width}x{num_images}'
            scan_info.scan_resolution_set = (image_width, num_images)
            scan_info.scan_jpeg_height = jpeg_height
        else:
            scan_info.scan_resolution = f'{image_width}'
            # if abs(scan_w - int(scan_w)) < 1e-5: # ends in .0
            #     scan_info.scan_size = f'{scan_w:.0f}'
            # else:
            scan_info.scan_size = f'{scan_w:.1f}'
            scan_info.scan_size_set = scan_w
            scan_info.scan_axial_res = scan_axial_res
            for attr in scan_info.regist_info.__dict__.keys():
                setattr(scan_info.regist_info, attr, 0.5)
    
    current_time = datetime.now()
    current_time = current_time.replace(microsecond=0)
    general_info.export_date = current_time.date()
    general_info.export_time = current_time.time()
    general_info.data_no = os.path.basename(filename).split('.')[0]

    # Always populate seg_data for OCT scans (needed for thickness calculations)
    if capture_info.capture_mode not in ('Fundus only', 'Fundus Photo only'):
        seg_data = contour_all.reshape((image_width, num_images, n_contour), order='F').copy()
        if n_contour > 0:
            seg_data = jpeg_height - seg_data
            seg_data[seg_data <= 0] = 1
            seg_data[seg_data > jpeg_height] = 1
            seg_data = np.flip(seg_data, 1)
        scan_info.seg_data = seg_data

        # Optionally populate oct3d volume if requested
        if read_oct3d and read_images:
            # oct_vol_size = oct3d.shape
            scan_info.oct3d = oct3d

        # contour_mask = contour_all_mask.reshape((image_width, num_images, n_contour_mask), order='F').copy()
        # scan_info.contour_mask = contour_mask

    return general_info, patient_info, capture_info, scan_info, disc_info

def read_fda_file_info(file_id):
    file_id.read(8)
    analysis_sw_ver = ""
    for _ in range(32):
        analysis_sw_ver += chr(struct.unpack('B', file_id.read(1))[0])
    analysis_sw_ver = analysis_sw_ver.rstrip('\x00')
    return analysis_sw_ver

def read_hw_info(file_id):
    instrument = ""
    for _ in range(16):
        instrument += chr(struct.unpack('B', file_id.read(1))[0])
    # instrument = instrument.rstrip('\x00')
    instrument = instrument.split('\x00', 1)[0]
    serial_no = ""
    for _ in range(16):
        serial_no += chr(struct.unpack('B', file_id.read(1))[0])
    serial_no = serial_no.rstrip('\x00')
    file_id.read(100)
    capture_sw_ver = ""
    for _ in range(16):
        capture_sw_ver += chr(struct.unpack('B', file_id.read(1))[0])
    capture_sw_ver = capture_sw_ver.rstrip('\x00')
    return instrument, serial_no, capture_sw_ver

def read_patient_info(file_id):
    """Read the ``@PATIENT_INFO_02`` block (plaintext patient demographics).

    Parameters
    ----------
    file_id : BinaryIO
        Open file handle positioned at the start of the block payload
        (i.e. after the block label and the 4-byte length field).

    Returns
    -------
    patient_id : str
    first_name : str
    last_name : str
    gender : str
        ``'Male'``, ``'Female'``, or ``'n/a'``.
    birth_date : datetime.date or None
        ``None`` when any of the year/month/day components stored in the file
        is zero (i.e. the patient's date of birth was not recorded).  Callers
        must guard against ``None`` before using ``relativedelta`` to compute
        patient age.
    """
    patient_id = ""
    for i in range(32):
        patient_id += chr(struct.unpack('B', file_id.read(1))[0])
    patient_id = patient_id.rstrip('\x00')
    first_name = ""
    for i in range(32):
        first_name += chr(struct.unpack('B', file_id.read(1))[0])
    first_name = first_name.rstrip('\x00')
    last_name = ""
    for i in range(32):
        last_name += chr(struct.unpack('B', file_id.read(1))[0])
    last_name = last_name.rstrip('\x00')
    file_id.read(8)
    gender = struct.unpack('B', file_id.read(1))[0]
    if gender == 1:
        gender = 'Male'
    elif gender == 2:
        gender = 'Female'
    else:
        gender = 'n/a'
    birth_year = struct.unpack('h', file_id.read(2))[0]
    birth_month = struct.unpack('h', file_id.read(2))[0]
    birth_day = struct.unpack('h', file_id.read(2))[0]
    if (birth_year == 0) or (birth_month == 0) or (birth_day == 0):
        birth_date = None
    else:
        birth_date = date(birth_year, birth_month, birth_day)

    return patient_id, first_name, last_name, gender, birth_date

def decrypt_pat_info(b: bytes) -> bytes:
    secret = b'D1F4940E-D16F-467A-A53D-5F90CEA7DCE8'
    if IS_WINDOWS:
        # Use native Windows wincrypto
        sha1 = CryptCreateHash(CALG_SHA1)
        CryptHashData(sha1, secret)
        sha1_key = CryptDeriveKey(sha1, CALG_RC4)
        return CryptDecrypt(sha1_key, b)
    else:
        # Use PyCryptodome on Mac to simulate Windows behavior
        hasher = SHA1.new()
        hasher.update(secret)
        sha1_result = hasher.digest()
        derived_key = sha1_result[:16] # Match CAPI 128-bit derivation
        cipher = ARC4.new(derived_key)
        return cipher.decrypt(b)

def read_patient_info_03(file_id):
    """Read the ``@PATIENT_INFO_03`` block (RC4-encrypted patient demographics).

    Decrypts the 615-byte payload using the device-specific RC4 key via
    ``decrypt_pat_info`` (uses Win32 CryptAPI on Windows, PyCryptodome on
    macOS/Linux).  Byte offsets within the decrypted buffer are fixed by the
    Topcon FDA specification.

    Parameters
    ----------
    file_id : BinaryIO
        Open file handle positioned at the start of the block payload.

    Returns
    -------
    patient_id, first_name, last_name, gender, birth_date
        Same types as ``read_patient_info``.  ``birth_date`` is ``None``
        when any date component is zero (DOB not recorded); callers must
        guard against ``None`` before computing age with ``relativedelta``.

        Returns ``None`` (not a tuple) if ``file_id.read(615)`` yields an
        empty bytes object (truncated file).
    """
    binary_data = file_id.read(615)
    if not binary_data:
        # Return a sentinel tuple so the caller's tuple-unpack never receives None.
        # A truncated @PATIENT_INFO_03 block means demographics are unavailable;
        # the caller will leave patient_info fields as None.
        return None, None, None, None, None

    decrypted_data = decrypt_pat_info(binary_data)

    patient_id = decrypted_data[0:32].decode('utf-8').rstrip('\x00')
    first_name = decrypted_data[32:64].decode('utf-8').rstrip('\x00')
    last_name = decrypted_data[64:96].decode('utf-8').rstrip('\x00')
    gender = ord(decrypted_data[104:105].decode('utf-8'))
    if gender == 1:
        gender = 'Male'
    elif gender == 2:
        gender = 'Female'
    else:
        gender = 'n/a'
    birth_year = struct.unpack('h', decrypted_data[105:107])[0]
    birth_month = struct.unpack('h', decrypted_data[107:109])[0]
    birth_day = struct.unpack('h', decrypted_data[109:111])[0]
    if (birth_year == 0) or (birth_month == 0) or (birth_day == 0):
        birth_date = None
    else:
        birth_date = date(birth_year, birth_month, birth_day)
    return patient_id, first_name, last_name, gender, birth_date

def read_patientext_info(file_id):
    ethnicity = ""
    for _ in range(32):
        ethnicity += chr(struct.unpack('B', file_id.read(1))[0])
    ethnicity = ethnicity.rstrip('\x00')    
    return ethnicity

def read_scan_info(file_id, lab_name):
    if not (lab_name == '@PARAM_SCAN_04'):
        file_id.read(5)
    file_id.read(3)
    fixation = struct.unpack('b', file_id.read(1))[0]
    if fixation == 0:
        fixation = 'Center'
    elif fixation == 1:
        fixation = 'Disc'
    elif fixation == 2:
        fixation = 'Macula'
    elif fixation == 3:
        fixation = 'Wide'
    elif fixation == 4:
        fixation = 'Center of Cornea'
    elif fixation == 15:
        fixation = 'External'
    else:
        fixation = 'Others'
    mirror_pos = struct.unpack('I', file_id.read(4))[0]
    if not (lab_name == '@PARAM_SCAN_04'):
        file_id.read(4)
    file_id.read(4)
    scan_w = struct.unpack('d', file_id.read(8))[0]
    scan_h = struct.unpack('d', file_id.read(8))[0]
    scan_axial_res = struct.unpack('d', file_id.read(8))[0]
    file_id.read(16)
    ref_position = struct.unpack('B', file_id.read(1))[0]
    file_id.read(6)
    scan_protocol = struct.unpack('B', file_id.read(1))[0]
    focus_mode = 'n/a'
    if ref_position == 0:
        if scan_protocol == 5: # anterior scan
            focus_mode = 'Deep pos'
        else:
            focus_mode = 'Choroidal'
    elif ref_position == 1:
        if scan_protocol == 5:
            focus_mode = 'Cornea'
        else:
            focus_mode = 'Vitreous'

    if scan_protocol == 1:
        scan_protocol = 'Macula'
    elif scan_protocol == 2:
        scan_protocol = 'Glaucoma'
    elif scan_protocol == 3:
        scan_protocol = 'Fundus Only'
    elif scan_protocol == 4:
        scan_protocol = 'Selected'
    elif scan_protocol == 5:
        scan_protocol = 'Anterior'
    else: # 0, 255, or others
        scan_protocol = 'Unknown'

    return fixation, mirror_pos, scan_w, scan_h, scan_axial_res, focus_mode, scan_protocol

def read_capture_info(file_id):
    eye = struct.unpack('B', file_id.read(1))[0]
    if eye == 0:
        eye = 'R'
    elif eye == 1:
        eye = 'L'
    capture_mode = struct.unpack('B', file_id.read(1))[0]
    if capture_mode == 0:
        capture_mode = 'OCT only'
    elif capture_mode == 1:
        capture_mode = 'Fundus only'
    elif capture_mode == 2:
        capture_mode = 'OCT+Fundus'
    else:
        capture_mode = 'n/a'
    file_id.read(4)
    capture_label = ""
    for i in range(100):
        capture_label += chr(struct.unpack('B', file_id.read(1))[0])
    capture_label = capture_label.rstrip('\x00')
    capture_year = struct.unpack('h', file_id.read(2))[0]
    capture_month = struct.unpack('h', file_id.read(2))[0]
    capture_day = struct.unpack('h', file_id.read(2))[0]
    capture_date = date(capture_year, capture_month, capture_day)
    capture_hour = struct.unpack('h', file_id.read(2))[0]
    capture_minute = struct.unpack('h', file_id.read(2))[0]
    capture_second = struct.unpack('h', file_id.read(2))[0]
    capture_time = time(capture_hour, capture_minute, capture_second)
    return eye, capture_mode, capture_label, capture_date, capture_time

def read_oct_jpeg(file_id):
    mode = struct.unpack('B', file_id.read(1))[0]    
    scan_mode = scan_modes[mode] if mode < len(scan_modes) else 'Unknown'
    file_id.read(8)
    width = struct.unpack('I', file_id.read(4))[0]
    height = struct.unpack('I', file_id.read(4))[0]
    nimages = struct.unpack('I', file_id.read(4))[0]

    return scan_mode, width, nimages, height

def read_top_q(file_id):
    file_id.read(4)
    top_q = struct.unpack('f', file_id.read(4))[0]
    return top_q

def read_oct_jpeg_octdata(file_id):
    mode = struct.unpack('B', file_id.read(1))[0]    
    scan_mode = scan_modes[mode] if mode < len(scan_modes) else 'Unknown'
    file_id.read(8)
    width = struct.unpack('I', file_id.read(4))[0]
    height = struct.unpack('I', file_id.read(4))[0]
    nimages = struct.unpack('I', file_id.read(4))[0]

    format = struct.unpack('B', file_id.read(1))[0]
    file_id.read(3)
    
    oct3d = np.empty(shape=(height, width, nimages), dtype=np.uint8) 
    for i in range(nimages):
        total_bytes = struct.unpack('I', file_id.read(4))[0]
        temp = array('B')
        temp.fromfile(file_id, total_bytes)
        image = BytesIO(temp)
        oct3d[:,:,i] = Image.open(image)

    return scan_mode, width, nimages, oct3d, height

def read_regist_info(file_id):
    regist_info = RegistInfo()
    file_id.read(73)    
    regist_info.disc_center_auto_x = struct.unpack('d', file_id.read(8))[0]
    regist_info.disc_center_auto_y = struct.unpack('d', file_id.read(8))[0]
    regist_info.disc_center_manual_x = struct.unpack('d', file_id.read(8))[0]
    regist_info.disc_center_manual_y = struct.unpack('d', file_id.read(8))[0]
    file_id.read(48)
    regist_info.fovea_auto_x = struct.unpack('d', file_id.read(8))[0]
    regist_info.fovea_auto_y = struct.unpack('d', file_id.read(8))[0]
    regist_info.fovea_manual_x = struct.unpack('d', file_id.read(8))[0]
    regist_info.fovea_manual_y = struct.unpack('d', file_id.read(8))[0]
    
    for attr, value in regist_info.__dict__.items():
        if isinstance(value, float):
            setattr(regist_info, attr, round(value, 6))
    return regist_info

def read_fast_q(file_id):
    file_id.read(16)
    q_mean = struct.unpack('f', file_id.read(4))[0]
    z_mean = struct.unpack('f', file_id.read(4))[0]
    
    return round(q_mean, 2), z_mean

def read_report_info(file_id):
    file_id.read(2)
    analysis_mode = struct.unpack('B', file_id.read(1))[0]
    if analysis_mode == 0:
        analysis_mode = 'Fine'
    elif analysis_mode == 5:
        analysis_mode = 'Basic'
    else:
        analysis_mode = 'n/a'
    return analysis_mode

def read_reanalysis_info(file_id):
    re_year = struct.unpack('h', file_id.read(2))[0]
    re_month = struct.unpack('h', file_id.read(2))[0]
    re_day = struct.unpack('h', file_id.read(2))[0]
    re_hour = struct.unpack('h', file_id.read(2))[0]
    re_minute = struct.unpack('h', file_id.read(2))[0]
    re_second = struct.unpack('h', file_id.read(2))[0]
    re_datetime = datetime(re_year, re_month, re_day, re_hour, re_minute, re_second)
    return re_datetime

def read_contour(file_id):
    file_id.read(22)
    width = struct.unpack('I', file_id.read(4))[0]
    height = struct.unpack('I', file_id.read(4))[0]
    data_sz = struct.unpack('I', file_id.read(4))[0]
    contour = array('H')
    contour.fromfile(file_id, width * height)
    return contour

def read_disc_info(file_id):
    # Parse version string (first 16 bytes of the header)
    disc_seg_version = file_id.read(16).decode('ascii', errors='replace').rstrip('\x00')
    
    # Skip 16 bytes (cup axis endpoints or other metadata; zero-filled in tested files)
    file_id.read(16)
    
    # Parse disc major/minor axis endpoints (4 points × 2 coords = 8 integers = 32 bytes)
    # Each pair is stored as (y, x) in the file
    disc_left_y = struct.unpack('I', file_id.read(4))[0]
    disc_left_x = struct.unpack('I', file_id.read(4))[0]
    disc_bottom_y = struct.unpack('I', file_id.read(4))[0]
    disc_bottom_x = struct.unpack('I', file_id.read(4))[0]
    disc_right_y = struct.unpack('I', file_id.read(4))[0]
    disc_right_x = struct.unpack('I', file_id.read(4))[0]
    disc_top_y = struct.unpack('I', file_id.read(4))[0]
    disc_top_x = struct.unpack('I', file_id.read(4))[0]
    
    # Parse diameter, area, and volume measurements
    horizontal_disc_diameter = struct.unpack('d', file_id.read(8))[0]
    vertical_disc_diameter = struct.unpack('d', file_id.read(8))[0]
    actual_disc_area = struct.unpack('d', file_id.read(8))[0]
    disc_projected_area = struct.unpack('d', file_id.read(8))[0]
    disc_volume = struct.unpack('d', file_id.read(8))[0]
    cup_area = struct.unpack('d', file_id.read(8))[0]
    cup_volume = struct.unpack('d', file_id.read(8))[0]
    rim_area = struct.unpack('d', file_id.read(8))[0]
    
    # Skip 24 bytes (max cup depth, avg cup depth, etc. - all zeros)
    file_id.read(24)
    
    # Parse dimensions and coordinate count
    width = struct.unpack('I', file_id.read(4))[0]
    height = struct.unpack('I', file_id.read(4))[0]
    disc_points = struct.unpack('I', file_id.read(4))[0]
    
    # Parse disc edge coordinates (x,y pairs)
    disc_seg_data = array('d')
    disc_seg_data.fromfile(file_id, disc_points * 2)
    
    # Parse cup edge coordinates (same format: count then x,y pairs)
    cup_points = struct.unpack('I', file_id.read(4))[0]
    cup_seg_data = array('d')
    cup_seg_data.fromfile(file_id, cup_points * 2)
    
    # Parse reference surface offset (final double)
    reference_surface_offset = struct.unpack('d', file_id.read(8))[0]
    
    return disc_seg_version, disc_left_x, disc_left_y, disc_bottom_x, disc_bottom_y, disc_right_x, disc_right_y, disc_top_x, disc_top_y, horizontal_disc_diameter, vertical_disc_diameter, actual_disc_area, disc_projected_area, disc_volume, cup_area, cup_volume, rim_area, disc_seg_data, cup_seg_data, reference_surface_offset
    
def read_littmann_info(file_id):
    axial_length = struct.unpack('d', file_id.read(8))[0]
    horizontal_corneal_radius = struct.unpack('d', file_id.read(8))[0]
    astimatism_deg = struct.unpack('d', file_id.read(8))[0]
    astigmatic_axis = struct.unpack('d', file_id.read(8))[0]
    spherical_power = struct.unpack('d', file_id.read(8))[0]
    IOL_information = struct.unpack('B', file_id.read(1))[0]
    file_id.read(3)  # Skip 3 bytes padding
    correction_lens_info = struct.unpack('B', file_id.read(1))[0]  # 255 = "Not Set"
    file_id.read(3)  # Skip 3 bytes padding
    correction_method = struct.unpack('B', file_id.read(1))[0]  # 1
    file_id.read(3)  # Skip 3 bytes padding (remaining)

    if correction_lens_info == 255:
        correction_lens_info = 'Not Set'
    else:
        correction_lens_info = 'do not know what other options can it take'

    if correction_method == 1:
        correction_method = 'New type Littmann used'
    else:
        correction_method = 'do not know what other options can it take'
    return axial_length, horizontal_corneal_radius, astimatism_deg, astigmatic_axis, spherical_power, IOL_information, correction_lens_info, correction_method


# def read_mask(file_id):
#     file_id.read(4)
#     width = struct.unpack('I', file_id.read(4))[0]
#     height = struct.unpack('I', file_id.read(4))[0]
#     data_sz = struct.unpack('I', file_id.read(4))[0]
#     contour = array('H')
#     contour.fromfile(file_id, width * height)
#     return contour

if __name__ == "__main__":
    # file = '//laj-fs2/SecureData/RWD/Site003/Location001/fda_deid/O56DDAZN.fda'
    file = '//laj-fs2/SecureData/Big Data/fda_files/Site001/Location001/2094.fda'
    file = '/Users/mestevesmiranda/Library/CloudStorage/OneDrive-Topcon/Desktop/New folder (3)/990323_2262.fda'
    print(file)
    general_info, patient_info, capture_info, scan_info, disc_info = read_fda_common_info(file, read_oct3d=True, read_images=True)
    print(general_info.__dict__)
    print(patient_info.__dict__)
    print(capture_info.__dict__)
    print(scan_info.__dict__)
    print(disc_info.__dict__)
    print(scan_info.regist_info.__dict__)