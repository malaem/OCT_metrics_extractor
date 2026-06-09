"""OCT en face registration.

Required inputs:
- reference_image
- moving_image

Required output:
- output_registered

Optional outputs (disabled by default):
- output_overlay_before
- output_overlay_after
- output_transform_csv

Sharing-friendly setup:
- Keep this script and an `Images` folder together.
- Put your two input files inside `Images`.
- Edit only the filenames in `CONFIG` (no long absolute paths needed).
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import csv
import time

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "Images"


@dataclass(frozen=True)
class RegistrationOptions:
	"""Tunable hyper-parameters for the KAZE-based image registration pipeline.

	All fields are frozen after construction; create a ``replace``-d copy to
	override individual settings without mutating the shared default instance.

	Attributes
	----------
	use_preprocessing : bool
		Apply ``_preprocessing_exclude`` before feature detection.  Should
		always be ``True`` in production; setting ``False`` is only valid for
		diagnostic comparisons (not supported by ``kaze_benchmark_pair``).
	preprocessing_exclude_flg : {0, 1}
		Passed to ``_preprocessing_exclude``: 1 = subtract Gaussian blur
		(emphasise local contrast), 0 = use blur suppression (invert the
		residual).  Production default is 1.
	kaze_octave : int
		Number of KAZE octaves (``nOctaves`` parameter).
	kaze_scale : int
		Number of octave layers (``nOctaveLayers`` parameter).
	kaze_thresh : float
		KAZE detector response threshold; lower → more keypoints.
	kaze_filter : str
		Diffusivity type: ``'sharpedge'`` / ``'pm_g1'`` / ``'pm_g2'`` /
		``'weickert'`` / ``'charbonnier'``.
	match_thresh : float
		Maximum absolute L2 descriptor distance for a match to be kept.
	max_ratio : float
		Lowe ratio-test threshold (``m.distance < max_ratio * n.distance``).
	ransac_confidence : float
		RANSAC confidence level in percent (passed as fraction to OpenCV).
	ransac_num_trials : int
		Maximum RANSAC iterations.
	ransac_max_distance : float
		RANSAC reprojection-error threshold in pixels.
	"""
	use_preprocessing: bool = True
	preprocessing_exclude_flg: int = 1
	kaze_octave: int = 4
	kaze_scale: int = 10
	kaze_thresh: float = 0.000001
	kaze_filter: str = "sharpedge"
	match_thresh: float = 15.0
	max_ratio: float = 0.9
	ransac_confidence: float = 99.0
	ransac_num_trials: int = 20000
	ransac_max_distance: float = 1.5


@dataclass
class Transform:
	"""Output of ``compute_registration``: the estimated partial-affine transform.

	Attributes
	----------
	matrix : np.ndarray, shape (2, 3), float64
		Affine matrix ``M`` as returned by ``cv2.estimateAffinePartial2D``.
		Maps *moving*-image pixel coordinates to *reference*-image pixel
		coordinates: ``M @ [x, y, 1]^T → [x', y']``.
	translation : list[float, float]
		``[tx, ty]`` — the translation components extracted from ``matrix``
		(``M[0,2]`` and ``M[1,2]`` respectively).  Stored separately for
		convenience; always in sync with ``matrix``.
	rotation_deg : float
		Estimated in-plane rotation in degrees (``atan2(M[1,0], M[0,0])``
		in degrees).
	scale : list[float, float]
		``[scale_x, scale_y]`` — isotropic scale components (``hypot(a, c)``
		and ``hypot(b, d)``).  For a valid similarity transform these should
		be equal to within float32 tolerance.
	output_shape : tuple[int, int]
		``(height, width)`` of the reference image; used by ``apply_transform``
		to set the output canvas size.
	num_good_matches : int
		Number of matches surviving both the distance and ratio tests.
	num_inliers : int
		Number of RANSAC inliers used to estimate the final transform.
	time_processing : float
		Wall-clock time (seconds) from the start of ``compute_registration``
		until the ``Transform`` was populated.
	"""
	matrix: np.ndarray
	translation: Tuple[float, float]
	rotation_deg: float
	scale: Tuple[float, float]
	output_shape: Tuple[int, int]
	num_good_matches: int
	num_inliers: int
	time_processing: float


@dataclass(frozen=True)
class Config:
	"""Paths and options for a standalone registration run via ``_main``.

	Used only when this module is executed as a script (``python -m …``) or
	when called via the ``CONFIG`` sentinel for quick smoke tests.  Production
	pipeline code uses ``compute_registration`` / ``apply_transform`` directly
	and does not interact with this class.
	"""
	reference_image: Path
	moving_image: Path
	output_registered: Path
	output_overlay_before: Optional[Path] = None
	output_overlay_after: Optional[Path] = None
	output_transform_csv: Optional[Path] = None
	overlay_alpha: float = 1.0


CONFIG = Config(
	reference_image=IMAGES_DIR / "BD1277_20171130_A.png",
	moving_image=IMAGES_DIR / "BD1277_20190213_B.png",
	output_registered=IMAGES_DIR / "registered_to_reference.png",
)


def _preprocessing_exclude(gray: np.ndarray, flg: int = 0):
	"""CLAHE-based contrast enhancement tuned for KAZE feature detection.

	Normalises the input to [0, 1], subtracts or adds a Gaussian background
	(controlled by ``flg``), clips, rescales to uint8, then applies CLAHE
	to equalise local contrast across an 8×8 tile grid.  The result strongly
	enhances vessel-shadow edges and suppresses large-scale intensity gradients.

	Parameters
	----------
	gray : np.ndarray
		Single-channel image (integer or floating-point).
	flg : {0, 1}
		``1`` → residual = blur − original (emphasise features *brighter* than
		their local neighbourhood).
		``0`` → residual = original − blur (emphasise features *darker* than
		their neighbourhood, e.g. vessel shadows in OCT en face).

	Returns
	-------
	np.ndarray, dtype uint8
		Contrast-enhanced image ready for KAZE feature detection.
	"""
	if np.issubdtype(gray.dtype, np.integer):
		img = gray.astype(np.float32) / float(np.iinfo(gray.dtype).max)
	else:
		img = gray.astype(np.float32)

	img_min = float(np.min(img))
	img_max = float(np.max(img))
	img_norm = (img - img_min) / (img_max - img_min) if img_max > img_min else np.zeros_like(img, dtype=np.float32)

	n = 37
	sigma = (n / 2.0 - 1.0) * 0.3 + 0.8
	im_enhance_f = cv2.GaussianBlur(
		img_norm,
		(n, n),
		sigmaX=sigma,
		sigmaY=sigma,
		borderType=cv2.BORDER_REPLICATE,
	)

	im_exclude = im_enhance_f - img_norm if int(flg) == 1 else img_norm - im_enhance_f
	im_exclude_u8 = np.clip(np.round(np.clip(im_exclude, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)

	tile_grid = (8, 8)
	height, width = im_exclude_u8.shape
	tile_width = int(np.ceil(width / tile_grid[0]))
	tile_height = int(np.ceil(height / tile_grid[1]))
	tile_area = max(1, tile_width * tile_height)
	opencv_clip_limit = max(1.0, 0.2 * tile_area / 256.0)

	clahe = cv2.createCLAHE(clipLimit=float(opencv_clip_limit), tileGridSize=tile_grid)
	return clahe.apply(im_exclude_u8)


def _create_kaze(options: RegistrationOptions):
	"""Instantiate a ``cv2.KAZE`` detector configured from *options*.

	The ``'sharpedge'`` filter alias maps to ``KAZE_DIFF_PM_G2`` which gives
	good edge preservation while remaining robust to noise.  Raises
	``ValueError`` for unrecognised filter names.
	"""
	diffusion_map = {
		"pm_g1": cv2.KAZE_DIFF_PM_G1,
		"pm_g2": cv2.KAZE_DIFF_PM_G2,
		"weickert": cv2.KAZE_DIFF_WEICKERT,
		"charbonnier": cv2.KAZE_DIFF_CHARBONNIER,
		"sharpedge": cv2.KAZE_DIFF_PM_G2,
	}
	diffusivity = diffusion_map.get(options.kaze_filter.lower())
	if diffusivity is None:
		raise ValueError("Unsupported kaze_filter.")

	return cv2.KAZE_create(
		upright=True,
		extended=True,
		threshold=float(options.kaze_thresh),
		nOctaves=int(options.kaze_octave),
		nOctaveLayers=int(options.kaze_scale),
		diffusivity=diffusivity,
	)


def compute_registration(
	reference: np.ndarray,
	follow_up: np.ndarray,
	options: RegistrationOptions = RegistrationOptions(),
) -> Transform:
	"""Estimate a partial-affine (similarity) transform from *follow_up* to *reference*.

	Pipeline
	--------
	1. Optional CLAHE preprocessing via ``_preprocessing_exclude`` (enabled by
	   default; should always be ``True`` in production).
	2. KAZE feature detection and description on both images.
	3. Brute-force L2 matching with Lowe ratio test + absolute distance cap.
	4. ``cv2.estimateAffinePartial2D`` (RANSAC) to fit a 4-DOF similarity
	   transform (rotation + isotropic scale + translation).
	5. Similarity-constraint check: ``scale_x`` and ``scale_y`` must agree
	   within tolerance (deferred to float32 tolerance, ~1e-4, because
	   ``estimateAffinePartial2D`` returns a float32 matrix).

	Parameters
	----------
	reference : np.ndarray
		Reference (fixed) image — any dtype, single channel.
	follow_up : np.ndarray
		Moving image to register onto *reference*.
	options : RegistrationOptions, optional
		Hyper-parameter bundle.  Defaults to the production preset.

	Returns
	-------
	Transform
		Fully populated transform dataclass.  Pass to ``apply_transform`` to
		warp the moving image (or any co-registered array) into reference space.

	Raises
	------
	ValueError
		If ``preprocessing_exclude_flg`` is not 0 or 1.
	RuntimeError
		If too few keypoints/matches are found, RANSAC fails, or the
		estimated matrix violates the similarity constraint.
	"""
	if options.preprocessing_exclude_flg not in (0, 1):
		raise ValueError("preprocessing_exclude_flg must be 0 or 1.")

	start_time = time.perf_counter()

	if options.use_preprocessing:
		ref_feature = _preprocessing_exclude(reference, flg=options.preprocessing_exclude_flg)
		mov_feature = _preprocessing_exclude(follow_up, flg=options.preprocessing_exclude_flg)
	else:
		ref_feature = reference
		mov_feature = follow_up

	kaze = _create_kaze(options)
	kp_ref, des_ref = kaze.detectAndCompute(ref_feature, None)
	kp_mov, des_mov = kaze.detectAndCompute(mov_feature, None)
	if des_ref is None or des_mov is None or len(kp_ref) < 4 or len(kp_mov) < 4:
		raise RuntimeError("Not enough KAZE keypoints/descriptors found in one or both images.")

	matcher = cv2.BFMatcher(cv2.NORM_L2)
	knn = matcher.knnMatch(des_mov, des_ref, k=2)

	good_matches = []
	for pair in knn:
		if len(pair) < 2:
			continue
		m, n = pair
		if m.distance <= options.match_thresh and m.distance < options.max_ratio * n.distance:
			good_matches.append(m)

	if len(good_matches) < 3:
		raise RuntimeError(f"Not enough good matches for affine transform: {len(good_matches)} found.")

	src_pts = np.float32([kp_mov[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
	dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

	affine_matrix, inlier_mask = cv2.estimateAffinePartial2D(
		src_pts,
		dst_pts,
		method=cv2.RANSAC,
		ransacReprojThreshold=float(options.ransac_max_distance),
		maxIters=int(options.ransac_num_trials),
		confidence=float(options.ransac_confidence) / 100.0,
	)
	if affine_matrix is None:
		raise RuntimeError("Failed to estimate affine transform.")

	a = float(affine_matrix[0, 0])
	b = float(affine_matrix[0, 1])
	tx = float(affine_matrix[0, 2])
	c = float(affine_matrix[1, 0])
	d = float(affine_matrix[1, 1])
	ty = float(affine_matrix[1, 2])

	scale_x = float(np.hypot(a, c))
	scale_y = float(np.hypot(b, d))
	if not np.isclose(scale_x, scale_y, rtol=1e-4, atol=1e-4):
		num_inliers_diag = int(np.sum(inlier_mask)) if inlier_mask is not None else 0
		raise RuntimeError(
			f"Similarity constraint violated: scale_x={scale_x:.8f} and scale_y={scale_y:.8f} differ beyond float32 tolerance. "
			f"(good_matches={len(good_matches)}, inliers={num_inliers_diag})"
		)
	scale = scale_x
	if not (0.5 < scale < 2.0):
		num_inliers_diag = int(np.sum(inlier_mask)) if inlier_mask is not None else 0
		raise RuntimeError(
			f"Degenerate similarity transform: scale={scale:.6f} is outside valid range (0.5, 2.0). "
			f"(good_matches={len(good_matches)}, inliers={num_inliers_diag})"
		)
	if scale > 1e-12:
		rotation_deg = float(np.degrees(np.arctan2(c, a)))
	else:
		rotation_deg = 0.0

	num_inliers = int(np.sum(inlier_mask)) if inlier_mask is not None else 0

	processing_time_seconds = time.perf_counter() - start_time

	return Transform(
		matrix=affine_matrix,
		translation=[tx, ty],
		rotation_deg=rotation_deg,
		scale=[scale_x, scale_y],
		output_shape=reference.shape[:2],
		num_good_matches=len(good_matches),
		num_inliers=num_inliers,
		time_processing=processing_time_seconds
	)


def apply_transform(transform: Transform, image: np.ndarray) -> np.ndarray:
	"""Warp *image* into reference space using a pre-computed ``Transform``.

	Parameters
	----------
	transform : Transform
		The transform returned by ``compute_registration``.
	image : np.ndarray
		Image to warp (typically the moving en face or RNFL thickness map).
		Must be single-channel; can be any dtype.

	Returns
	-------
	np.ndarray
		Warped image at the same spatial resolution as the reference
		(``transform.output_shape`` = (height, width)).  Border pixels that
		fall outside the source boundary are filled with ``NaN`` for
		floating-point images, or ``0`` for integer images (NaN is not
		representable in integer dtypes).  Using NaN rather than 0.0 for
		float images ensures that missing-data pixels are distinguishable from
		valid zero-thickness RNFL values.
	"""
	h, w = transform.output_shape
	# Use NaN as the fill value for floating-point images so that pixels warped
	# outside the source boundary are clearly identifiable as non-data rather than
	# being silently filled with 0.0 (which is a valid RNFL thickness value).
	# Integer images fall back to 0 because NaN is not representable.
	border_value = float('nan') if np.issubdtype(image.dtype, np.floating) else 0
	return cv2.warpAffine(image, transform.matrix, (w, h), borderValue=border_value)


def _make_overlay(reference_gray: np.ndarray, moving_gray: np.ndarray, alpha: float = 1.0):
	"""Blend two grayscale images into a false-colour (magenta / green) overlay.

	The reference is placed in the green channel; the moving image (scaled by
	*alpha*) is placed in both the red and blue channels, producing a
	magenta-on-green colour scheme that makes misalignment visually obvious.

	Parameters
	----------
	reference_gray : np.ndarray
		Fixed (reference) single-channel image.
	moving_gray : np.ndarray
		Moving image; resized to match reference dimensions if needed.
	alpha : float, default 1.0
		Brightness scale applied to the moving image channel.

	Returns
	-------
	np.ndarray, dtype uint8, shape (H, W, 3)
		RGB overlay image ready for ``cv2.imwrite``.
	"""
	height, width = reference_gray.shape[:2]
	if moving_gray.shape[:2] != (height, width):
		moving_gray = cv2.resize(moving_gray, (width, height), interpolation=cv2.INTER_LINEAR)

	reference_norm = cv2.normalize(reference_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
	moving_norm = cv2.normalize(moving_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
	moving_scaled = np.clip(moving_norm.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

	overlay = np.zeros((height, width, 3), dtype=np.uint8)
	overlay[:, :, 0] = moving_scaled
	overlay[:, :, 1] = reference_norm
	overlay[:, :, 2] = moving_scaled
	return overlay


def _save_if_enabled(path: Optional[Path], image: Optional[np.ndarray]):
	"""Write *image* to *path* using ``cv2.imwrite``; no-op if either argument is ``None``."""
	if path is not None and image is not None:
		cv2.imwrite(str(path), image)


def _save_transform_csv(path: Optional[Path], transform: Transform):
	"""Write the estimated transform parameters to a two-row CSV file.

	No-op when *path* is ``None``.

	Output columns
	--------------
	``m00, m01, m02`` — first row of the 2×3 affine matrix (a, b, tx).
	``m10, m11, m12`` — second row (c, d, ty).
	``translation_x, translation_y`` — convenience copy of (m02, m12) so
		downstream tools can read the translation without parsing the matrix;
		extracted from ``transform.translation[0/1]`` (not the deprecated
		``translation_x / translation_y`` attribute names that do not exist).
	``rotation_deg``  — in-plane rotation in degrees.
	``scale``         — the full scale list ``[scale_x, scale_y]``.

	The parent directory of *path* is created if it does not exist.
	"""
	if path is None:
		return

	m = transform.matrix
	a, b, c, d = float(m[0, 0]), float(m[0, 1]), float(m[1, 0]), float(m[1, 1])

	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(
			[
				"m00",
				"m01",
				"m02",
				"m10",
				"m11",
				"m12",
				"translation_x",
				"translation_y",
				"rotation_deg",
				"scale",
			]
		)
		writer.writerow(
			[
				a, b, transform.translation[0],
				c, d, transform.translation[1],
				transform.translation[0], transform.translation[1],
				transform.rotation_deg, transform.scale,
			]
		)


def _main():
	"""Command-line entry point for standalone image registration.

	Accepts ``--ref`` and ``--fu`` arguments for reference and follow-up image
	paths.  Falls back to the hard-coded ``CONFIG`` sentinel if neither is
	provided (useful for quick smoke tests during development).

	Outputs the registered image alongside optional overlay PNGs and a
	transform CSV as configured in ``CONFIG``.
	"""
	parser = argparse.ArgumentParser(description="Register a follow-up OCT en face image to a reference.")
	parser.add_argument("--ref", metavar="PATH", help="Path to the reference image.")
	parser.add_argument("--fu", metavar="PATH", help="Path to the follow-up (moving) image.")
	args = parser.parse_args()

	if args.ref is not None and args.fu is not None:
		ref_path = Path(args.ref)
		fu_path = Path(args.fu)
		output_path = fu_path.with_name(fu_path.stem + "_registered.png")
	else:
		cfg = CONFIG
		ref_path = cfg.reference_image
		fu_path = cfg.moving_image
		output_path = cfg.output_registered

	reference = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
	moving = cv2.imread(str(fu_path), cv2.IMREAD_GRAYSCALE)
	if reference is None:
		raise FileNotFoundError(f"Could not read reference image: {ref_path}")
	if moving is None:
		raise FileNotFoundError(f"Could not read follow-up image: {fu_path}")

	ref_f32 = reference.astype(np.float32)
	mov_f32 = moving.astype(np.float32)

	start_time = time.perf_counter()

	transform = compute_registration(ref_f32, mov_f32)
	registered_f32 = apply_transform(transform, mov_f32)

	registered_u8 = np.clip(registered_f32, 0, 255).astype(np.uint8)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	cv2.imwrite(str(output_path), registered_u8)

	processing_time_seconds = time.perf_counter() - start_time
	print(f"Good matches: {transform.num_good_matches}")
	print(f"Inliers: {transform.num_inliers}")
	print(f"Registered image saved to: {output_path.resolve()}")
	print(f"Processing time: {processing_time_seconds:.3f} seconds")


if __name__ == "__main__":
	_main()
