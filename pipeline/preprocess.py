"""
preprocess.py - Image preprocessing for HD-GLIO.

HD-GLIO (nnU-Net based) expects four NIfTI files (T1, T1c, T2, FLAIR) that are

  * oriented like the FSL MNI152 template ("standard" orientation; in
    nibabel axcodes: L, A, S),
  * brain-extracted with all non-brain voxels set to 0,
  * co-registered to each other (the README prescribes rigid 6-DOF
    registration of T1c/T2/FLAIR onto T1 with spline interpolation).

This module reproduces that preprocessing without FSL:

  * `reorient_to_fsl_std`  - pure numpy flip/permute reorientation to the
    FSL standard orientation (equivalent of `fslreorient2std`).
  * `rigid_register`       - SimpleITK rigid (6 DOF) mutual-information
    registration (equivalent of `flirt -dof 6`).
  * `run_hd_bet`           - wrapper around the installed `hd-bet` command
    (same group's brain-extraction network).

Transform bookkeeping is kept in *world* coordinates (sitk LPS): the
reorientation is a pure re-indexing of the voxel array (the world space
does not change), and the registration transform maps world points between
the (reoriented) image geometries. Mapping results back to the original
DICOM grid therefore only needs the inverse registration transform.
"""

import logging
import os
import shutil
import subprocess
import sys

import numpy as np
import SimpleITK as sitk

log = logging.getLogger("glio.preprocess")

# sitk orientation string that corresponds to the FSL/MNI152 standard
# (nibabel axcodes ('L', 'A', 'S') when the image is written to NIfTI).
FSL_STD_TARGET = ("L", "A", "S")


def reorient_to_fsl_std(image):
    """Reorient a sitk image (LPS world) to FSL-standard orientation ('LAS').

    Pure axis permutation + flipping (exactly what fslreorient2std does):
    each voxel axis is mapped to the standard axis its direction is closest
    to, and flipped so it points in the standard direction. No resampling.

    Target world directions (LPS): axis0 -> +x (L), axis1 -> -y (A),
    axis2 -> +z (S).

    Returns (new_image, info) where info describes permutation/flips.
    """
    arr = sitk.GetArrayFromImage(image)            # [z, y, x]
    size = np.array(image.GetSize(), dtype=int)    # (x, y, z)
    spacing = np.array(image.GetSpacing(), dtype=float)
    origin = np.asarray(image.GetOrigin(), dtype=float)
    D = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)

    # target world direction of each output axis, in LPS
    target = np.array([[1.0, 0.0, 0.0],   # L  = +x LPS
                       [0.0, -1.0, 0.0],  # A  = -y LPS
                       [0.0, 0.0, 1.0]])  # S  = +z LPS

    # for each target axis t pick the source axis whose direction matches best
    src_axes = []
    used = set()
    for t in range(3):
        cands = [(abs(float(np.dot(D[:, i], target[t]))), i) for i in range(3) if i not in used]
        best_cos, best_i = max(cands)
        used.add(best_i)
        sign = 1.0 if float(np.dot(D[:, best_i], target[t])) >= 0 else -1.0
        src_axes.append((best_i, sign))

    # sitk axis t  <-  old sitk axis p, sign s
    # numpy array axes are [z, y, x]  ->  numpy axis n = 2 - sitk axis
    perm_np = tuple(2 - p for (p, _s) in reversed(src_axes))  # new numpy axis 0..2
    # (new numpy axis 0 = new sitk axis 2 = src_axes[2][0], etc.)
    new_arr = np.transpose(arr, perm_np)
    flip_np_axes = tuple(2 - t for t, (_p, s) in enumerate(src_axes) if s < 0)
    if flip_np_axes:
        new_arr = np.flip(new_arr, axis=flip_np_axes)

    # direction: new axis t direction = s * old direction of axis p
    new_cols = []
    for (p, s) in src_axes:
        new_cols.append(s * D[:, p])
    new_D = np.column_stack(new_cols)

    # origin: world position of new voxel (0,0,0)
    new_origin = origin.copy()
    for t, (p, s) in enumerate(src_axes):
        if s < 0:
            # new index 0 along axis t corresponds to old index N_p - 1
            new_origin += (size[p] - 1) * spacing[p] * D[:, p]

    new_spacing = tuple(float(spacing[p]) for (p, _s) in src_axes)

    new_image = sitk.GetImageFromArray(np.ascontiguousarray(new_arr))
    new_image.SetOrigin(tuple(float(v) for v in new_origin))
    new_image.SetSpacing(new_spacing)
    new_image.SetDirection(tuple(float(v) for v in new_D.ravel()))

    info = dict(
        src_axes=src_axes,                              # [(source sitk axis, sign)]
        direction_before=D.copy(),
        direction_after=new_D.copy(),
        size=tuple(int(v) for v in size),
    )
    return new_image, info


def reorient_mask_back(mask_t1_las, target_native_image, reg_transform):
    """Resample an HD-GLIO segmentation back onto the native DICOM grid.

    mask_t1_las        : sitk image (uint8) of the segmentation, on the
                         (reoriented) T1 grid.
    target_native_image: sitk image of the target series on its original
                         DICOM grid (LPS world).
    reg_transform      : registration transform T (fixed=T1-LAS, moving=
                         target-LAS), i.e. T maps T1-LAS world -> target-LAS
                         world (sitk Resample convention).

    The reorientation is a pure re-indexing and therefore a no-op in world
    space; mapping the mask back only needs the inverse registration:
      output(p_native) = mask(T^-1(p_native))
    The result is a sitk image on the target's native grid, array order
    [z, y, x] matching the sorted DICOM slices.
    """
    inv = reg_transform.GetInverse()
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(target_native_image)
    resampler.SetTransform(inv)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputPixelType(sitk.sitkUInt8)
    return resampler.Execute(mask_t1_las)


def rigid_register(moving, fixed, config=None):
    """Rigid (6 DOF) registration of *moving* onto *fixed* (both sitk, LPS).

    Returns the sitk transform T that maps fixed-world -> moving-world
    points (the sitk Resample convention), i.e. use
    sitk.Resample(moving, fixed, T) to bring the moving image onto the
    fixed grid.

    Mutual information metric, multi-resolution, gradient-descent optimizer.
    The initial transform is the geometry-centered identity, which is a good
    start for same-session acquisitions sharing a Frame of Reference.
    """
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.25, sitk.sitkWallClock)
    R.SetInterpolator(sitk.sitkLinear)

    init = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R.SetInitialTransform(init, inPlace=False)

    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-5, numberOfIterations=200,
        gradientMagnitudeTolerance=1e-8, relaxationFactor=0.5)
    R.SetOptimizerScalesFromPhysicalShift()

    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    final = R.Execute(fixed, moving)
    final.FlattenTransform()
    log.info("registration metric after optimization: %.4f", R.GetMetricValue())
    return final


def resample_onto_reference(image, reference, transform=None, interpolation=None,
                            default_value=0.0, pixel_type=None):
    """Resample *image* onto the grid of *reference* (optionally via transform)."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    if transform is not None:
        resampler.SetTransform(transform)
    if interpolation is None:
        interpolation = sitk.sitkBSpline
    resampler.SetInterpolator(interpolation)
    resampler.SetDefaultPixelValue(default_value)
    if pixel_type is not None:
        resampler.SetOutputPixelType(pixel_type)
    return resampler.Execute(image)


def run_hd_bet(input_path, output_dir, device=0, hdbet_exe=None, save_bet_mask=True):
    """Run HD-BET brain extraction on a NIfTI file via its CLI.

    Matches the installed HD-BET 2.x CLI (`hd-bet -i in.nii.gz -o out.nii.gz
    --save_bet_mask`, single-file mode; -device defaults to 'cuda'):

        <output_dir>/<stem>.nii.gz        skull-stripped image
        <output_dir>/<stem>_bet.nii.gz    brain mask (kept via --save_bet_mask)

    GPU selection is done via CUDA_VISIBLE_DEVICES (the CLI has no GPU-index
    flag; its default device is cuda:0).

    Returns (bet_image_path, mask_image_path). Both share the input's grid.
    """
    os.makedirs(output_dir, exist_ok=True)
    # The env's own hd-bet first: a process manager may start the env's python.exe without
    # `conda activate`, so PATH holds base miniconda's Scripts - whose hd-bet is a
    # different package (brainles_hd_bet) with an incompatible API.
    env_dir = os.path.dirname(sys.executable)
    own = [p for p in (os.path.join(env_dir, "Scripts", "hd-bet.exe"),   # Windows conda env
                       os.path.join(env_dir, "hd-bet"))                  # Linux/macOS bin/
           if os.path.isfile(p)]
    exe = hdbet_exe or (own[0] if own else shutil.which("hd-bet"))
    if exe is None:
        raise RuntimeError("hd-bet executable not found; install it via 'pip install hd-bet' "
                           "into the pipeline environment")

    base = os.path.basename(input_path)
    stem = base[:-7] if base.endswith(".nii.gz") else os.path.splitext(base)[0]
    out_path = os.path.abspath(os.path.join(output_dir, stem + ".nii.gz"))
    mask_path = out_path[:-7] + "_bet.nii.gz"     # HD-BET 2.x keeps the mask here
    for p in (out_path, mask_path):
        if os.path.isfile(p):
            os.remove(p)

    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", str(device))
    cmd = [exe, "-i", input_path, "-o", out_path, "--save_bet_mask"]
    log.info("running HD-BET: %s", " ".join(cmd))
    try:
        # utf-8/replace: the tools print progress glyphs cp1252 can't decode, which
        # killed the reader thread (and would have hidden the error text on failure)
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=1800, env=env)
    except subprocess.TimeoutExpired:
        raise RuntimeError("HD-BET timed out after 1800 s")
    if proc.returncode != 0:
        tail = (proc.stdout or "") + "\n" + (proc.stderr or "")
        raise RuntimeError("HD-BET failed (rc=%d):\n%s" % (proc.returncode, tail[-4000:]))
    if not os.path.isfile(out_path) or (save_bet_mask and not os.path.isfile(mask_path)):
        tail = (proc.stdout or "") + "\n" + (proc.stderr or "")
        raise RuntimeError("HD-BET produced no output under %s:\n%s"
                           % (output_dir, tail[-3000:]))
    return out_path, mask_path