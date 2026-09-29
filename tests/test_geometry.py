"""Geometry self-test for the HD-GLIO pipeline (no model weights needed).

Builds synthetic oblique DICOM MR series on disk, then verifies:

  A. DICOM -> sitk geometry follows the DICOM image-plane equation
     P(i,j,k) = IPP_k + i*PixelSpacing[1]*iop_row + j*PixelSpacing[0]*iop_col
  B. reorient_to_fsl_std produces LAS axcodes and is a pure re-indexing
     (world space unchanged; voxel values correspond exactly).
  C. reorient_mask_back maps a mask on the T1-LAS grid back onto a second
     series' native grid using the inverse of a KNOWN transform (validates
     the sitk Resample transform-direction convention end-to-end).
  D. the written RTSTRUCT references the original SOP instances / FoR and
     its contour points lie on the referenced slices' planes.

Run:  conda run -n glio python tests/test_geometry.py
"""

import os
import shutil
import sys

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian
import SimpleITK as sitk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline import dicom_io, preprocess, rtstruct                  # noqa: E402

OUT = os.path.join(ROOT, "test_output", "geometry_test")
FOREF = "1.2.840.10008.15.1.1.9000.1"
MR_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.4"


def _rotation_matrix(rx_deg, ry_deg, rz_deg):
    """Rotation matrix whose columns are the sitk axis directions (LPS)."""
    ax, ay, az = np.deg2rad([rx_deg, ry_deg, rz_deg])
    cx, sx_, cy, sy_, cz, sz_ = (np.cos(ax), np.sin(ax), np.cos(ay),
                                 np.sin(ay), np.cos(az), np.sin(az))
    Rx = np.array([[1, 0, 0], [0, cx, -sx_], [0, sx_, cx]])
    Ry = np.array([[cy, 0, sy_], [0, 1, 0], [-sy_, 0, cy]])
    Rz = np.array([[cz, -sz_, 0], [sz_, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def make_series(base_dir, series_number, description, nx=12, ny=13, nz=14,
                ps_row=0.8, ps_col=1.3, dz=2.0, rot=(0.0, 0.0, 0.0),
                ipp0=(10.0, -20.0, 5.0), normal_sign=1.0, contrast=False):
    """Write a synthetic MR DICOM series; returns (folder, series_uid, study_uid)."""
    R = _rotation_matrix(*rot)
    iop = np.concatenate([R[:, 0], R[:, 1]]).astype(float)
    normal = R[:, 2] * normal_sign

    folder = os.path.join(base_dir, "series_%d" % series_number)
    os.makedirs(folder, exist_ok=True)
    series_uid = "1.2.840.10008.15.1.3.%d" % series_number
    study_uid = "1.2.840.10008.15.1.2.7777"

    rng = np.random.RandomState(series_number)
    for k in range(nz):
        arr = (rng.random((ny, nx)) * 400 + 100).astype(np.uint16)
        ds = FileDataset(None, {}, file_meta=FileMetaDataset(), preamble=b"\x00" * 128)
        ds.file_meta.MediaStorageSOPClassUID = MR_IMAGE_STORAGE
        sop = "1.2.840.10008.15.1.4.%d.%d" % (series_number, k)
        ds.file_meta.MediaStorageSOPInstanceUID = sop
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.SOPClassUID = MR_IMAGE_STORAGE
        ds.SOPInstanceUID = sop
        ds.Modality = "MR"
        ds.SeriesInstanceUID = series_uid
        ds.StudyInstanceUID = study_uid
        ds.FrameOfReferenceUID = FOREF
        ds.SeriesNumber = series_number
        ds.SeriesDescription = description
        ds.PatientName = "TEST^GEOMETRY"
        ds.PatientID = "GEO_TEST_001"
        ds.PatientBirthDate = "19700101"
        ds.PatientSex = "O"
        ds.StudyDate = "20260805"
        ds.StudyTime = "130000"
        ds.StudyID = "1"
        ds.AccessionNumber = "ACC001"
        ds.ReferringPhysicianName = ""
        ds.InstitutionName = "Test Hospital"
        ds.ImagePositionPatient = [float(v) for v in (np.asarray(ipp0) + k * dz * normal)]
        ds.ImageOrientationPatient = [float(v) for v in iop]
        ds.PixelSpacing = [float(ps_row), float(ps_col)]
        ds.SliceThickness = float(abs(dz))
        ds.SpacingBetweenSlices = float(abs(dz))
        ds.InstanceNumber = k + 1
        ds.Rows = ny
        ds.Columns = nx
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = 0.0
        if contrast:
            ds.ContrastBolusAgent = "GADO"
        ds.PixelData = arr.tobytes()
        # note: TransferSyntaxUID in file_meta decides the encoding; the legacy
        # ds.is_little_endian / ds.is_implicit_VR setters are gone in pydicom 3.x.
        ds.save_as(os.path.join(folder, "IM_%04d.dcm" % k), enforce_file_format=False)
    return folder, series_uid, study_uid


def world_native(series, i, j, k):
    """DICOM image-plane equation, independent of sitk."""
    s = series.slices[k]
    return (np.asarray(s.ipp) + i * s.pixel_spacing[1] * s.iop_row
            + j * s.pixel_spacing[0] * s.iop_col)


def axcodes_of(img):
    """Nibabel-style axcodes of a sitk image (LPS world: +x=L, +y=P, +z=S)."""
    D = np.asarray(img.GetDirection()).reshape(3, 3)
    dirs = {"L": (1, 0, 0), "R": (-1, 0, 0), "P": (0, 1, 0), "A": (0, -1, 0),
            "S": (0, 0, 1), "I": (0, 0, -1)}
    codes = []
    for t in range(3):
        col = D[:, t]
        best, best_dot = "?", -2.0
        for code, vec in dirs.items():
            c = float(np.dot(col, vec))       # signed: L vs R differ by sign
            if c > best_dot:
                best, best_dot = code, c
        codes.append(best)
    return tuple(codes)


def mask_centroid_from_image(img):
    arr = sitk.GetArrayFromImage(img)          # [z, y, x]
    z, y, x = np.nonzero(arr)
    pts = np.array([img.TransformIndexToPhysicalPoint((int(c), int(b), int(a)))
                    for c, b, a in zip(x, y, z)])
    return pts.mean(axis=0)


def sphere_mask_on_image(img, center_world, radius_mm):
    """uint8 mask (same grid/geometry as img) of a sphere in world space."""
    nx, ny, nz = img.GetSize()
    z, y, x = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx),
                          indexing="ij")
    pts = np.array([img.TransformIndexToPhysicalPoint((int(c), int(b), int(a)))
                    for c, b, a in zip(x.ravel(), y.ravel(), z.ravel())])
    d = np.linalg.norm(pts - np.asarray(center_world), axis=1)
    arr = (d < radius_mm).astype(np.uint8).reshape(nz, ny, nx)   # [z, y, x]
    mask = sitk.GetImageFromArray(arr)
    mask.CopyInformation(img)
    return mask


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT, ignore_errors=True)

    # T1: axial-oblique (25 deg + 10 deg tilt), rectangular pixels.
    # T1c: slightly rotated vs T1, isotropic pixels, inverted slice normal,
    # large enough to fully cover the T1 volume (the back-mapped mask must
    # land inside the target's field of view).
    make_series(OUT, 3, "t1_space_TEST", nx=10, ny=11, nz=14,
                ps_row=0.8, ps_col=1.3, dz=2.0,
                rot=(10.0, 0.0, 25.0), normal_sign=1.0)
    make_series(OUT, 17, "t1_mprage_TEST", nx=64, ny=64, nz=40,
                ps_row=1.0, ps_col=1.0, dz=1.0,
                rot=(10.0, 0.0, 28.0), normal_sign=-1.0,
                ipp0=(8.0, -31.0, 27.0), contrast=True)

    series_list = dicom_io.collect_series(OUT)
    by_num = {s.series_number: s for s in series_list}
    s_t1, s_t1c = by_num[3], by_num[17]
    assert s_t1.n_slices == 14 and s_t1c.n_slices == 40
    assert s_t1.frame_of_reference_uid == FOREF

    # ---- A. DICOM -> sitk geometry ------------------------------------------
    img_t1, _ = dicom_io.load_series_image(s_t1)
    nx, ny, nz = img_t1.GetSize()
    assert (nx, ny, nz) == (10, 11, 14), "size must be (cols, rows, slices)"
    sx, sy, sz = img_t1.GetSpacing()
    assert abs(sx - 1.3) < 1e-9 and abs(sy - 0.8) < 1e-9, \
        "spacing must be (col_spacing, row_spacing)"
    for (i, j, k) in [(0, 0, 0), (5, 3, 2), (9, 9, 13), (4, 7, 8)]:
        w = np.asarray(img_t1.TransformIndexToPhysicalPoint((i, j, k)))
        expect = world_native(s_t1, i, j, k)
        assert np.linalg.norm(w - expect) < 1e-4, \
            "voxel->world mismatch at %s: %s vs %s" % ((i, j, k), w, expect)
    print("[A] DICOM geometry (origin/spacing/direction) OK")

    # ---- B. reorientation is a pure re-indexing ------------------------------
    las_t1, _info = preprocess.reorient_to_fsl_std(img_t1)
    assert axcodes_of(las_t1) == ("L", "A", "S"), axcodes_of(las_t1)
    rng = np.random.RandomState(3)
    las_arr = sitk.GetArrayFromImage(las_t1)
    nat_arr = sitk.GetArrayFromImage(img_t1)
    for _ in range(200):
        (i, j, k) = (int(rng.randint(0, nx)), int(rng.randint(0, ny)),
                     int(rng.randint(0, nz)))
        w = img_t1.TransformIndexToPhysicalPoint((i, j, k))
        ci, cj, ck = las_t1.TransformPhysicalPointToContinuousIndex(w)
        assert max(abs(ci - round(ci)), abs(cj - round(cj)),
                   abs(ck - round(ck))) < 1e-4, \
            "LAS grid must coincide with the native grid in world space"
        v_las = las_arr[int(round(ck)), int(round(cj)), int(round(ci))]
        assert abs(v_las - nat_arr[k, j, i]) < 1e-3, "value mismatch after reorientation"
    print("[B] reorientation to LAS: axcodes OK, world-identity value check OK")

    # ---- C. back-mapping with a KNOWN transform ------------------------------
    center = np.asarray(img_t1.TransformIndexToPhysicalPoint((5, 5, 7)))
    seg_las = sphere_mask_on_image(las_t1, center, 3.5)

    # known transform: translation by (5, 0, 0) mm mapping T1-LAS world -> target world
    T = sitk.TranslationTransform(3, (5.0, 0.0, 0.0))
    target_native, _ = dicom_io.load_series_image(s_t1c)
    back = preprocess.reorient_mask_back(seg_las, target_native, T)
    assert back.GetSize() == target_native.GetSize()
    c_expected = center + np.asarray([5.0, 0.0, 0.0])
    c_back = mask_centroid_from_image(back)
    dist = float(np.linalg.norm(np.asarray(c_back) - c_expected))
    assert dist < 1.5, "back-mapped centroid off by %.3f mm (transform convention bug?)" % dist
    print("[C] mask back-mapping: centroid within %.3f mm of expected" % dist)

    # ---- D. RTSTRUCT ----------------------------------------------------------
    mask_native = sitk.GetArrayFromImage(back) == 1
    label_masks = {1: mask_native}
    roi_specs = {1: {"name": "HDGLIO_ET", "color": (255, 64, 64),
                     "interpreted_type": "GTV"}}
    rt_path = os.path.join(OUT, "test_rtstruct.dcm")
    rtstruct.write_rtstruct(label_masks, s_t1c, rt_path, roi_specs)

    rtds = pydicom.dcmread(rt_path)
    assert str(rtds.SOPClassUID) == "1.2.840.10008.5.1.4.1.1.481.3"
    assert str(rtds.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID) == FOREF
    ref_uids = {str(x.ReferencedSOPInstanceUID)
                for x in rtds.ReferencedFrameOfReferenceSequence[0]
                .RTReferencedStudySequence[0].RTReferencedSeriesSequence[0]
                .ContourImageSequence}
    dicom_uids = {m.sop_instance_uid for m in s_t1c.slices}
    assert ref_uids == dicom_uids, "RTSTRUCT must reference every slice SOP"
    assert str(rtds.StudyInstanceUID) == s_t1c.study_uid

    # contour points: planar with the referenced slice, inside the image bounds
    n_pts = 0
    for roi_c in rtds.ROIContourSequence:
        for c in roi_c.ContourSequence:
            ref_uid = str(c.ContourImageSequence[0].ReferencedSOPInstanceUID)
            k = next(n for n, m in enumerate(s_t1c.slices)
                     if m.sop_instance_uid == ref_uid)
            sl = s_t1c.slices[k]
            pts = np.asarray(c.ContourData, dtype=float).reshape(-1, 3)
            nrm = np.cross(sl.iop_row, sl.iop_col)
            planar = np.abs((pts - np.asarray(sl.ipp)) @ nrm)
            assert planar.max() < 0.3, \
                "contour deviates %.3f mm from its slice plane" % planar.max()
            n_pts += len(pts)
    assert n_pts > 0
    print("[D] RTSTRUCT: %d contour points, all planar on referenced slices" % n_pts)

    print("\nALL GEOMETRY CHECKS PASSED")


if __name__ == "__main__":
    main()