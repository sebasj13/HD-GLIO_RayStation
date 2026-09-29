"""
dicom_io.py - DICOM MR series reading for the HD-GLIO pipeline.

Responsibilities:
  * Discover and group DICOM MR image series inside a job's series folder.
  * Load a series into a SimpleITK image with geometry taken directly from
    the DICOM headers (ImagePositionPatient / ImageOrientationPatient),
    so the pipeline works for tilted (oblique) Siemens acquisitions.
  * Keep per-slice metadata (SOPInstanceUID, ImagePositionPatient) so the
    RTSTRUCT writer can reference the original slices exactly.

Geometry model (all in mm, DICOM LPS world coordinates):
  * x+ = patient Left, y+ = Posterior, z+ = Superior (LPS).
  * A series is treated as a regular stack: IPP_k = IPP_0 + k * dz * d2.
    Deviations beyond GRID_TOLERANCE_MM raise an error rather than silently
    producing shifted contours.

Array conventions:
  * sitk images: index (x=column, y=row, z=slice), LPS world.
  * numpy from sitk: [z, y, x].
"""

import logging
import os
from dataclasses import dataclass

import numpy as np
import pydicom
import SimpleITK as sitk

log = logging.getLogger("glio.dicom_io")

MR_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.4"
GRID_TOLERANCE_MM = 0.25          # max deviation of IPP_k from linear model
IOP_TOLERANCE = 2e-3              # max IOP deviation across slices


@dataclass
class SliceMeta:
    """Metadata of one DICOM slice within a series."""
    file_path: str
    sop_instance_uid: str
    ipp: np.ndarray            # ImagePositionPatient, mm, LPS (len 3)
    iop_row: np.ndarray        # IOP first triplet  (direction of +column)
    iop_col: np.ndarray        # IOP second triplet (direction of +row)
    pixel_spacing: tuple       # (row spacing, col spacing) mm per DICOM tag


@dataclass
class DicomSeries:
    """A DICOM MR image series discovered in a folder."""
    series_uid: str
    description: str
    series_number: int
    modality: str
    folder: str
    slices: list               # list[SliceMeta], in sorted stack order
    frame_of_reference_uid: str
    study_uid: str
    contrast_agent: str        # ContrastBolusAgent, "" if none
    spacing_between_slices: float

    @property
    def n_slices(self):
        return len(self.slices)


def read_series_metadata(folder):
    """Scan a folder (recursively) for DICOM files and group them by series.

    Returns a dict {series_uid: [SliceMeta, ...]}. Files that are not plain
    MR image storage objects (e.g. Presentation State) are ignored.
    """
    groups = {}
    for root, _dirs, names in os.walk(folder):
        for name in names:
            path = os.path.join(root, name)
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True, force=False)
            except Exception:
                continue
            # MR image storage only (drops Presentation State etc.).
            # Note: with stop_before_pixels=True the PixelData element is not
            # even parsed - never test for its presence here.
            if getattr(ds, "SOPClassUID", None) != MR_IMAGE_STORAGE:
                continue
            uid = str(ds.SeriesInstanceUID)
            meta = SliceMeta(
                file_path=path,
                sop_instance_uid=str(ds.SOPInstanceUID),
                ipp=np.asarray(ds.ImagePositionPatient, dtype=float),
                iop_row=np.asarray(ds.ImageOrientationPatient[0:3], dtype=float),
                iop_col=np.asarray(ds.ImageOrientationPatient[3:6], dtype=float),
                pixel_spacing=tuple(float(v) for v in ds.PixelSpacing),
            )
            groups.setdefault(uid, []).append(meta)
    return groups


def _slice_sort_key(meta, normal):
    return float(np.dot(meta.ipp, normal))


def collect_series(folder):
    """Return a list of DicomSeries found under *folder*, sorted by SeriesNumber."""
    grouped = read_series_metadata(folder)
    out = []
    for uid, slices in grouped.items():
        try:
            first = pydicom.dcmread(slices[0].file_path, stop_before_pixels=True)
        except Exception:
            continue
        desc = str(getattr(first, "SeriesDescription", ""))
        num = int(getattr(first, "SeriesNumber", 0) or 0)
        contrast = str(getattr(first, "ContrastBolusAgent", "") or "")
        spacing_between = float(getattr(first, "SpacingBetweenSlices",
                                        getattr(first, "SliceThickness", 1.0)) or 1.0)
        # Sort the slices into a consistent stack order (along the slice normal)
        normal = np.cross(np.asarray(first.ImageOrientationPatient[0:3], dtype=float),
                          np.asarray(first.ImageOrientationPatient[3:6], dtype=float))
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            raise ValueError("degenerate ImageOrientationPatient in series %s" % uid)
        normal = normal / norm
        slices.sort(key=lambda m: _slice_sort_key(m, normal))
        out.append(DicomSeries(
            series_uid=uid,
            description=desc,
            series_number=num,
            modality=str(getattr(first, "Modality", "MR")),
            folder=folder,
            slices=slices,
            frame_of_reference_uid=str(getattr(first, "FrameOfReferenceUID", "")),
            study_uid=str(getattr(first, "StudyInstanceUID", "")),
            contrast_agent=contrast,
            spacing_between_slices=spacing_between,
        ))
    out.sort(key=lambda s: s.series_number)
    return out


def stack_axes(s0):
    """Return the unit direction vectors (d0, d1, d2) of a series.

    d0 = +column (along a row), d1 = +row (down a column), d2 = slice normal,
    computed from the first slice's ImageOrientationPatient. All in LPS mm.
    """
    d0 = np.asarray(s0.iop_row, dtype=float)
    d1 = np.asarray(s0.iop_col, dtype=float)
    d2 = np.cross(d0, d1)
    d2 /= np.linalg.norm(d2)
    return d0, d1, d2


def check_series_consistency(series):
    """Validate geometry of a DicomSeries; returns dz (mm, signed slice spacing).

    Checks:
      * orthonormal IOP, constant IOP and pixel spacing across slices
      * IPP_k lies on the linear model IPP_0 + k*dz*d2 (regular stack)
    """
    s0 = series.slices[0]
    if abs(np.linalg.norm(s0.iop_row) - 1) > IOP_TOLERANCE or \
            abs(np.linalg.norm(s0.iop_col) - 1) > IOP_TOLERANCE:
        raise ValueError("ImageOrientationPatient is not orthonormal in series %s" % series.series_uid)
    if abs(float(np.dot(s0.iop_row, s0.iop_col))) > IOP_TOLERANCE:
        raise ValueError("ImageOrientationPatient vectors are not orthogonal in series %s"
                         % series.series_uid)

    d0, d1, d2 = stack_axes(s0)
    sx, sy = s0.pixel_spacing[0], s0.pixel_spacing[1]

    proj = np.array([float(np.dot(m.ipp, d2)) for m in series.slices])
    if len(proj) > 1:
        dz_vals = np.diff(proj)
        dz = float(np.mean(dz_vals))
        if abs(dz) < 1e-4:
            raise ValueError("slice positions are not distinct in series %s" % series.series_uid)
        if float(np.max(np.abs(dz_vals - dz))) > GRID_TOLERANCE_MM:
            raise ValueError("non-uniform slice spacing (%.3f..%.3f mm) in series %s"
                             % (float(np.min(dz_vals)), float(np.max(dz_vals)), series.series_uid))
    else:
        dz = series.spacing_between_slices

    for n, s in enumerate(series.slices):
        if float(np.max(np.abs(np.concatenate([s0.iop_row, s0.iop_col]) -
                               np.concatenate([s.iop_row, s.iop_col])))) > IOP_TOLERANCE:
            raise ValueError("ImageOrientationPatient varies across slices in series %s (slice %d)"
                             % (series.series_uid, n))
        if abs(s.pixel_spacing[0] - sx) > 1e-3 or abs(s.pixel_spacing[1] - sy) > 1e-3:
            raise ValueError("PixelSpacing varies across slices in series %s (slice %d)"
                             % (series.series_uid, n))
        expect = np.asarray(s0.ipp) + n * dz * d2
        dev = float(np.linalg.norm(s.ipp - expect))
        if dev > GRID_TOLERANCE_MM:
            raise ValueError("slice %d deviates %.3f mm from regular stack model in series %s"
                             % (n, dev, series.series_uid))
    return dz, (d0, d1, d2)


def load_series_image(series):
    """Load a DICOM series as a SimpleITK float32 image on the native DICOM grid.

    Returns (sitk.Image, slice_meta_list). The sitk index order is
    (x=column, y=row, z=slice); voxel (0,0,0) center = IPP of the first slice.
    Pixel values are float32 with RescaleSlope/Intercept applied (GDCM).
    """
    dz, (d0, d1, d2) = check_series_consistency(series)
    s0 = series.slices[0]

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([s.file_path for s in series.slices])
    image = reader.Execute()

    # Override geometry with values derived straight from the headers
    # (GDCM handles pixel decoding and rescale; we trust IOP/IPP/spacing here).
    # sitk spacing[0] is the step per *column* index -> PixelSpacing[1];
    # spacing[1] is the step per *row* index -> PixelSpacing[0].
    direction = np.column_stack([d0, d1, d2])          # columns = axis directions
    image.SetOrigin(tuple(float(v) for v in s0.ipp))
    image.SetSpacing((float(s0.pixel_spacing[1]), float(s0.pixel_spacing[0]),
                      float(abs(dz))))
    image.SetDirection(tuple(float(v) for v in direction.ravel()))
    if image.GetSize()[2] != len(series.slices):
        raise ValueError("slice count mismatch while loading series %s" % series.series_uid)
    return image, series.slices


def classify_series(series_list):
    """Map DICOM series to the HD-GLIO slots (T1, T1c, T2, FLAIR).

    Rules (case-insensitive on SeriesDescription, contrast flag from
    ContrastBolusAgent (0018,0010)):
      * FLAIR  : contains "flair", or ("t2" and "dark-fluid")
      * T2     : contains "t2" but neither "dark-fluid" nor "flair"
      * T1     : contains "t1"; the pre-contrast one is T1, the post-contrast
                 one (ContrastBolusAgent set) is T1c.
    Ties are broken by slice count, then in-plane resolution.
    Returns dict slot -> DicomSeries. Raises when a slot cannot be filled.
    """
    def desc(s):
        return str(s.description or "").lower()

    def has_contrast(s):
        return bool(s.contrast_agent.strip())

    def res_key(s):
        return -float(s.slices[0].pixel_spacing[0])

    t1_like = [s for s in series_list if "t1" in desc(s)]
    t2_like = [s for s in series_list if "t2" in desc(s)]
    flair = [s for s in t2_like if "flair" in desc(s) or "dark-fluid" in desc(s)]
    plain_t2 = [s for s in t2_like if s not in flair]

    mapping = {}
    if flair:
        flair.sort(key=lambda s: (-s.n_slices, res_key(s)))
        mapping["FLAIR"] = flair[0]
    if plain_t2:
        plain_t2.sort(key=lambda s: (-s.n_slices, res_key(s)))
        mapping["T2"] = plain_t2[0]

    pre = [s for s in t1_like if not has_contrast(s)]
    post = [s for s in t1_like if has_contrast(s)]
    if pre:
        pre.sort(key=lambda s: (-s.n_slices, res_key(s)))
        mapping["T1"] = pre[0]
    if post:
        post.sort(key=lambda s: (-s.n_slices, res_key(s)))
        mapping["T1c"] = post[0]

    missing = [slot for slot in ("T1", "T1c", "T2", "FLAIR") if slot not in mapping]
    if missing:
        raise ValueError(
            "cannot map all HD-GLIO slots; missing: %s. Found series: %s"
            % (", ".join(missing),
               "; ".join("%r (series %s, %d slices)"
                         % (desc(s) or "<no description>", s.series_number, s.n_slices)
                         for s in series_list)))
    return mapping