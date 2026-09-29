"""
rtstruct.py - Binary segmentation mask -> DICOM RTSTRUCT, and PNG previews.

The mask must be defined on the *native DICOM grid* of the target series
(array order [slice, row, col], slice k == k-th file of the sorted series).
Contours are generated per slice with marching squares (skimage), converted
to patient coordinates (mm, LPS) using the slice's own ImagePositionPatient
and ImageOrientationPatient, and written as CLOSED_PLANAR contour items.

The writer is deliberately dependency-free beyond pydicom/numpy/skimage:
a hand-rolled writer gives exact control over Frame of Reference, referenced
SOP instances, ROI names and colors.
"""

import datetime
import logging

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

log = logging.getLogger("glio.rtstruct")

RTSTRUCT_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.3"
MR_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.4"
STUDY_COMPONENT_SOP_CLASS = "1.2.840.10008.3.1.2.3.1"


def _world_point(slice_meta, row, col):
    """Patient coordinate (mm, LPS) of an (row, col) image index on a slice.

    IPP is the center of the (row=0, col=0) voxel; a +column index step moves
    along iop_row with the column spacing (PixelSpacing[1]), a +row index step
    along iop_col with the row spacing (PixelSpacing[0]).
    """
    sx = slice_meta.pixel_spacing[1]   # step per +column index
    sy = slice_meta.pixel_spacing[0]   # step per +row index
    p = (np.asarray(slice_meta.ipp, dtype=float)
         + col * sx * np.asarray(slice_meta.iop_row, dtype=float)
         + row * sy * np.asarray(slice_meta.iop_col, dtype=float))
    return p


def _mask_to_contours(mask):
    """Yield (slice_index, [(row, col), ...]) polygons from a boolean mask."""
    from skimage import measure

    for k in range(mask.shape[0]):
        m = mask[k].astype(np.float32)
        if not m.any():
            continue
        for contour in measure.find_contours(m, 0.5):
            if len(contour) < 3:
                continue
            yield k, contour


def write_rtstruct(label_masks, series, output_path, roi_specs,
                   structure_set_label="HD-GLIO",
                   series_description="HD-GLIO auto contours"):
    """Write an RT Structure Set for the given label masks.

    label_masks : dict {label_int: np.ndarray bool, shape (nz, ny, nx)}
                  defined on the native grid of *series* (array order
                  [z, y, x], slice k == sorted DICOM slice k).
    series      : pipeline.dicom_io.DicomSeries of the target MR series.
    roi_specs   : dict {label_int: {"name": str, "color": (r, g, b),
                                    "interpreted_type": str}}
    """
    if not label_masks or not any(m.any() for m in label_masks.values()):
        raise ValueError("segmentation masks are empty; refusing to write an empty RTSTRUCT")

    s0_full = pydicom.dcmread(series.slices[0].file_path, stop_before_pixels=True)

    ds = FileDataset(None, {}, file_meta=FileMetaDataset(), preamble=b"\x00" * 128)
    now = datetime.datetime.now()
    ds.file_meta.MediaStorageSOPClassUID = RTSTRUCT_SOP_CLASS
    media_sop_uid = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = media_sop_uid
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds.SpecificCharacterSet = "ISO_IR 100"
    ds.SOPClassUID = RTSTRUCT_SOP_CLASS
    ds.SOPInstanceUID = media_sop_uid
    ds.Modality = "RTSTRUCT"
    ds.PatientName = str(getattr(s0_full, "PatientName", ""))
    ds.PatientID = str(getattr(s0_full, "PatientID", ""))
    ds.PatientBirthDate = str(getattr(s0_full, "PatientBirthDate", ""))
    ds.PatientSex = str(getattr(s0_full, "PatientSex", ""))
    ds.StudyInstanceUID = series.study_uid
    ds.StudyDate = str(getattr(s0_full, "StudyDate", ""))
    ds.StudyTime = str(getattr(s0_full, "StudyTime", ""))
    ds.StudyID = str(getattr(s0_full, "StudyID", ""))
    ds.AccessionNumber = str(getattr(s0_full, "AccessionNumber", ""))
    ds.ReferringPhysicianName = str(getattr(s0_full, "ReferringPhysicianName", ""))
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = 9000 + (int(series.series_number) % 1000)
    ds.SeriesDescription = series_description
    ds.StructureSetLabel = structure_set_label[:16]
    ds.StructureSetName = structure_set_label
    ds.StructureSetDate = now.strftime("%Y%m%d")
    ds.StructureSetTime = now.strftime("%H%M%S")
    ds.InstanceCreationDate = now.strftime("%Y%m%d")
    ds.InstanceCreationTime = now.strftime("%H%M%S")
    ds.Manufacturer = "HD-GLIO RayStation pipeline"
    ds.InstitutionName = str(getattr(s0_full, "InstitutionName", ""))

    # ---- Referenced frame of reference ------------------------------------
    ref_for = Dataset()
    ref_for.FrameOfReferenceUID = series.frame_of_reference_uid
    ref_study = Dataset()
    ref_study.ReferencedSOPClassUID = STUDY_COMPONENT_SOP_CLASS
    ref_study.ReferencedSOPInstanceUID = series.study_uid
    ref_series = Dataset()
    ref_series.SeriesInstanceUID = series.series_uid
    img_seq = []
    for s in series.slices:
        item = Dataset()
        item.ReferencedSOPClassUID = MR_IMAGE_STORAGE
        item.ReferencedSOPInstanceUID = s.sop_instance_uid
        img_seq.append(item)
    ref_series.ContourImageSequence = Sequence(img_seq)
    ref_study.RTReferencedSeriesSequence = Sequence([ref_series])
    ref_for.RTReferencedStudySequence = Sequence([ref_study])
    ds.ReferencedFrameOfReferenceSequence = Sequence([ref_for])

    # ---- ROIs --------------------------------------------------------------
    structure_set_roi_sequence = []
    roi_contour_sequence = []
    rt_roi_observations = []

    for label in sorted(roi_specs):
        specs = roi_specs[label]
        mask = label_masks.get(int(label))
        if mask is None or not mask.any():
            log.warning("label %d has no voxels; skipping ROI %s", label, specs["name"])
            continue

        roi_number = len(structure_set_roi_sequence) + 1
        roi = Dataset()
        roi.ROINumber = roi_number
        roi.ReferencedFrameOfReferenceUID = series.frame_of_reference_uid
        roi.ROIName = specs["name"]
        roi.ROIDescription = "HD-GLIO automatic segmentation (label %d)" % int(label)
        roi.ROIGenerationAlgorithm = "AUTOMATIC"
        structure_set_roi_sequence.append(roi)

        contour_seq = []
        n_points_total = 0
        for k, contour in _mask_to_contours(mask):
            pts = []
            for (row, col) in contour:
                p = _world_point(series.slices[k], float(row), float(col))
                pts.extend([float(p[0]), float(p[1]), float(p[2])])
            item = Dataset()
            item.ContourGeometricType = "CLOSED_PLANAR"
            item.NumberOfContourPoints = len(contour)
            item.ContourData = [float(v) for v in pts]
            cimg = Dataset()
            cimg.ReferencedSOPClassUID = MR_IMAGE_STORAGE
            cimg.ReferencedSOPInstanceUID = series.slices[k].sop_instance_uid
            item.ContourImageSequence = Sequence([cimg])
            contour_seq.append(item)
            n_points_total += len(contour)

        if not contour_seq:
            log.warning("ROI %s: no slice contours after marching squares; skipped", specs["name"])
            structure_set_roi_sequence.pop()
            continue

        rc = Dataset()
        rc.ReferencedROINumber = roi_number
        color = tuple(int(c) for c in specs.get("color", (255, 64, 64)))
        rc.ROIDisplayColor = list(color)
        rc.ContourSequence = Sequence(contour_seq)
        roi_contour_sequence.append(rc)

        obs = Dataset()
        obs.ObservationNumber = roi_number
        obs.ReferencedROINumber = roi_number
        obs.RTROIInterpretedType = str(specs.get("interpreted_type", "ORGAN"))
        obs.ROIObservationLabel = "HD-GLIO label %d" % int(label)
        rt_roi_observations.append(obs)
        log.info("RTSTRUCT ROI %s (number %s): %d contour items, %d points",
                 specs["name"], roi_number, len(contour_seq), n_points_total)

    if not structure_set_roi_sequence:
        raise ValueError("no non-empty ROIs produced by the segmentation")

    ds.StructureSetROISequence = Sequence(structure_set_roi_sequence)
    ds.ROIContourSequence = Sequence(roi_contour_sequence)
    ds.RTROIObservationsSequence = Sequence(rt_roi_observations)

    ds.save_as(output_path, enforce_file_format=False)
    log.info("RTSTRUCT written: %s", output_path)
    return output_path


def voxel_volume_ml(sx, sy, sz):
    """Volume of one voxel (mm^3) in mL."""
    return (sx * sy * sz) / 1000.0