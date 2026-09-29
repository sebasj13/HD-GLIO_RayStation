"""
run_case.py - End-to-end orchestration for one HD-GLIO job.

Input layout (created by the RayStation script or the CLI test runner):

    <input_dir>/
        job.json                     (optional job metadata)
        series_T1/ ... *.dcm         (one folder per HD-GLIO slot;
        series_T1c/  ...              RS export: series_<SLOT>)
        series_T2/
        series_FLAIR/

If job.json carries a slot->series mapping, it is honored; otherwise the
series are auto-classified from SeriesDescription + ContrastBolusAgent.

Output layout:

    <output_dir>/
        status.json                  (poll this from RayStation)
        HDGLIO_<job_id>.dcm          (RTSTRUCT on the target series grid)
        preview_*.png                (QA overlays)
    <work_dir>/                      (intermediates: NIfTI, BET, seg)

The pipeline:
  1. DICOM -> NIfTI (native geometry)
  2. reorient to FSL/MNI152-standard orientation (LAS)
  3. HD-BET brain extraction (all four sequences)
  4. rigid 6-DOF MI registration of T1c/T2/FLAIR onto T1
  5. resample onto the T1 grid, apply the T1 brain mask
  6. hd_glio_predict  (labels: 1 = non-enhancing T2/FLAIR abnormality,
                                  2 = contrast-enhancing tumor)
  7. map the segmentation back onto the native grid of the target series
  8. write the RTSTRUCT + previews + status
"""

import json
import logging
import os
import time
import traceback

import numpy as np
import SimpleITK as sitk

from . import dicom_io
from . import preprocess
from . import run_glio
from . import rtstruct

log = logging.getLogger("glio.run_case")

PIPELINE_VERSION = "1.0.0"

DEFAULT_ROI_SPECS = {
    1: {"name": "HDGLIO_NET", "color": (64, 128, 255), "interpreted_type": "ORGAN"},
    2: {"name": "HDGLIO_ET", "color": (255, 64, 64), "interpreted_type": "GTV"},
}

# HD-GLIO label semantics (verified against the installed hd_glio package).
LABEL_NAMES = {1: "non-enhancing T2/FLAIR abnormality", 2: "contrast-enhancing tumor"}

SLOTS = ("T1", "T1c", "T2", "FLAIR")


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_config_path():
    """config.json in the project root (site copy, not tracked), else config.example.json."""
    site = os.path.join(ROOT, "config.json")
    return site if os.path.isfile(site) else os.path.join(ROOT, "config.example.json")


def load_config(config_path=None):
    """Load the pipeline config (default: default_config_path())."""
    config_path = config_path or default_config_path()
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_status(job_dir, state, message="", extra=None, critical=True):
    """Atomically write output/status.json (the RayStation poll target).

    RayStation polls this file over SMB; Windows refuses os.replace while a
    reader holds the target open (WinError 32), hence the retries. Progress
    updates pass critical=False - a lost progress line must never kill a job.
    """
    payload = dict(state=state, message=message, updated=time.strftime("%Y-%m-%dT%H:%M:%S"))
    if extra:
        payload.update(extra)
    os.makedirs(os.path.join(job_dir, "output"), exist_ok=True)
    tmp = os.path.join(job_dir, "output", "status.json.%d.tmp" % os.getpid())
    final = os.path.join(job_dir, "output", "status.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    for attempt in range(10):
        try:
            os.replace(tmp, final)
            return final
        except OSError as exc:
            last = exc
            time.sleep(0.5)
    try:
        os.remove(tmp)
    except OSError:
        pass
    if critical:
        raise last
    log.warning("status.json not writable, continuing: %s", last)
    return None


def write_preview(image, label_masks, series, output_path, roi_specs, slice_idx=None):
    """Write an axial QA overlay PNG of the segmentation onto the native image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = sitk.GetArrayFromImage(image)
    nz = img.shape[0]
    if slice_idx is None:
        slice_idx = int(np.argmax([mask[k].sum() for k in range(nz)
                                   for mask in label_masks.values() if mask.any()] or [0]))
    fig, ax = plt.subplots(figsize=(6, 6))
    vmin = float(np.percentile(img[slice_idx], 1))
    vmax = float(np.percentile(img[slice_idx], 99))
    ax.imshow(img[slice_idx], cmap="gray", origin="upper", vmin=vmin, vmax=vmax)
    for label, mask in label_masks.items():
        if not mask.any():
            continue
        rgb = roi_specs.get(label, {}).get("color", (0, 255, 0))
        ax.contour(mask[slice_idx].astype(np.float32), levels=[0.5],
                   colors=[tuple(c / 255.0 for c in rgb)], linewidths=0.8)
    ax.set_title("%s (slice %d/%d)" % (series.description, slice_idx + 1, nz), fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def mapping_from_job(job_meta, mr_series):
    """{slot: series} from the user's choice in job.json (matched by SeriesInstanceUID).

    The user picked these series in RayStation, so an incomplete match is an
    error - silently auto-classifying would override their choice.
    """
    canonical = {s.lower(): s for s in SLOTS}
    by_uid = {s.series_uid: s for s in mr_series}
    mapping = {}
    for slot, info in job_meta["series"].items():
        key = canonical.get(str(slot).lower())
        uid = str(info.get("series_uid", ""))
        if key and uid in by_uid:
            mapping[key] = by_uid[uid]
    missing = [s for s in SLOTS if s not in mapping]
    if missing:
        raise ValueError("job.json: series for %s not found among the exported series "
                         "(matched by SeriesInstanceUID)" % ", ".join(missing))
    return mapping


PROGRESS_TOTAL = 10


def process_job(input_dir, output_dir, work_dir, config, job_meta=None, progress=None):
    """Run the full pipeline for one job. Raises on failure.

    job_meta: parsed input/job.json (None for the plain CLI path).
    progress: optional callback(step, total, message) - the watcher turns it
    into status.json updates for the RayStation progress bar.
    Returns a result dict; the caller is responsible for status.json.
    """
    t_start = time.time()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    device = int(config.get("gpu_device", 0))
    roi_cfg = config.get("roi_specs")
    roi_specs = {int(k): dict(v) for k, v in roi_cfg.items()} if roi_cfg else dict(DEFAULT_ROI_SPECS)
    target_slot = str(config.get("target_series_slot", "T1c"))
    matches = [s for s in SLOTS if s.lower() == target_slot.lower()]
    if not matches:
        raise ValueError("target_series_slot must be one of %s" % ", ".join(SLOTS))
    target_slot = matches[0]

    job_id = (job_meta or {}).get("job_id") or os.path.basename(os.path.dirname(input_dir))

    def step(i, message):
        log.info("[%d/%d] %s", i, PROGRESS_TOTAL, message)
        if progress is not None:
            progress(i, PROGRESS_TOTAL, message)

    # ---- 1. collect + map series -------------------------------------------
    step(1, "Reading DICOM series")
    log.info("job %s: collecting series from %s", job_id, input_dir)
    series_list = dicom_io.collect_series(input_dir)
    mr_series = [s for s in series_list if s.modality == "MR"]
    if not mr_series:
        raise ValueError("no MR image series found in %s" % input_dir)
    for s in mr_series:
        log.info("  found: series %s %-45s slices=%3d contrast=%r",
                 s.series_number, s.description, s.n_slices, s.contrast_agent or "-")

    if job_meta and job_meta.get("series"):
        mapping = mapping_from_job(job_meta, mr_series)
    else:
        mapping = dicom_io.classify_series(mr_series)
    for slot in SLOTS:
        s = mapping[slot]
        log.info("  slot %-5s -> series %s %r (%d slices, %s)",
                 slot, s.series_number, s.description, s.n_slices,
                 "post-contrast" if s.contrast_agent.strip() else "pre-contrast")

    target_series = mapping[target_slot]
    if target_series.frame_of_reference_uid != mapping["T1"].frame_of_reference_uid:
        raise ValueError("target series and T1 live in different frames of reference; "
                         "the exported data is not one co-registered examination set")

    # ---- 2. DICOM -> NIfTI + reorient to FSL standard -----------------------
    step(2, "NIfTI conversion")
    raw_las = {}
    raw_nifti_dir = os.path.join(work_dir, "raw_nifti")
    os.makedirs(raw_nifti_dir, exist_ok=True)
    for slot in SLOTS:
        series = mapping[slot]
        image, _ = dicom_io.load_series_image(series)
        las, _info = preprocess.reorient_to_fsl_std(image)
        path = os.path.join(raw_nifti_dir, "%s.nii.gz" % slot)
        sitk.WriteImage(sitk.Cast(las, sitk.sitkFloat32), path)
        raw_las[slot] = las
        log.info("  %s: native %s -> LAS %s (voxel %.3f x %.3f x %.3f mm)",
                 slot, str(image.GetSize()), str(las.GetSize()), *las.GetSpacing())
        del image

    # ---- 3. HD-BET brain extraction (on the reoriented images) --------------
    bet_dir = os.path.join(work_dir, "bet")
    bet_images = {}
    t1_mask_img = None
    hdbet_exe = config.get("hdbet_exe") or None
    for i_slot, slot in enumerate(SLOTS):
        step(3 + i_slot, "Skull stripping (HD-BET) %s" % slot)
        bet_path, mask_path = preprocess.run_hd_bet(
            os.path.join(raw_nifti_dir, "%s.nii.gz" % slot), bet_dir, device=device,
            hdbet_exe=hdbet_exe, save_bet_mask=True)
        bet_images[slot] = sitk.ReadImage(bet_path, sitk.sitkFloat32)
        mask_img = sitk.ReadImage(mask_path, sitk.sitkUInt8)
        mask_img.CopyInformation(raw_las[slot])
        if int(sitk.GetArrayFromImage(mask_img).sum()) == 0:
            raise RuntimeError("HD-BET produced an empty brain mask for %s" % slot)
        if slot == "T1":
            t1_mask_img = mask_img
        log.info("  HD-BET %s done", slot)

    # ---- 4+5. register T1c/T2/FLAIR onto T1, resample, brain-mask -----------
    step(7, "Registration onto T1")
    fixed = bet_images["T1"]
    t1_ref = raw_las["T1"]
    t1_mask = sitk.Cast(t1_mask_img, sitk.sitkFloat32)
    transforms = {"T1": sitk.Transform(3, sitk.sitkIdentity)}

    nnunet_dir = os.path.join(work_dir, "nnunet_input")
    os.makedirs(nnunet_dir, exist_ok=True)
    inputs = {"T1": os.path.join(nnunet_dir, "T1.nii.gz")}
    sitk.WriteImage(sitk.Cast(bet_images["T1"], sitk.sitkFloat32), inputs["T1"])

    for slot in ("T1c", "T2", "FLAIR"):
        T = preprocess.rigid_register(bet_images[slot], fixed)
        transforms[slot] = T
        moved = preprocess.resample_onto_reference(raw_las[slot], t1_ref, transform=T,
                                                   interpolation=sitk.sitkBSpline,
                                                   pixel_type=sitk.sitkFloat32)
        moved = moved * t1_mask
        out = os.path.join(nnunet_dir, "%s.nii.gz" % slot)
        sitk.WriteImage(sitk.Cast(moved, sitk.sitkFloat32), out)
        inputs[slot] = out
        log.info("  %s registered onto T1 grid (T1 brain mask applied)", slot)

    # ---- 6. HD-GLIO inference -----------------------------------------------
    step(8, "Segmentation (HD-GLIO)")
    seg_dir = os.path.join(work_dir, "segmentation")
    os.makedirs(seg_dir, exist_ok=True)
    seg_path = os.path.join(seg_dir, "HDGLIO_seg.nii.gz")
    run_glio.run_hd_glio(inputs["T1"], inputs["T1c"], inputs["T2"], inputs["FLAIR"],
                         seg_path, device=device)
    seg_las = sitk.ReadImage(seg_path, sitk.sitkUInt8)
    seg_arr = sitk.GetArrayFromImage(seg_las)
    labels_present = sorted(int(v) for v in np.unique(seg_arr) if v > 0)
    if not labels_present:
        raise RuntimeError("HD-GLIO returned an empty segmentation")
    log.info("HD-GLIO labels present: %s", labels_present)

    # ---- 7. map back onto the native grid of the target series --------------
    step(9, "Mapping back onto %s" % target_slot)
    target_native, _ = dicom_io.load_series_image(target_series)
    # transforms[slot] maps T1-LAS world -> slot-LAS world (sitk convention),
    # so its inverse maps target world -> T1-LAS world - exactly what
    # reorient_mask_back needs (see preprocess.reorient_mask_back docstring).
    mask_native_img = preprocess.reorient_mask_back(seg_las, target_native,
                                                    transforms[target_slot])
    mask_native = sitk.GetArrayFromImage(mask_native_img)     # [z, y, x] uint8

    sx, sy, sz = target_native.GetSpacing()
    label_masks = {}
    volumes = {}
    for label in labels_present:
        specs = roi_specs.get(label, {"name": "HDGLIO_LABEL%d" % label,
                                      "color": (128, 255, 128),
                                      "interpreted_type": "ORGAN"})
        mask = mask_native == np.uint8(label)
        label_masks[int(label)] = mask
        vol = float(int(mask.sum()) * rtstruct.voxel_volume_ml(sx, sy, sz))
        volumes[specs["name"]] = vol
        log.info("label %d (%s): %d voxels = %.2f mL on %s grid",
                 label, specs["name"], int(mask.sum()), vol, target_slot)

    # ---- 8. RTSTRUCT + previews ---------------------------------------------
    step(10, "Writing RTSTRUCT")
    rt_path = os.path.join(output_dir, "HDGLIO_%s.dcm" % job_id)
    rtstruct.write_rtstruct(label_masks, target_series, rt_path, roi_specs)

    preview_paths = []
    try:
        for label, mask in label_masks.items():
            if not mask.any():
                continue
            p = os.path.join(output_dir, "preview_%s.png"
                             % roi_specs.get(label, {}).get("name", label))
            write_preview(target_native, {label: mask}, target_series, p, roi_specs)
            preview_paths.append(p)
    except Exception as exc:                                  # previews are optional
        log.warning("preview rendering failed: %s", exc)

    import pydicom
    rtds = pydicom.dcmread(rt_path, stop_before_pixels=True)

    result = dict(
        job_id=job_id,
        state="done",
        message="HD-GLIO segmentation finished: %s"
                % ", ".join("%s %.1f mL" % (n, v) for n, v in sorted(volumes.items())),
        finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
        runtime_s=round(time.time() - t_start, 1),
        volumes_ml=volumes,
        rtstruct_file=os.path.join("output", os.path.basename(rt_path)),
        rtstruct_sop_uid=str(rtds.SOPInstanceUID),
        rtstruct_series_uid=str(rtds.SeriesInstanceUID),
        target_series_uid=target_series.series_uid,
        target_series_description=target_series.description,
        frame_of_reference_uid=target_series.frame_of_reference_uid,
        labels_present=labels_present,
        pipeline_version=PIPELINE_VERSION,
    )
    return result


def run_cli(dicom_folder, output_root, config_path=None):
    """CLI entry point: run the pipeline directly on a folder of DICOM series.

    Used for the TestData end-to-end test - same code path as the watcher.
    """
    import shutil

    config = load_config(config_path)

    job_dir = os.path.join(output_root, "cli_job")
    input_dir = os.path.join(job_dir, "input")
    output_dir = os.path.join(job_dir, "output")
    work_dir = os.path.join(job_dir, "work")
    os.makedirs(input_dir, exist_ok=True)
    if not os.listdir(input_dir):
        shutil.copytree(dicom_folder, input_dir, dirs_exist_ok=True)

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)-12s %(levelname)s %(message)s")
    try:
        result = process_job(input_dir, output_dir, work_dir, config, job_meta=None)
        write_status(job_dir, "done", result.get("message", ""), extra=result)
        return result
    except Exception as exc:
        log.exception("job failed")
        write_status(job_dir, "error", str(exc), extra={"traceback": traceback.format_exc()})
        raise