# HD-GLIO Auto-Contour for RayStation

[![tests](https://github.com/sebasj13/HD-GLIO_RayStation/actions/workflows/tests.yml/badge.svg)](https://github.com/sebasj13/HD-GLIO_RayStation/actions/workflows/tests.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Glioma contours in RayStation from a single script. The user picks four MR
series (T1, T1c, T2, FLAIR) in a small window and presses Start. A GPU
workstation then runs [HD-GLIO](https://github.com/CCI-Bonn/HD-GLIO) on
them, and a few minutes later the result appears in the case as a structure
set - the enhancing tumour and the T2/FLAIR abnormality as two ROIs on the
contrast-enhanced T1.

HD-GLIO itself expects preprocessed NIfTI files: skull-stripped, co-registered
and in FSL standard orientation. This repository closes the gap between that
and a planning system. It handles the DICOM export, the preprocessing, the
inference and the way back into RayStation, including the geometry on
oblique acquisitions and an RTSTRUCT that references the original images.

> [!WARNING]
> **Research use only - not a medical device.** HD-GLIO and this integration
> are research software without regulatory clearance. Every contour is a
> proposal that must be reviewed and edited by a qualified clinician before
> any clinical use. You are responsible for validating the pipeline at your
> site and for the protection of the patient data it moves.

## How it works

```
RayStation (TPS client)          GPU workstation                    RayStation
┌──────────────────────┐   ┌───────────────────────────┐   ┌──────────────────────┐
│ HD-GLIO_AutoContour  │   │ glio_watcher.py           │   │ HD-GLIO_AutoContour  │
│  - series selection  │──▶│  - DICOM → NIfTI          │──▶│  - RTSTRUCT import   │
│  - DICOM export      │   │  - HD-BET skull-strip     │   │  - ROI volumes in    │
│  - job.json          │   │  - rigid co-registration  │   │    the status bar    │
└──────────────────────┘   │  - HD-GLIO (nnU-Net)      │   └──────────────────────┘
         shared folder      │  - mask → RTSTRUCT        │
                            └───────────────────────────┘
```

The two sides only share a folder. The RayStation script exports the four
series into a new job folder and writes `job.json` last. A watcher on the GPU
workstation picks the job up, reports its progress in `status.json` and
writes the RTSTRUCT next to it. RayStation imports the structure set, shows
the ROI volumes in the status bar and deletes the job folder, so no patient
data stays on the share.

| Label | ROI | Meaning |
|---|---|---|
| 1 | `HDGLIO_NET` | non-enhancing T2/FLAIR signal abnormality |
| 2 | `HDGLIO_ET` | contrast-enhancing tumour |

A case takes about 3.5-5 min on one GPU, most of it HD-BET. Tested with
RayStation 2024B (CPython 3.11 scripting), Windows 11, HD-GLIO v2 weights,
HD-BET 2.x and one NVIDIA GPU.

Building this took more than wiring three tools together: Windows process
managers, SMB shares and a few RayStation API quirks each broke a production
run once. The notes in [docs/LESSONS_LEARNED.md](docs/LESSONS_LEARNED.md) may
save you the same afternoons.

## Quick start

```powershell
git clone --recursive https://github.com/sebasj13/HD-GLIO_RayStation.git
cd HD-GLIO_RayStation
powershell -ExecutionPolicy Bypass -File .\setup_glio_env.ps1   # conda env "glio", ~10 min
copy config.example.json config.json                             # set job_root
powershell -ExecutionPolicy Bypass -File .\start_watcher.ps1 --once
```

Then set `TRANSFER_DIR` in `raystation\HD-GLIO_AutoContour.py` to the same
folder and run the script from RayStation. The sections below go through
each step.

## Setup

### 1. GPU workstation

You need Windows 10/11 with an NVIDIA GPU (driver supporting CUDA 12.4),
Miniconda or Anaconda, and git.

HD-GLIO is included as a git submodule (`third_party/HD-GLIO`, pinned to a
known-good commit of [CCI-Bonn/HD-GLIO](https://github.com/CCI-Bonn/HD-GLIO)).
If you cloned without `--recursive`, run `git submodule update --init
--recursive` - the setup script also does this if the folder is empty.

`setup_glio_env.ps1` creates the conda env `glio` with PyTorch (CUDA 12.4),
HD-GLIO from the submodule (which pulls nnU-Net v1), HD-BET 2.x and the
DICOM tooling. The install order matters, and the comments in the script
explain why.

### 2. Model weights

Both packages download their weights on first use. Behind a TLS-intercepting
proxy this fails (`SSLError` via certifi, `CRYPT_E_NO_REVOCATION_CHECK` via
SChannel). In that case download them once with `curl.exe --ssl-no-revoke`
and extract:

| Weights | Zip (zenodo) | Extract into |
|---|---|---|
| HD-GLIO v2 | `https://zenodo.org/record/4014850/files/hd_glio_v2_params.zip?download=1` | `%USERPROFILE%\hd_glio_params\` (contains `fold_0\`, `plans.pkl`, `version`=2) |
| HD-BET 2.0 | `https://zenodo.org/records/14445620/files/release_v1.5.0.zip?download=1` | `%USERPROFILE%\hd-bet_params\release_2.0.0\` (contains `fold_all\checkpoint_final.pth`) |

### 3. Site configuration

Copy `config.example.json` to `config.json` and adapt it. `config.json` is
git-ignored, so pulling updates never overwrites it (without it, the
pipeline falls back to the example). These are the only site-specific
values:

| Where | Value | Meaning |
|---|---|---|
| `config.json` → `job_root` | e.g. `\\fileserver\share\transfer\hdglio` | Shared folder both RayStation and the GPU workstation can write |
| `config.json` → `heartbeat_file` | `null` → `<job_root>\heartbeat_GlioAutoContour.txt` | Rewritten every 60 s. The RayStation window shows the watcher red when it is older than 3 min. Point your monitoring at it. |
| `config.json` → `gpu_device`, `roi_specs`, `target_series_slot` | `0`, names/colours, `T1c` | ROI names must not clash with existing ROIs in your cases |
| `HD-GLIO_AutoContour.py` → `TRANSFER_DIR` | same folder as `job_root` | RayStation scripts usually run from the script database, so this is a constant rather than a value read from `config.json` |
| `HD-GLIO_AutoContour.py` → `HEARTBEAT_FILE` | `TRANSFER_DIR\heartbeat_GlioAutoContour.txt` | Change it only if you set `heartbeat_file` above |
| `HD-GLIO_AutoContour.py` → `LOG_DIR` | `%TEMP%\hdglio_logs` on the RayStation host | One log per run. **The file names contain the patient ID**, so put this on a location your data-protection rules allow, or somewhere central if you want all logs in one place. |

### 4. Watcher

Run the watcher permanently, by hand or under any process manager or
service wrapper:

```
C:\path\to\miniconda3\envs\glio\python.exe -u glio_watcher.py
```

Without arguments the job folder comes from `job_root` and the GPU from
`gpu_device`; `--watch_dir` and `--gpu` override them. The watcher does
**not** need an activated conda env (tools are resolved next to
`sys.executable`) and copes with a UNC working directory. For a single test
job use `start_watcher.ps1 --once`.

### 5. RayStation

The RayStation scripting environment needs `customtkinter` and `pydicom` -
install them the way your site manages script packages. Then add
`raystation\HD-GLIO_AutoContour.py` to the script database (or run it from
disk) with `TRANSFER_DIR` set.

## Usage

1. Open the case with the four MR series. T1 and T1c must share a Frame of
   Reference.
2. Run `HD-GLIO_AutoContour` from the Scripting tab.
3. The window proposes a series for each slot and shows whether the watcher
   is alive. Correct the dropdowns if needed.
4. Press **Start**. The series are exported and the window closes. Progress,
   the import and finally one line with the ROI volumes appear in the
   RayStation status bar.

The proposal follows the series descriptions:

| Slot | Used by HD-GLIO as | Auto-mapping rule |
|---|---|---|
| T1   | pre-contrast T1      | description contains `t1`, no contrast agent |
| T1c  | post-contrast T1     | description contains `t1`, ContrastBolusAgent set or `_KM`/`_Gd` in the name |
| T2   | T2                   | `t2` without `dark-fluid`/`flair` |
| FLAIR| T2/FLAIR abnormality | `flair` or (`t2` + `dark-fluid`) |

T1c, T2 and FLAIR are preferred from the same Frame of Reference as the chosen
T1, so planning MRs or second sessions in the case are not proposed. The
rules were written for Siemens series descriptions. If your naming differs,
the user simply corrects the dropdowns - or you adapt `auto_map()` in the
RayStation script and `classify_series()` in `pipeline/dicom_io.py`.

## Pipeline

Per job, the watcher runs:

1. **DICOM → NIfTI** with the geometry taken straight from IOP/IPP, so
   oblique acquisitions stay correct.
2. **Reorientation to FSL/MNI152 standard orientation** (the `fslreorient2std`
   equivalent), since HD-GLIO was trained on FSL-preprocessed data.
3. **HD-BET brain extraction** on all four sequences.
4. **Rigid 6-DOF mutual-information registration** (SimpleITK) of
   T1c/T2/FLAIR onto T1, spline resampling onto the T1 grid, and everything
   outside the T1 brain mask set to 0.
5. **`hd_glio_predict`** on the GPU.
6. **Back-mapping** of the segmentation onto the native grid of the target
   series (T1c by default) through the inverse registration, and an
   **RTSTRUCT** that references the original SOP instances and Frame of
   Reference.

Intermediate files (local input copy, NIfTI, brain masks, segmentation) live
in `%TEMP%\glio_work\<job_id>_<pid>` on the GPU workstation, not on the
share, and are deleted after every job (`--keep_work` keeps them for
debugging). Failed jobs stay on the share for debugging; delete them by hand.

### Job folder contract

```
<job_root>/<PatientID>_<ts>/
├── input/
│   ├── job.json            written LAST by the RS script (atomic rename)
│   ├── claim.lock          watcher claim (O_EXCL, stale after 12 h)
│   └── series_<SLOT>/*.dcm exported DICOM (SLOT ∈ T1, T1c, T2, FLAIR)
├── output/
│   ├── status.json         poll target: processing (+ step/total/message) | done | error
│   ├── HDGLIO_<job_id>.dcm RTSTRUCT on the target series' native grid
│   └── preview_*.png       QA overlays (red = ET, blue = NET)
└── log.txt                 per-job pipeline log
```

## Repository layout

| Path | Purpose |
|---|---|
| `raystation/HD-GLIO_AutoContour.py` | The RayStation script: series selection → export → progress → import |
| `glio_watcher.py` | The watcher: polls the job folder, runs the pipeline, writes the heartbeat |
| `pipeline/` | DICOM↔NIfTI, preprocessing, HD-BET/HD-GLIO calls, RTSTRUCT writer |
| `pipeline/hd_glio_predict_win.py` | `hd_glio_predict` with threads instead of worker processes (nnU-Net v1 cannot pickle its trainer on Windows) |
| `third_party/HD-GLIO` | HD-GLIO as a git submodule |
| `config.example.json` | Watcher config template |
| `setup_glio_env.ps1`, `start_watcher.ps1`, `run_test.ps1` | Environment setup, manual watcher start, end-to-end CLI test |
| `tests/` | Self-tests on synthetic data (no GPU, no weights) |
| `docs/LESSONS_LEARNED.md` | What broke in deployment, and why |

## Tests

```powershell
conda run -n glio python tests/test_geometry.py      # synthetic oblique DICOM, no weights
conda run -n glio python tests/test_job_mapping.py
powershell -ExecutionPolicy Bypass -File .\run_test.ps1   # full pipeline on .\TestData
```

The two self-tests need only `numpy SimpleITK pydicom scikit-image` and run
in CI on Ubuntu and Windows with every push. For the end-to-end test, put an
anonymised four-series MR case into `TestData\` (not part of the repository)
and check the `preview_*.png` overlays in `test_output\cli_job\output\`.

## Updating HD-GLIO

```powershell
git -C third_party/HD-GLIO fetch
git -C third_party/HD-GLIO checkout <commit-or-tag>
git add third_party/HD-GLIO
git commit -m "Bump HD-GLIO"
conda run -n glio python -m pip install .\third_party\HD-GLIO
```

## License

Apache License 2.0, see [LICENSE](LICENSE). HD-GLIO, HD-BET and nnU-Net are
separate projects under their own Apache-2.0 licenses. Their model weights
are downloaded from zenodo under the terms given there.

## Citation

If you use this in research, please cite the underlying methods:

- **HD-GLIO**: Kickingereder P, Isensee F, et al. Automated quantitative
  tumour response assessment of MRI in neuro-oncology with artificial neural
  networks. *Lancet Oncol.* 2019;20(5):728-740.
  https://github.com/CCI-Bonn/HD-GLIO
- **HD-BET**: Isensee F, et al. Automated brain extraction of multisequence
  MRI using artificial neural networks. *Hum Brain Mapp.*
  2019;40(17):4952-4964. https://github.com/MIC-DKFZ/HD-BET
- **nnU-Net**: Isensee F, et al. nnU-Net: a self-configuring method for deep
  learning-based biomedical image segmentation. *Nat Methods.*
  2021;18:203-211.
