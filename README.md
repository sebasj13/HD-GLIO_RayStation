# HD-GLIO Auto-Contour for RayStation

Automated glioma segmentation with [HD-GLIO](https://github.com/CCI-Bonn/HD-GLIO)
wired end-to-end into RayStation: pick four MR series in a small GUI, a GPU
workstation segments them, and the result comes back into the case as an
RTSTRUCT - all from a single RayStation script.

> [!WARNING]
> **Research use only - not a medical device.** HD-GLIO and this integration
> are research software without regulatory clearance. Every contour is a
> proposal that must be reviewed and edited by a qualified clinician before
> any clinical use. You are responsible for validating the pipeline at your
> site and for the data protection of the patient data it moves.

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

Segmentation labels: **1 = non-enhancing T2/FLAIR signal abnormality**
(`HDGLIO_NET`), **2 = contrast-enhancing tumor** (`HDGLIO_ET`).

Tested with RayStation 2024B (CPython 3.11 scripting), Windows 11,
HD-GLIO v2 weights, HD-BET 2.x, one NVIDIA GPU.

Lessons learned while building and deploying this (Windows, SMB shares,
process managers, RayStation scripting API quirks):
[docs/LESSONS_LEARNED.md](docs/LESSONS_LEARNED.md).

## Getting the code

HD-GLIO is included as a git submodule (`third_party/HD-GLIO`, pinned to a
known-good commit of [CCI-Bonn/HD-GLIO](https://github.com/CCI-Bonn/HD-GLIO)),
so clone recursively:

```powershell
git clone --recursive https://github.com/sebasj13/HD-GLIO_RayStation.git
# already cloned without --recursive:
git submodule update --init --recursive
```

## Components

| File | Purpose |
|---|---|
| `config.example.json` | Watcher config template: job root, heartbeat, GPU device, ROI names/colors, target series |
| `glio_watcher.py` | **The watcher** - polls the job folder, runs the pipeline, writes a heartbeat |
| `pipeline/` | DICOM↔NIfTI, preprocessing, HD-BET/HD-GLIO calls, RTSTRUCT writer |
| `pipeline/hd_glio_predict_win.py` | `hd_glio_predict` with threads instead of worker processes (nnU-Net v1 can't pickle its trainer on Windows) |
| `raystation/HD-GLIO_AutoContour.py` | **The RayStation script** - series selection → export → progress → import |
| `third_party/HD-GLIO` | HD-GLIO (git submodule), installed into the `glio` env by the setup script |
| `setup_glio_env.ps1` | Creates the `glio` conda environment |
| `start_watcher.ps1` | Starts the watcher by hand; extra arguments pass through |
| `run_test.ps1` | End-to-end CLI test on your own `TestData\` folder (no RayStation) |
| `tests/test_geometry.py` | Self-test: DICOM↔NIfTI geometry, mask back-mapping, RTSTRUCT (synthetic data, no weights) |
| `tests/test_job_mapping.py` | Self-test: the series choice from job.json is honoured |

## Setup

### 1. GPU workstation (watcher)

Requirements: Windows 10/11 with an NVIDIA GPU (driver supporting CUDA 12.4),
Miniconda/Anaconda, git.

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_glio_env.ps1     # ~10 min, once
copy config.example.json config.json                               # then adapt, see below
```

The setup script installs PyTorch (CUDA 12.4), HD-GLIO from the submodule
(which pulls nnU-Net v1), HD-BET 2.x and the DICOM tooling into a conda env
called `glio`.

### 2. Model weights

The packages download their weights on first use. Behind a TLS-intercepting
proxy that fails (`SSLError` via certifi, `CRYPT_E_NO_REVOCATION_CHECK` via
SChannel); in that case download once with `curl.exe --ssl-no-revoke` and
extract:

| Weights | Zip (zenodo) | Extract into |
|---|---|---|
| HD-GLIO v2 | `https://zenodo.org/record/4014850/files/hd_glio_v2_params.zip?download=1` | `%USERPROFILE%\hd_glio_params\` (contains `fold_0\`, `plans.pkl`, `version`=2) |
| HD-BET 2.0 | `https://zenodo.org/records/14445620/files/release_v1.5.0.zip?download=1` | `%USERPROFILE%\hd-bet_params\release_2.0.0\` (contains `fold_all\checkpoint_final.pth`) |

### 3. Site configuration

Adapt these before the first run - they are the only site-specific values:

| Where | Value | Meaning |
|---|---|---|
| `config.json` → `job_root` | e.g. `\\fileserver\share\transfer\hdglio` | Shared folder both RayStation and the GPU workstation can write |
| `config.json` → `heartbeat_file` | `null` → `<job_root>\heartbeat_GlioAutoContour.txt` | Rewritten every 60 s; the RS GUI shows the watcher red when it is older than 3 min. Point your monitoring at it. |
| `config.json` → `gpu_device`, `roi_specs`, `target_series_slot` | `0`, names/colors, `T1c` | ROI names must not clash with existing ROIs in your cases |
| `raystation/HD-GLIO_AutoContour.py` → `TRANSFER_DIR` | same folder as `job_root` | RayStation scripts usually run from the script database, so this is a constant, not read from `config.json` |
| `raystation/HD-GLIO_AutoContour.py` → `HEARTBEAT_FILE`, `LOG_DIR` | default: heartbeat in `TRANSFER_DIR`, logs in the local `%TEMP%\hdglio_logs` | Change if you set `heartbeat_file`, or want the script logs somewhere central. Log file names contain the patient ID. |

`config.json` is git-ignored, so pulling updates never overwrites it; without
it the pipeline falls back to `config.example.json`.

### 4. Run the watcher

Run the watcher permanently, by hand or under any process manager / service
wrapper:

```
C:\path\to\miniconda3\envs\glio\python.exe -u glio_watcher.py
```

Without arguments the job folder comes from `config.json` (`job_root`) and
the GPU from `gpu_device`; `--watch_dir` / `--gpu` override. The watcher
does **not** need an activated conda env (tools are resolved next to
`sys.executable`) and copes with a UNC working directory - see
[docs/LESSONS_LEARNED.md](docs/LESSONS_LEARNED.md) for why that matters.

By hand for tests: `powershell -ExecutionPolicy Bypass -File .\start_watcher.ps1 --once`.

### 5. RayStation

- The RayStation scripting environment needs `customtkinter` and `pydicom`
  (install them into the RS CPython environment the way your site manages
  script packages).
- Add `raystation\HD-GLIO_AutoContour.py` to the script database (or run it
  from disk), with `TRANSFER_DIR` set as above.

## RayStation usage

1. Open the case with the 4 MR series (T1 and T1c in the same Frame of Reference).
2. Scripting tab → `HD-GLIO_AutoContour`.
3. The window proposes a series per slot and shows whether the watcher is
   alive; correct the dropdowns if needed.
4. **Start**: the series are exported, then the window closes. Watcher
   progress, the RTSTRUCT import and finally one line with the ROI volumes
   appear in RayStation's status bar; the script then ends.

## Job folder contract

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

Intermediate work (local copy of the input, NIfTI, BET masks, segmentation)
runs in `%TEMP%\glio_work\<job_id>_<pid>` on the GPU workstation, not on the
share, and is deleted after every job (`--keep_work` keeps it for
debugging). After a successful import the RS script deletes the whole job
folder, so no patient data stays on the share. Failed jobs stay for
debugging; delete them by hand.

## Pipeline (watcher side, per job)

1. **DICOM → NIfTI** with geometry straight from IOP/IPP (oblique-safe).
2. **Reorient to FSL/MNI152-standard orientation** (`fslreorient2std`
   equivalent). HD-GLIO was trained on FSL-preprocessed data.
3. **HD-BET brain extraction** on all four sequences.
4. **Rigid 6-DOF mutual-information registration** (SimpleITK) of
   T1c/T2/FLAIR onto T1; spline resampling onto the T1 grid, everything
   outside the T1 brain mask set to 0 (HD-GLIO expects that).
5. **`hd_glio_predict`** (subprocess, GPU).
6. **Map the segmentation back** onto the native grid of the target series
   (T1c by default) via the inverse registration, and write an **RTSTRUCT**
   that references the original SOP instances and Frame of Reference.

Runtime on one GPU: about 4-5 min per case (most of it HD-BET).

## Sequences / slots

| Slot | Used by HD-GLIO as | Auto-mapping rule |
|---|---|---|
| T1   | pre-contrast T1      | description contains `t1`, no contrast agent |
| T1c  | post-contrast T1     | description contains `t1`, ContrastBolusAgent set or `_KM`/`_Gd` in the name |
| T2   | T2                   | `t2` without `dark-fluid`/`flair` |
| FLAIR| T2/FLAIR abnormality | `flair` or (`t2` + `dark-fluid`) |

The GUI prefers series in the same Frame of Reference as the chosen T1, so
planning MRs or second sessions in the case are not proposed. The rules
were written for Siemens series descriptions; if your naming differs, the
user simply corrects the dropdowns, or adapt `auto_map()` in the RS script
and `classify_series()` in `pipeline/dicom_io.py`.

## Tests

```powershell
conda run -n glio python tests/test_geometry.py      # synthetic data, no weights, ~2 min
conda run -n glio python tests/test_job_mapping.py
powershell -ExecutionPolicy Bypass -File .\run_test.ps1   # full pipeline on .\TestData (bring your own)
```

The two self-tests need only `numpy SimpleITK pydicom scikit-image` and run
in CI on every push. `TestData\` is not part of this repository - put an
anonymised four-series MR case there. Output lands in
`test_output\cli_job\output\`; check the `preview_*.png` overlays.

## Updating HD-GLIO

```powershell
git -C third_party/HD-GLIO fetch
git -C third_party/HD-GLIO checkout <commit-or-tag>
git add third_party/HD-GLIO; git commit -m "Bump HD-GLIO"
conda run -n glio python -m pip install .\third_party\HD-GLIO
```

## License

Apache License 2.0, see [LICENSE](LICENSE). HD-GLIO, HD-BET and nnU-Net are
separate projects under their own (Apache-2.0) licenses; their model weights
are downloaded from zenodo under the terms given there.

## Credits

If you use this in research, please cite HD-GLIO:

- HD-GLIO: Kickingereder P, Isensee F, et al. *Automated quantitative tumour
  response assessment of MRI in neuro-oncology with artificial neural
  networks.* Lancet Oncol. 2019;20(5):728-740.
  https://github.com/CCI-Bonn/HD-GLIO (Apache-2.0)
- HD-BET: Isensee F, et al. *Automated brain extraction of multisequence MRI
  using artificial neural networks.* Hum Brain Mapp. 2019;40(17):4952-4964.
  https://github.com/MIC-DKFZ/HD-BET (Apache-2.0)
- nnU-Net: Isensee F, et al. *nnU-Net: a self-configuring method for deep
  learning-based biomedical image segmentation.* Nat Methods. 2021;18:203-211.
