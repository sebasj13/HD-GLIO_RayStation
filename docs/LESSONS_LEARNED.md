# Lessons learned

Notes from building this integration and taking it into clinical-network
operation. Most of these cost a failed run to find; none were visible in
tests that ran from an activated conda environment on a local drive.

## 1. Model and pipeline

**Check the label convention against images, not against memory.** HD-GLIO
writes 1 = non-enhancing T2/FLAIR abnormality, 2 = contrast-enhancing
tumor. An early version had them swapped; the only visible symptom was
implausible volumes (66 ml "enhancing" vs 0.6 ml "non-enhancing" in a
post-resection case). The QA overlay PNGs made it obvious. The per-class
Dice in the model's `postprocessing.json` is a second hint (the large
oedema class scores higher). Keep ROI colours in the config and draw the
previews from it, so preview and RTSTRUCT can never disagree.

**Resample moving images to float.** `ResampleImageFilter` keeps the input
pixel type; MR series are usually `uint16`. Multiplying that by a float
brain mask fails in SimpleITK, and B-spline overshoot would wrap around in
unsigned integers anyway. Pass `sitk.sitkFloat32` as output type.

**nnU-Net v1 inference does not run on Windows as shipped.**
`predict_cases` hands `trainer.preprocess_patient` - a bound method, so the
whole trainer including the CUDA network and a lambda softmax - to a
`multiprocessing.Process`. That only works with `fork`; Windows `spawn`
dies with `Can't pickle <lambda> ... nd_softmax`. For one case at a time
threads cost nothing: `pipeline/hd_glio_predict_win.py` swaps
`Process`/`Queue`/`Pool` in `nnunet.inference.predict` for thread-based
equivalents and then calls the stock `hd_glio_predict` entry point.

**Honour the user's choice explicitly.** The RS side sends slot → series UID
in `job.json`. A case-sensitive key comparison (`slot.upper()` turning
`T1c` into `T1C`) silently discarded that mapping and fell back to
auto-classification. It happened to pick the same series, which is exactly
why it went unnoticed. An incomplete mapping is now an error, and
`tests/test_job_mapping.py` covers it.

## 2. DICOM RTSTRUCT with pydicom

pydicom accepts any CamelCase attribute on a `Dataset` but only writes
known DICOM keywords; unknown ones are dropped silently with a warning.
Two of ours were wrong:

- `RTROIGenerationAlgorithm` → correct keyword `ROIGenerationAlgorithm`
  (Type 2, so the file was non-conformant without it).
- `Label` → `ROIObservationLabel`.

`RTROIInterpretedType` is VR CS: upper case, digits, space, underscore
only, and should be a defined term - `GTV`, not `GTVt`. Treat every
pydicom warning in the log as a bug.

## 3. Running under a process manager on Windows

Our watcher is kept alive by an in-house process manager. It starts
programs as `<python_path> -u <script>` and differs from an interactive
shell in three ways, each of which broke a production run:

1. **No `conda activate`.** `python_path` points at the env's `python.exe`,
   but `PATH` is the manager's, with the *base* env's `Scripts` first.
   `shutil.which("hd-bet")` therefore found the base env's `hd-bet`, a
   different package (`brainles_hd_bet`) with an incompatible API →
   `ImportError`. Resolve tools next to `sys.executable` first
   (`<env>\Scripts\hd-bet.exe`) and only then fall back to `PATH`.
2. **UNC working directory.** The manager uses the script's folder - a UNC
   path - as cwd. Every `shell=True` call then gets cmd.exe's warning
   ("UNC paths are not supported ...", localised, with non-ASCII
   characters) prepended to its output. nnU-Net v2, which HD-BET uses,
   runs `subprocess.getoutput(['hostname'])` at import time and crashed
   decoding that text. The watcher now does
   `os.chdir(tempfile.gettempdir())` at start; all its paths are absolute.
3. **Console encoding.** With `text=True`, `subprocess.run` decodes child
   output with the ANSI code page (cp1252). Tool progress output contains
   bytes cp1252 cannot decode; the reader thread dies, and on a failing
   tool the error text is lost. Always pass
   `encoding="utf-8", errors="replace"`.

**Test under the real launch conditions.** Reproduce them exactly: the
env's `python.exe` started directly, UNC cwd, base env first on `PATH`,
the manager's environment variables (here `PYTHONIOENCODING=utf-8`). Start
such a test via a *different* path to the script than the managed
instance uses - stopping the managed program also killed our test
process, because the manager matched it by script path.

A heartbeat file rewritten every 60 s (by a daemon thread, so a long job
does not stop it) is both the input for the site's monitoring and the
"watcher alive" indicator in the RayStation GUI.

## 4. SMB shares

**Do intermediate work locally.** With the job folder on the share,
reading ~260 DICOM files several times and writing the NIfTIs took 262 s
of a 9.6 min run; locally the same steps take ~8 s. The watcher now copies
the input once to `%TEMP%\glio_work\<job>_<pid>` (the copy itself is
~80 s - per-file latency, parallel copies would help) and deletes that
folder after every job, success or error. Leftovers of a crashed watcher
are removed at the next start once older than 12 h, so two watcher
instances never delete each other's live work.

**Atomic status files need retries on the writer and tolerance on the
reader.** `os.replace(tmp, status.json)` fails with WinError 32 while the
RayStation side has the file open for polling; retry a few times, and never
let a failed *progress* write kill the job (`critical=False`). The reader in
turn sees the file briefly missing or half-written; it must keep its last
state instead of falling back to "waiting" - otherwise the progress bar
jumps back mid-run.

**Claims via `O_CREAT | O_EXCL`** are atomic on SMB and are enough to stop
two watchers from processing the same job.

## 5. RayStation scripting API (2024B, CPython 3.11)

- `set_progress(message, percentage)` maps to
  `SetProgress(String, Int32, String)`: a **float percentage raises**
  `TypeError: No method matches given arguments`. Pass `int(round(p))`, and
  wrap the call - a status-bar update must never end a run.
- `exam.GetStoredDicomTagValueForVerification(Group=, Element=)` returns
  `{"<tag name>": value}`, e.g. `{"Contrast/Bolus Agent": "Gadovist"}`.
  A **present-but-empty Type-2 tag returns `None` as value** (confirmed by
  raw dump); a missing tag raises. `str(None)` is `"None"` - truthy - which
  made every MR look contrast-enhanced. Normalise `None`/empty first.
- Contrast is not reliably tagged; the series name (`..._KM`, `..._Gd`)
  is a useful second signal.
- A case often holds several MR sessions plus a planning MR. Propose
  T1c/T2/FLAIR from the **same Frame of Reference as the chosen T1**;
  otherwise the largest post-contrast T1 is typically the planning MR.
- `patient.ImportDataFromPath(...)` requires a saved patient: call
  `patient.Save()` right before it. Re-importing an existing series raises
  "already imported"; treat that as a skip.
- RayStation API calls must stay on the script's main thread. The GUI
  (CustomTkinter) drives the work as a generator via `after()`, which keeps
  the window responsive without threads. Once the job is handed over, the
  window closes and the remaining steps run headless with the status bar
  as the only display - users found a window that merely mirrors the
  status bar redundant.
- Scripts usually run from the RayStation script database, where
  `__file__` does not point at the repository. Keep site constants inline
  in the RS script instead of reading `config.json`.

## 6. Testing strategy that worked

- **Geometry self-test on synthetic oblique data** (no weights): DICOM →
  NIfTI → reorientation → mask back-mapping → RTSTRUCT, with centroid and
  planarity asserts.
- **A mock of the RayStation API with real semantics**: tag dicts keyed by
  name, `None` for empty tags, an exception for missing ones, an
  int-only `set_progress`, a case with a second session in another Frame of
  Reference, and a fake watcher that briefly leaves `status.json`
  unreadable. It drives the real GUI end to end and asserts export order,
  monotonic progress, window closed after handoff, import, and cleanup.
  Every RS-side bug we hit in production became a behaviour of the mock.
- **A single-job watcher run (`--once`) under the production launch
  conditions** before every release that touches the watcher.
