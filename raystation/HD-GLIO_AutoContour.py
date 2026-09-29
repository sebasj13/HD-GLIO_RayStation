r"""
HD-GLIO Auto-Contour - RayStation side.

Run from the Scripting tab with a brain tumour case open.

Flow:
  1. Window: list the case's MR series and propose one for each of the four
     HD-GLIO slots (T1, T1c, T2, FLAIR); the user corrects via dropdowns.
  2. "Start": export the four series with ScriptableDicomExport into a new
     job folder on the transfer share, then write input/job.json (only then
     does the watcher pick the job up). The window closes afterwards.
  3. No window, RS status bar only: poll output/status.json - the GPU
     watcher (glio_watcher.py) reports step/total/message.
  4. Import the RTSTRUCT with ImportDataFromPath, show the volumes in the
     status bar, delete the job folder (no patient data stays on the share),
     end of script. Errors from step 3 on are ordinary script errors.

Modifies the patient: patient.Save() + ImportDataFromPath (structure set).
Everything up to and including the export is read-only.

No threading: RS calls must run on the main thread. The work is a
generator: until the handoff Tk drives it via after() (the window stays
responsive), afterwards finish() drives it with time.sleep.

Log: stdout is also written to LOG_DIR\glio_<patient>_<ts>.log.
"""

import datetime
import json
import os
import re
import shutil
import sys
import tempfile
import time
import traceback

try:
    from raystation import get_current, set_progress
except ImportError:
    from connect import get_current, set_progress

import customtkinter as ctk

# ---------------------------------------------------------------- Config ---
# site-specific (see README, Site configuration). RS scripts usually run from
# the script database, so these are constants instead of values read from
# config.json. TRANSFER_DIR must match job_root in the watcher's config.json.
TRANSFER_DIR = r"\\fileserver\share\transfer\hdglio"
# the watcher's heartbeat_file; its default is <job_root>\heartbeat_GlioAutoContour.txt
HEARTBEAT_FILE = os.path.join(TRANSFER_DIR, "heartbeat_GlioAutoContour.txt")
# per-run script log (file name contains the patient ID - keep it on a protected location)
LOG_DIR = os.path.join(tempfile.gettempdir(), "hdglio_logs")
HEARTBEAT_STALE_S = 180     # the watcher writes every 60 s
POLL_MS = 5000
HANDOFF = -1                # generator signal: job handed over, close the window
TIMEOUT_MIN = 45            # typical runtime 5-10 min
ROI_PREFIX = "HDGLIO_"      # HDGLIO_ET (contrast-enhancing), HDGLIO_NET (T2/FLAIR)
# ---------------------------------------------------------------------------

SLOTS = ("T1", "T1c", "T2", "FLAIR")
SLOT_LABELS = {"T1": "T1 native", "T1c": "T1 + contrast", "T2": "T2", "FLAIR": "FLAIR"}
NONE_CHOICE = "- please select -"

COLOR_OK = "#49D17D"
COLOR_BAD = "#FF5C77"
COLOR_MUTED = "#AAB6C8"


class _Tee(object):
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass


def safe_name(text):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(text)) or "unknown"


def stored_tag(exam, group, element):
    """Stored DICOM attribute of an examination (None if missing or empty).

    RS returns {"<tag name>": value}, e.g. {"Series Description": "..."}. Type-2
    tags like ContrastBolusAgent are often present but empty - the value is then
    None or "", and str(None) would be "None", i.e. falsely "contrast".
    """
    try:
        d = exam.GetStoredDicomTagValueForVerification(Group=group, Element=element)
        v = list(d.values())[0] if d else None
    except Exception:
        return None
    v = "" if v is None else str(v).strip()
    return None if v.lower() in ("", "none") else v


def has_contrast(e):
    # tag or naming convention (..._KM, ..._Gd; KM = German "Kontrastmittel")
    return bool(e["contrast"]) or bool(re.search(r"(^|[_\s-])(km|gd|kontrast)([_\s-]|$)",
                                                  e["description"].lower()))


def enumerate_mr_series(case):
    """One dict per MR examination with the tags the slot mapping needs."""
    out = []
    for exam in case.Examinations:
        try:
            stack = exam.Series[0].ImageStack
        except Exception:
            continue
        if not stack:
            continue
        if str(stored_tag(exam, 0x0008, 0x0060) or "").upper() != "MR":
            continue
        for_uid = stored_tag(exam, 0x0020, 0x0052) or ""
        if not for_uid:
            try:
                for_uid = str(exam.EquipmentInfo.FrameOfReference)
            except Exception:
                pass
        out.append(dict(
            exam=exam,
            name=str(exam.Name),
            series_uid=stored_tag(exam, 0x0020, 0x000E) or "",
            series_number=stored_tag(exam, 0x0020, 0x0011) or "",
            study_uid=stored_tag(exam, 0x0020, 0x000D) or "",
            description=stored_tag(exam, 0x0008, 0x103E) or "",
            contrast=stored_tag(exam, 0x0018, 0x0010) or "",
            for_uid=for_uid,
            n_slices=int(len(stack.SlicePositions)),
            pixel_size=float(stack.PixelSize.x),
        ))
    out.sort(key=lambda e: int(e["series_number"]) if e["series_number"].isdigit() else 0)
    return out


def auto_map(series):
    """Proposal {slot: entry} using the same rules as the pipeline; may be incomplete."""
    def desc(e):
        return e["description"].lower()

    def best(cands):
        return sorted(cands, key=lambda e: (-e["n_slices"], -e["pixel_size"]))[0] if cands else None

    t2_like = [e for e in series if "t2" in desc(e)]
    flair = [e for e in t2_like if "flair" in desc(e) or "dark-fluid" in desc(e)]
    flair += [e for e in series if "flair" in desc(e) and e not in flair]
    t1_like = [e for e in series if "t1" in desc(e)]
    t1 = best([e for e in t1_like if not has_contrast(e)])

    def same_session(cands):
        # several sessions / a planning MR in the case: prefer the T1's Frame of Reference
        if t1 is None:
            return cands
        return [e for e in cands if e["for_uid"] == t1["for_uid"]] or cands

    mapping = {
        "T1": t1,
        "T1c": best(same_session([e for e in t1_like if has_contrast(e)])),
        "T2": best(same_session([e for e in t2_like if e not in flair])),
        "FLAIR": best(same_session(flair)),
    }
    return {k: v for k, v in mapping.items() if v is not None}


def choice_label(e):
    return "%s  |  Series %s  |  %s  |  %d sl.%s" % (
        e["name"], e["series_number"] or "?", e["description"] or "<no description>",
        e["n_slices"], "  |  contrast" if has_contrast(e) else "")


def heartbeat_age_s():
    try:
        return time.time() - os.path.getmtime(HEARTBEAT_FILE)
    except OSError:
        return None


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:          # not there yet, or being replaced right now
        return None


class GlioApp(object):
    def __init__(self, patient, case, series, log_path):
        self.patient = patient
        self.case = case
        self.series = series
        self.log_path = log_path
        self.by_label = {choice_label(e): e for e in series}
        self._gen = None
        self.running = False
        self.handed_off = False
        self.job_dir = None

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        self.app = ctk.CTk()
        self.app.title("HD-GLIO Auto-Contour")
        self.app.attributes("-topmost", True)
        self.app.protocol("WM_DELETE_WINDOW", self._close)

        ctk.CTkLabel(self.app, text="HD-GLIO Auto-Contour (glioma)",
                     font=ctk.CTkFont(size=18, weight="bold")).pack(padx=16, pady=(14, 2), anchor="w")
        ctk.CTkLabel(self.app, text="Patient %s  |  Case %s" % (
            patient.PatientID, case.CaseName), text_color=COLOR_MUTED).pack(padx=16, anchor="w")
        self.watcher_lbl = ctk.CTkLabel(self.app, text="")
        self.watcher_lbl.pack(padx=16, pady=(2, 8), anchor="w")

        # reserve the footer first so it is never cut off
        footer = ctk.CTkFrame(self.app, fg_color="transparent")
        footer.pack(side="bottom", fill="x", padx=16, pady=(4, 14))
        self.close_btn = ctk.CTkButton(footer, text="Close", width=110, command=self._close,
                                       fg_color="#7A2230", hover_color="#963043")
        self.close_btn.pack(side="right")
        self.start_btn = ctk.CTkButton(footer, text="Start", width=140, command=self._start)
        self.start_btn.pack(side="right", padx=8)

        self.status_lbl = ctk.CTkLabel(self.app, text="Check the series, then Start.",
                                       wraplength=640, justify="left", anchor="w")
        self.status_lbl.pack(side="bottom", fill="x", padx=16)
        self.bar = ctk.CTkProgressBar(self.app, height=14)
        self.bar.set(0)
        self.bar.pack(side="bottom", fill="x", padx=16, pady=(6, 4))

        grid = ctk.CTkFrame(self.app)
        grid.pack(side="top", fill="x", padx=16, pady=4)
        values = [NONE_CHOICE] + list(self.by_label)
        proposal = auto_map(series)
        self.menus = {}
        for r, slot in enumerate(SLOTS):
            ctk.CTkLabel(grid, text=SLOT_LABELS[slot], width=80, anchor="w").grid(
                row=r, column=0, padx=(10, 6), pady=5, sticky="w")
            m = ctk.CTkOptionMenu(grid, values=values, width=600, dynamic_resizing=False)
            m.set(choice_label(proposal[slot]) if slot in proposal else NONE_CHOICE)
            m.grid(row=r, column=1, padx=(0, 10), pady=5, sticky="w")
            self.menus[slot] = m
        ctk.CTkLabel(self.app, text="Contours (%sET, %sNET) are created on the T1 + contrast series. "
                     "The window closes after the export; progress is shown in the "
                     "RS status bar (approx. 5-10 min)." % (ROI_PREFIX, ROI_PREFIX),
                     text_color=COLOR_MUTED).pack(padx=16, pady=(2, 4), anchor="w")

        self._refresh_watcher()

    # ------------------------------------------------------------- UI ---
    def _refresh_watcher(self):
        age = heartbeat_age_s()
        if age is not None and age < HEARTBEAT_STALE_S:
            self.watcher_lbl.configure(text="Watcher active (heartbeat %d s ago)" % age,
                                       text_color=COLOR_OK)
        else:
            self.watcher_lbl.configure(
                text="Watcher not reachable (%s) - the job will wait until it runs."
                % ("no heartbeat" if age is None else "last heartbeat %d min ago" % (age // 60)),
                text_color=COLOR_BAD)
        self.app.after(30000, self._refresh_watcher)

    def _report(self, pct, text):
        """RS status bar + log - the only display once the job is handed to the watcher."""
        try:
            # RS 2024B: SetProgress(String, Int32, String) - a float raises TypeError
            set_progress("HD-GLIO: %s" % text.split("\n")[0], int(round(pct)))
        except Exception as exc:        # the status bar is cosmetic and must never end the run
            print("(set_progress: %s)" % exc)
        if text != getattr(self, "_last_text", None):
            print("[%3d%%] %s" % (pct, text))
            self._last_text = text

    def _show(self, pct, text, color=None):
        self.bar.set(max(0.0, min(pct, 100.0)) / 100.0)
        self.status_lbl.configure(text=text, text_color=color or ("gray90", "gray90"))
        self._report(pct, text)
        self.app.update_idletasks()

    def _selection(self):
        sel = {}
        for slot in SLOTS:
            label = self.menus[slot].get()
            if label == NONE_CHOICE:
                raise ValueError("No series selected for %s." % SLOT_LABELS[slot])
            sel[slot] = self.by_label[label]
        if len({e["series_uid"] for e in sel.values()}) != len(SLOTS):
            raise ValueError("Each series may be assigned to one slot only.")
        if sel["T1"]["for_uid"] != sel["T1c"]["for_uid"]:
            raise ValueError("T1 and T1 + contrast are in different Frames of Reference - "
                             "the pipeline needs one co-registered examination set.")
        return sel

    def _start(self):
        try:
            sel = self._selection()
        except ValueError as exc:
            self._show(0, str(exc), COLOR_BAD)
            return
        print("Mapping:")
        for slot in SLOTS:
            print("  %-5s -> %s" % (slot, choice_label(sel[slot])))
        for m in self.menus.values():
            m.configure(state="disabled")
        self.start_btn.configure(state="disabled")
        self.running = True
        self._gen = self._work(sel)
        self.app.after(1, self._tick)

    def _tick(self):
        try:
            pct, text, delay = next(self._gen)
        except StopIteration:
            self.running = False
            return
        except Exception as exc:
            traceback.print_exc()
            self.running = False
            self._show(self.bar.get() * 100, "ERROR: %s\nLog: %s" % (exc, self.log_path), COLOR_BAD)
            return
        if delay == HANDOFF:
            # the watcher has the job: close the window, finish() does the rest via the status bar
            self._report(pct, text)
            self.running = False
            self.handed_off = True
            self.app.destroy()
            return
        self._show(pct, text)
        self.app.after(max(int(delay), 1), self._tick)

    def _close(self):
        if self.running:
            # the GPU job is not cancelled; the watcher finishes it and the job
            # folder stays on the share until it is cleaned up by hand.
            print("Window closed during the run - the job keeps running: %s" % self.job_dir)
        self.app.destroy()

    def run(self):
        self.app.mainloop()

    def finish(self):
        """After the window has closed: poll, import, clean up.

        Tk is gone; errors propagate as script errors and RS displays them.
        """
        if not self.handed_off:
            return
        for pct, text, delay in self._gen:
            self._report(pct, text)
            time.sleep(max(delay, 1) / 1000.0)

    # ------------------------------------------------------------- Work ---
    def _work(self, sel):
        patient, case = self.patient, self.case
        pid = str(patient.PatientID)
        job_id = "%s_%s" % (safe_name(pid), datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.job_dir = job_dir = os.path.join(TRANSFER_DIR, job_id)
        input_dir = os.path.join(job_dir, "input")
        output_dir = os.path.join(job_dir, "output")
        print("Job: %s" % job_dir)

        # ---- 1. Export (0-20 %) -------------------------------------------
        for i, slot in enumerate(SLOTS):
            e = sel[slot]
            yield 5 * i, "Exporting %s (%s) ..." % (SLOT_LABELS[slot], e["name"]), 1
            folder = os.path.join(input_dir, "series_%s" % slot)
            os.makedirs(folder, exist_ok=True)
            case.ScriptableDicomExport(ExportFolderPath=folder, Examinations=[e["name"]],
                                       DicomFilter="", IgnorePreConditionWarnings=True)

        job_meta = dict(
            job_id=job_id,
            patient_id=pid,
            case_name=str(case.CaseName),
            series={slot: dict(series_uid=sel[slot]["series_uid"], exam_name=sel[slot]["name"],
                               description=sel[slot]["description"],
                               series_number=sel[slot]["series_number"],
                               contrast=bool(sel[slot]["contrast"]))
                    for slot in SLOTS},
            created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        tmp = os.path.join(input_dir, "job.json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(job_meta, f, indent=2, ensure_ascii=False)
        os.replace(tmp, os.path.join(input_dir, "job.json"))     # only now does the watcher see it

        # ---- 2. poll the watcher (22-90 %) ---------------------------------
        status_path = os.path.join(output_dir, "status.json")
        deadline = time.time() + TIMEOUT_MIN * 60
        yield 22, "Job handed over - waiting for the watcher ...", HANDOFF
        shown = (22, "Waiting for the watcher ...")
        while True:
            if time.time() > deadline:
                raise RuntimeError("Timeout after %d min - is the HD-GLIO watcher running? "
                                   "Job: %s" % (TIMEOUT_MIN, job_dir))
            status = read_json(status_path)
            if status is None:
                # not there yet or being replaced by the watcher: keep the last display,
                # otherwise the bar jumps back to 22 % mid-run
                yield shown[0], shown[1], POLL_MS
                continue
            state = status.get("state")
            if state == "error":
                raise RuntimeError("Watcher reports an error: %s (details: %s)"
                                   % (status.get("message"), os.path.join(job_dir, "log.txt")))
            if state == "done":
                break
            if state == "processing" and status.get("total"):
                step, total = int(status.get("step", 0)), int(status["total"])
                shown = (25 + 65.0 * step / total,
                         "[%d/%d] %s" % (step, total, status.get("message", "")))
            yield shown[0], shown[1], POLL_MS

        # ---- 3. Import (92 %) ----------------------------------------------
        yield 92, "Importing RTSTRUCT ...", 1
        import pydicom
        rt_path = os.path.join(job_dir, str(status.get("rtstruct_file", "")))
        rtds = pydicom.dcmread(rt_path, stop_before_pixels=True)
        patient.Save()                      # the import requires a saved patient
        try:
            patient.ImportDataFromPath(
                Path=output_dir, CaseName=str(case.CaseName),
                SeriesOrInstances=[{"PatientID": str(rtds.PatientID),
                                    "StudyInstanceUID": str(rtds.StudyInstanceUID),
                                    "SeriesInstanceUID": str(rtds.SeriesInstanceUID)}])
        except Exception as exc:
            if "already imported" not in str(exc).lower():
                raise
            print("RTSTRUCT was already imported - skipped.")
        patient.Save()

        # ---- 4. volumes + cleanup -----------------------------------------
        target = [e for e in self.series if e["series_uid"] == status.get("target_series_uid")]
        vols = []
        if target:
            ss = case.PatientModel.StructureSets[target[0]["name"]]
            for roi in case.PatientModel.RegionsOfInterest:
                if not str(roi.Name).startswith(ROI_PREFIX):
                    continue
                try:
                    rg = ss.RoiGeometries[roi.Name]
                    if rg.HasContours():
                        vols.append("%s %.1f ml" % (roi.Name, rg.GetRoiVolume()))
                except Exception:
                    pass
        shutil.rmtree(job_dir, ignore_errors=True)
        # one line: the RS status bar only shows the first
        yield 100, "Done on %s: %s (%.0f min)" % (
            target[0]["name"] if target else "?", ", ".join(vols) or "no volumes readable",
            float(status.get("runtime_s", 0)) / 60.0), 1


def main():
    patient = get_current("Patient")
    case = get_current("Case")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, "glio_%s_%s.log" % (safe_name(patient.PatientID), ts))
    log_file = None
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")
        sys.stdout = _Tee(sys.__stdout__, log_file)
    except OSError:
        log_path = "(no log)"
    try:
        print("HD-GLIO Auto-Contour  Patient %s  Case %s" % (patient.PatientID, case.CaseName))
        series = enumerate_mr_series(case)
        if not series:
            raise RuntimeError("No MR series in the current case.")
        for e in series:
            print("  " + choice_label(e))
        app = GlioApp(patient, case, series, log_path)
        app.run()           # selection + export; closes after the handoff to the watcher
        app.finish()        # poll + import via the RS status bar, then end of script
    finally:
        sys.stdout = sys.__stdout__
        if log_file:
            log_file.close()


if __name__ == "__main__":
    main()
