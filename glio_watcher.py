r"""
glio_watcher.py - Folder watcher for the HD-GLIO RayStation pipeline.

Runs with the `glio` conda environment's interpreter (GPU, PyTorch,
hd_glio, hd-bet). It is meant to run permanently (service or process manager), started
without arguments and without `conda activate`:

    C:\path\to\miniconda3\envs\glio\python.exe -u glio_watcher.py

job root and GPU then come from config.json (falls back to
config.example.json); --watch_dir / --gpu override.

Heartbeat: config "heartbeat_file" (default <job_root>/heartbeat_GlioAutoContour.txt)
is rewritten every 60 s with a plain timestamp; the RayStation GUI reads its
age to tell the user whether the watcher is up.

Job folders look like:

    <job_root>/<job_id>/
        input/job.json          written by the RayStation script AFTER the
                                DICOM export is complete
        input/series_<SLOT>/*.dcm
        output/status.json      written here when done/failed
        output/HDGLIO_<job_id>.dcm
        log.txt                 per-job log

Intermediates (local input copy, NIfTI, BET masks, segmentation) live in
LOCAL_WORK_ROOT\<job_id>_<pid> and are deleted after every job.

The watcher polls <job_root> every few seconds, claims a job by creating
input/claim.lock exclusively (stale locks > 12 h are taken over), runs the
pipeline in-process, and writes output/status.json with state
"processing" (+ step/total/message) -> "done" | "error". RayStation polls
status.json, imports the RTSTRUCT and deletes the job folder.

Claiming is idempotent-safe: if the pipeline crashes hard (power loss), the
lock file stays; the job can be re-run by deleting input/claim.lock and
output/status.json, or it is picked up again after the stale timeout.
"""

import argparse
import json
import logging
import os
import shutil
import signal
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from pipeline import run_case as run_case_mod                      # noqa: E402
from pipeline.run_case import default_config_path, process_job     # noqa: E402

log = logging.getLogger("glio.watcher")

STALE_LOCK_SECONDS = 12 * 3600
LOCAL_WORK_ROOT = os.path.join(tempfile.gettempdir(), "glio_work")
HEARTBEAT_NAME = "heartbeat_GlioAutoContour.txt"


def parse_args():
    p = argparse.ArgumentParser(description="HD-GLIO RayStation pipeline watcher")
    p.add_argument("--config", default=None,
                   help="config file (default: config.json, else config.example.json)")
    p.add_argument("--watch_dir", default=None, help="job root override (config: job_root)")
    p.add_argument("--gpu", type=int, default=None, help="GPU index override (config: gpu_device)")
    p.add_argument("--keep_work", action="store_true",
                   help="keep the local work folder (%%TEMP%%\\glio_work\\<job>_<pid>) for debugging")
    p.add_argument("--once", action="store_true",
                   help="process at most one job and exit (no polling loop)")
    p.add_argument("--poll", type=float, default=None, help="poll interval override (s)")
    return p.parse_args()


def setup_logging():
    handlers = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.append(logging.FileHandler(os.path.join(
            ROOT, "logs", "glio_watcher.log"), encoding="utf-8"))
    except OSError:
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        handlers=handlers)
    # nnU-Net/HD-BET are chatty at INFO; keep their noise at WARNING
    for noisy in ("nnunet", "hd_bet", "BatchGenerator", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def heartbeat(stop_event, heartbeat_file):
    while not stop_event.is_set():
        try:
            with open(heartbeat_file, "w") as f:
                f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        except Exception:
            pass
        stop_event.wait(60)


def _claim_job(job_dir):
    """Create input/claim.lock exclusively; None if already claimed."""
    lock = os.path.join(job_dir, "input", "claim.lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps({"pid": os.getpid(),
                                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}))
        return lock
    except FileExistsError:
        # already claimed - take it over if it looks stale (crashed watcher)
        try:
            with open(lock, "r", encoding="utf-8") as f:
                data = json.load(f)
            ts = datetime.strptime(str(data.get("time")), "%Y-%m-%d %H:%M:%S")
            if (datetime.now() - ts).total_seconds() > STALE_LOCK_SECONDS:
                log.warning("taking over stale claim lock for %s", job_dir)
                os.remove(lock)
                return _claim_job(job_dir)
        except Exception:
            pass
        return None


def find_jobs(job_root):
    """Yield (job_dir, job_meta) for every ready, unclaimed job."""
    if not os.path.isdir(job_root):
        return
    for name in sorted(os.listdir(job_root)):
        job_dir = os.path.join(job_root, name)
        if not os.path.isdir(job_dir):
            continue
        job_json = os.path.join(job_dir, "input", "job.json")
        if not os.path.isfile(job_json):
            continue
        status_json = os.path.join(job_dir, "output", "status.json")
        if os.path.isfile(status_json):
            # already processed to completion (or failed) - never re-run these
            try:
                with open(status_json, "r", encoding="utf-8") as f:
                    if json.load(f).get("state") in ("done", "error"):
                        continue
            except Exception:
                continue
        yield job_dir, job_json


def cleanup_local_work():
    """Remove local work folders of crashed runs (a live job never gets this old)."""
    if not os.path.isdir(LOCAL_WORK_ROOT):
        return
    for name in os.listdir(LOCAL_WORK_ROOT):
        path = os.path.join(LOCAL_WORK_ROOT, name)
        try:
            if time.time() - os.path.getmtime(path) > STALE_LOCK_SECONDS:
                shutil.rmtree(path, ignore_errors=True)
                log.info("removed stale local work folder %s", path)
        except OSError:
            pass


def process_one(job_dir, job_json_path, config, keep_work=False):
    job_id = os.path.basename(job_dir)
    input_dir = os.path.join(job_dir, "input")
    output_dir = os.path.join(job_dir, "output")
    # Intermediate work runs locally: reading ~260 DICOMs several times and writing
    # the NIfTIs over SMB cost ~4 min per job (63 s + 199 s vs ~7 s locally).
    # Only output/ (RTSTRUCT, previews, status) and log.txt stay on the share.
    local_dir = os.path.join(LOCAL_WORK_ROOT, "%s_%d" % (job_id, os.getpid()))
    local_input = os.path.join(local_dir, "input")
    work_dir = os.path.join(local_dir, "work")

    with open(job_json_path, "r", encoding="utf-8") as f:
        job_meta = json.load(f)

    run_case_mod.write_status(job_dir, "processing", "job claimed by watcher "
                              "pid %d" % os.getpid(),
                              extra={"started": time.strftime("%Y-%m-%dT%H:%M:%S")})
    # per-job log
    job_log_path = os.path.join(job_dir, "log.txt")
    fh = logging.FileHandler(job_log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(name)-12s %(levelname)s %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(fh)
    try:
        def progress(step, total, message):
            run_case_mod.write_status(job_dir, "processing", message,
                                      extra={"step": step, "total": total}, critical=False)

        t0 = time.time()
        shutil.copytree(input_dir, local_input,
                        ignore=shutil.ignore_patterns("claim.lock", "job.json*"),
                        dirs_exist_ok=True)
        log.info("input copied to %s (%.0f s)", local_input, time.time() - t0)

        result = process_job(local_input, output_dir, work_dir, config, job_meta=job_meta,
                             progress=progress)
        run_case_mod.write_status(job_dir, "done", result.get("message", ""), extra=result)
        log.info("job %s DONE: %s", job_id, result.get("message", ""))
        return result
    except Exception as exc:
        tb = traceback.format_exc()
        log.error("job %s FAILED: %s\n%s", job_id, exc, tb)
        run_case_mod.write_status(job_dir, "error", str(exc),
                                  extra={"traceback": tb,
                                         "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
        return None
    finally:
        # local copy of patient data + intermediates: always removed, also on error
        if keep_work:
            log.info("--keep_work: local work kept in %s", local_dir)
        else:
            shutil.rmtree(local_dir, ignore_errors=True)
            if os.path.exists(local_dir):
                log.warning("local work folder could not be removed completely: %s "
                            "(retried at next watcher start after 12 h)", local_dir)
        root_logger.removeHandler(fh)
        fh.close()


def main():
    args = parse_args()
    args.config = args.config or default_config_path()
    config = run_case_mod.load_config(args.config)
    job_root = args.watch_dir or config.get("job_root") or os.path.join(ROOT, "jobs")
    if args.gpu is not None:
        config["gpu_device"] = args.gpu
    poll_interval = float(args.poll if args.poll else config.get("poll_interval_s", 5.0))
    job_root = os.path.abspath(job_root)
    # site-specific: point your monitoring at this file (see README, Site configuration)
    heartbeat_file = config.get("heartbeat_file") or os.path.join(job_root, HEARTBEAT_NAME)

    # Process managers may start us with the UNC script folder as cwd. cmd.exe then prefixes
    # every shell call with a localised "UNC paths are not supported ..." warning - nnU-Net v2
    # (inside HD-BET) runs `hostname` through the shell and dies decoding its
    # non-ASCII characters. All paths here are absolute, so a local cwd costs nothing.
    os.chdir(tempfile.gettempdir())

    os.makedirs(job_root, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    setup_logging()
    log.info("HD-GLIO watcher starting (config=%s, job_root=%s, poll=%.1fs, heartbeat=%s)",
             args.config, job_root, poll_interval, heartbeat_file)
    cleanup_local_work()

    stop_event = threading.Event()
    threading.Thread(target=heartbeat, args=(stop_event, heartbeat_file), daemon=True).start()

    signal.signal(signal.SIGINT, lambda *_: stop_event.set())

    processed = 0
    try:
        while not stop_event.is_set():
            found = False
            for job_dir, job_json in find_jobs(job_root):
                found = True
                lock = _claim_job(job_dir)
                if lock is None:
                    continue
                log.info("claiming job %s", os.path.basename(job_dir))
                process_one(job_dir, job_json, config, keep_work=args.keep_work)
                processed += 1
                if args.once:
                    break
            if args.once:
                break
            stop_event.wait(poll_interval)
    except KeyboardInterrupt:
        log.info("watcher stopped by user")

    log.info("watcher stopped (%d jobs processed)", processed)


if __name__ == "__main__":
    main()