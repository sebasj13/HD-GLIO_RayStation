"""
run_glio.py - Wrapper around the HD-GLIO prediction command.

Runs `hd_glio_predict` (installed from the third_party/HD-GLIO submodule,
see setup_glio_env.ps1) as a subprocess so that nnU-Net's noisy stdout
stays out of the pipeline log and GPU memory is cleanly released afterwards. Goes through
hd_glio_predict_win.py (threads instead of worker processes), since the
stock CLI can't pickle its trainer on Windows.
"""

import logging
import os
import subprocess
import sys

log = logging.getLogger("glio.run_glio")


def run_hd_glio(t1, t1c, t2, flair, output_file, device=0, timeout_s=3600):
    """Run HD-GLIO for one case.

    All inputs must be preprocessed NIfTI files (FSL-standard orientation,
    brain-extracted, co-registered onto the T1 grid). The output is written
    to *output_file* (.nii.gz, same grid as the T1 input).

    Returns the path to the output segmentation.
    """
    if os.path.exists(output_file):
        os.remove(output_file)

    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", str(device))
    env.setdefault("PYTHONUNBUFFERED", "1")

    shim = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hd_glio_predict_win.py")
    cmd = [sys.executable, shim,
           "-t1", t1, "-t1c", t1c, "-t2", t2, "-flair", flair, "-o", output_file]
    log.info("running HD-GLIO: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout_s, env=env)
    except subprocess.TimeoutExpired:
        raise RuntimeError("HD-GLIO prediction timed out after %s s" % timeout_s)

    if proc.returncode != 0:
        tail = (proc.stdout or "") + "\n" + (proc.stderr or "")
        raise RuntimeError("HD-GLIO failed (rc=%d):\n%s" % (proc.returncode, tail[-5000:]))
    if not os.path.isfile(output_file):
        raise RuntimeError("HD-GLIO did not produce %s\n--- stdout tail ---\n%s"
                           % (output_file, (proc.stdout or "")[-3000:]))
    log.info("HD-GLIO output: %s", output_file)
    return output_file