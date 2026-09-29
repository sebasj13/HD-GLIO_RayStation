# start_watcher.ps1 - start the HD-GLIO watcher by hand (conda env `glio`).
#
# In operation, run glio_watcher.py permanently (service / process manager).
# This script is for tests and debugging; do not run it in parallel with the
# permanent instance. Extra arguments pass through, e.g. --once or --keep_work.
# Job folder and GPU come from config.json (job_root, gpu_device); --watch_dir / --gpu override.

$env:PYTHONUNBUFFERED = "1"
conda run --live-stream -n glio python -u "$PSScriptRoot\glio_watcher.py" @args
