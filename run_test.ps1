# run_test.ps1 - end-to-end CLI test: runs TestData through the full pipeline
# (same code path the watcher uses) and writes RTSTRUCT + previews to
# .\test_output\cli_job\output\.
#
# Requires the `glio` conda env (setup_glio_env.ps1).

$env:PYTHONUNBUFFERED = "1"
conda run --live-stream -n glio python -u -m pipeline.run_case_cli --clean
exit $LASTEXITCODE