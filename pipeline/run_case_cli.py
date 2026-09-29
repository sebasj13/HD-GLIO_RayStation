"""
run_case_cli.py - CLI test runner: pushes a DICOM folder through the full
HD-GLIO pipeline (same code path the watcher uses) without RayStation.

    conda run -n glio python -m pipeline.run_case_cli [--dicom TestData] [--out test_output]
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    parser = argparse.ArgumentParser(description="HD-GLIO CLI test run")
    parser.add_argument("--dicom", default=os.path.join(ROOT, "TestData"),
                        help="folder containing the DICOM MR series")
    parser.add_argument("--out", default=os.path.join(ROOT, "test_output"),
                        help="output root for the job folder")
    parser.add_argument("--config", default=None,
                        help="config file (default: config.json, else config.example.json)")
    parser.add_argument("--clean", action="store_true",
                        help="remove previous CLI output first")
    args = parser.parse_args()

    from pipeline.run_case import run_cli

    if args.clean:
        prior = os.path.join(args.out, "cli_job")
        if os.path.isdir(prior):
            import shutil
            shutil.rmtree(prior, ignore_errors=True)

    result = run_cli(args.dicom, args.out, config_path=args.config)
    print("=" * 78)
    print("DONE: %s" % result.get("message"))
    print("RTSTRUCT: %s" % os.path.join(args.out, "cli_job", result["rtstruct_file"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())