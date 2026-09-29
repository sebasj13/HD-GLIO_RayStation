"""Self-check: job.json slot mapping (the RS user's series choice) is honoured.

    python tests/test_job_mapping.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.run_case import mapping_from_job  # noqa: E402

series = [types.SimpleNamespace(series_uid=u) for u in ("u5", "u17", "u14", "u7")]
job = {"series": {"T1": {"series_uid": "u5"}, "T1c": {"series_uid": "u17"},
                  "T2": {"series_uid": "u14"}, "FLAIR": {"series_uid": "u7"}}}

# mixed-case slot "T1c" must map (was lost to slot.upper() -> "T1C")
m = mapping_from_job(job, series)
assert {k: v.series_uid for k, v in m.items()} == {"T1": "u5", "T1c": "u17", "T2": "u14", "FLAIR": "u7"}, m

# a UID that was not exported -> error, never a silent auto-classification
job["series"]["T1c"]["series_uid"] = "other"
try:
    mapping_from_job(job, series)
    raise AssertionError("expected ValueError")
except ValueError as exc:
    assert "T1c" in str(exc), exc

print("JOB MAPPING OK")
