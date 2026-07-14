"""
Ground-truth check for attribution.py, run against the engineered
fixtures from generate_data.py (timeseries.csv / touchpoints.csv).

Confirms:
  - 2026-05-20 is attributed to the 2026-05-19 LinkedIn touchpoint.
  - 2026-06-10 is attributed to the 2026-06-09 Press release touchpoint.
  - 2026-06-23 is marked unexplained (no touchpoint in window).
"""

import pandas as pd

from attribution import UNEXPLAINED_LABEL, attribute_spikes
from spike_detector import find_spikes

ts = pd.read_csv("timeseries.csv")
tp = pd.read_csv("touchpoints.csv")

spikes = find_spikes(ts)
results = {r.spike_date.date().isoformat(): r for r in attribute_spikes(spikes, tp)}

checks = [
    ("2026-05-20", "LinkedIn post: launch announcement", False),
    ("2026-06-10", "Press release: Series A funding", False),
    ("2026-06-23", UNEXPLAINED_LABEL, True),
]

failures = []
for spike_date, expected_cause, expect_unexplained in checks:
    if spike_date not in results:
        failures.append(f"{spike_date}: not present in spike_detector output at all")
        continue
    r = results[spike_date]
    if r.is_unexplained != expect_unexplained:
        failures.append(
            f"{spike_date}: expected is_unexplained={expect_unexplained}, got {r.is_unexplained}"
        )
    elif r.top_cause != expected_cause:
        failures.append(f"{spike_date}: expected cause {expected_cause!r}, got {r.top_cause!r}")
    else:
        print(f"PASS  {spike_date}: {r.top_cause}")

if failures:
    print("\nFAILURES:")
    for f in failures:
        print(f"  - {f}")
    raise SystemExit(1)

print("\nAll ground-truth attribution checks passed.")
