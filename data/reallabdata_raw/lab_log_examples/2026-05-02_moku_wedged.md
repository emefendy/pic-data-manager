# Operator log — 2026-05-02 02:14 → 02:18

Moku wedged mid-FRA. Specifically: the `fra.get_data(timeout=5)` call hung on iteration 47 of 80.
No exception thrown — the python process just stalled. Killed the process at 02:18 and ran
moku_lock.py --status — lock still held by the wedged PID.

Forced lock release, restarted Moku, re-claimed lock, resumed FRA from iteration 47. Lost ~4 minutes
of wall time, no data lost. Same pattern as the 2026-03-14 wedge; suspect Moku firmware bug under
long FRA runs. See ~/.claude/projects/.../memory/moku_wedging.md for the workaround.
