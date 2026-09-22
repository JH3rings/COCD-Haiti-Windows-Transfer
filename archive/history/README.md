# Historical archive

Everything below this directory is retained for provenance and recovery. It is
not part of the executable final GRSL method and must not be imported by final
entry points.

- `code/` contains superseded self-owned architectures and their original
  runners/tests, including Additive Cross-Orbit, complement/EAR, CGSearch,
  R0-R3/R3g, and the old DA T0/T1/DAStudent line.
- `docs/` contains old handoffs, checkpoint ledgers, migration logs, planning
  prompts, and experiment summaries. Their recorded numbers are historical
  evidence and were not rewritten.
- `protocols/` contains the old protocol JSON files associated with those
  architectures.
- `reports/` contains the old architecture audits, method comparisons, and
  protocol plans. Their recorded values were moved unchanged.

The active method is documented at the repository root and in `docs/`, and its
code is under `cocd/windows_main/*_v2.py` plus the retained shared primitives.
Do not delete this archive when preparing a paper supplement unless the Git
history and a separate research backup have both been preserved.
