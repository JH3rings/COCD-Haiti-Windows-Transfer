# Git delivery scope

Backup branch: `archive/pre-grsl-paper-freeze-20260930`. Working branch: `paper/grsl-final-freeze`.

The first preservation commit records preexisting tracked modifications/deletions; those deletions were already present before this freeze. The next backup commit captures 2,152 existing compact code/metadata/figure files (545,870,878 bytes before Git compression). Raw data, checkpoints and probability arrays remain local. No history is rewritten. Some bundled-runtime files accidentally captured with the existing-source snapshot are untracked in the final tree and retained locally; the backup still preserves them.

This freeze moves 22 obsolete files, preserving their SHA256 values; it changes root navigation, adds FINAL source comments, creates the release and stops tracking local node_modules. Removing those files from the Git index does not remove them from disk. It changes no scientific computation or manuscript/PDF bytes. Existing baseline implementations and experiment outputs are retained. See archive_manifest.csv, artifact_manifest.csv and verification.json.

Final evidence differences: nine claim mismatches are recorded for future manuscript correction, not silently edited. Same-protocol results, configured-protocol-different results and historical reported-only results are separate CSVs. Qualitative human review remains pending.

The staged patch is available through Git review. Commit and ordinary non-force push are explicitly authorized by the supplied task. No merge into the default branch or history rewrite is requested.
