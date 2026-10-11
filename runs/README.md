# Run registry
One JSON manifest per run (or per sweep), written automatically by `rled.provenance.write_manifest`.
Manifests are small and **always committed**; large outputs live in `results/` (git-ignored).

Naming: `runs/<YYYYMMDD>-<phase>-<short-name>-<git7>.json`
