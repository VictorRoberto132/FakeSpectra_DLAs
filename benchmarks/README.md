# benchmarks

Comparisons between the original `fake_spectra` (upstream v2.2.4) and the HCD
fork (`~/fake_spectra_fork`).

| File | What it does |
|---|---|
| `build_upstream.sh` | exports tag v2.2.4 from the fork repo and compiles it in `fake_spectra_upstream/` (nothing is installed) |
| `extract_skewers.py` | extracts one skewer (or a table of them) with a chosen build, times `init` / `get_tau` / `get_col_density`, writes a `.npz` |
| `01_upstream_vs_fork_single_skewer.ipynb` | one skewer through a random halo: flux, HCD/Forest decomposition and timing |
| `02_random_skewers_sf_neutral.ipynb` | many skewers at random positions: isolates `sf_neutral`, the 7/8 tau bug and the decomposition |
| `time_pipeline_workers.py` | runs the pipeline worker (`process_single_halo`) on N halos with N processes; wall time, time per halo, peak memory |
| `03_cores_scaling_TNG50-2.ipynb` | scans the number of workers (and OpenMP threads per worker) to find the fastest `--cores` for the pipeline |
| `test_per_skewer_r200.py` | checks the fork's per-sightline `halo_r200`: backup vs new, halos one by one vs in one `Spectra`, random sightlines |
| `04_profile_one_halo_TNG50-2.ipynb` | `cProfile` of one halo: time in snapshot reading vs interpolation vs HCD decomposition |
| `test_sf_gas_model.py` | checks the fork's star-forming gas options (`sf_criterion`, `sf_temp`, `h2_model`) against an independent implementation |
| `test_halo_groups.py` | checks that the grouped pipeline worker (`--halos_per_task K`) gives the same records as one halo per task |
| `05_halo_groups_TNG50-2.ipynb` | throughput with W workers × K halos per task (one snapshot read per group) |

The two builds share the package name, so every extraction runs in its own
subprocess. `extract_skewers.py` removes the fork's editable-install import
hook before importing, and checks which copy was actually loaded.

```bash
bash benchmarks/build_upstream.sh        # once (or after changing the tag)
```

`fake_spectra_upstream/` and `results/` are not versioned.
