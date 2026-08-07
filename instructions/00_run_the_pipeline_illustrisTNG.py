#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
00_run_the_pipeline_illustrisTNG.py
===================================

ONE-STOP DRIVER for the Lyman-alpha skewer extraction around IllustrisTNG
halos. Edit the CONFIG block below, then:

    python 00_run_the_pipeline_illustrisTNG.py

It replaces the manual loop

    for i in {0..4}; do python run_parallel_illustris.py --run_id $i; done

and, on top of it, checks the setup before burning hours of CPU, creates a
dedicated output folder inside the project root, and writes a JSON record of
exactly what was run.

WHAT IT DOES
    1. Sanity-checks the environment: the tools/ modules, the HCD fork of
       fake_spectra, the simulation directories, the core count.
    2. Creates <project root>/<OUTPUT_DIRNAME>/ for the results.
    3. Runs tools/run_parallel_illustris.py once per realization, each with
       its own seed, writing one .pkl per realization into that folder.
    4. Writes run_config.json next to the data, and prints a summary.

WHERE THINGS LIVE

    FakeSpectra_DLAs/
    ├── instructions/
    │   └── 00_run_the_pipeline_illustrisTNG.py   <- you are here
    ├── tools/
    │   ├── utils.py
    │   ├── radii_skewers.py
    │   └── run_parallel_illustris.py             <- does the actual work
    └── <OUTPUT_DIRNAME>/                         <- created by this script
        ├── run_config.json
        ├── TNG50-4_z3_..._run0_seed42.pkl
        └── ...

The output folder is placed in the PROJECT ROOT, not inside instructions/.

READING THE RESULTS

    import pandas as pd, sys
    sys.path.append("tools")
    import radii_skewers as rs

    df = pd.read_pickle("<OUTPUT_DIRNAME>/..._run0_seed42.pkl")
    rs.add_moments(df)          # eq_width, std_dev, skewness, T_hi, ...
    rs.add_concentration(df)    # halo_v200, halo_c200

PREREQUISITE
    The HCD fork of fake_spectra must be installed:
        pip install "git+https://github.com/VictorRoberto132/fake_spectra@v2.2.4-hcd1"
    Fork of sbird/fake_spectra v2.2.4 (MIT). To see exactly what changed:
        git clone https://github.com/VictorRoberto132/fake_spectra
        cd fake_spectra && git diff v2.2.4..v2.2.4-hcd1 -- fake_spectra/spectra.py
    This script verifies it and refuses to start otherwise.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ==============================================================================
#                                   CONFIG
#                    ~~~ THIS IS THE ONLY PART YOU EDIT ~~~
# ==============================================================================

# --- Simulation -------------------------------------------------------------
# BASE_PATH is the directory that CONTAINS groups_XXX/ and snapdir_XXX/.
# SIM_NAME is only a label: it goes into the filenames and df.attrs.
BASE_PATH = "/home/STORAGE/TNG50-3"
SIM_NAME  = "TNG50-3"
SIM_TYPE  = "TNG"          # "TNG" (lengths in ckpc/h) or "MN5" (Mpc)
SNAP_NUM  = 25             # 25 -> z ~ 3.0 in TNG50

# --- Realizations -----------------------------------------------------------
# How many independent samples of the halo population to draw. Each one gets
# its own seed and its own .pkl, which is what the variance analysis compares.
N_RUNS = 5

# --- Halo sampling ----------------------------------------------------------
# Bin EDGES in log10(M_sun), physical. N edges -> N-1 bins.
MASS_BIN_EDGES_LOG   = [10.0, 10.5, 11.0, 11.5, 12.0, 13.0]
N_HALOS_PER_BIN      = 150   # halos drawn per bin
EXHAUSTIVE_THRESHOLD = 100   # bins with <= this many halos are taken WHOLE

# --- Impact parameter grid --------------------------------------------------
# NUM_PARAM_IMPACTS log-spaced values between the two limits, in units of
# R200, PLUS b = 0 (the central sightline) added on top.
# Skewers per halo = 3 + 12 * NUM_PARAM_IMPACTS.
NUM_PARAM_IMPACTS   = 30
RANGE_PARAM_IMPACTS = [0.01, 50.0]

# --- Decomposition ----------------------------------------------------------
#   "HCD_only"    -> only cells belonging to the HCD structure
#   "Forest_only" -> the complementary field (pure forest)
#   "None"        -> stock fake_spectra, all cells
# More than one mode suffixes the columns (flux_HCD_only, ...).
MODES = ["HCD_only"]

# --- Runtime ----------------------------------------------------------------
NUM_CORES        = 20      # worker processes
SECONDS_PER_HALO = 300.0   # only for the wall-clock estimate at startup

# --- Output -----------------------------------------------------------------
# Folder created inside the PROJECT ROOT (FakeSpectra_DLAs/).
# None -> auto: "data_<SIM_NAME>_snap<NNN>_<N_RUNS>runs"
OUTPUT_DIRNAME = None

# Refuse to start if the folder already holds .pkl files (protects finished
# runs). Set to False to add/overwrite realizations in an existing folder.
FAIL_IF_OUTPUT_NOT_EMPTY = True

# ==============================================================================
#                          END OF CONFIG - no need to edit below
# ==============================================================================


class UI:
    CYAN = '\033[96m'; GREEN = '\033[92m'; WARN = '\033[93m'
    FAIL = '\033[91m'; RESET = '\033[0m'; BOLD = '\033[1m'; DIM = '\033[2m'


def banner(text, colour=UI.CYAN):
    print(f"\n{colour}{UI.BOLD}{'=' * 74}{UI.RESET}")
    print(f"{colour}{UI.BOLD}  {text}{UI.RESET}")
    print(f"{colour}{UI.BOLD}{'=' * 74}{UI.RESET}")


def ok(msg):    print(f"  {UI.GREEN}[ok]{UI.RESET}   {msg}")
def warn(msg):  print(f"  {UI.WARN}[warn]{UI.RESET} {msg}")
def fail(msg):  print(f"  {UI.FAIL}[FAIL]{UI.RESET} {msg}")


# ------------------------------------------------------------------ paths ---
THIS_FILE = Path(__file__).resolve()
PROJ_ROOT = THIS_FILE.parent.parent          # instructions/ -> FakeSpectra_DLAs/
TOOLS_DIR = PROJ_ROOT / "tools"
RUNNER    = TOOLS_DIR / "run_parallel_illustris.py"


def resolve_output_dir():
    name = OUTPUT_DIRNAME or f"data_{SIM_NAME}_snap{SNAP_NUM:03d}_{N_RUNS}runs"
    return PROJ_ROOT / name


def preflight():
    """Cheap checks that would otherwise fail hours in, or silently."""
    banner("PREFLIGHT CHECKS")
    problems = []

    # --- project layout ---
    if RUNNER.is_file():
        ok(f"runner found: {RUNNER}")
    else:
        fail(f"missing {RUNNER}")
        problems.append(
                "install the fork:\n"
                "      pip install \"git+https://github.com/VictorRoberto132/fake_spectra@v2.2.4-hcd1\"")
    for mod in ("utils.py", "radii_skewers.py"):
        if (TOOLS_DIR / mod).is_file():
            ok(f"tools/{mod}")
        else:
            fail(f"missing tools/{mod}")
            problems.append(f"tools/{mod} not found")

    # --- the HCD fork of fake_spectra ---
    try:
        import inspect
        import fake_spectra.spectra as _sp
        if 'filter_mode' in inspect.signature(_sp.Spectra.__init__).parameters:
            ok(f"HCD fork of fake_spectra installed ({_sp.__file__})")
        else:
            fail("fake_spectra is the STOCK upstream package, not the HCD fork")
            problems.append(
                "install the fork:\n"
                "      SP=$(python -c 'import fake_spectra.spectra as s; print(s.__file__)')\n"
                "      cp \"$SP\" \"$SP.backup\"\n"
                f"      cp {TOOLS_DIR / 'spectra_corrected.py'} \"$SP\"")
    except ImportError as e:
        fail(f"cannot import fake_spectra ({e})")
        problems.append("fake_spectra is not installed in this environment")

    # --- simulation data ---
    base = Path(os.path.expanduser(BASE_PATH))
    if base.is_dir():
        ok(f"base path: {base}")
    else:
        fail(f"BASE_PATH does not exist: {base}")
        problems.append(f"BASE_PATH not found: {base}")

    grp0 = base / f"groups_{SNAP_NUM:03d}" / f"fof_subhalo_tab_{SNAP_NUM:03d}.0.hdf5"
    if grp0.is_file():
        ok(f"group catalogue: {grp0.name}")
    else:
        fail(f"group catalogue not found: {grp0}")
        problems.append("check BASE_PATH and SNAP_NUM (groups_XXX/fof_subhalo_tab_XXX.0.hdf5)")

    snapdir = base / f"snapdir_{SNAP_NUM:03d}"
    if snapdir.is_dir():
        ok(f"snapshot dir: {snapdir.name}")
    else:
        warn(f"snapdir_{SNAP_NUM:03d}/ not found -- fake_spectra may still find "
             f"the particles elsewhere, but check this if extraction fails")

    # --- config sanity ---
    if len(MASS_BIN_EDGES_LOG) < 2:
        problems.append("MASS_BIN_EDGES_LOG needs at least two edges")
    elif any(b <= a for a, b in zip(MASS_BIN_EDGES_LOG, MASS_BIN_EDGES_LOG[1:])):
        problems.append(f"MASS_BIN_EDGES_LOG must increase: {MASS_BIN_EDGES_LOG}")
    else:
        ok(f"{len(MASS_BIN_EDGES_LOG) - 1} mass bins, {N_HALOS_PER_BIN}/bin")

    if N_RUNS < 1:
        problems.append("N_RUNS must be >= 1")
    if NUM_PARAM_IMPACTS < 1:
        problems.append("NUM_PARAM_IMPACTS must be >= 1")
    if not (0 < RANGE_PARAM_IMPACTS[0] < RANGE_PARAM_IMPACTS[1]):
        problems.append(f"RANGE_PARAM_IMPACTS must be 0 < min < max, "
                        f"got {RANGE_PARAM_IMPACTS}")

    n_cpu = os.cpu_count() or 1
    if NUM_CORES > n_cpu:
        warn(f"NUM_CORES={NUM_CORES} exceeds the {n_cpu} CPUs visible here; "
             f"the machine will thrash")
    else:
        ok(f"{NUM_CORES} worker processes (machine has {n_cpu} CPUs)")

    # --- disk ---
    try:
        free_gb = shutil.disk_usage(PROJ_ROOT).free / 1e9
        (ok if free_gb > 5 else warn)(f"{free_gb:.1f} GB free on the project disk")
    except OSError:
        pass

    if problems:
        print()
        fail("cannot start:")
        for p in problems:
            print(f"    - {p}")
        print()
        sys.exit(1)
    print()


def build_command(run_id, outdir):
    """The exact argv handed to run_parallel_illustris.py for one realization."""
    return [
        sys.executable, str(RUNNER),
        "--run_id",              str(run_id),
        "--sim",                 SIM_NAME,
        "--base_path",           str(Path(os.path.expanduser(BASE_PATH))),
        "--sim_type",            SIM_TYPE,
        "--snap",                str(SNAP_NUM),
        "--modes",               ",".join(MODES),
        "--mass_bins",           ",".join(str(e) for e in MASS_BIN_EDGES_LOG),
        "--n_per_bin",           str(N_HALOS_PER_BIN),
        "--exhaustive_threshold", str(EXHAUSTIVE_THRESHOLD),
        "--n_impacts",           str(NUM_PARAM_IMPACTS),
        "--impact_min",          str(RANGE_PARAM_IMPACTS[0]),
        "--impact_max",          str(RANGE_PARAM_IMPACTS[1]),
        "--cores",               str(NUM_CORES),
        "--seconds_per_halo",    str(SECONDS_PER_HALO),
        "--outdir",              str(outdir),
    ]


def write_run_config(outdir, run_ids, commands):
    """Provenance: what was run, with which knobs, when."""
    cfg = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "driver": str(THIS_FILE),
        "python": sys.executable,
        "simulation": {
            "sim_name": SIM_NAME, "sim_type": SIM_TYPE,
            "base_path": str(Path(os.path.expanduser(BASE_PATH))),
            "snap_num": SNAP_NUM,
        },
        "sampling": {
            "mass_bin_edges_log10_msun": MASS_BIN_EDGES_LOG,
            "n_halos_per_bin": N_HALOS_PER_BIN,
            "exhaustive_threshold": EXHAUSTIVE_THRESHOLD,
        },
        "impacts": {
            "num_param_impacts": NUM_PARAM_IMPACTS,
            "range_param_impacts_r200": RANGE_PARAM_IMPACTS,
            "skewers_per_halo": 3 + 12 * NUM_PARAM_IMPACTS,
        },
        "modes": list(MODES),
        "runtime": {"num_cores": NUM_CORES},
        "runs": run_ids,
        "commands": [" ".join(c) for c in commands],
    }
    path = Path(outdir) / "run_config.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Run the full TNG skewer-extraction pipeline. "
                    "Edit the CONFIG block at the top of this file.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Run the checks and print the commands, extract nothing.")
    parser.add_argument("--runs", type=str, default=None,
                        help="Comma-separated run_ids to (re)do, e.g. '2,4'. "
                             "Default: 0..N_RUNS-1. Asking for specific runs "
                             "implies you mean it, so the non-empty-folder "
                             "guard is skipped.")
    parser.add_argument("--force", action="store_true",
                        help="Write into a folder that already holds .pkl files.")
    args = parser.parse_args()

    run_ids = ([int(r) for r in args.runs.split(",") if r.strip()]
               if args.runs else list(range(N_RUNS)))

    banner("LYMAN-ALPHA SKEWER PIPELINE — IllustrisTNG")
    print(f"  {UI.BOLD}Simulation   :{UI.RESET} {SIM_NAME} ({SIM_TYPE}), snap {SNAP_NUM}")
    print(f"  {UI.BOLD}Data         :{UI.RESET} {BASE_PATH}")
    print(f"  {UI.BOLD}Realizations :{UI.RESET} {len(run_ids)}  (run_ids {run_ids})")
    print(f"  {UI.BOLD}Mass bins    :{UI.RESET} "
          f"{len(MASS_BIN_EDGES_LOG) - 1} in log10(M_sun) {MASS_BIN_EDGES_LOG}")
    print(f"  {UI.BOLD}Impacts      :{UI.RESET} b=0 + {NUM_PARAM_IMPACTS} log-spaced "
          f"{RANGE_PARAM_IMPACTS[0]}–{RANGE_PARAM_IMPACTS[1]} R200 "
          f"({3 + 12 * NUM_PARAM_IMPACTS} skewers/halo)")
    print(f"  {UI.BOLD}Modes        :{UI.RESET} {', '.join(MODES)}")
    print(f"  {UI.BOLD}Cores        :{UI.RESET} {NUM_CORES}")

    preflight()

    # --- output folder ---
    # The guard protects a FINISHED set from being clobbered by an accidental
    # re-run. Targeting specific realizations (--runs) or passing --force means
    # you asked for it on purpose, so it steps aside.
    outdir = resolve_output_dir()
    existing = sorted(outdir.glob("*.pkl")) if outdir.exists() else []
    guard_on = FAIL_IF_OUTPUT_NOT_EMPTY and args.runs is None and not args.force
    if existing and guard_on:
        fail(f"{outdir} already contains {len(existing)} .pkl file(s).")
        print(f"    Move them, pick another OUTPUT_DIRNAME, pass --force, or set "
              f"FAIL_IF_OUTPUT_NOT_EMPTY = False.\n")
        sys.exit(1)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"  {UI.BOLD}Output dir   :{UI.RESET} {outdir}")
    if existing:
        warn(f"folder already holds {len(existing)} .pkl file(s); "
             f"same-named files will be overwritten")

    commands = [build_command(rid, outdir) for rid in run_ids]

    if args.dry_run:
        banner("DRY RUN — commands that WOULD be executed", UI.WARN)
        for rid, cmd in zip(run_ids, commands):
            print(f"\n  {UI.BOLD}run {rid}:{UI.RESET}\n    {' '.join(cmd)}")
        cfg_path = write_run_config(outdir, run_ids, commands)
        print(f"\n  config written to {cfg_path}\n")
        return 0

    cfg_path = write_run_config(outdir, run_ids, commands)
    print(f"  {UI.BOLD}Provenance   :{UI.RESET} {cfg_path}")

    # --- the actual loop ---
    results = []
    t0 = time.time()
    for i, (rid, cmd) in enumerate(zip(run_ids, commands), start=1):
        banner(f"REALIZATION {i}/{len(run_ids)}  (run_id {rid})", UI.CYAN)
        print(f"{UI.DIM}  {' '.join(cmd)}{UI.RESET}\n")
        t_run = time.time()
        # cwd=tools so `import utils` resolves; output streams straight through
        # so the tqdm bar behaves exactly as when you run it by hand.
        proc = subprocess.run(cmd, cwd=str(TOOLS_DIR))
        dt = (time.time() - t_run) / 60.0
        results.append((rid, proc.returncode, dt))
        if proc.returncode == 0:
            print(f"\n{UI.GREEN}{UI.BOLD}  run {rid} finished in {dt:.1f} min"
                  f"{UI.RESET}")
        else:
            print(f"\n{UI.FAIL}{UI.BOLD}  run {rid} FAILED "
                  f"(exit {proc.returncode}) after {dt:.1f} min{UI.RESET}")
            print(f"{UI.FAIL}  see {TOOLS_DIR / 'extraction_errors_illustris_extended.log'}"
                  f"{UI.RESET}")

    # --- summary ---
    total = (time.time() - t0) / 60.0
    n_ok = sum(1 for _, rc, _ in results if rc == 0)
    banner("SUMMARY", UI.GREEN if n_ok == len(results) else UI.WARN)
    for rid, rc, dt in results:
        mark = f"{UI.GREEN}OK  {UI.RESET}" if rc == 0 else f"{UI.FAIL}FAIL{UI.RESET}"
        print(f"  run {rid:<3} {mark}  {dt:6.1f} min")
    print(f"\n  {UI.BOLD}{n_ok}/{len(results)} realizations succeeded "
          f"in {total:.1f} min{UI.RESET}")

    produced = sorted(outdir.glob("*.pkl"))
    print(f"  {UI.BOLD}{len(produced)} pickle(s) in{UI.RESET} {outdir}")
    for p in produced:
        print(f"    {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")

    print(f"\n  {UI.BOLD}Next:{UI.RESET}")
    print(f"    import pandas as pd, sys; sys.path.append('{TOOLS_DIR}')")
    print(f"    import radii_skewers as rs")
    print(f"    df = pd.read_pickle('{produced[0] if produced else 'FILE.pkl'}')")
    print(f"    rs.add_moments(df); rs.add_concentration(df)\n")

    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())