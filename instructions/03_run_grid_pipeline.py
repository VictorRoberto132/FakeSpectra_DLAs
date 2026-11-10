"""
03_run_grid_pipeline.py
=======================
DRIVER for the Lyman-alpha skewer extraction on a UNIFORM GRID of the
IllustrisTNG box (no haloes). Edit the CONFIG block below, then:

    python 03_run_grid_pipeline.py              # run
    python 03_run_grid_pipeline.py --dry_run    # checks + command only

Same structure as 00_run_the_pipeline_illustrisTNG.py: it checks the setup,
creates a dedicated output folder in the project root, runs the worker
tools/run_parallel_grid_illustris.py and writes run_config.json.

WHAT THE PIPELINE COMPUTES
    N_SIDE x N_SIDE sightlines on each face of the box, one face per axis
    (3 * N_SIDE^2 sightlines in total), each one crossing the whole box.
    Every sightline is extracted in all the CASES in the same run:

        total   all cells (stock fake_spectra)               -> LyA with HCDs
        noHCD   Forest_only, absorbers N_HI >= 10^17.2 removed -> LyA without HCDs
        noDLA   Forest_only, absorbers N_HI >= 10^20.3 removed -> LyA without DLAs

    The absorber definition is the one of the halo pipeline (velocity-space
    peaks + blend test, star-forming gas model of tools/utils.py), with the
    threshold of each case and NO halo window: every absorber along the
    sightline counts. The threshold is set on the Spectra object, the fork is
    not modified (see tools/run_parallel_grid_illustris.py). Nothing is zeroed
    and the spectra are NOT centred (native box velocity axis, 0 ... vmax),
    which is what the P1D needs. The grid is deterministic: one run, no seed.

OUTPUT
    <project root>/<OUTPUT_DIRNAME>/
        run_config.json
        <SIM_NAME>_z3_Grid<N>x<N>x3_total_noHCD_noDLA.pkl

    One row per sightline, same conventions as the halo pipeline: per-case
    columns with suffix _<case> (flux_total, flux_noHCD, flux_noDLA,
    n_hi_col_*, temp_*, log_n_hi_*, is_hcd_*), shared grids in df.attrs
    (v_grid, nbins, vmax, pixel_length_cm, cases, ...). See the docstring of
    tools/run_parallel_grid_illustris.py. Analysis: notebooks_grid/.

SIZE
    3 * N_SIDE^2 rows x 3 cases x 3 arrays (flux, n_hi_col, temp) x nbins
    float32. For N_SIDE = 256 and nbins = 396 (TNG50 at 10 km/s): ~2.8 GB.

PREREQUISITE
    The HCD fork of fake_spectra (see README, "Installation").
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
BASE_PATH = "/home/STORAGE/TNG50-2"
SIM_NAME  = "TNG50-2"
SIM_TYPE  = "TNG"          # "TNG" (lengths in ckpc/h) or "MN5" (Mpc)
SNAP_NUM  = 25             # 25 -> z ~ 3.0 in TNG50

# --- Grid -------------------------------------------------------------------
# N_SIDE x N_SIDE sightlines per face, 3 faces (x, y, z):
# total = 3 * N_SIDE^2 (256 -> 196,608 sightlines).
N_SIDE  = 256
RES_KMS = 10.0             # pixel width, km/s (same as the halo pipeline)

# --- Cases ------------------------------------------------------------------
# case name -> (filter_mode, log10 N_HI threshold of the removed absorbers).
# All in the same run (one Spectra object per case and per chunk). The name is
# the column suffix (flux_total, flux_noHCD, ...). Mode "None" takes no
# threshold. "HCD_only" with a threshold keeps only those absorbers instead.
CASES = {
    "total": ("None", None),           # LyA with HCDs
    "noHCD": ("Forest_only", 17.2),    # LyA without HCDs (log N_HI >= 17.2 removed)
    "noDLA": ("Forest_only", 20.3),    # LyA without DLAs (log N_HI >= 20.3 removed)
}

# --- Runtime ----------------------------------------------------------------
NUM_CORES  = 16            # worker processes
CHUNK_SIZE = 2048          # sightlines per task: each task reads the snapshot
                           # once per case, so larger chunks = fewer reads
# Only for the wall-clock estimate (seconds per sightline per worker, all
# cases). TNG50-4: 0.2 s, measured on one chunk of 2048 (total + noHCD + noDLA,
# 407 s, 1 process). TNG50-2: ~0.4 s of compute plus the snapshot reads
# (349 GB on disk), which dominate when 16 workers read at once: expect
# several hours, not the printed estimate.
SECONDS_PER_SKEWER = 0.2

# --- Output -----------------------------------------------------------------
# Folder created inside the PROJECT ROOT (FakeSpectra_DLAs/).
# None -> auto: "data_<SIM_NAME>_snap<NNN>_grid<N_SIDE>_peak_plus_blend_sfneutral_1e4K_H2"
# (the label says which HCD method and star-forming gas model were used, as in
# the halo-pipeline folders; df.attrs also records them).
OUTPUT_DIRNAME = None

# Refuse to start if the folder already holds .pkl files (protects finished
# runs). Set to False (or pass --force) to overwrite.
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
RUNNER    = TOOLS_DIR / "run_parallel_grid_illustris.py"
N_SKEWERS = 3 * N_SIDE**2


def resolve_output_dir():
    name = OUTPUT_DIRNAME or (f"data_{SIM_NAME}_snap{SNAP_NUM:03d}_grid{N_SIDE}"
                              f"_peak_plus_blend_sfneutral_1e4K_H2")
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
        problems.append("tools/run_parallel_grid_illustris.py not found")
    if (TOOLS_DIR / "utils.py").is_file():
        ok("tools/utils.py")
    else:
        fail("missing tools/utils.py")
        problems.append("tools/utils.py not found")

    # --- the HCD fork of fake_spectra ---
    try:
        import inspect
        import fake_spectra.spectra as _sp
        if 'filter_mode' in inspect.signature(_sp.Spectra.__init__).parameters:
            ok(f"HCD fork of fake_spectra installed ({_sp.__file__})")
            bad = [m for m, _ in CASES.values() if m not in _sp.FILTER_MODES]
            if bad:
                problems.append(f"unknown modes in CASES {bad}; valid: {_sp.FILTER_MODES}")
        else:
            fail("fake_spectra is the STOCK upstream package, not the HCD fork")
            problems.append("install the HCD fork (see README, 'Installation')")
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

    # The box size is read from the group catalogue header
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
        warn(f"snapdir_{SNAP_NUM:03d}/ not found -- check this if extraction fails")

    # --- config sanity ---
    if N_SIDE < 1:
        problems.append("N_SIDE must be >= 1")
    if CHUNK_SIZE < 1:
        problems.append("CHUNK_SIZE must be >= 1")
    if not CASES:
        problems.append("CASES is empty")
    for name, (mode, thr) in CASES.items():
        if mode != "None" and thr is not None and not 10 < thr < 25:
            problems.append(f"case {name}: log N_HI threshold {thr} looks wrong")
    ok(f"{N_SIDE}x{N_SIDE} x 3 axes = {N_SKEWERS:,} sightlines, "
       f"{-(-N_SKEWERS // CHUNK_SIZE)} chunks of {CHUNK_SIZE}")

    n_cpu = os.cpu_count() or 1
    if NUM_CORES > n_cpu:
        warn(f"NUM_CORES={NUM_CORES} exceeds the {n_cpu} CPUs visible here")
    else:
        ok(f"{NUM_CORES} worker processes (machine has {n_cpu} CPUs)")

    # --- disk: rough size of the output (3 float32 arrays per mode, nbins ~ 400) ---
    size_gb = N_SKEWERS * len(CASES) * 3 * 400 * 4 / 1e9
    try:
        free_gb = shutil.disk_usage(PROJ_ROOT).free / 1e9
        (ok if free_gb > 2 * size_gb else warn)(
            f"{free_gb:.1f} GB free on the project disk (output ~{size_gb:.1f} GB)")
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


def cases_arg():
    """CASES as the worker's --cases string: name:mode[:threshold],..."""
    return ",".join(f"{n}:{m}" + ("" if t is None else f":{t}")
                    for n, (m, t) in CASES.items())


def build_command(outdir):
    """The exact argv handed to run_parallel_grid_illustris.py."""
    return [
        sys.executable, str(RUNNER),
        "--sim",                SIM_NAME,
        "--base_path",          str(Path(os.path.expanduser(BASE_PATH))),
        "--sim_type",           SIM_TYPE,
        "--snap",               str(SNAP_NUM),
        "--cases",              cases_arg(),
        "--n_side",             str(N_SIDE),
        "--chunk_size",         str(CHUNK_SIZE),
        "--res",                str(RES_KMS),
        "--cores",              str(NUM_CORES),
        "--seconds_per_skewer", str(SECONDS_PER_SKEWER),
        "--outdir",             str(outdir),
    ]


def extraction_settings():
    """Star-forming gas model, fixed in tools/utils.py (not a CONFIG knob):
    recorded so that the folder name is not the only label of the method."""
    sys.path.insert(0, str(TOOLS_DIR))
    import utils
    return {"halo_window": False, "sf_neutral": utils.SF_NEUTRAL, **utils.SF_GAS_MODEL}


def write_run_config(outdir, command):
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
        "grid": {
            "n_side": N_SIDE,
            "n_axes": 3,
            "n_skewers": N_SKEWERS,
            "res_kms": RES_KMS,
        },
        "cases": {n: {"filter_mode": m, "log_nhi_thresh": t}
                  for n, (m, t) in CASES.items()},
        "runtime": {"num_cores": NUM_CORES, "chunk_size": CHUNK_SIZE},
        "extraction": extraction_settings(),
        "command": " ".join(command),
    }
    path = Path(outdir) / "run_config.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Run the uniform-grid skewer extraction. "
                    "Edit the CONFIG block at the top of this file.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Run the checks and print the command, extract nothing.")
    parser.add_argument("--force", action="store_true",
                        help="Write into a folder that already holds .pkl files.")
    args = parser.parse_args()

    banner("LYMAN-ALPHA SKEWER PIPELINE — UNIFORM GRID, IllustrisTNG")
    print(f"  {UI.BOLD}Simulation   :{UI.RESET} {SIM_NAME} ({SIM_TYPE}), snap {SNAP_NUM}")
    print(f"  {UI.BOLD}Data         :{UI.RESET} {BASE_PATH}")
    print(f"  {UI.BOLD}Grid         :{UI.RESET} {N_SIDE}x{N_SIDE} per face x 3 axes "
          f"= {N_SKEWERS:,} sightlines")
    for name, (mode, thr) in CASES.items():
        print(f"  {UI.BOLD}Case {name:<8}:{UI.RESET} {mode}"
              + ("" if thr is None else f", removes log N_HI >= {thr}"))
    print(f"  {UI.BOLD}Cores        :{UI.RESET} {NUM_CORES}")
    print(f"  {UI.BOLD}Estimate     :{UI.RESET} "
          f"~{N_SKEWERS * SECONDS_PER_SKEWER / NUM_CORES / 60:.0f} min")

    preflight()

    # --- output folder (the guard protects a finished run) ---
    outdir = resolve_output_dir()
    existing = sorted(outdir.glob("*.pkl")) if outdir.exists() else []
    if existing and FAIL_IF_OUTPUT_NOT_EMPTY and not args.force:
        fail(f"{outdir} already contains {len(existing)} .pkl file(s).")
        print(f"    Move them, pick another OUTPUT_DIRNAME, pass --force, or set "
              f"FAIL_IF_OUTPUT_NOT_EMPTY = False.\n")
        sys.exit(1)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"  {UI.BOLD}Output dir   :{UI.RESET} {outdir}")
    if existing:
        warn(f"folder already holds {len(existing)} .pkl file(s); "
             f"a same-named file will be overwritten")

    command = build_command(outdir)

    if args.dry_run:
        banner("DRY RUN — command that WOULD be executed", UI.WARN)
        print(f"\n    {' '.join(command)}")
        cfg_path = write_run_config(outdir, command)
        print(f"\n  config written to {cfg_path}\n")
        return 0

    cfg_path = write_run_config(outdir, command)
    print(f"  {UI.BOLD}Provenance   :{UI.RESET} {cfg_path}")

    # --- the run ---
    banner("GRID EXTRACTION", UI.CYAN)
    print(f"{UI.DIM}  {' '.join(command)}{UI.RESET}\n")
    t0 = time.time()
    # cwd=tools so `import utils` resolves; output streams straight through
    proc = subprocess.run(command, cwd=str(TOOLS_DIR))
    dt = (time.time() - t0) / 60.0

    # --- summary ---
    banner("SUMMARY", UI.GREEN if proc.returncode == 0 else UI.WARN)
    if proc.returncode == 0:
        print(f"  {UI.GREEN}{UI.BOLD}finished in {dt:.1f} min{UI.RESET}")
    else:
        print(f"  {UI.FAIL}{UI.BOLD}FAILED or incomplete (exit {proc.returncode}) "
              f"after {dt:.1f} min{UI.RESET}")
        print(f"  {UI.FAIL}see {TOOLS_DIR / 'extraction_errors_grid_illustris.log'}{UI.RESET}")

    produced = sorted(outdir.glob("*.pkl"))
    print(f"  {UI.BOLD}{len(produced)} pickle(s) in{UI.RESET} {outdir}")
    for p in produced:
        print(f"    {p.name}  ({p.stat().st_size / 1e9:.2f} GB)")
    print(f"\n  {UI.BOLD}Next:{UI.RESET} notebooks_grid/01_P1D_grid.ipynb\n")

    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
