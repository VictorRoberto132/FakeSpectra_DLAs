# -*- coding: utf-8 -*-
"""
run_parallel_grid_illustris.py

Extracts Lyman-alpha skewers on a uniform grid that covers each face of the
simulation box (n_side x n_side sightlines per face, one face per axis, so
3 * n_side^2 skewers), for several CASES in the same run. A case is a
decomposition mode of the fork plus the log10 N_HI threshold that defines
which absorbers are removed (or kept):

    total : mode "None"                  -> all cells
    noHCD : mode "Forest_only", 17.2     -> HCDs (N_HI >= 10^17.2) removed
    noDLA : mode "Forest_only", 20.3     -> only DLAs (N_HI >= 10^20.3) removed

Called by the driver `instructions/03_run_grid_pipeline.py` (edit the CONFIG
block there, not here). It can also be run by hand:

    python run_parallel_grid_illustris.py --sim TNG50-4 --n_side 256 \
        --cases total:None,noHCD:Forest_only:17.2,noDLA:Forest_only:20.3 \
        --cores 16 --outdir DIR

THE THRESHOLD, WITHOUT TOUCHING THE FORK
    In the fork, a velocity-space feature (basin) is an HCD when its column
    reaches Spectra.hcd_selection_thresh (= 10^17.2, set in __init__); this is
    the ONLY place where the threshold enters the mask (spectra.py,
    _hcd_peak_mask: `accepted = basin_N >= self.hcd_selection_thresh`). The mask
    is built lazily on the first get_tau, so setting the attribute right after
    the constructor changes the definition for that Spectra object only.
    The blend test still applies: weaker features that an observer could not
    separate from an accepted DLA (the flux does not recover between them) are
    removed together with it, as for the 17.2 cut.

DIFFERENCES FROM THE HALO PIPELINE (run_parallel_illustris.py)
    * No halo: the HCD window is OFF (halo_r200=None in the fork), so every HCD
      along the sightline (anywhere in the box) is removed (Forest_only) or
      kept (HCD_only). The definition is the same: velocity-space peaks with
      N_HI >= threshold, plus the blend test (hcd_algorithm="peaks+blend").
    * No centring: the spectra stay on the native box velocity axis,
      v = 0 ... vmax (periodic), which is what the 1D flux power needs.
    * No 17.2 cut on the sightline: nothing is zeroed. A grid sightline with
      no HCD keeps its whole absorption in Forest_only. log_n_hi is the raw
      integrated column of each case (0.0 only if it is empty).
    * No randomness: the grid is deterministic, so there is no seed.

OUTPUT (one .pkl, same conventions as the halo pipeline)
    One row per sightline. Scalars: skewer_id, axis (0 = x, 1 = y, 2 = z),
    grid_i, grid_j (indices on the face), pos_x, pos_y, pos_z (code units;
    the coordinate along the line of sight is 0, the sightline spans the box).
    Per case (suffix _<case>, e.g. flux_total, flux_noHCD, flux_noDLA):
    flux, n_hi_col, temp (float32 arrays on df.attrs['v_grid']), log_n_hi,
    is_hcd. df.attrs: the same grids and constants as the halo pipeline, plus
    'cases' ({case: {'filter_mode', 'log_nhi_thresh'}}), 'modes' (the case
    names, i.e. the column suffixes, as rs.add_moments expects), the grid
    settings (n_side, grid_spacing) and halo_window = False.
"""

import argparse
import contextlib
import io
import logging
import multiprocessing
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- IMPORT PROJECT MODULES ---
try:
    import utils
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import utils
import fake_spectra.spectra as spectra


# ==============================================================================
# TERMINAL UI COLOURS
# ==============================================================================
class TerminalUI:
    CYAN    = '\033[96m'
    GREEN   = '\033[92m'
    WARNING = '\033[93m'
    FAIL    = '\033[91m'
    RESET   = '\033[0m'
    BOLD    = '\033[1m'


# ==============================================================================
# LOGGING (errors only, to file)
# ==============================================================================
# force=True: `import utils` already configured the root logger (INFO, to the
# terminal), which would make this call a no-op. filemode='w' starts a fresh
# file: do not os.remove() it afterwards, the handler would keep writing into
# the deleted file.
LOG_FILE = "extraction_errors_grid_illustris.log"
logging.basicConfig(
    filename=LOG_FILE, filemode='w', level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s', force=True,
)


# ==============================================================================
# 1. DEFAULT CONFIGURATION   (every entry has a --flag override)
# ==============================================================================
SNAP_NUM  = 25
SIM_NAME  = "TNG50-4"
BASE_PATH = f"/home/STORAGE/{SIM_NAME}"
SIM_TYPE  = "TNG"

# case name -> (filter_mode, log10 N_HI threshold; None for mode "None")
CASES = {
    "total": ("None", None),
    "noHCD": ("Forest_only", 17.2),
    "noDLA": ("Forest_only", 20.3),
}
N_SIDE     = 256        # sightlines per side of each face
CHUNK_SIZE = 2048       # sightlines per task (= per Spectra object)
RES_KMS    = 10.0       # pixel width, km/s (same as the halo pipeline)
NUM_CORES  = 16

# Wall-clock estimate only: seconds per sightline per worker, all cases.
SECONDS_PER_SKEWER = 0.2


# ==============================================================================
# 2. GRID GEOMETRY
# ==============================================================================
def get_box_size(base_path, snap_num):
    """BoxSize (code length units) from the header of the group catalogue."""
    fname = os.path.join(base_path, f"groups_{snap_num:03d}",
                         f"fof_subhalo_tab_{snap_num:03d}.0.hdf5")
    with h5py.File(fname, 'r') as f:
        return float(f['Header'].attrs['BoxSize'])


def build_grid(box_size, n_side):
    """Sightlines of the grid, for the 3 axes.

    Returns arrays with one entry per sightline:
        axis   (0, 1, 2) line-of-sight axis
        grid_i, grid_j  indices on the face (along the two other axes, in order)
        cofm   (N, 3) position, code units; 0 along the line of sight
    skewer_id = axis * n_side^2 + grid_i * n_side + grid_j (= the row index).
    """
    # Cell-centred grid: (k + 0.5) * spacing, k = 0 ... n_side - 1
    coords = (np.arange(n_side) + 0.5) * box_size / n_side
    gi, gj = np.meshgrid(np.arange(n_side), np.arange(n_side), indexing='ij')
    gi, gj = gi.ravel(), gj.ravel()

    axis_list, i_list, j_list, cofm_list = [], [], [], []
    for ax in range(3):
        free = [k for k in range(3) if k != ax]     # the two axes of the face
        cofm = np.zeros((n_side * n_side, 3))
        cofm[:, free[0]] = coords[gi]
        cofm[:, free[1]] = coords[gj]
        axis_list.append(np.full(n_side * n_side, ax))
        i_list.append(gi)
        j_list.append(gj)
        cofm_list.append(cofm)

    return (np.concatenate(axis_list), np.concatenate(i_list),
            np.concatenate(j_list), np.concatenate(cofm_list))


# ==============================================================================
# 3. WORKER FUNCTION
# ==============================================================================
def parse_cases(text):
    """'total:None,noHCD:Forest_only:17.2' -> {name: (mode, threshold)}."""
    cases = {}
    for item in text.split(","):
        parts = [p.strip() for p in item.split(":")]
        if not parts[0]:
            continue
        mode = utils.LEGACY_FILTER_MODES.get(parts[1], parts[1])
        thresh = float(parts[2]) if len(parts) > 2 else None
        cases[parts[0]] = (mode, thresh)
    return cases


def extract_case(ctx, axis_1idx, cofm, mode, log_nhi_thresh, res):
    """One Spectra object (one read of the snapshot) for one case.

    halo_r200=None switches the HCD window off: these are not halo sightlines.
    The star-forming gas model is the one of the halo pipeline.
    log_nhi_thresh sets which features are HCDs for the filtered modes
    (None = the fork's default, 17.2).
    """
    # The fork prints "Interpolation x percent done" for mode "None" even with
    # quiet=True: discard the worker's stdout, so that only the tqdm bar shows
    with contextlib.redirect_stdout(io.StringIO()):
        return _extract_case(ctx, axis_1idx, cofm, mode, log_nhi_thresh, res)


def _extract_case(ctx, axis_1idx, cofm, mode, log_nhi_thresh, res):
    ss = spectra.Spectra(
        int(ctx.snap_num), ctx.base_path, cofm=cofm, axis=axis_1idx,
        res=res,
        savefile="unused_grid.hdf5",       # never written
        reload_file=True,
        kernel=None,                       # Arepo -> tophat, as the halo pipeline
        MPI=None,
        quiet=True,
        filter_mode=mode,
        halo_r200=None,
        units=ctx.units,
        **utils.SPECTRA_GAS_KWARGS,
    )
    # Threshold of the HCD selection. Must be set BEFORE the first get_*:
    # the mask is computed (and cached) on the first call.
    if mode != "None" and log_nhi_thresh is not None:
        ss.hcd_selection_thresh = 10.0**log_nhi_thresh
    # tau first: in the filtered modes it caches the particles and the HCD
    # mask, which colden and temp then reuse
    tau    = ss.get_tau("H", 1, 1215)
    colden = ss.get_col_density("H", 1)
    temp   = ss.get_temp("H", 1)
    return ss, tau, colden, temp


def snapshot_meta(ss, ctx, cases):
    """Grids and constants shared by every sightline (go to df.attrs)."""
    nbins, vmax = int(ss.nbins), float(ss.vmax)
    # Native box velocity axis (no halo centring): pixel k starts at k * dvbin
    v_grid = np.arange(nbins) * float(ss.dvbin)
    return {
        'v_grid': v_grid,
        'wave_grid': utils.LYA_REST_ANG * np.exp(v_grid / utils.C_LIGHT_KMS),
        'chi_grid': v_grid / ss.velfac,
        'nbins': nbins,
        'vmax': vmax,
        'dvbin': float(ss.dvbin),
        'velfac': float(ss.velfac),
        'box_size': float(ss.box),
        'redshift': float(ss.red),
        'lam_rest': utils.LYA_REST_ANG,
        # n_HI [cm^-3] = n_hi_col [cm^-2] / pixel_length_cm
        'pixel_length_cm': float(ss.dvbin / ss.velfac * ss.rscale),
        'hcd_log_nhi_cut': float(utils.HCD_LOG_NHI),
        # Column suffixes (case names) and what each case is
        'modes': tuple(cases),
        'cases': {name: {'filter_mode': mode, 'log_nhi_thresh': thr}
                  for name, (mode, thr) in cases.items()},
        # Extraction settings, read back from the Spectra object
        'hcd_algorithm': ss.hcd_algorithm,
        'halo_window': False,              # halo_r200=None: no HCD window
        'sf_neutral': bool(ss.gasprop.sf_neutral),
        'sf_criterion': ss.gasprop.sf_criterion,
        'sf_temp': ss.gasprop.sf_temp,
        'h2_model': ss.gasprop.h2_model,
        'length_unit_kpc': ctx.length_unit_kpc,
        'velocity_frame': "box",           # v_grid = 0 ... vmax, not halo-centred
    }


def process_grid_chunk(args):
    """Worker: all cases for one chunk of grid sightlines.

    Receives (chunk_id, grid, cfg): grid holds this chunk's slice of the grid
    (rows = indices in the full grid, axis, grid_i, grid_j, cofm); cfg carries
    the rest (fork/spawn safe, as in run_parallel_illustris.py).
    Returns (chunk_id, records, meta, status).
    """
    chunk_id, grid, cfg = args
    cases = cfg['cases']

    try:
        ctx  = utils.SimulationContext(cfg['base_path'], cfg['snap_num'],
                                       sim_type=cfg['sim_type'])
        rows = grid['rows']
        axis = grid['axis']
        cofm = grid['cofm'].astype(np.float64)
        axis_1idx = (axis + 1).astype(np.int32)    # fake_spectra axes are 1..3

        # One record per sightline: geometry first
        records = []
        for k, r in enumerate(rows):
            records.append({
                'skewer_id': int(r),
                'axis': int(axis[k]),
                'grid_i': int(grid['grid_i'][k]),
                'grid_j': int(grid['grid_j'][k]),
                'pos_x': float(cofm[k, 0]),
                'pos_y': float(cofm[k, 1]),
                'pos_z': float(cofm[k, 2]),
            })

        meta = None
        for name, (mode, thresh) in cases.items():
            ss, tau, colden, temp = extract_case(ctx, axis_1idx, cofm, mode, thresh,
                                                 cfg['res'])
            # The threshold really used by this Spectra object
            if mode != "None":
                assert np.isclose(ss.hcd_selection_thresh,
                                  10.0**(thresh if thresh is not None
                                                  else spectra.HCD_SELECTION_LOGNHI))
            if meta is None:
                meta = snapshot_meta(ss, ctx, cases)
            sfx = f"_{name}"        # always suffixed by the case name
            for k, rec in enumerate(records):
                tot = float(np.sum(colden[k]))
                log_n = np.log10(tot) if tot > 0 else 0.0
                rec[f'log_n_hi{sfx}'] = log_n
                rec[f'is_hcd{sfx}']   = bool(log_n >= utils.HCD_LOG_NHI)
                rec[f'flux{sfx}']     = np.exp(-tau[k]).astype(np.float32)
                rec[f'n_hi_col{sfx}'] = colden[k].astype(np.float32)
                rec[f'temp{sfx}']     = temp[k].astype(np.float32)
            del ss, tau, colden, temp

    except Exception as e:
        logging.error(f"Chunk {chunk_id}: {str(e)}", exc_info=True)
        return (chunk_id, [], None, "ERROR")

    return (chunk_id, records, meta, "SUCCESS")


# ==============================================================================
# 4. MAIN BLOCK
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract Lyman-alpha skewers on a uniform grid of the TNG box.")
    parser.add_argument("--sim", type=str, default=None,
                        help=f"Simulation label. Default: {SIM_NAME}.")
    parser.add_argument("--base_path", type=str, default=None,
                        help="Root holding groups_XXX/ and snapdir_XXX/. "
                             "Default: /home/STORAGE/<sim>.")
    parser.add_argument("--sim_type", type=str, default=None, choices=["TNG", "MN5"],
                        help=f"Unit system. Default: {SIM_TYPE}.")
    parser.add_argument("--snap", type=int, default=None,
                        help=f"Snapshot number. Default: {SNAP_NUM}.")
    parser.add_argument("--cases", type=str, default=None,
                        help="Comma-separated name:mode[:log_nhi_thresh], e.g. "
                             "'total:None,noHCD:Forest_only:17.2,noDLA:Forest_only:20.3' "
                             "(the default).")
    parser.add_argument("--n_side", type=int, default=None,
                        help=f"Sightlines per side of each face. Default: {N_SIDE}.")
    parser.add_argument("--chunk_size", type=int, default=None,
                        help=f"Sightlines per task. Default: {CHUNK_SIZE}.")
    parser.add_argument("--res", type=float, default=None,
                        help=f"Pixel width, km/s. Default: {RES_KMS}.")
    parser.add_argument("--cores", type=int, default=None,
                        help=f"Worker processes. Default: {NUM_CORES}.")
    parser.add_argument("--seconds_per_skewer", type=float, default=None,
                        help="Only for the wall-clock estimate. "
                             f"Default: {SECONDS_PER_SKEWER}.")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Directory for the output .pkl (created if missing). "
                             "Default: <project root>/notebooks_grid.")
    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # Apply overrides (--sim moves BASE_PATH unless --base_path is given)
    # --------------------------------------------------------------------------
    if args.sim is not None:
        SIM_NAME  = args.sim
        BASE_PATH = f"/home/STORAGE/{SIM_NAME}"
    if args.base_path is not None:
        BASE_PATH = os.path.abspath(os.path.expanduser(args.base_path))
    if args.sim_type is not None:
        SIM_TYPE = args.sim_type
    if args.snap is not None:
        SNAP_NUM = args.snap
    if args.cases is not None:
        CASES = parse_cases(args.cases)
    if args.n_side is not None:
        N_SIDE = args.n_side
    if args.chunk_size is not None:
        CHUNK_SIZE = args.chunk_size
    if args.res is not None:
        RES_KMS = args.res
    if args.cores is not None:
        NUM_CORES = args.cores
    if args.seconds_per_skewer is not None:
        SECONDS_PER_SKEWER = args.seconds_per_skewer

    bad = [m for m, _ in CASES.values() if m not in spectra.FILTER_MODES]
    if bad or not CASES:
        parser.error(f"unknown mode(s) {bad}; valid: {spectra.FILTER_MODES}")
    if N_SIDE < 1 or CHUNK_SIZE < 1:
        parser.error("--n_side and --chunk_size must be >= 1")

    # --------------------------------------------------------------------------
    # Grid and tasks
    # --------------------------------------------------------------------------
    BOX_SIZE = get_box_size(BASE_PATH, SNAP_NUM)
    AXIS, GRID_I, GRID_J, COFM = build_grid(BOX_SIZE, N_SIDE)
    n_skewers = len(AXIS)

    # Everything the worker needs, shipped explicitly (fork/spawn safe).
    WORKER_CFG = {
        'base_path': BASE_PATH,
        'snap_num' : SNAP_NUM,
        'sim_type' : SIM_TYPE,
        'cases'    : CASES,
        'res'      : RES_KMS,
    }
    # Consecutive rows of the grid: neighbouring sightlines of the same face,
    # so each Spectra object only keeps the cells of one slab of the box.
    # Each task gets only its own slice of the grid.
    chunk_args = []
    for cid, i in enumerate(range(0, n_skewers, CHUNK_SIZE)):
        sl = slice(i, min(i + CHUNK_SIZE, n_skewers))
        grid = {'rows': np.arange(n_skewers)[sl], 'axis': AXIS[sl],
                'grid_i': GRID_I[sl], 'grid_j': GRID_J[sl], 'cofm': COFM[sl]}
        chunk_args.append((cid, grid, WORKER_CFG))
    n_chunks = len(chunk_args)
    est_mins = n_skewers * SECONDS_PER_SKEWER / NUM_CORES / 60.0

    TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJ_ROOT = os.path.dirname(TOOLS_DIR)
    OUTPUT_DIR = (os.path.abspath(os.path.expanduser(args.outdir))
                  if args.outdir is not None
                  else os.path.join(PROJ_ROOT, "notebooks_grid"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # Header printout
    # --------------------------------------------------------------------------
    print(f"\n{TerminalUI.CYAN}{TerminalUI.BOLD}{'='*70}{TerminalUI.RESET}")
    print(f"{TerminalUI.CYAN}{TerminalUI.BOLD}"
          f"   ILLUSTRIS TNG — UNIFORM-GRID SKEWER EXTRACTION{TerminalUI.RESET}")
    print(f"{TerminalUI.CYAN}{TerminalUI.BOLD}{'='*70}{TerminalUI.RESET}")
    print(f"{TerminalUI.BOLD}--> Simulation       :{TerminalUI.RESET} "
          f"{TerminalUI.GREEN}{SIM_NAME}  (Snap {SNAP_NUM}, {SIM_TYPE}){TerminalUI.RESET}")
    print(f"{TerminalUI.BOLD}--> Base path        :{TerminalUI.RESET} {BASE_PATH}")
    print(f"{TerminalUI.BOLD}--> Box size         :{TerminalUI.RESET} {BOX_SIZE:.2f} code units")
    print(f"{TerminalUI.BOLD}--> Grid             :{TerminalUI.RESET} "
          f"{N_SIDE}x{N_SIDE} per face x 3 axes = {n_skewers:,} skewers "
          f"(spacing {BOX_SIZE / N_SIDE:.2f} code units)")
    for name, (mode, thr) in CASES.items():
        print(f"{TerminalUI.BOLD}--> Case {name:<12}:{TerminalUI.RESET} "
              f"{mode}" + ("" if thr is None else f", log N_HI >= {thr}"))
    print(f"{TerminalUI.BOLD}--> Pixel width      :{TerminalUI.RESET} {RES_KMS} km/s")
    print(f"{TerminalUI.BOLD}--> Parallel Cores   :{TerminalUI.RESET} {NUM_CORES}")
    print(f"{TerminalUI.BOLD}--> Chunks           :{TerminalUI.RESET} "
          f"{n_chunks} of up to {CHUNK_SIZE} skewers")
    print(f"{TerminalUI.BOLD}--> Output dir       :{TerminalUI.RESET} {OUTPUT_DIR}")
    print(f"{TerminalUI.WARNING}{TerminalUI.BOLD}"
          f"--> Estimated Time   : ~{est_mins:.1f} min{TerminalUI.RESET}")
    print(f"{TerminalUI.CYAN}{'-'*70}{TerminalUI.RESET}\n")

    # --------------------------------------------------------------------------
    # Parallel extraction
    # --------------------------------------------------------------------------
    all_data      = []
    run_meta      = None
    success_count = 0
    fail_count    = 0
    start_time    = time.time()

    bar_fmt = (
        f"{TerminalUI.GREEN}{{l_bar}}{{bar:40}}{TerminalUI.RESET}"
        f"| {{n_fmt}}/{{total_fmt}} chunks "
        f"[{{elapsed}}<{{remaining}}, {{rate_fmt}}{{postfix}}]"
    )

    with multiprocessing.Pool(processes=NUM_CORES) as pool:
        with tqdm(total=n_chunks, bar_format=bar_fmt) as pbar:
            for cid, records, meta, status in pool.imap_unordered(process_grid_chunk,
                                                                  chunk_args):
                if status == "SUCCESS" and records:
                    all_data.extend(records)
                    if run_meta is None:
                        run_meta = meta
                    success_count += len(records)
                else:
                    fail_count += 1
                pbar.set_postfix({"OK skewers": f"{success_count:,}",
                                  "Err chunks": fail_count})
                pbar.update(1)

    # --------------------------------------------------------------------------
    # Save
    # --------------------------------------------------------------------------
    print(f"\n\n{TerminalUI.CYAN}{'-'*70}{TerminalUI.RESET}")
    print(f"{TerminalUI.BOLD}--> Assembling DataFrame...{TerminalUI.RESET}")

    df = pd.DataFrame(all_data)
    del all_data

    if df.empty:
        print(f"\n{TerminalUI.FAIL}{TerminalUI.BOLD}"
              f"[X] No data produced. Check '{LOG_FILE}'.{TerminalUI.RESET}")
        sys.exit(1)

    df = df.sort_values(by='skewer_id').reset_index(drop=True)

    # Shared grids and snapshot constants: stored once, not per row
    df.attrs.update(run_meta)
    df.attrs['sim_name']     = SIM_NAME
    df.attrs['snap_num']     = SNAP_NUM
    df.attrs['n_side']       = N_SIDE
    df.attrs['grid_spacing'] = BOX_SIZE / N_SIDE      # code length units
    df.attrs['n_expected']   = n_skewers              # < len(df) if chunks failed

    output_filename = f"{SIM_NAME}_z3_Grid{N_SIDE}x{N_SIDE}x3_{'_'.join(CASES)}.pkl"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    print(f"{TerminalUI.BOLD}--> Saving to: {output_path}{TerminalUI.RESET}")
    df.to_pickle(output_path)

    elapsed = (time.time() - start_time) / 60.0
    print(f"\n{TerminalUI.GREEN}{TerminalUI.BOLD}"
          f"[OK] Grid extraction finished in {elapsed:.1f} min.{TerminalUI.RESET}")
    print(f"{TerminalUI.BOLD}    File        :{TerminalUI.RESET} {output_filename}")
    print(f"{TerminalUI.BOLD}    Rows        :{TerminalUI.RESET} {len(df):,} / {n_skewers:,}")
    print(f"{TerminalUI.BOLD}    Size        :{TerminalUI.RESET} "
          f"{os.path.getsize(output_path) / 1e9:.2f} GB")
    print(f"{TerminalUI.BOLD}    Columns     :{TerminalUI.RESET} {list(df.columns)}")
    print(f"{TerminalUI.BOLD}    df.attrs    :{TerminalUI.RESET} {sorted(df.attrs)}")
    if fail_count > 0:
        print(f"{TerminalUI.WARNING}{TerminalUI.BOLD}"
              f"[!] {fail_count} chunks failed ({n_skewers - len(df):,} skewers missing). "
              f"See '{LOG_FILE}'.{TerminalUI.RESET}")
        sys.exit(1)
