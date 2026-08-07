# -*- coding: utf-8 -*-
"""
run_parallel_grid_TNG.py

Extracts Lyman-alpha skewers on a uniform 2D grid across the full simulation
box for IllustrisTNG.

By default runs in TWO modes:
  1. "DLAs"   — same region-growing + fallback as the halo-driven pipeline,
                 ensuring apple-to-apple comparison with run_parallel_illustris.py
  2. "None"   — raw unfiltered IGM for general statistics (flux PDF, etc.)

Temperature (HI-density-weighted, in Kelvin) is included in the output.
No spectral moments or kinematic centering are computed — only raw grids.

Usage:
    python run_parallel_grid_TNG.py
    python run_parallel_grid_TNG.py --sim TNG50-3 --n_side 150 --cores 20
    python run_parallel_grid_TNG.py --modes DLAs None No_DLAs
"""

import argparse
import os
import sys
import time
import logging
import multiprocessing
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Import project modules
# ---------------------------------------------------------------------------
try:
    import utils
    import fake_spectra.spectra as spectra
    import fake_spectra.unitsystem as unitsystem
except ImportError:
    sys.path.append(os.path.abspath("."))
    import utils
    import fake_spectra.spectra as spectra
    import fake_spectra.unitsystem as unitsystem


# ===========================================================================
# Terminal colours
# ===========================================================================
class TerminalUI:
    CYAN    = '\033[96m'
    GREEN   = '\033[92m'
    WARNING = '\033[93m'
    FAIL    = '\033[91m'
    RESET   = '\033[0m'
    BOLD    = '\033[1m'


# ===========================================================================
# Logging (errors only, to file)
# ===========================================================================
LOG_FILE = "extraction_grid_tng.log"
logging.basicConfig(
    filename=LOG_FILE, filemode='w', level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


# ===========================================================================
# 1.  DEFAULT CONFIGURATION
# ===========================================================================
SNAP_NUM        = 25            # z ≈ 3.0 in IllustrisTNG
SIM_NAME        = "TNG50-3"
BASE_PATH       = f"/home/STORAGE/{SIM_NAME}"
SIM_TYPE        = "TNG"

N_SIDE_DEFAULT  = 500           # 500×500 = 250 000 skewers per axis
CHUNK_SIZE      = 2000          # skewers handed to fake_spectra at once
NUM_CORES       = 20

# Default modes: "DLAs" for halo-pipeline-consistent comparison,
# "None" for raw IGM statistics.
# "No_DLAs" available but not run by default (expensive third pass).
FILTER_MODES    = ["DLAs"]

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(TOOLS_DIR)


# ===========================================================================
# 2.  HELPER FUNCTIONS
# ===========================================================================
def get_box_size(base_path, snap_num):
    """Read BoxSize from the first group-catalog (or snapshot) chunk."""
    sn = f"{snap_num:03d}" if isinstance(snap_num, int) else snap_num
    candidates = [
        os.path.join(base_path, f"groups_{sn}", f"fof_subhalo_tab_{sn}.0.hdf5"),
        os.path.join(base_path, f"snapdir_{sn}", f"snap_{sn}.0.hdf5"),
    ]
    for c in candidates:
        if os.path.exists(c):
            with h5py.File(c, 'r') as f:
                return float(f['Header'].attrs['BoxSize'])
    raise FileNotFoundError(
        f"Could not find a header file in {base_path} for snap {snap_num}"
    )


def generate_grid_skewers(box_size, n_side):
    """
    Build a uniform 2-D grid of sightlines for each of the three Cartesian
    axes.  Returns a flat list of dicts, each fully specifying one skewer.

    For axis=1 (propagation along X), the skewer position is defined by
    (Y, Z) on the grid; the X coordinate is set to NaN because fake_spectra
    integrates the full box length along the propagation axis and ignores it.
    """
    spacing = box_size / n_side
    c1d = np.linspace(spacing / 2.0, box_size - spacing / 2.0, n_side)
    ga, gb = np.meshgrid(c1d, c1d)
    ga, gb = ga.ravel(), gb.ravel()
    n_per_axis = n_side * n_side

    skewers = []
    sid = 0

    # axis 1 → propagates along X; grid lives in (Y, Z)
    for i in range(n_per_axis):
        skewers.append({
            'skewer_idx': sid, 'axis': 1,
            'pos_x_code': np.nan,
            'pos_y_code': ga[i],
            'pos_z_code': gb[i],
        })
        sid += 1

    # axis 2 → propagates along Y; grid lives in (X, Z)
    for i in range(n_per_axis):
        skewers.append({
            'skewer_idx': sid, 'axis': 2,
            'pos_x_code': ga[i],
            'pos_y_code': np.nan,
            'pos_z_code': gb[i],
        })
        sid += 1

    # axis 3 → propagates along Z; grid lives in (X, Y)
    for i in range(n_per_axis):
        skewers.append({
            'skewer_idx': sid, 'axis': 3,
            'pos_x_code': ga[i],
            'pos_y_code': gb[i],
            'pos_z_code': np.nan,
        })
        sid += 1

    return skewers, n_per_axis


def _build_cofm_array(skewers_chunk, box_size):
    """
    Convert a list of skewer dicts into the (N, 3) cofm array that
    fake_spectra expects.  NaN entries (the propagation-axis coordinate)
    are replaced by box_size/2 — a dummy value the interpolator ignores.
    """
    n = len(skewers_chunk)
    cofm = np.zeros((n, 3), dtype=np.float64)
    axis = np.zeros(n, dtype=np.int32)

    for i, sk in enumerate(skewers_chunk):
        x = sk['pos_x_code']
        y = sk['pos_y_code']
        z = sk['pos_z_code']
        cofm[i, 0] = box_size / 2.0 if np.isnan(x) else x
        cofm[i, 1] = box_size / 2.0 if np.isnan(y) else y
        cofm[i, 2] = box_size / 2.0 if np.isnan(z) else z
        axis[i] = sk['axis']

    return cofm, axis


def _get_unit_system(sim_type):
    """Return the appropriate UnitSystem for the simulation type."""
    if sim_type == "TNG":
        return unitsystem.UnitSystem(
            UnitLength_in_cm=3.085678e21,
            UnitMass_in_g=1.989e43,
            UnitVelocity_in_cm_per_s=1e5,
        )
    elif sim_type == "MN5":
        return unitsystem.UnitSystem(
            UnitLength_in_cm=3.085678e24,
            UnitMass_in_g=1.989e43,
            UnitVelocity_in_cm_per_s=1e5,
        )
    return None


# ===========================================================================
# 3.  WORKER FUNCTION
# ===========================================================================
def process_grid_chunk(args):
    """
    Worker: extract tau, column density, and temperature for one chunk of
    grid skewers under a specific filter_mode.

    CRITICAL DESIGN NOTE:
        When filter_mode="DLAs", the spectra.py code applies region-growing
        to each LOS independently.  If NO HCD structure is found along a
        given LOS, the fallback logic interpolates ALL particles (identical
        to filter_mode="None").  This ensures bit-for-bit consistency with
        the halo-driven pipeline (run_parallel_illustris.py), where the same
        fallback applies to skewers at large impact parameters that do not
        intersect any HCD.

    Parameters
    ----------
    args : tuple
        (chunk_id, skewers_chunk, box_size, filter_mode)

    Returns
    -------
    (chunk_id, list_of_dicts, status_string)
    """
    chunk_id, skewers_chunk, box_size, filter_mode = args
    n_skewers = len(skewers_chunk)

    ctx = utils.SimulationContext(BASE_PATH, SNAP_NUM, sim_type=SIM_TYPE)
    cofm_array, axis_array = _build_cofm_array(skewers_chunk, box_size)
    units = _get_unit_system(ctx.sim_type)

    save_file = f"temp_grid_{chunk_id}_{filter_mode}_{np.random.randint(100000)}.hdf5"

    try:
        # -------------------------------------------------------------------
        # Build Spectra object
        # -------------------------------------------------------------------
        # When filter_mode is "None", skip the expensive root_scalar
        # threshold computation by passing a dummy numeric value.
        dla_thresh = 0.001 if filter_mode == "None" else "auto"

        ss = spectra.Spectra(
            int(ctx.snap_num),
            ctx.base_path,
            cofm=cofm_array,
            axis=axis_array,
            res=10.0,
            savefile=save_file,
            reload_file=True,
            kernel=None,
            MPI=None,
            quiet=True,
            filter_mode=filter_mode,
            dla_phys_thresh=dla_thresh,
            units=units,
        )

        # -------------------------------------------------------------------
        # Extract physical quantities
        # -------------------------------------------------------------------
        tau_batch     = ss.get_tau("H", 1, 1215)
        col_den_batch = ss.get_col_density("H", 1)
        temp_batch    = ss.get_temp("H", 1)       # HI-density-weighted T [K]

        vmax   = ss.vmax
        nbins  = ss.nbins
        lam_rest = 1215.67
        c_light  = 299792.458

        # Velocity / wavelength / comoving-distance axes (same for all skewers)
        v_axis    = np.arange(nbins, dtype=np.float32) * (vmax / nbins)
        wave_axis = (lam_rest * np.exp(v_axis / c_light)).astype(np.float32)
        chi_axis  = (v_axis / (vmax / box_size)).astype(np.float32)

        # -------------------------------------------------------------------
        # Pack results
        # -------------------------------------------------------------------
        results = []
        for idx in range(n_skewers):
            flux  = np.exp(-tau_batch[idx]).astype(np.float32)
            n_hi  = col_den_batch[idx].astype(np.float32)
            t_k   = temp_batch[idx].astype(np.float32)

            n_hi_sum = float(np.sum(n_hi))

            row = skewers_chunk[idx].copy()
            row['chunk_id']      = chunk_id
            row['filter_mode']   = filter_mode
            row['total_col_den'] = n_hi_sum
            row['log_n_hi']      = np.log10(n_hi_sum) if n_hi_sum > 0 else 0.0
            row['flux']          = flux
            row['n_hi_col']      = n_hi
            row['temp_k']        = t_k
            row['wave']          = wave_axis
            row['vel_axis']      = v_axis
            row['chi_axis']      = chi_axis
            results.append(row)

    except Exception as e:
        logging.error(f"Chunk {chunk_id} [{filter_mode}]: {e}", exc_info=True)
        return (chunk_id, [], "ERROR")

    finally:
        if os.path.exists(save_file):
            os.remove(save_file)

    return (chunk_id, results, "SUCCESS")


# ===========================================================================
# 4.  MAIN
# ===========================================================================
if __name__ == "__main__":

    # -----------------------------------------------------------------------
    # CLI overrides
    # -----------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Uniform-grid skewer extraction for IllustrisTNG."
    )
    parser.add_argument("--sim",    type=str, default=None,
                        help="Override SIM_NAME (e.g. TNG50-3).")
    parser.add_argument("--n_side", type=int, default=N_SIDE_DEFAULT,
                        help="Grid side length (n_side x n_side per axis).")
    parser.add_argument("--cores",  type=int, default=NUM_CORES,
                        help="Number of parallel workers.")
    parser.add_argument("--modes",  type=str, nargs='+',
                        default=FILTER_MODES,
                        choices=["None", "DLAs", "No_DLAs"],
                        help="Filter modes to extract.")
    cli = parser.parse_args()

    if cli.sim is not None:
        SIM_NAME  = cli.sim
        BASE_PATH = f"/home/STORAGE/{SIM_NAME}"

    n_side    = cli.n_side
    num_cores = cli.cores
    modes     = cli.modes

    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    # -----------------------------------------------------------------------
    # Read box size and build the master skewer list
    # -----------------------------------------------------------------------
    BOX_SIZE = get_box_size(BASE_PATH, SNAP_NUM)

    master_skewers, n_per_axis = generate_grid_skewers(BOX_SIZE, n_side)
    total_skewers = len(master_skewers)

    # -----------------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------------
    print(f"\n{TerminalUI.CYAN}{TerminalUI.BOLD}{'='*70}{TerminalUI.RESET}")
    print(f"{TerminalUI.CYAN}{TerminalUI.BOLD}"
          f"   UNIFORM-GRID SKEWER EXTRACTION  ({SIM_NAME}, z≈3)"
          f"{TerminalUI.RESET}")
    print(f"{TerminalUI.CYAN}{TerminalUI.BOLD}{'='*70}{TerminalUI.RESET}")
    print(f"{TerminalUI.BOLD}--> Box Size          :{TerminalUI.RESET} "
          f"{BOX_SIZE:.2f} code units (ckpc/h)")
    print(f"{TerminalUI.BOLD}--> Grid              :{TerminalUI.RESET} "
          f"{n_side}×{n_side} = {n_per_axis:,} skewers/axis  "
          f"(spacing ≈ {BOX_SIZE/n_side:.3f} code units)")
    print(f"{TerminalUI.BOLD}--> Total skewers     :{TerminalUI.RESET} "
          f"{total_skewers:,} (3 axes)")
    print(f"{TerminalUI.BOLD}--> Chunk size        :{TerminalUI.RESET} "
          f"{CHUNK_SIZE}")
    print(f"{TerminalUI.BOLD}--> Filter modes      :{TerminalUI.RESET} "
          f"{modes}")
    print(f"{TerminalUI.BOLD}--> Parallel cores    :{TerminalUI.RESET} "
          f"{num_cores}")

    # Explain the comparison logic
    if "DLAs" in modes:
        print(f"\n{TerminalUI.WARNING}{TerminalUI.BOLD}"
              f"    NOTE: 'DLAs' mode uses the same region-growing + fallback\n"
              f"    as run_parallel_illustris.py, ensuring apple-to-apple\n"
              f"    comparison at all impact parameters."
              f"{TerminalUI.RESET}")

    print(f"{TerminalUI.CYAN}{'-'*70}{TerminalUI.RESET}\n")

    # -----------------------------------------------------------------------
    # Loop over filter modes
    # -----------------------------------------------------------------------
    for fmode in modes:

        print(f"\n{TerminalUI.CYAN}{TerminalUI.BOLD}"
              f">>> Starting filter_mode = \"{fmode}\" "
              f"({total_skewers:,} skewers)"
              f"{TerminalUI.RESET}")

        # Build work items for this mode
        chunks = []
        for i in range(0, total_skewers, CHUNK_SIZE):
            chunk_slice = master_skewers[i : i + CHUNK_SIZE]
            cid = i // CHUNK_SIZE
            chunks.append((cid, chunk_slice, BOX_SIZE, fmode))

        n_chunks = len(chunks)
        all_data      = []
        success_count = 0
        fail_count    = 0
        start_time    = time.time()

        bar_fmt = (
            f"{TerminalUI.GREEN}{{l_bar}}{{bar:40}}{TerminalUI.RESET}"
            f"| {{n_fmt}}/{{total_fmt}} chunks "
            f"[{{elapsed}}<{{remaining}}, {{rate_fmt}}{{postfix}}]"
        )

        with multiprocessing.Pool(processes=num_cores) as pool:
            it = pool.imap_unordered(process_grid_chunk, chunks)

            with tqdm(total=n_chunks, bar_format=bar_fmt,
                      desc=f"[{fmode}]") as pbar:
                for cid, result_list, status in it:
                    if status == "SUCCESS" and result_list:
                        all_data.extend(result_list)
                        success_count += len(result_list)
                    else:
                        fail_count += 1

                    pbar.set_postfix({
                        "OK": f"{success_count:,}",
                        "Err": fail_count,
                    })
                    pbar.update(1)

        # -------------------------------------------------------------------
        # Save this mode
        # -------------------------------------------------------------------
        elapsed = (time.time() - start_time) / 60.0

        if all_data:
            df = pd.DataFrame(all_data)
            df = df.sort_values(by=['axis', 'skewer_idx']).reset_index(drop=True)

            mode_tag = {
                "None":    "Raw",
                "DLAs":    "DLAMode",
                "No_DLAs": "ForestOnly",
            }
            tag = mode_tag.get(fmode, fmode)

            fname = f"{SIM_NAME}_z3_Grid_{n_side}x{n_side}_{tag}.pkl"
            fpath = os.path.join(PROJ_ROOT, "notebooks", fname)

            df.to_pickle(fpath)

            print(f"\n{TerminalUI.GREEN}{TerminalUI.BOLD}"
                  f"[✔] {fmode} done in {elapsed:.1f} min  →  {fpath}"
                  f"{TerminalUI.RESET}")
            print(f"    Rows: {len(df):,}   "
                  f"Memory: {df.memory_usage(deep=True).sum()/1e6:.1f} MB")

            if fail_count > 0:
                print(f"{TerminalUI.WARNING}"
                      f"    {fail_count} chunks failed — see {LOG_FILE}"
                      f"{TerminalUI.RESET}")
        else:
            print(f"\n{TerminalUI.FAIL}{TerminalUI.BOLD}"
                  f"[✘] {fmode}: no data produced. Check {LOG_FILE}."
                  f"{TerminalUI.RESET}")

    # -----------------------------------------------------------------------
    print(f"\n{TerminalUI.CYAN}{TerminalUI.BOLD}{'='*70}{TerminalUI.RESET}")
    print(f"{TerminalUI.GREEN}{TerminalUI.BOLD}"
          f"All requested modes completed."
          f"{TerminalUI.RESET}")