# -*- coding: utf-8 -*-
"""
run_parallel_illustris.py

Extracts Lyman-alpha skewers at multiple impact parameters around TNG halos,
using stratified mass-bin sampling with controlled randomness for variance
estimation across several independent realizations.

Usage (one call per realization):

    python run_parallel_illustris.py --run_id 0   # seed 42
    python run_parallel_illustris.py --run_id 1   # seed 137
    python run_parallel_illustris.py --run_id 2   # seed 2025
    python run_parallel_illustris.py --run_id 3   # seed 999
    python run_parallel_illustris.py --run_id 4   # seed 31415

Each call produces a separate .pkl file labelled with the run_id and seed.

Every constant in section 1 can be overridden on the command line; the
defaults below reproduce the historical behaviour exactly, so the bare
`--run_id N` calls above are unchanged. The driver script
`instructions/00_run_the_pipeline_illustrisTNG.py` uses those flags to run
the whole set from one config block -- edit that file instead of this one if
you just want to point the pipeline at your own simulation.

    --base_path DIR        snapshot + group catalogue root (default:
                           /home/STORAGE/<sim>)
    --sim NAME             simulation label (default TNG50-4)
    --sim_type {TNG,MN5}   unit system / header quirks
    --snap N               snapshot number (default 25, z~3)
    --run_id N             realization index, used for the label and the seed
    --seed N               explicit seed, overrides the RUN_SEEDS mapping
    --modes LIST           comma-separated decomposition modes
    --mass_bins EDGES      comma-separated log10(M_sun) bin EDGES, e.g.
                           "10.0,10.5,11.0,11.5,12.0,13.0" -> 5 bins
    --n_per_bin N          halos drawn per bin
    --exhaustive_threshold N  bins with fewer halos than this are taken whole
    --n_impacts N          number of log-spaced impact parameters (b=0 extra)
    --impact_min X --impact_max X   impact range in units of R200
    --cores N              worker processes
    --outdir DIR           where the .pkl goes (default: <proj_root>/notebooks)

UNIT CONVENTION:
    IllustrisTNG stores Group_M_Crit200 in units of 10^10 M_sun/h.
    All mass bin edges in this script are defined in log10(M_sun) [physical],
    and are converted to code units using h from the snapshot header.

Bin strategy (Proposal A, the default):
    [10^10.0 - 10^10.5]  ~7,666 halos in TNG50-2  -> sample 150  (rich)
    [10^10.5 - 10^11.0]  ~2,466 halos in TNG50-2  -> sample 150  (rich)
    [10^11.0 - 10^11.5]  ~  698 halos in TNG50-2  -> sample 150  (rich)
    [10^11.5 - 10^12.0]  ~  214 halos in TNG50-2  -> sample 150  (moderate)
    [10^12.0 - 10^13.0]  ~   37 halos in TNG50-2  -> ALL         (exhaustive)

RENAMED (nomenclature only, no numbers or logic touched):
    Filter modes: "DLAs" -> "HCD_only", "No_DLAs" -> "Forest_only",
    "None" unchanged.  Legacy spellings are still accepted on --modes and
    silently normalised.  The output filename tag changed accordingly:
    "DLAMode" -> "HCDMode".  Notebook globs looking for *_DLAMode_* only see
    OLD runs; use *_HCDMode_* for new runs (or *Mode_Stratified_run* for both).

OUTPUT SCHEMA (changed)

    Shared grids (v_grid, wave_grid, chi_grid) and snapshot constants live in
    `df.attrs`, not in every row:

        df = pd.read_pickle(path)
        df.attrs['v_grid']            # (nbins,) halo-centred velocity, km/s
        df.attrs['pixel_length_cm']   # n_HI [cm^-3] = n_hi_col / this

    Dropped, because they were exactly derivable:
        wave, vel_axis, chi_axis      -> df.attrs
        vel_aligned                   -> v_grid - vel_peak_shift
        total_col_den                 -> radii_skewers.total_col_den(log_n_hi)
        eq_width, std_dev, skewness,
        kurtosis, mean_wave, mean_vel,
        vel_coherent                  -> radii_skewers.add_moments(df)
        halo_v200, halo_c200          -> radii_skewers.add_concentration(df)

    Added:
        temp        HI-weighted temperature per pixel [K]
        is_hcd      did the sightline reach the 10^17.2 threshold
        position_tag
        halo_vmax   SubhaloVmax of the CENTRAL subhalo [km/s, physical],
                    resolved via GroupFirstSub across the subhalo chunks.
                    The only non-derivable ingredient of the concentration:
                    c200 = radii_skewers.add_concentration(df), which needs
                    nothing beyond (halo_vmax, halo_mass, halo_r200,
                    redshift, attrs['length_unit_kpc']) -- h cancels.
                    NaN when the halo has no central subhalo.

    Fixed:
        impact_kpc  was `impact_code * 1000`, which assumed Mpc code units.
                    TNG lengths are ckpc/h, so it was 1000x too large.
        halo_r200   was misnamed `halo_r200_mpc`; it is in code length units.
"""

import argparse
import random
import numpy as np
import pandas as pd
import os
import sys
import time
import logging
import multiprocessing
import h5py
from tqdm import tqdm

# --- IMPORT PROJECT MODULES ---
try:
    import utils
    import radii_skewers
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(os.path.abspath("."))
    import utils
    import radii_skewers

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
# LOGGING
# ==============================================================================
LOG_FILE = "extraction_errors_illustris_extended.log"
logging.basicConfig(
    filename=LOG_FILE, filemode='w', level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# ==============================================================================
# 1. GENERAL CONFIGURATION   (every entry has a --flag override)
# ==============================================================================
SNAP_NUM  = 25
SIM_NAME  = "TNG50-4"
BASE_PATH = f"/home/STORAGE/{SIM_NAME}"
SIM_TYPE  = "TNG"

#: Canonical seeds for the 5-realization variance study. Kept so that
#: `--run_id 0..4` reproduces the historical runs bit-for-bit. Higher run_ids
#: get a deterministic seed from _seed_for_run_id (still reproducible).
RUN_SEEDS = {
    0: 42,
    1: 137,
    2: 2025,
    3: 999,
    4: 31415,
}

# Decomposition modes to extract.  They all come out of a SINGLE pass over the
# snapshot, so ("None", "HCD_only", "Forest_only") costs essentially the same
# disk I/O as ("HCD_only",) alone -- only extra C interpolation calls.
#   "HCD_only"    -> only cells assigned to the HCD structure
#                    (formerly called "DLAs")
#   "Forest_only" -> the complementary set (pure forest)
#                    (formerly called "No_DLAs")
#   "None"        -> stock fake_spectra, all cells
# Legacy spellings passed via --modes are silently normalised.
# With more than one mode the columns get a suffix: flux_HCD_only,
# log_n_hi_HCD_only, ...
MODES = ("HCD_only",)

#: Legacy --modes spellings, mapped to the current names (pure aliasing).
_LEGACY_MODES = {"DLAs": "HCD_only", "No_DLAs": "Forest_only"}

# Mass bin edges in log10(M_sun) — PHYSICAL solar masses.
# Conversion to code units (10^10 M_sun/h) happens inside the sampler.
MASS_BINS_LOG = [
    (10.0, 10.5),
    (10.5, 11.0),
    (11.0, 11.5),
    (11.5, 12.0),
    (12.0, 13.0),
]
N_HALOS_PER_BIN      = 150
EXHAUSTIVE_THRESHOLD = 100

# Impact parameter grid
NUM_PARAM_IMPACTS   = 30
RANGE_PARAM_IMPACTS = [0.01, 50.0]

NUM_CORES = 17

# Wall-clock estimate. The extraction is batched: one snapshot read per halo,
# so the cost scales with halos, not with skewers. Calibrate this once by
# timing a single halo and edit the number (or pass --seconds_per_halo).
SECONDS_PER_HALO = 300.0

# Where the .pkl lands. None -> <project root>/notebooks, the historical spot.
OUTPUT_DIR = None


def build_r_multipliers(n_impacts, impact_min, impact_max):
    """Impact-parameter grid: b=0 plus `n_impacts` log-spaced multiples of R200."""
    log_steps = np.logspace(np.log10(impact_min), np.log10(impact_max), n_impacts)
    return np.insert(log_steps, 0, 0.0)


def build_mass_bins(edges_log):
    """Consecutive (low, high) pairs from a monotonic list of log10(M_sun) edges."""
    edges = [float(e) for e in edges_log]
    if len(edges) < 2:
        raise ValueError("--mass_bins needs at least two edges, e.g. '10.0,11.0'")
    if any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError(f"--mass_bins edges must increase strictly, got {edges}")
    return [(lo, hi) for lo, hi in zip(edges[:-1], edges[1:])]


def _seed_for_run_id(run_id):
    """Reproducible seed for any run_id.

    run_id 0..4 keep their historical seeds so old runs can be reproduced;
    beyond that the seed is a fixed deterministic function of the index, so
    two people asking for 10 realizations get the same 10 seeds.
    """
    if run_id in RUN_SEEDS:
        return RUN_SEEDS[run_id]
    return int((2654435761 * (run_id + 1) + 40503) % (2**31 - 1))


R_MULTIPLIERS = build_r_multipliers(NUM_PARAM_IMPACTS, *RANGE_PARAM_IMPACTS)

# ==============================================================================
# 2. STRATIFIED MASS BIN SAMPLER
# ==============================================================================
def get_stratified_halo_indices(base_dir, snap_number, bins_log,
                                n_per_bin, exhaustive_threshold, rng_seed):
    """
    Scan all FoF group catalog chunks, bin halos by M_Crit200, and return
    a randomly sampled (or exhaustive) flat list of global halo IDs.

    UNIT HANDLING:
        Group_M_Crit200 is stored in 10^10 M_sun/h.
        bins_log are in log10(M_sun) [physical].
        We read h from the snapshot header and convert:
            M_code = M_sun * h / 1e10
    """
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    group_dir = os.path.join(base_dir, f"groups_{snap_number:03d}")
    file0     = os.path.join(
        group_dir, f"fof_subhalo_tab_{snap_number:03d}.0.hdf5"
    )
    if not os.path.exists(file0):
        raise FileNotFoundError(
            f"FoF catalog not found at: {file0}\n"
            f"Check BASE_PATH='{base_dir}' and SNAP_NUM={snap_number}."
        )

    with h5py.File(file0, 'r') as f:
        total_halos = int(f['Header'].attrs['Ngroups_Total'])
        num_files   = int(f['Header'].attrs['NumFiles'])
        hubble_h    = float(f['Header'].attrs['HubbleParam'])

    # Convert physical M_sun bin edges to code units: 10^10 M_sun/h
    bins_code = [
        ((10**low) * hubble_h / 1e10, (10**high) * hubble_h / 1e10)
        for low, high in bins_log
    ]

    print(
        f"{TerminalUI.CYAN}Scanning {total_halos:,} halos across "
        f"{num_files} catalog chunk(s)...{TerminalUI.RESET}"
    )
    print(
        f"{TerminalUI.CYAN}  h = {hubble_h}, converting bins to code units "
        f"(10^10 M_sun/h){TerminalUI.RESET}"
    )

    bin_populations = {i: [] for i in range(len(bins_code))}
    current_id_offset = 0

    for i in range(num_files):
        filepath = os.path.join(
            group_dir, f"fof_subhalo_tab_{snap_number:03d}.{i}.hdf5"
        )
        if not os.path.exists(filepath):
            continue
        with h5py.File(filepath, 'r') as f:
            if 'Group' not in f or 'Group_M_Crit200' not in f['Group']:
                continue
            masses     = f['Group']['Group_M_Crit200'][:]
            global_ids = np.arange(
                current_id_offset, current_id_offset + len(masses)
            )
            for bin_idx, (m_low, m_high) in enumerate(bins_code):
                mask = (masses >= m_low) & (masses < m_high)
                bin_populations[bin_idx].extend(global_ids[mask].tolist())
            current_id_offset += len(masses)

    # ------------------------------------------------------------------
    # Sample from each bin
    # ------------------------------------------------------------------
    final_indices = []
    sampling_info = {}

    print(
        f"\n{TerminalUI.BOLD}"
        f"--- Mass Stratification (seed={rng_seed}, h={hubble_h}, "
        f"sim={os.path.basename(base_dir)}) ---"
        f"{TerminalUI.RESET}"
    )

    for bin_idx, (m_low, m_high) in enumerate(bins_log):
        available = len(bin_populations[bin_idx])

        if available <= exhaustive_threshold:
            sampled = list(bin_populations[bin_idx])
            mode    = "exhaustive"
            color   = TerminalUI.CYAN
            tag     = "[EXHAUSTIVE]"
        elif available < n_per_bin:
            sampled = list(bin_populations[bin_idx])
            mode    = "limited"
            color   = TerminalUI.WARNING
            tag     = f"[LIMITED — only {available}]"
        else:
            sampled = random.sample(bin_populations[bin_idx], n_per_bin)
            mode    = "sampled"
            color   = TerminalUI.GREEN
            tag     = ""

        final_indices.extend(sampled)
        sampling_info[bin_idx] = {
            'log_range': (m_low, m_high),
            'available': available,
            'sampled':   len(sampled),
            'mode':      mode,
        }
        print(
            f"  Bin [10^{m_low:.1f} - 10^{m_high:.1f}] M_sun : "
            f"{color}{available:>6,} available -> "
            f"{len(sampled):>4} drawn  {tag}{TerminalUI.RESET}"
        )

    random.shuffle(final_indices)
    return final_indices, total_halos, sampling_info

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================
def estimate_time(n_halos, n_cores, seconds_per_halo=SECONDS_PER_HALO):
    """Rough wall-clock estimate, in minutes."""
    return (n_halos * seconds_per_halo) / n_cores / 60.0

# ==============================================================================
# 4. WORKER FUNCTION
# ==============================================================================
def process_single_halo(args):
    """Worker: extracts all skewers for one halo.

    Receives (halo_id, global_seed, cfg) where cfg carries everything the
    worker needs. The config travels IN the argument tuple rather than through
    module globals so the worker behaves identically under fork and spawn
    start methods (globals rebound in __main__ are not seen by spawned
    children).

    Returns (halo_id, rows, meta, status_string).
    """
    halo_id, global_seed, cfg = args

    worker_seed = int((global_seed * 1000003 + halo_id) % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)

    modes         = cfg['modes']
    r_multipliers = cfg['r_multipliers']

    results_buffer = []
    meta = None
    try:
        ctx         = utils.SimulationContext(cfg['base_path'], cfg['snap_num'],
                                              sim_type=cfg['sim_type'])
        halo_finder = utils.HaloFinder(ctx)
        halo_info   = halo_finder.get_halo_properties(halo_id)

        geom = radii_skewers.SkewerGeometry(ctx, halo_info)

        all_skewers = []
        for r_mult in r_multipliers:
            all_skewers.extend(geom.generate_skewers(r_multiplier=r_mult))

        meta, rows = geom.run_analysis_batched(all_skewers, modes=modes)
        v_grid = meta['v_grid']
        single = (len(modes) == 1)

        for res in rows:
            rec = {
                # --- Halo identification ---
                'halo_id'      : halo_id,
                'halo_mass'    : res['halo_mass'],
                'halo_r200'    : res['halo_r200'],       # code length units
                'halo_vmax'    : res['halo_vmax'],       # km/s (physical), central
                                                         # subhalo; NaN if none.
                                                         # c200 on demand:
                                                         # radii_skewers.add_concentration
                'redshift'     : meta['redshift'],       # kept as a column so that
                                                         # concatenating snapshots is safe
                # --- Skewer geometry ---
                'skewer_id'    : res['skewer_id'],
                'axis'         : res['axis'],
                'position_tag' : res['position_tag'],
                'r_multiplier' : res['r_multiplier'],
                'impact_kpc'   : (res['r_multiplier'] * res['halo_r200']
                                  * meta['length_unit_kpc']),
            }

            for mode in modes:
                sfx = "" if single else f"_{mode}"
                flux = res[f'flux{sfx}']

                # --- Column density ---
                rec[f'log_n_hi{sfx}'] = res[f'log_n_hi{sfx}']
                rec[f'is_hcd{sfx}']   = res[f'is_hcd{sfx}']

                # --- Kinematics: trough closest to the halo rest frame.
                #     The aligned axis is v_grid - vel_peak_shift, so we only
                #     keep the scalar. ---
                rec[f'vel_peak_shift{sfx}'] = radii_skewers.peak_shift(flux, v_grid)

                # --- Native spectral grids (float32 for size) ---
                rec[f'flux{sfx}']     = np.asarray(flux, dtype=np.float32)
                rec[f'n_hi_col{sfx}'] = np.asarray(res[f'n_hi_col{sfx}'], dtype=np.float32)
                if f'temp{sfx}' in res:
                    rec[f'temp{sfx}'] = np.asarray(res[f'temp{sfx}'], dtype=np.float32)

            results_buffer.append(rec)

    except Exception as e:
        logging.error(f"Halo ID {halo_id}: {str(e)}", exc_info=True)
        return (halo_id, [], None, "ERROR")

    return (halo_id, results_buffer, meta, "SUCCESS")

# ==============================================================================
# 5. MAIN BLOCK
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract Lyman-alpha skewers around TNG halos."
    )
    # --- realization ---
    parser.add_argument(
        "--run_id", type=int, default=0,
        help=f"Realization index (label + seed). Canonical seeds: {RUN_SEEDS}; "
             "higher indices get a deterministic seed. Default: 0.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Explicit RNG seed, overriding the run_id -> seed mapping.",
    )
    # --- simulation ---
    parser.add_argument(
        "--sim", type=str, default=None,
        help=f"Simulation label (e.g. --sim TNG50-3). Default: {SIM_NAME}.",
    )
    parser.add_argument(
        "--base_path", type=str, default=None,
        help="Root holding groups_XXX/ and snapdir_XXX/. "
             "Default: /home/STORAGE/<sim>.",
    )
    parser.add_argument(
        "--sim_type", type=str, default=None, choices=["TNG", "MN5"],
        help=f"Unit system / header quirks. Default: {SIM_TYPE}.",
    )
    parser.add_argument(
        "--snap", type=int, default=None,
        help=f"Snapshot number. Default: {SNAP_NUM} (z~3 for TNG50).",
    )
    # --- extraction ---
    parser.add_argument(
        "--modes", type=str, default=None,
        help="Comma-separated decomposition modes, e.g. 'HCD_only,Forest_only,None'. "
             "Legacy names 'DLAs'/'No_DLAs' are accepted and normalised. "
             f"Default: {','.join(MODES)}.",
    )
    parser.add_argument(
        "--mass_bins", type=str, default=None,
        help="Comma-separated log10(M_sun) bin EDGES, e.g. "
             "'10.0,10.5,11.0,11.5,12.0,13.0' (5 bins). "
             "Default: the Proposal A edges.",
    )
    parser.add_argument(
        "--n_per_bin", type=int, default=None,
        help=f"Halos drawn per mass bin. Default: {N_HALOS_PER_BIN}.",
    )
    parser.add_argument(
        "--exhaustive_threshold", type=int, default=None,
        help="Bins with at most this many halos are taken whole. "
             f"Default: {EXHAUSTIVE_THRESHOLD}.",
    )
    parser.add_argument(
        "--n_impacts", type=int, default=None,
        help="Number of log-spaced impact parameters (b=0 is added on top). "
             f"Default: {NUM_PARAM_IMPACTS}.",
    )
    parser.add_argument(
        "--impact_min", type=float, default=None,
        help=f"Smallest impact parameter, in R200. Default: {RANGE_PARAM_IMPACTS[0]}.",
    )
    parser.add_argument(
        "--impact_max", type=float, default=None,
        help=f"Largest impact parameter, in R200. Default: {RANGE_PARAM_IMPACTS[1]}.",
    )
    # --- runtime ---
    parser.add_argument(
        "--cores", type=int, default=None,
        help=f"Worker processes. Default: {NUM_CORES}.",
    )
    parser.add_argument(
        "--seconds_per_halo", type=float, default=None,
        help="Only used for the wall-clock estimate printed at startup. "
             f"Default: {SECONDS_PER_HALO}.",
    )
    parser.add_argument(
        "--outdir", type=str, default=None,
        help="Directory for the output .pkl (created if missing). "
             "Default: <project root>/notebooks.",
    )
    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # Apply overrides. --sim moves BASE_PATH with it unless --base_path is given
    # explicitly, which keeps the historical `--sim TNG50-3` shortcut working.
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
    if args.modes is not None:
        MODES = tuple(_LEGACY_MODES.get(m.strip(), m.strip())
                      for m in args.modes.split(",") if m.strip())
    if args.mass_bins is not None:
        MASS_BINS_LOG = build_mass_bins(
            [e for e in args.mass_bins.split(",") if e.strip()])
    if args.n_per_bin is not None:
        N_HALOS_PER_BIN = args.n_per_bin
    if args.exhaustive_threshold is not None:
        EXHAUSTIVE_THRESHOLD = args.exhaustive_threshold
    if args.n_impacts is not None:
        NUM_PARAM_IMPACTS = args.n_impacts
    if args.impact_min is not None:
        RANGE_PARAM_IMPACTS[0] = args.impact_min
    if args.impact_max is not None:
        RANGE_PARAM_IMPACTS[1] = args.impact_max
    if args.cores is not None:
        NUM_CORES = args.cores
    if args.seconds_per_halo is not None:
        SECONDS_PER_HALO = args.seconds_per_halo

    # Rebuild anything derived from the (possibly overridden) knobs.
    R_MULTIPLIERS = build_r_multipliers(NUM_PARAM_IMPACTS, *RANGE_PARAM_IMPACTS)

    GLOBAL_SEED = args.seed if args.seed is not None else _seed_for_run_id(args.run_id)

    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    # --------------------------------------------------------------------------
    # Stratified sampling
    # --------------------------------------------------------------------------
    ALL_MATCH_INDICES, TOTAL_HALOS_IN_BOX, SAMPLING_INFO = \
        get_stratified_halo_indices(
            base_dir             = BASE_PATH,
            snap_number          = SNAP_NUM,
            bins_log             = MASS_BINS_LOG,
            n_per_bin            = N_HALOS_PER_BIN,
            exhaustive_threshold = EXHAUSTIVE_THRESHOLD,
            rng_seed             = GLOBAL_SEED,
        )

    n_halos  = len(ALL_MATCH_INDICES)
    est_mins = estimate_time(n_halos, NUM_CORES, SECONDS_PER_HALO)

    # Everything the worker needs, shipped explicitly (fork/spawn safe).
    WORKER_CFG = {
        'base_path'    : BASE_PATH,
        'snap_num'     : SNAP_NUM,
        'sim_type'     : SIM_TYPE,
        'modes'        : MODES,
        'r_multipliers': R_MULTIPLIERS,
    }
    worker_args = [(hid, GLOBAL_SEED, WORKER_CFG) for hid in ALL_MATCH_INDICES]

    # --------------------------------------------------------------------------
    # Output location
    # --------------------------------------------------------------------------
    TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJ_ROOT = os.path.dirname(TOOLS_DIR)

    OUTPUT_DIR = (os.path.abspath(os.path.expanduser(args.outdir))
                  if args.outdir is not None
                  else os.path.join(PROJ_ROOT, "notebooks"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # Header printout
    # --------------------------------------------------------------------------
    print(f"\n{TerminalUI.CYAN}{TerminalUI.BOLD}{'='*70}{TerminalUI.RESET}")
    print(f"{TerminalUI.CYAN}{TerminalUI.BOLD}"
          f"   ILLUSTRIS TNG — HCD-MODE SKEWER EXTRACTION"
          f"{TerminalUI.RESET}")
    print(f"{TerminalUI.CYAN}{TerminalUI.BOLD}{'='*70}{TerminalUI.RESET}")
    print(f"{TerminalUI.BOLD}--> Simulation       :{TerminalUI.RESET} "
          f"{TerminalUI.GREEN}{SIM_NAME}  (Snap {SNAP_NUM}, {SIM_TYPE})"
          f"{TerminalUI.RESET}")
    print(f"{TerminalUI.BOLD}--> Base path        :{TerminalUI.RESET} {BASE_PATH}")
    print(f"{TerminalUI.BOLD}--> Run ID / Seed    :{TerminalUI.RESET} "
          f"{args.run_id} / {GLOBAL_SEED}")
    print(f"{TerminalUI.BOLD}--> Decomposition    :{TerminalUI.RESET} "
          f"{', '.join(MODES)}")
    print(f"{TerminalUI.BOLD}--> Bin strategy     :{TerminalUI.RESET} "
          f"{len(MASS_BINS_LOG)} bins, "
          f"{N_HALOS_PER_BIN}/bin, exhaust thresh={EXHAUSTIVE_THRESHOLD}")
    for bin_idx, info in SAMPLING_INFO.items():
        lo, hi = info['log_range']
        avail  = info['available']
        drawn  = info['sampled']
        mode   = info['mode']
        color  = (TerminalUI.CYAN    if mode == "exhaustive" else
                  TerminalUI.WARNING if mode == "limited"    else
                  TerminalUI.GREEN)
        print(f"    [10^{lo:.1f}-10^{hi:.1f}]  "
              f"{color}{avail:>6,} avail -> {drawn:>4} drawn  "
              f"[{mode}]{TerminalUI.RESET}")
    print(f"{TerminalUI.BOLD}--> Total Halos      :{TerminalUI.RESET} {n_halos}")
    print(f"{TerminalUI.BOLD}--> Impact params    :{TerminalUI.RESET} "
          f"{len(R_MULTIPLIERS)} steps  "
          f"(b=0 + {NUM_PARAM_IMPACTS} log-spaced from "
          f"{RANGE_PARAM_IMPACTS[0]} to {RANGE_PARAM_IMPACTS[1]} R200)")
    print(f"{TerminalUI.BOLD}--> Parallel Cores   :{TerminalUI.RESET} {NUM_CORES}")
    print(f"{TerminalUI.BOLD}--> Output dir       :{TerminalUI.RESET} {OUTPUT_DIR}")
    print(f"{TerminalUI.WARNING}{TerminalUI.BOLD}"
          f"--> Estimated Time   : ~{est_mins:.1f} min"
          f"{TerminalUI.RESET}")
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
        f"| {{n_fmt}}/{{total_fmt}} halos "
        f"[{{elapsed}}<{{remaining}}, {{rate_fmt}}{{postfix}}]"
    )

    with multiprocessing.Pool(processes=NUM_CORES) as pool:
        iterator = pool.imap_unordered(process_single_halo, worker_args)

        with tqdm(total=n_halos, bar_format=bar_fmt) as pbar:
            for halo_id, result_list, meta, status in iterator:
                if status == "SUCCESS" and result_list:
                    all_data.extend(result_list)
                    if run_meta is None:
                        # Snapshot-level: identical for every halo.
                        run_meta = meta
                    success_count += 1
                else:
                    fail_count += 1

                pbar.set_postfix({
                    "Last ID": f"{halo_id:<6}",
                    "OK":      success_count,
                    "Err":     fail_count,
                })
                pbar.update(1)

    # --------------------------------------------------------------------------
    # Save
    # --------------------------------------------------------------------------
    print(f"\n\n{TerminalUI.CYAN}{'-'*70}{TerminalUI.RESET}")
    print(f"{TerminalUI.BOLD}--> Assembling DataFrame...{TerminalUI.RESET}")

    df = pd.DataFrame(all_data)

    if not df.empty:
        df = df.sort_values(by=['halo_id', 'r_multiplier', 'skewer_id'])

        # Shared grids and snapshot constants: stored once, not per row.
        # They survive to_pickle/read_pickle and sort_values.
        df.attrs.update(run_meta)
        df.attrs['sim_name'] = SIM_NAME
        df.attrs['snap_num'] = SNAP_NUM
        df.attrs['run_id']   = args.run_id
        df.attrs['seed']     = GLOBAL_SEED

        output_filename = (
            f"{SIM_NAME}_z3_{n_halos}Halos_HCDMode_Stratified"
            f"_run{args.run_id}_seed{GLOBAL_SEED}.pkl"
        )
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        print(f"{TerminalUI.BOLD}--> Saving to: {output_path}{TerminalUI.RESET}")
        df.to_pickle(output_path)

        elapsed = (time.time() - start_time) / 60.0
        print(f"\n{TerminalUI.GREEN}{TerminalUI.BOLD}"
              f"[OK] Pipeline finished in {elapsed:.1f} min."
              f"{TerminalUI.RESET}")
        print(f"{TerminalUI.BOLD}    File        :{TerminalUI.RESET} "
              f"{output_filename}")
        print(f"{TerminalUI.BOLD}    Total Rows  :{TerminalUI.RESET} "
              f"{len(df):,}")
        print(f"{TerminalUI.BOLD}    Memory      :{TerminalUI.RESET} "
              f"{df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
        print(f"{TerminalUI.BOLD}    Halos OK    :{TerminalUI.RESET} "
              f"{success_count}")
        print(f"{TerminalUI.BOLD}    Columns     :{TerminalUI.RESET} "
              f"{list(df.columns)}")
        print(f"{TerminalUI.BOLD}    df.attrs    :{TerminalUI.RESET} "
              f"{sorted(df.attrs)}")
        if fail_count > 0:
            print(f"{TerminalUI.WARNING}{TerminalUI.BOLD}"
                  f"[!] {fail_count} halos failed. "
                  f"See '{LOG_FILE}'.{TerminalUI.RESET}")
    else:
        print(f"\n{TerminalUI.FAIL}{TerminalUI.BOLD}"
              f"[X] No data produced. Check '{LOG_FILE}'."
              f"{TerminalUI.RESET}")
        sys.exit(1)