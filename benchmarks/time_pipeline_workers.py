"""Time the production pipeline worker with N parallel processes.

Runs tools/run_parallel_illustris.process_single_halo -- the exact function the
pipeline maps over its halos -- on a list of halos with a multiprocessing.Pool
of N workers, and writes a .json with the wall time, the time of each halo and
the peak memory. The extracted rows are thrown away.

With --halos_per_task K > 1 it runs process_halo_group instead, on groups of K
consecutive halos (one snapshot read per group), as the pipeline does with
--halos_per_task K. The time of a halo is then the time of its group.

Example:
    python time_pipeline_workers.py --sim TNG50-2 --halo_ids halos.npy \
        --n_workers 8 --omp_threads 1 --out N8_omp1.json

--omp_threads 0 leaves OMP_NUM_THREADS unset, i.e. the pipeline as it runs
today (every worker opens one OpenMP thread per machine core).
"""
import argparse
import os
import sys

# OMP_NUM_THREADS must be set before numpy / fake_spectra are imported,
# so parse it by hand first.
_omp = next((a.split("=")[1] for a in sys.argv if a.startswith("--omp_threads=")), None)
if _omp is None and "--omp_threads" in sys.argv:
    _omp = sys.argv[sys.argv.index("--omp_threads") + 1]
if _omp is not None and int(_omp) > 0:
    os.environ["OMP_NUM_THREADS"] = _omp

import json
import multiprocessing
import threading
import time
from pathlib import Path

import numpy as np
import psutil
from tqdm import tqdm

# Import the pipeline module from tools/
PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT / "tools"))
import run_parallel_illustris as rpi


def timed_halo(args):
    """Run one halo with the pipeline worker; return its status and timing."""
    t0 = time.time()
    halo_id, rows, _, status = rpi.process_single_halo(args)
    return dict(halo_id=int(halo_id), status=status, n_rows=len(rows),
                start=t0, end=time.time())


def timed_group(args):
    """Run one group of halos with the grouped worker; one timing entry per halo."""
    t0 = time.time()
    results = rpi.process_halo_group(args)
    t1 = time.time()
    return [dict(halo_id=int(halo_id), status=status, n_rows=len(rows),
                 start=t0, end=t1) for halo_id, rows, _, status in results]


def memory_sampler(stop, out, dt=2.0):
    """Track the peak of system 'used' memory (no page cache) while running."""
    base = psutil.virtual_memory().used
    peak = 0
    while not stop.is_set():
        peak = max(peak, psutil.virtual_memory().used - base)
        time.sleep(dt)
    out["peak_mem_gb"] = peak / 1e9


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sim", required=True)
    p.add_argument("--snap", type=int, default=25)
    p.add_argument("--halo_ids", required=True, help=".npy with the global FoF ids to run")
    p.add_argument("--n_workers", type=int, required=True, help="processes in the Pool (= --cores of the pipeline)")
    p.add_argument("--omp_threads", type=int, default=1, help="OpenMP threads per worker; 0 = unset (pipeline default)")
    p.add_argument("--halos_per_task", type=int, default=1,
                   help="halos per task (= --halos_per_task of the pipeline); 1 = one halo per task")
    p.add_argument("--out", required=True, help="output .json")
    args = p.parse_args()

    # Same worker configuration as the production runs (run_config.json):
    # HCD_only, b = 0 plus 30 log-spaced impact parameters from 0.01 to 50 R200.
    cfg = {
        "base_path": f"/home/STORAGE/{args.sim}",
        "snap_num": args.snap,
        "sim_type": "TNG",
        "modes": ("HCD_only",),
        "r_multipliers": rpi.build_r_multipliers(30, 0.01, 50.0),
    }
    halo_ids = np.load(args.halo_ids).astype(int)
    worker_args = [(hid, 42, cfg) for hid in halo_ids]
    k = args.halos_per_task
    group_args = [([int(h) for h in halo_ids[i:i + k]], 42, cfg)
                  for i in range(0, len(halo_ids), k)]

    # Peak memory, sampled in a background thread
    mem = {}
    stop = threading.Event()
    sampler = threading.Thread(target=memory_sampler, args=(stop, mem))
    sampler.start()

    t0 = time.time()
    with multiprocessing.Pool(processes=args.n_workers) as pool:
        # Progress over tasks (shown when the script runs in a terminal)
        if k == 1:
            halos = list(tqdm(pool.imap_unordered(timed_halo, worker_args),
                              total=len(worker_args), desc=f"N={args.n_workers} halos"))
        else:
            groups = list(tqdm(pool.imap_unordered(timed_group, group_args),
                               total=len(group_args), desc=f"N={args.n_workers} groups of {k}"))
            halos = [h for g in groups for h in g]
    wall = time.time() - t0

    stop.set()
    sampler.join()

    result = dict(sim=args.sim, n_workers=args.n_workers, omp_threads=args.omp_threads,
                  halos_per_task=k, n_halos=len(halo_ids), wall=wall, halos=halos, **mem)
    Path(args.out).write_text(json.dumps(result, indent=1))
    n_ok = sum(h["status"] == "SUCCESS" for h in halos)
    print(json.dumps(dict(n_workers=args.n_workers, omp_threads=args.omp_threads, halos_per_task=k,
                          wall=round(wall, 1), ok=n_ok, failed=len(halos) - n_ok,
                          peak_mem_gb=round(mem["peak_mem_gb"], 1))))


if __name__ == "__main__":
    main()
