"""Check that the grouped pipeline worker gives the same records as one halo per task.

Two steps (the old and new tools/ share module names, so each extraction runs
in its own process):

    # 1. extract: one .pkl of records per call
    python test_halo_groups.py extract --tools_dir ../tools --worker single --out new_single.pkl
    python test_halo_groups.py extract --tools_dir ../tools --worker group  --out new_group.pkl
    python test_halo_groups.py extract --tools_dir <copy of old tools> --worker single --out old_single.pkl

    # 2. compare every column of every record
    python test_halo_groups.py compare old_single.pkl new_single.pkl new_group.pkl
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np


def extract(args):
    # Import the worker from the requested tools/ folder
    sys.path.insert(0, str(Path(args.tools_dir).resolve()))
    import run_parallel_illustris as rpi

    cfg = {
        "base_path": f"/home/STORAGE/{args.sim}",
        "snap_num": args.snap,
        "sim_type": "TNG",
        "modes": tuple(args.modes.split(",")),
        "r_multipliers": rpi.build_r_multipliers(args.n_impacts, 0.01, 50.0),
    }
    halo_ids = [int(h) for h in args.halo_ids.split(",")]

    t0 = time.time()
    if args.worker == "single":
        results = [rpi.process_single_halo((hid, 42, cfg)) for hid in halo_ids]
    else:
        results = rpi.process_halo_group((halo_ids, 42, cfg))
    print(f"{args.worker}: {time.time() - t0:.1f} s, status = {[r[3] for r in results]}")

    # Records of each halo, keyed by halo id
    records = {int(hid): rows for hid, rows, _, status in results}
    Path(args.out).write_bytes(pickle.dumps(records))


def compare(args):
    ref_file, *others = args.files
    ref = pickle.loads(Path(ref_file).read_bytes())
    all_ok = True
    for other_file in others:
        other = pickle.loads(Path(other_file).read_bytes())
        print(f"\n{ref_file}  vs  {other_file}")
        assert sorted(ref) == sorted(other), "different halos"
        for hid in sorted(ref):
            a, b = ref[hid], other[hid]
            ok = len(a) == len(b) and len(a) > 0
            n_diff = 0
            for ra, rb in zip(a, b):
                ok &= ra.keys() == rb.keys()
                for key in ra:
                    va, vb = np.asarray(ra[key]), np.asarray(rb[key])
                    same = (np.array_equal(va, vb, equal_nan=True)
                            if va.dtype.kind in "fc" else np.array_equal(va, vb))
                    if not same:
                        n_diff += 1
            ok &= n_diff == 0
            all_ok &= ok
            print(f"  halo {hid}: {len(a)} records, {n_diff} differing fields -> "
                  f"{'identical' if ok else 'DIFFERENT'}")
    print("\nPASS" if all_ok else "\nFAIL")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract")
    e.add_argument("--tools_dir", required=True)
    e.add_argument("--worker", choices=["single", "group"], required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--sim", default="TNG50-4")
    e.add_argument("--snap", type=int, default=25)
    e.add_argument("--halo_ids", default="5,50,300")
    e.add_argument("--n_impacts", type=int, default=30)
    e.add_argument("--modes", default="HCD_only")

    c = sub.add_parser("compare")
    c.add_argument("files", nargs="+", help="reference first")

    args = p.parse_args()
    extract(args) if args.cmd == "extract" else compare(args)


if __name__ == "__main__":
    main()
