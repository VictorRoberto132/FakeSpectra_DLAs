"""Check the fork change "halo_r200 per sightline" (2026-10-25).

Compares, for HCD_only and Forest_only, tau / colden / temp of:
  1. the backup (one R200 per Spectra) vs the new spectra.py, one halo at a
     time with a scalar R200                    -> must be bit-for-bit equal;
  2. the new spectra.py, halos one at a time vs all halos in ONE Spectra
     with one R200 per sightline                -> must agree;
  3. control: all halos in one Spectra with the R200 of the first halo for
     every sightline                            -> should differ (the window matters);
  4. random sightlines, halo_r200=None, backup vs new -> must be bit-for-bit equal.

Example:
    python test_per_skewer_r200.py --sim TNG50-4 --halo_ids 5,50,300 --n_impacts 8
"""
import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np

# Pipeline modules from tools/
PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT / "tools"))
import utils
import radii_skewers
import run_parallel_illustris as rpi

import fake_spectra.spectra as new_mod
old_mod = importlib.import_module("fake_spectra.spectra_backup_2026-10-25_single_r200")


def extract(mod, ctx, cofm, axis, mode, r200):
    """tau, colden, temp of one Spectra object, same kwargs as utils.SpectrumEngine._build_spectra."""
    ss = mod.Spectra(int(ctx.snap_num), ctx.base_path, cofm=cofm, axis=axis,
                     res=10.0, savefile="unused.hdf5", reload_file=True,
                     kernel=None, MPI=None, quiet=True, filter_mode=mode,
                     hcd_phys_thresh="auto", units=ctx.units,
                     halo_r200=r200, halo_radius_cut=10.0)
    return dict(tau=ss.get_tau("H", 1, 1215),
                colden=ss.get_col_density("H", 1),
                temp=ss.get_temp("H", 1))


def compare(a, b):
    """Max |a-b| relative to max |a|, and whether the arrays are identical."""
    scale = max(np.abs(a).max(), 1e-300)
    return np.abs(a - b).max() / scale, np.array_equal(a, b)


def report(label, ref, other):
    # One line per product; returns True if every product is identical
    ok = True
    for key in ("tau", "colden", "temp"):
        rel, same = compare(ref[key], other[key])
        ok &= same
        print(f"  {label:34s} {key:7s} max rel diff = {rel:.2e}   identical = {same}")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sim", default="TNG50-4")
    p.add_argument("--snap", type=int, default=25)
    p.add_argument("--halo_ids", default="5,50,300", help="global FoF ids, comma-separated")
    p.add_argument("--n_impacts", type=int, default=8, help="impact parameters per halo (production: 30)")
    p.add_argument("--n_random", type=int, default=50, help="random sightlines for test 4")
    args = p.parse_args()

    ctx = utils.SimulationContext(f"/home/STORAGE/{args.sim}", args.snap, sim_type="TNG")
    finder = utils.HaloFinder(ctx)
    r_mults = rpi.build_r_multipliers(args.n_impacts, 0.01, 50.0)

    # Skewers of each halo, built exactly as radii_skewers.run_analysis_batched does
    halos = []
    for hid in [int(h) for h in args.halo_ids.split(",")]:
        info = finder.get_halo_properties(hid)
        geom = radii_skewers.SkewerGeometry(ctx, info)
        sk = [s for r in r_mults for s in geom.generate_skewers(r_multiplier=r)]
        cofm = np.array([np.mod(np.asarray(info['cm'], dtype=np.float64) + s['offset_vec'],
                                info['box_size']) for s in sk])
        axis = np.array([s['axis'] + 1 for s in sk], dtype=np.int32)
        halos.append(dict(id=hid, r200=float(info['r200']), cofm=cofm, axis=axis,
                          box=float(info['box_size'])))
        print(f"halo {hid}: R200 = {info['r200']:.2f}, {len(sk)} skewers")

    # Everything in one Spectra: concatenated sightlines, one R200 per sightline
    cofm_all = np.concatenate([h["cofm"] for h in halos])
    axis_all = np.concatenate([h["axis"] for h in halos])
    r200_all = np.concatenate([np.full(len(h["cofm"]), h["r200"]) for h in halos])
    cuts = np.cumsum([0] + [len(h["cofm"]) for h in halos])

    all_ok = True
    for mode in ("HCD_only", "Forest_only"):
        print(f"\n=== {mode} ===")
        t0 = time.time()
        sep_old = [extract(old_mod, ctx, h["cofm"], h["axis"], mode, h["r200"]) for h in halos]
        sep_new = [extract(new_mod, ctx, h["cofm"], h["axis"], mode, h["r200"]) for h in halos]
        t_sep = (time.time() - t0) / 2
        t0 = time.time()
        joint = extract(new_mod, ctx, cofm_all, axis_all, mode, r200_all)
        t_joint = time.time() - t0
        control = extract(new_mod, ctx, cofm_all, axis_all, mode, halos[0]["r200"])
        print(f"  time: {len(halos)} halos one by one = {t_sep:.1f} s, all in one Spectra = {t_joint:.1f} s")

        # Test 1 (must be identical) and test 2 (must agree), halo by halo
        for i, h in enumerate(halos):
            sl = slice(cuts[i], cuts[i + 1])
            print(f" halo {h['id']}")
            all_ok &= report("1. backup vs new (scalar R200)", sep_old[i], sep_new[i])
            report("2. one by one vs joint (R200 array)", sep_new[i],
                   {k: v[sl] for k, v in joint.items()})

        # Test 3: with the wrong R200, the halos after the first should change
        n_changed = sum(not np.array_equal(control["colden"][cuts[i]:cuts[i + 1]],
                                           sep_new[i]["colden"])
                        for i in range(1, len(halos)))
        print(f"  3. control (R200 of halo {halos[0]['id']} for all): "
              f"{n_changed} of {len(halos) - 1} other halos changed")

    # Test 4: random sightlines, no window, backup vs new
    print("\n=== random sightlines, halo_r200=None, Forest_only ===")
    rng = np.random.default_rng(1)
    box = halos[0]["box"]
    cofm_r = rng.uniform(0, box, size=(args.n_random, 3))
    axis_r = rng.integers(1, 4, size=args.n_random).astype(np.int32)
    all_ok &= report("4. backup vs new (no window)",
                     extract(old_mod, ctx, cofm_r, axis_r, "Forest_only", None),
                     extract(new_mod, ctx, cofm_r, axis_r, "Forest_only", None))

    # Wrong-length R200 must be refused
    try:
        extract(new_mod, ctx, cofm_all, axis_all, "HCD_only", r200_all[:-1])
        print("\nwrong-length halo_r200: NOT refused")
        all_ok = False
    except ValueError as e:
        print(f"\nwrong-length halo_r200 refused: {e}")

    print("\nbit-for-bit tests (1, 4):", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
