"""Extract skewers with a chosen fake_spectra build and time each stage.

The upstream package and the HCD fork are both called `fake_spectra` and both
ship a compiled `_spectra_priv`, so they cannot live in the same Python
process. The notebook therefore runs this script once per configuration, in
a fresh subprocess, and reads back the .npz it writes.

One skewer:   --axis 3 --cofm x y z          (arrays saved as 1D, nbins)
Many skewers: --skewers file.npy, columns (axis, x, y, z)   (arrays saved as 2D)

Timed stages:
    init   -> Spectra(...) constructor (opens the snapshot, sets up grids)
    tau    -> get_tau("H", 1, 1215)   (one pass over the snapshot chunks)
    colden -> get_col_density("H", 1) (another pass)

This script deliberately does NOT import tools/utils.py: utils refuses to
import without the fork, and the upstream run must not depend on it.
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def select_fake_spectra(impl_dir):
    """Make `import fake_spectra` load the package found in impl_dir."""
    # The fork is installed in editable mode. Its import hook lives in
    # sys.meta_path and wins over sys.path, so it must be removed first,
    # otherwise the fork is loaded even when we ask for the upstream build.
    sys.meta_path = [f for f in sys.meta_path
                     if "__editable___fake_spectra" not in getattr(f, "__module__", "")]
    sys.path.insert(0, impl_dir)

    import fake_spectra
    # Fail loudly if the wrong copy was picked up
    loaded = os.path.realpath(fake_spectra.__file__)
    if not loaded.startswith(os.path.realpath(impl_dir)):
        raise RuntimeError(f"Wanted fake_spectra from {impl_dir}, got {loaded}")
    return loaded


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--impl_dir", required=True, help="folder that contains the fake_spectra/ package")
    p.add_argument("--base", required=True, help="simulation BASE_PATH (has snapdir_NNN/)")
    p.add_argument("--snap", type=int, required=True)
    p.add_argument("--axis", type=int, help="1 = x, 2 = y, 3 = z (fake_spectra convention)")
    p.add_argument("--cofm", type=float, nargs=3, help="sightline position, code units")
    p.add_argument("--skewers", help=".npy with one row (axis, x, y, z) per skewer; replaces --axis/--cofm")
    p.add_argument("--mode", default="None", help="filter_mode: None, HCD_only, Forest_only (fork only)")
    p.add_argument("--r200", type=float, default=None, help="halo R200, code units (fork filter only)")
    p.add_argument("--radius_cut", type=float, default=10.0, help="window, in R200 (fork filter only)")
    p.add_argument("--res", type=float, default=10.0, help="pixel width, km/s")
    p.add_argument("--sf_neutral", type=int, default=1,
                   help="1 (fake_spectra default): star-forming gas is set self-shielded; 0: keep the snapshot value")
    p.add_argument("--out", required=True, help="output .npz")
    args = p.parse_args()

    loaded = select_fake_spectra(args.impl_dir)
    from fake_spectra import spectra, unitsystem

    # TNG code units: kpc/h, 1e10 Msun/h, km/s (same as tools/utils.py, "TNG")
    units = unitsystem.UnitSystem(UnitLength_in_cm=3.085678e21,
                                  UnitMass_in_g=1.989e43,
                                  UnitVelocity_in_cm_per_s=1e5)

    # Arguments shared by both builds (same as SpectrumEngine._build_spectra)
    kwargs = dict(res=args.res, savefile="unused_benchmark.hdf5", reload_file=True,
                  kernel=None, MPI=None, quiet=True, units=units,
                  sf_neutral=bool(args.sf_neutral))
    # The decomposition arguments only exist in the fork
    if args.mode != "None":
        kwargs.update(filter_mode=args.mode, hcd_phys_thresh="auto",
                      halo_r200=args.r200, halo_radius_cut=args.radius_cut)

    # Sightlines: either one (--axis/--cofm) or a table of them (--skewers)
    if args.skewers:
        table = np.load(args.skewers)
        axis = table[:, 0].astype(np.int32)
        cofm = table[:, 1:4].astype(np.float64)
    else:
        axis = np.array([args.axis], dtype=np.int32)
        cofm = np.array([args.cofm], dtype=np.float64)
    keep = slice(None) if args.skewers else 0      # 2D for a table, 1D for one skewer

    # --- timed extraction -------------------------------------------------
    t0 = time.perf_counter()
    ss = spectra.Spectra(args.snap, args.base, cofm=cofm, axis=axis, **kwargs)
    t1 = time.perf_counter()
    tau = ss.get_tau("H", 1, 1215)[keep]
    t2 = time.perf_counter()
    colden = ss.get_col_density("H", 1)[keep]
    t3 = time.perf_counter()

    timings = dict(init=t1 - t0, tau=t2 - t1, colden=t3 - t2, total=t3 - t0)

    np.savez(args.out, tau=tau, colden=colden, vmax=ss.vmax, nbins=ss.nbins,
             kernel_int=ss.kernel_int, timings=json.dumps(timings),
             loaded_from=loaded, mode=args.mode, sf_neutral=args.sf_neutral)
    print(json.dumps(dict(out=args.out, loaded_from=loaded, **timings)))


if __name__ == "__main__":
    main()
