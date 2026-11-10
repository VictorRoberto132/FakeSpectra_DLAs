"""Check the fork's star-forming gas options (sf_criterion / sf_temp / h2_model).

  1. the fork options vs an independent implementation (a GasProperties
     subclass written from Bird et al. 2014, eq. 4)      -> must be identical;
  2. what each piece changes: sf_neutral off / on / on + 1e4 K + H2, with the
     density criterion and with StarFormationRate > 0;
  3. inconsistent settings (H2 without sf_neutral) must be refused.

Example:
    python test_sf_gas_model.py --sim TNG50-4 --halo_ids 5,50,300
"""
import argparse
import sys
from pathlib import Path

import numpy as np

PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ_ROOT / "tools"))
import utils
import radii_skewers
import fake_spectra.spectra as spectra
from fake_spectra.gas_properties import GasProperties


class ReferenceSFGas(GasProperties):
    """Independent version: sf_neutral (density threshold) + T = 1e4 K + BR06 H2."""
    def _sf(self, part_type, segment):
        dens = self.absnap.get_data(part_type, "Density", segment=segment)
        conv = np.float32(self.units.UnitDensity_in_cgs*self.hubble**2/self.units.protonmass*(1+self.redshift)**3)
        return dens > self.PhysDensThresh/0.76/conv, dens*conv*0.76

    def get_temp(self, part_type, segment):
        T = self.absnap.get_temp(part_type, segment=segment, units=self.units)
        T[self._sf(part_type, segment)[0]] = 1e4
        return T

    def get_reproc_HI(self, part_type, segment):
        x = super().get_reproc_HI(part_type, segment)     # upstream sf_neutral
        sf, nH = self._sf(part_type, segment)
        x[sf] = x[sf]*(1.0 - 1.0/(1.0 + (35.0*(0.1/nH[sf])**(5./3))**0.92))
        return x


def extract(ctx, cofm, axis, mode, r200, **gas):
    ss = spectra.Spectra(int(ctx.snap_num), ctx.base_path, cofm=cofm, axis=axis, res=10.0,
                         savefile="unused.hdf5", reload_file=True, kernel=None, MPI=None,
                         quiet=True, filter_mode=mode, units=ctx.units,
                         halo_r200=r200, halo_radius_cut=utils.HALO_RADIUS_CUT, **gas)
    return dict(tau=ss.get_tau("H", 1, 1215), colden=ss.get_col_density("H", 1),
                temp=ss.get_temp("H", 1))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--sim", default="TNG50-4")
    p.add_argument("--snap", type=int, default=25)
    p.add_argument("--halo_ids", default="5,50,300")
    args = p.parse_args()

    ctx = utils.SimulationContext(f"/home/STORAGE/{args.sim}", args.snap, sim_type="TNG")
    finder = utils.HaloFinder(ctx)
    cofm, axis, r200, b = [], [], [], []
    for hid in [int(h) for h in args.halo_ids.split(",")]:
        info = finder.get_halo_properties(hid)
        geom = radii_skewers.SkewerGeometry(ctx, info)
        sk = [s for r in (0.0, 0.05, 0.2, 1.0) for s in geom.generate_skewers(r_multiplier=r)]
        a, c = geom._los_arrays(sk)
        axis.append(a); cofm.append(c)
        r200 += [info['r200']] * len(sk); b += [s['r_multiplier'] for s in sk]
    cofm, axis, r200, b = np.concatenate(cofm), np.concatenate(axis), np.array(r200), np.array(b)
    print(f"{args.sim}: {len(b)} skewers, window {utils.HALO_RADIUS_CUT} R200")

    ok = True
    # 1. fork options (density criterion) == independent implementation
    for mode in ("None", "HCD_only"):
        fork = extract(ctx, cofm, axis, mode, r200, sf_neutral=True,
                       gasprop_args=dict(sf_criterion="density", sf_temp=1e4, h2_model="BR06"))
        ref = extract(ctx, cofm, axis, mode, r200, sf_neutral=True, gasprop=ReferenceSFGas)
        for k in fork:
            same = np.array_equal(fork[k], ref[k])
            ok &= same
            print(f"  1. {mode:8s} {k:6s} fork options == reference: {same}")

    # 2. what each piece changes (total field, mode None)
    variants = {
        "sf off":                     dict(sf_neutral=False),
        "sf on (upstream)":           dict(sf_neutral=True),
        "sf on + 1e4K + H2, density": dict(sf_neutral=True, gasprop_args=dict(sf_criterion="density", sf_temp=1e4, h2_model="BR06")),
        "sf on + 1e4K + H2, sfr":     dict(sf_neutral=True, gasprop_args=dict(utils.SF_GAS_MODEL)),
    }
    print("\n  2. log N_HI and N_HI-weighted T (mode None)")
    for name, kw in variants.items():
        r = extract(ctx, cofm, axis, "None", r200, **kw)
        col = r["colden"].sum(axis=1)
        N = np.log10(np.maximum(col, 1))
        T = (r["colden"]*r["temp"]).sum(axis=1)/np.maximum(col, 1e-30)
        line = "  ".join(f"b={bb:<4g} N {np.median(N[b == bb]):5.2f} (max {N[b == bb].max():5.2f}) T {np.median(T[b == bb]):8.3g}"
                         for bb in (0.0, 0.05, 0.2, 1.0))
        print(f"     {name:28s} {line}")

    # 3. H2 / sf_temp without sf_neutral must be refused
    try:
        extract(ctx, cofm[:1], axis[:1], "None", None, sf_neutral=False, gasprop_args=dict(h2_model="BR06"))
        print("\n  3. h2_model without sf_neutral: NOT refused"); ok = False
    except ValueError as e:
        print(f"\n  3. refused as expected: {e}")

    print("\nidentity tests (1, 3):", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
