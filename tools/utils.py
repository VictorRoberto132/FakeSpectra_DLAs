"""
Halo / snapshot bookkeeping and the mock-spectrum extraction engine.

This version talks to the CLASSIC Spectra API of the HCD fork of spectra.py
(filter_mode / get_tau / get_col_density / get_temp). It does NOT require
compute_products -- that refactor was dropped.

Filter modes (renamed): "HCD_only" (formerly "DLAs"), "Forest_only" (formerly
"No_DLAs") and "None" (stock, all cells). The legacy spellings are still
accepted everywhere and silently normalised -- pure renaming, no thresholds or
logic changed.

REQUIREMENT: the HCD spectra.py must be installed inside the fake_spectra
package (it replaces fake_spectra/spectra.py). The guard right below the
imports fails loudly at import time if the stock upstream Spectra is found,
instead of dying later inside a worker with a cryptic AttributeError.
"""
import inspect
import logging
import os

import h5py
import numpy as np

import fake_spectra.spectra as spectra
import fake_spectra.abstractsnapshot as absn
import fake_spectra.unitsystem as unitsystem

# ---------------------------------------------------------------------------
# Fail early if the installed Spectra is not the HCD fork.
# ---------------------------------------------------------------------------
if 'filter_mode' not in inspect.signature(spectra.Spectra.__init__).parameters:
    raise ImportError(
        "The installed fake_spectra.spectra.Spectra has no 'filter_mode' argument: "
        "this is the stock upstream code, not the HCD fork. Install the fork with:\n"
        "    pip install \"git+https://github.com/VictorRoberto132/fake_spectra@v2.2.4-hcd1\"\n"
        "(installed copy found at %r)" % (spectra.__file__,))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

C_LIGHT_KMS = 299792.458
LYA_REST_ANG = 1215.67

#: SELECTION cut, log10 N_HI, applied to the INTEGRATED sightline column.
#: 17.2 = observational HCD definition. This is intentionally ABOVE the 16.5
#: particle-level SEED used inside spectra.py: the seed is a safety margin so
#: the region growing misses nothing; the selection here keeps only real HCDs.
HCD_LOG_NHI = spectra.HCD_SELECTION_LOGNHI if hasattr(spectra, "HCD_SELECTION_LOGNHI") else 17.2

#: Half-width of the HCD association window along the sightline, in R200.
#: 5 is the value of the canonical runs (*_peak_plus_blend); the *_10 / *_50
#: folders are sensitivity tests with 10 and 50.
HALO_RADIUS_CUT = 5.0

#: Star-forming gas, as in Bird et al. 2014 (Sec. 2.2): self-shielded / neutral
#: (sf_neutral), at 1e4 K instead of the 'effective' temperature of the equation
#: of state, and partly molecular (Blitz & Rosolowsky 2006; H2 does not absorb in
#: Lyman alpha). Star-forming = StarFormationRate > 0, the flag of the simulation.
#: See fake_spectra/gas_properties.py of the fork. Every Spectra object of the
#: project (pipeline and notebooks) is built with these, via SPECTRA_GAS_KWARGS.
SF_NEUTRAL = True
SF_GAS_MODEL = {"sf_criterion": "sfr", "sf_temp": 1e4, "h2_model": "BR06"}
SPECTRA_GAS_KWARGS = {"sf_neutral": SF_NEUTRAL, "gasprop_args": dict(SF_GAS_MODEL)}

# An older fork does not know these options and would crash deep inside a
# worker on the unknown keyword: fail here instead, with the reason.
import fake_spectra.gas_properties as _gas_properties
_missing = set(SF_GAS_MODEL) - set(inspect.signature(_gas_properties.GasProperties.__init__).parameters)
if _missing:
    raise ImportError("The installed fake_spectra fork has no %s option(s) in GasProperties "
                      "(star-forming gas model of 2026-10-25). Update the fork (installed copy: %r)."
                      % (sorted(_missing), _gas_properties.__file__))

#: Legacy filter-mode spellings, silently mapped to the current names.
#: Falls back to a hardcoded map if the installed fork predates the rename.
LEGACY_FILTER_MODES = getattr(spectra, "LEGACY_FILTER_MODES",
                              {"DLAs": "HCD_only", "No_DLAs": "Forest_only"})


class HaloMatcher:
    """Maps a cross-match index onto a global halo id, per simulation."""

    def __init__(self, xmatch_dir):
        self.xmatch_dir = xmatch_dir
        self.match_catalogs = {}

    def get_halo_id(self, sim_name, match_index):
        if sim_name not in self.match_catalogs:
            path = os.path.join(self.xmatch_dir, f"xmatch_{sim_name}.npy")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Match file not found at: {path}")
            logging.info("Loading match catalog for %s...", sim_name)
            data = np.load(path, allow_pickle=True)
            self.match_catalogs[sim_name] = data.item()['ind'] if data.ndim == 0 else data[0]['ind']
        catalog = self.match_catalogs[sim_name]
        if match_index < 0 or match_index >= len(catalog):
            raise ValueError(f"Match Index {match_index} is out of bounds.")
        return int(catalog[match_index])


class SimulationContext:
    """Paths, unit system and per-code quirks for one snapshot."""

    _UNITS = {
        "MN5": dict(UnitLength_in_cm=3.085678e24,   # Mpc
                    UnitMass_in_g=1.989e43,
                    UnitVelocity_in_cm_per_s=1e5),
        "TNG": dict(UnitLength_in_cm=3.085678e21,   # kpc
                    UnitMass_in_g=1.989e43,
                    UnitVelocity_in_cm_per_s=1e5),
    }

    def __init__(self, base_path, snap_num, sim_type="MN5"):
        self.base_path = base_path
        self.sim_type = sim_type.upper()
        self.snap_num = f"{snap_num:03d}" if isinstance(snap_num, int) else snap_num
        self.group_dir = f"groups_{self.snap_num}"
        self.snap_dir = f"snapdir_{self.snap_num}"
        if self.sim_type == "MN5":
            self._apply_mn5_patch()

    @property
    def units(self):
        cfg = self._UNITS.get(self.sim_type)
        return None if cfg is None else unitsystem.UnitSystem(**cfg)

    @property
    def length_unit_kpc(self):
        """How many (comoving/h) kpc one code length unit is worth."""
        cfg = self._UNITS.get(self.sim_type)
        return 1.0 if cfg is None else cfg["UnitLength_in_cm"] / 3.085678e21

    def _apply_mn5_patch(self):
        """Monkey patch absn.get_header_attr to handle MN5's missing keywords."""
        TargetClass = absn.AbstractSnapshot
        if not hasattr(TargetClass, '_original_get_header_attr'):
            TargetClass._original_get_header_attr = TargetClass.get_header_attr

        def patched_get_header_attr(instance, attr):
            try:
                return instance._original_get_header_attr(attr)
            except KeyError:
                pass
            if hasattr(instance, '_f_handle') and "Parameters" in instance._f_handle:
                try:
                    return instance._f_handle["Parameters"].attrs[attr]
                except KeyError:
                    pass
            if attr == "NumPart_Total_HighWord":
                return np.zeros(6, dtype=np.int32)
            raise KeyError(f"Could not find attribute '{attr}'")

        TargetClass.get_header_attr = patched_get_header_attr

    def get_group_file(self, index):
        return os.path.join(self.base_path, self.group_dir,
                            f"fof_subhalo_tab_{self.snap_num}.{index}.hdf5")


class HaloFinder:
    """Reads FoF group properties out of the (chunked) group catalogue."""

    def __init__(self, sim_context):
        self.sim = sim_context

    def _central_subhalo_vmax(self, global_sub_id, n_files):
        """SubhaloVmax [km/s, PHYSICAL] of one subhalo, by GLOBAL Subfind index.

        GroupFirstSub is a *global* subhalo index: the central subhalo of a
        group stored in chunk i may live in a later chunk, so the file walk
        has to be redone here, accumulating SUBHALO counts (which are
        independent of the group counts used in get_halo_properties).

        Returns NaN when the group has no subhalo (GroupFirstSub == -1) or
        when the catalogue carries no Subhalo table at all (e.g. MN5), so the
        rest of the pipeline keeps working unchanged.
        """
        if global_sub_id is None or global_sub_id < 0:
            return float('nan')
        offset = 0
        for i in range(n_files):
            g_file = self.sim.get_group_file(i)
            if not os.path.exists(g_file):
                continue
            with h5py.File(g_file, 'r') as f:
                hdr = f['Header'].attrs
                if 'Nsubgroups_ThisFile' in hdr:            # TNG / Arepo
                    n_here = int(hdr['Nsubgroups_ThisFile'])
                elif 'Nsubhalos_ThisFile' in hdr:           # Gadget-4 naming
                    n_here = int(hdr['Nsubhalos_ThisFile'])
                elif 'Subhalo' in f and 'SubhaloVmax' in f['Subhalo']:
                    n_here = len(f['Subhalo']['SubhaloVmax'])
                else:
                    n_here = 0
                if n_here == 0:
                    continue
                if global_sub_id < offset + n_here:
                    if 'Subhalo' not in f or 'SubhaloVmax' not in f['Subhalo']:
                        return float('nan')
                    return float(f['Subhalo']['SubhaloVmax'][int(global_sub_id - offset)])
                offset += n_here
        return float('nan')

    def get_halo_properties(self, global_halo_id):
        with h5py.File(self.sim.get_group_file(0), 'r') as f:
            hdr = f['Header'].attrs
            n_group_files = int(hdr.get('NumFiles', hdr.get('NumFilesPerSnapshot', 1)))
            box_size = hdr['BoxSize']
            scale_factor = hdr['Time']
            redshift = hdr['Redshift']

        props = None
        first_sub = -1
        offset = 0
        for i in range(n_group_files):
            g_file = self.sim.get_group_file(i)
            if not os.path.exists(g_file):
                continue
            with h5py.File(g_file, 'r') as f:
                if 'Group' not in f or 'Group_M_Crit200' not in f['Group']:
                    continue
                grp = f['Group']
                n_here = len(grp['Group_M_Crit200'])
                if global_halo_id < offset + n_here:
                    j = int(global_halo_id - offset)
                    # Central subhalo (global Subfind index); -1 = no subhalo.
                    first_sub = int(grp['GroupFirstSub'][j]) if 'GroupFirstSub' in grp else -1
                    props = {
                        'id': int(global_halo_id),
                        'cm': np.array(grp['GroupPos'][j]),
                        'mass': float(grp['Group_M_Crit200'][j]),
                        'r200': float(grp['Group_R_Crit200'][j]),
                        'vel_raw': np.array(grp['GroupVel'][j]),
                        'box_size': float(box_size),
                        'redshift': float(redshift),
                        'scale_factor': float(scale_factor),
                    }
                    break
                offset += n_here
        if props is None:
            raise ValueError(f"Halo {global_halo_id} not found in any group file.")

        # Vmax of the central subhalo [km/s, physical]. This is the ONLY
        # non-derivable ingredient of the halo concentration: V200 (and hence
        # c200) is an exact function of (vmax, mass, r200, redshift,
        # length_unit_kpc), all of which the pipeline already stores -- see
        # radii_skewers.add_concentration.
        props['vmax'] = self._central_subhalo_vmax(first_sub, n_group_files)
        return props


class SpectrumEngine:
    """Extract a batch of mock spectra around one halo, via the classic API.

    Cost model: each (filter_mode, product) pair is its own full pass over the
    snapshot chunks -- get_tau, get_col_density and get_temp each read the disk
    once. With products=("colden","tau","temp") an "HCD_only" batch therefore
    costs 3 reads; drop "temp" from products to go back to 2.
    """

    #: How the FoF catalogue stores GroupVel.
    #: TNG/Arepo Subfind documents GroupVel in km/s/a  ->  v_pec = GroupVel / a.
    #: The old pipeline used the PARTICLE convention (GroupVel * sqrt(a)),
    #: which at z=3 (a=0.25) differs by a factor of 8. Default is the spec-
    #: correct "km/s/a"; to reproduce the OLD centring exactly, set
    #:     eng.group_vel_units = "km*sqrt(a)/s"
    #: (or pass group_vel_units=... in the constructor).
    GROUP_VEL_UNITS = "km/s/a"

    def __init__(self, sim_context, halo_data, res=10.0, group_vel_units=None,
                 spectra_kwargs=None):
        self.sim = sim_context
        self.halo = halo_data
        self.res = res
        self.group_vel_units = group_vel_units or self.GROUP_VEL_UNITS
        self.spectra_kwargs = dict(spectra_kwargs or {})
        self.ss = None
        self.tau_batch = None
        self.col_den_batch = None
        self.temp_batch = None
        self.los_axis = None     # LOS axis (1..3) of each row of the batches
        self.c_light = C_LIGHT_KMS
        self.lam_rest = LYA_REST_ANG

    # ------------------------------------------------------------------
    def _build_spectra(self, axis_array, cofm_array, filter_mode, halo_r200=None):
        # halo_r200=None -> the R200 of this engine's halo (one halo per
        # Spectra). run_halo_group passes one R200 per sightline instead.
        if halo_r200 is None:
            halo_r200 = float(self.halo['r200'])
        kwargs = dict(
            res=self.res,
            savefile=f"unused_batch_{self.halo['id']}.hdf5",  # never written
            reload_file=True,
            kernel=None,
            MPI=None,
            quiet=True,
            filter_mode=filter_mode,
            hcd_phys_thresh="auto",
            units=self.sim.units,
            halo_r200=halo_r200,
            halo_radius_cut=HALO_RADIUS_CUT,
            sf_neutral=SF_NEUTRAL,
            gasprop_args=dict(SF_GAS_MODEL),
        )
        kwargs.update(self.spectra_kwargs)
        return spectra.Spectra(int(self.sim.snap_num), self.sim.base_path,
                               cofm=cofm_array, axis=axis_array, **kwargs)

    def generate_skewer_batch(self, axis_array, cofm_array, filter_mode="HCD_only",
                              with_temp=True):
        """Classic entry point: builds the Spectra object and fills the batches.

        tau_batch, col_den_batch and (optionally) temp_batch are RAW, i.e. NOT
        centred and NOT zeroed below the HCD cut -- the cut is applied
        non-destructively in get_centered_spectrum / run.
        """
        self.ss = self._build_spectra(axis_array, cofm_array, filter_mode)
        self.tau_batch = self.ss.get_tau("H", 1, 1215)
        self.col_den_batch = self.ss.get_col_density("H", 1)
        self.temp_batch = self.ss.get_temp("H", 1) if with_temp else None
        self.los_axis = self.ss.axis
        return self.ss

    # ------------------------------------------------------------------
    def _systemic_velocity(self, axis_1indexed):
        """Hubble flow + peculiar velocity of the halo, projected on the LOS axis."""
        k = axis_1indexed - 1
        a = self.halo['scale_factor']
        v_cosmo = (self.halo['cm'][k] / self.halo['box_size']) * self.ss.vmax
        raw = self.halo['vel_raw'][k]
        if self.group_vel_units == "km/s/a":
            v_pec = raw / a
        elif self.group_vel_units == "km*sqrt(a)/s":
            v_pec = raw * np.sqrt(a)
        elif self.group_vel_units == "km/s":
            v_pec = raw
        else:
            raise ValueError("Unknown group_vel_units %r" % self.group_vel_units)
        return v_cosmo + v_pec

    @staticmethod
    def _center(arr_1d, v_axis, v_grid, v_sys, vmax):
        """Shift one spectrum to the halo rest frame on the common grid.

        np.interp(..., period=vmax) wraps correctly at the box seam; the old
        argsort + plain-interp clamped there.
        """
        return np.interp(v_grid, v_axis - v_sys, arr_1d, period=vmax)

    def get_centered_spectrum(self, index=0, zero_below_cut=True):
        """Return (wave, v, chi, flux, colden, temp, lam_rest, is_hcd) for one skewer.

        THE OBSERVATIONAL CUT: if the integrated N_HI does not reach 10^17.2,
        the sightline merely grazed a dense 3D candidate; with
        zero_below_cut=True the returned flux/colden/temp are the empty-sky
        versions. Unlike the old version this does NOT mutate the cached
        batches, so calling it twice gives the same answer.
        """
        if self.tau_batch is None:
            raise RuntimeError("Run generate_skewer_batch() first before fetching spectra.")
        nbins, vmax = self.ss.nbins, self.ss.vmax
        v_axis = np.arange(nbins) * (vmax / nbins)
        v_grid = np.linspace(-vmax / 2.0, vmax / 2.0, nbins)

        tau_skewer = self.tau_batch[index, :]
        col_den_raw = self.col_den_batch[index, :]
        temp_raw = None if self.temp_batch is None else self.temp_batch[index, :]

        total_N_HI = float(np.sum(col_den_raw))
        is_hcd = (total_N_HI > 0) and (np.log10(total_N_HI) >= HCD_LOG_NHI)

        if zero_below_cut and not is_hcd:
            flux_raw = np.ones(nbins, dtype=np.float64)
            col_den_raw = np.zeros(nbins, dtype=np.float64)
            temp_raw = None if temp_raw is None else np.zeros(nbins, dtype=np.float64)
        else:
            flux_raw = np.exp(-tau_skewer)

        v_sys = self._systemic_velocity(int(self.los_axis[index]))
        flux_c = self._center(flux_raw, v_axis, v_grid, v_sys, vmax)
        colden_c = self._center(col_den_raw, v_axis, v_grid, v_sys, vmax)
        temp_c = (None if temp_raw is None
                  else self._center(temp_raw, v_axis, v_grid, v_sys, vmax))

        wave = self.lam_rest * np.exp(v_grid / self.c_light)
        chi = v_grid / self.ss.velfac   # comoving, code length units
        return wave, v_grid, chi, flux_c, colden_c, temp_c, self.lam_rest, is_hcd

    def get_total_col_den(self, index=0):
        """Integrated raw N_HI (before centring and before the 17.2 cut)."""
        if self.col_den_batch is None:
            return 0.0
        return float(np.sum(self.col_den_batch[index, :]))

    # ------------------------------------------------------------------
    def run(self, axis_array, cofm_array, modes=("HCD_only",),
            products=("colden", "tau", "temp"), zero_below_cut=True,
            hcd_log_nhi=HCD_LOG_NHI):
        """Extract the whole batch, for one or several filter modes.

        Legacy mode names ("DLAs", "No_DLAs") are accepted and normalised to
        ("HCD_only", "Forest_only") -- the returned blocks and any column
        suffixes always use the canonical names.

        Returns a dict with shared grids plus one block per mode:

            out["v_grid"], out["wave_grid"], ...
            out[mode]["flux" | "n_hi_col" | "temp" | "log_n_hi" | "is_hcd"]

        Each mode is a separate Spectra object, i.e. its own read of the
        snapshot (the fork caches the particles, so tau, colden and temp
        share that one read).
        """
        modes = tuple(LEGACY_FILTER_MODES.get(m, m) for m in tuple(modes))
        with_temp = "temp" in tuple(products)
        out = None
        for mode in modes:
            ss = self.generate_skewer_batch(axis_array, cofm_array,
                                            filter_mode=mode, with_temp=with_temp)
            if out is None:
                out = self._grids(ss, modes, hcd_log_nhi)
            out[mode] = self._mode_block(mode, len(axis_array), ss.nbins,
                                         with_temp, zero_below_cut)
        return out

    def _grids(self, ss, modes, hcd_log_nhi):
        """Grids and constants shared by every skewer of the snapshot."""
        nbins, vmax = ss.nbins, ss.vmax
        v_grid = np.linspace(-vmax / 2.0, vmax / 2.0, nbins)
        return {
            "v_grid": v_grid,
            "wave_grid": self.lam_rest * np.exp(v_grid / self.c_light),
            "chi_grid": v_grid / ss.velfac,
            "nbins": int(nbins),
            "vmax": float(vmax),
            "dvbin": float(ss.dvbin),
            "velfac": float(ss.velfac),
            "box_size": float(ss.box),
            "redshift": float(ss.red),
            "lam_rest": self.lam_rest,
            # n_HI [cm^-3] = n_hi_col [cm^-2] / pixel_length_cm
            "pixel_length_cm": float(ss.dvbin / ss.velfac * ss.rscale),
            "hcd_log_nhi_cut": float(hcd_log_nhi),
            "modes": modes,
            # Settings of the extraction itself, read back from the Spectra object
            # so they record what was really used (the folder name is not enough)
            "halo_radius_cut": float(ss.halo_radius_cut),
            "sf_neutral": bool(ss.gasprop.sf_neutral),
            "sf_criterion": ss.gasprop.sf_criterion,
            "sf_temp": ss.gasprop.sf_temp,
            "h2_model": ss.gasprop.h2_model,
        }

    def _mode_block(self, mode, n_los, nbins, with_temp, zero_below_cut):
        """Centred flux / n_hi_col / temp and log_n_hi, is_hcd of the current batches."""
        block = {
            "flux": np.empty((n_los, nbins)),
            "n_hi_col": np.empty((n_los, nbins)),
            "log_n_hi": np.zeros(n_los),
            "is_hcd": np.zeros(n_los, dtype=bool),
        }
        if with_temp:
            block["temp"] = np.empty((n_los, nbins))
        for i in range(n_los):
            # For the total field ("None") the 17.2 cut does not apply.
            zbc = zero_below_cut and (mode != "None")
            w, v, chi, flux_c, colden_c, temp_c, lam, is_hcd = \
                self.get_centered_spectrum(index=i, zero_below_cut=zbc)
            tot = self.get_total_col_den(index=i)
            block["flux"][i] = flux_c
            block["n_hi_col"][i] = colden_c
            block["is_hcd"][i] = is_hcd
            if tot > 0 and (is_hcd or not zbc):
                block["log_n_hi"][i] = np.log10(tot)
            if with_temp:
                block["temp"][i] = temp_c
        return block


def run_halo_group(sim_context, halos, axis_arrays, cofm_arrays, modes=("HCD_only",),
                   products=("colden", "tau", "temp"), zero_below_cut=True,
                   hcd_log_nhi=HCD_LOG_NHI, engine_kwargs=None):
    """SpectrumEngine.run for SEVERAL halos, with ONE Spectra object per mode.

    The sightlines of all halos go into the same Spectra, with one R200 per
    sightline (fork option), so the snapshot is read once for the whole group
    instead of once per halo. Each sightline keeps the HCD window of its own
    halo, and the centring is done afterwards halo by halo, so the result is
    the same as running the halos one at a time.

    halos, axis_arrays, cofm_arrays: one entry per halo (same as SpectrumEngine.run).
    Returns a list with one `out` dict per halo, as SpectrumEngine.run returns.
    """
    modes = tuple(LEGACY_FILTER_MODES.get(m, m) for m in tuple(modes))
    with_temp = "temp" in tuple(products)
    engines = [SpectrumEngine(sim_context, h, **(engine_kwargs or {})) for h in halos]

    # All sightlines in one array; cuts[k]:cuts[k+1] are the rows of halo k
    axis_all = np.concatenate(axis_arrays).astype(np.int32)
    cofm_all = np.concatenate(cofm_arrays).astype(np.float64)
    r200_all = np.concatenate([np.full(len(a), float(h['r200']))
                               for h, a in zip(halos, axis_arrays)])
    cuts = np.cumsum([0] + [len(a) for a in axis_arrays])

    outs = [None] * len(halos)
    for mode in modes:
        # One snapshot read for the whole group (built by the first engine:
        # only the kwargs are taken from it, the R200 comes per sightline)
        ss = engines[0]._build_spectra(axis_all, cofm_all, mode, halo_r200=r200_all)
        tau = ss.get_tau("H", 1, 1215)
        colden = ss.get_col_density("H", 1)
        temp = ss.get_temp("H", 1) if with_temp else None

        # Give each halo's engine its own rows, then centre as in run()
        for k, eng in enumerate(engines):
            sl = slice(cuts[k], cuts[k + 1])
            eng.ss = ss
            eng.tau_batch = tau[sl]
            eng.col_den_batch = colden[sl]
            eng.temp_batch = None if temp is None else temp[sl]
            eng.los_axis = ss.axis[sl]
            if outs[k] is None:
                outs[k] = eng._grids(ss, modes, hcd_log_nhi)
            outs[k][mode] = eng._mode_block(mode, cuts[k + 1] - cuts[k], ss.nbins,
                                            with_temp, zero_below_cut)
    return outs
