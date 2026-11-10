"""
Skewer geometry around a halo, batched extraction, and the spectral moments.

What changed
------------
* The velocity / wavelength / comoving axes are IDENTICAL for every skewer of a
  given snapshot (they depend only on vmax and nbins).  They used to be stored
  once per skewer; they now live once in the `meta` dict returned alongside the
  rows, and `run_parallel_illustris.py` parks them in `df.attrs`.
* Spectral moments are no longer stored in the pickle.  They are exact, cheap
  functions of (flux, n_hi_col, v_grid), so keeping them frozen in the file was
  duplication.  They live at the bottom of this module instead -- call
  `radii_skewers.add_moments(df)` in a notebook to get them back.
* Removed `total_col_den` (identical to 10**log_n_hi), `axis_name` (derivable
  from `axis`), `impact_code` (= r_multiplier * halo_r200) and `impact_kpc`
  (which was also wrong: it assumed Mpc code units, so for TNG -- where lengths
  are ckpc/h -- it inflated the impact parameter by 1000).
* `r_mult` renamed to `r_multiplier`, which is the name the notebooks already
  use; the rename that used to happen inside the worker is gone.
* Added the HI-weighted temperature per pixel.
* `run_analysis_batched` can return several decomposition modes
  (total / HCD-only / forest-only) from a single pass over the snapshot.
* Filter modes RENAMED: "DLAs" -> "HCD_only", "No_DLAs" -> "Forest_only",
  "None" unchanged.  Pure renaming (no thresholds or logic touched); the
  legacy spellings are still accepted at the entry points and normalised, so
  the canonical names are what land in meta['modes'] and column suffixes.
  NOTE: add_moments does NOT normalise the modes it reads from df.attrs, on
  purpose -- old pickles carry columns like flux_DLAs and the lookup must
  match what was written.
* Exceptions are re-raised instead of being swallowed into an empty result, so
  the worker's `logging.error(..., exc_info=True)` actually records a traceback.
* Added `halo_vmax` to the per-skewer rows: SubhaloVmax of the CENTRAL subhalo
  (km/s, physical), resolved by utils.HaloFinder via GroupFirstSub.  This is
  the only NON-derivable ingredient of the halo concentration.  The
  concentration itself is NOT stored: V200 and c200 are exact functions of
  (halo_vmax, halo_mass, halo_r200, redshift, length_unit_kpc) -- note that h
  cancels, since Group_M_Crit200 and Group_R_Crit200 both carry 1/h.  Call
  `radii_skewers.add_concentration(df)` after loading the pickle.
"""
import logging

import numpy as np

_utils = None


def _get_utils():
    """Import tools.utils only when spectra are actually extracted."""
    global _utils
    if _utils is None:
        try:
            import utils as _u
        except ImportError:
            from tools import utils as _u
        _utils = _u
    return _utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - RADIISK - %(levelname)s - %(message)s')

AXIS_NAMES = ("X", "Y", "Z")
C_LIGHT_KMS = 299792.458

# numpy 2 renamed trapz -> trapezoid
_trapz = getattr(np, "trapezoid", None) or np.trapz

#: Legacy filter-mode spellings ("DLAs", "No_DLAs"), normalised at the
#: extraction entry point.  Sourced from the fork via utils, with a hardcoded
#: fallback if the installed fork predates the rename.
LEGACY_FILTER_MODES = {"DLAs": "HCD_only", "No_DLAs": "Forest_only"}

#: Column-density thresholds separating the absorber regimes, in log10 N_HI.
#: Prochaska+2005 / Ho, Bird & Garnett 2021, as used throughout the dissertation.
#: Sightlines below the HCD threshold carry log_n_hi = 0.0 and land in Forest.
#: NOTE: these are OBSERVATIONAL N_HI classes (a DLA is a kind of HCD), not
#: filter modes -- unrelated to the HCD_only / Forest_only decomposition.
CATEGORY_EDGES = [0.0, 17.2, 19.0, 20.3, 21.0, np.inf]
CATEGORY_LABELS = ['Forest', 'LLS', 'Sub_DLA', 'Small_DLA', 'Large_DLA']


class SkewerGeometry:
    """Manages the spatial orientation of skewers around the halo, in code units."""

    def __init__(self, sim_context, halo_info):
        self.ctx = sim_context
        self.halo = halo_info
        self.box = halo_info['box_size']
        self.r200_code = float(halo_info.get('r200', 0.0))
        if self.r200_code <= 0:
            raise ValueError(
                f"Halo {halo_info.get('id')} has R200 = {self.r200_code} in the group "
                "catalogue. Refusing to fall back to a magic default; filter this halo "
                "out upstream instead.")

    # ------------------------------------------------------------------
    def generate_skewers(self, r_multiplier):
        """Skewer descriptors: 3 for r=0 (one per axis), 12 otherwise (4 offsets x 3 axes)."""
        impact = r_multiplier * self.r200_code
        skewers = []
        sk_id = 0
        if r_multiplier == 0.0:
            for ax in range(3):
                sk_id += 1
                skewers.append({'id': sk_id, 'axis': ax, 'r_multiplier': 0.0,
                                'position_tag': 'Center', 'offset_vec': np.zeros(3)})
            return skewers
        offsets = [(impact, 0.0, "Right"), (-impact, 0.0, "Left"),
                   (0.0, impact, "Top"), (0.0, -impact, "Bottom")]
        for ax in range(3):
            for o1, o2, tag in offsets:
                sk_id += 1
                vec = np.zeros(3)
                free = [k for k in range(3) if k != ax]
                vec[free[0]] = o1
                vec[free[1]] = o2
                skewers.append({'id': sk_id, 'axis': ax, 'r_multiplier': r_multiplier,
                                'position_tag': tag, 'offset_vec': vec})
        return skewers

    # ------------------------------------------------------------------
    def run_analysis_batched(self, skewers_list, modes=("HCD_only",),
                             products=("colden", "tau", "temp"),
                             zero_below_cut=True, engine_kwargs=None):
        """Extract every skewer of this halo in one read of the snapshot.

        Legacy mode names ("DLAs", "No_DLAs") are accepted and normalised to
        ("HCD_only", "Forest_only") before anything else, so meta['modes'] and
        the column suffixes always carry the canonical names.

        Returns
        -------
        meta : dict
            Grids and constants shared by every skewer of this snapshot.
        rows : list of dict
            One entry per skewer, with only the non-derivable quantities.
        """
        modes = tuple(LEGACY_FILTER_MODES.get(m, m) for m in tuple(modes))
        axis_array, cofm_array = self._los_arrays(skewers_list)

        print(f"--> [Halo {self.halo['id']}] extracting {len(skewers_list)} skewers "
              f"({', '.join(modes)}) in RAM...", flush=True)

        eng = _get_utils().SpectrumEngine(self.ctx, self.halo, **(engine_kwargs or {}))
        out = eng.run(axis_array, cofm_array, modes=modes, products=products,
                      zero_below_cut=zero_below_cut)
        return self._meta_and_rows(out, skewers_list, modes)

    def _los_arrays(self, skewers_list):
        """fake_spectra axis (1..3) and position of each skewer of this halo."""
        base_cm = np.asarray(self.halo['cm'], dtype=np.float64)

        cofm_list, axis_list = [], []
        for sk in skewers_list:
            new_cm = np.mod(base_cm + sk['offset_vec'], self.box)   # periodic boundary
            cofm_list.append(new_cm)
            axis_list.append(sk['axis'] + 1)                        # fake_spectra axes are 1..3
        cofm_array = np.array(cofm_list, dtype=np.float64)
        axis_array = np.array(axis_list, dtype=np.int32)
        return axis_array, cofm_array

    def _meta_and_rows(self, out, skewers_list, modes):
        """Shared grids (meta) and one row per skewer, from an engine `out` dict."""
        meta = {
            'v_grid': out['v_grid'],
            'wave_grid': out['wave_grid'],
            'chi_grid': out['chi_grid'],
            'nbins': out['nbins'],
            'vmax': out['vmax'],
            'dvbin': out['dvbin'],
            'velfac': out['velfac'],
            'box_size': out['box_size'],
            'redshift': out['redshift'],
            'lam_rest': out['lam_rest'],
            # n_HI [cm^-3] = n_hi_col [cm^-2] / pixel_length_cm
            'pixel_length_cm': out['pixel_length_cm'],
            'hcd_log_nhi_cut': out['hcd_log_nhi_cut'],
            # Extraction settings (HCD window, star-forming gas model)
            'halo_radius_cut': out['halo_radius_cut'],
            'sf_neutral': out['sf_neutral'],
            'sf_criterion': out['sf_criterion'],
            'sf_temp': out['sf_temp'],
            'h2_model': out['h2_model'],
            'length_unit_kpc': self.ctx.length_unit_kpc,
            'modes': modes,
        }

        single = (len(modes) == 1)
        rows = []
        for idx, sk in enumerate(skewers_list):
            res = {
                'skewer_id': sk['id'],
                'axis': sk['axis'],
                'r_multiplier': sk['r_multiplier'],
                'position_tag': sk['position_tag'],
                'halo_id': self.halo['id'],
                'halo_mass': self.halo['mass'],
                'halo_r200': self.halo['r200'],
                # SubhaloVmax of the central subhalo [km/s, physical].
                # NaN when the halo has no central subhalo or the catalogue
                # has no Subhalo table (MN5).  .get keeps hand-built
                # halo_info dicts (old notebooks) working.
                'halo_vmax': float(self.halo.get('vmax', np.nan)),
            }
            for mode in modes:
                blk = out[mode]
                sfx = "" if single else f"_{mode}"
                res[f'flux{sfx}'] = blk['flux'][idx]
                res[f'n_hi_col{sfx}'] = blk['n_hi_col'][idx]
                res[f'log_n_hi{sfx}'] = float(blk['log_n_hi'][idx])
                res[f'is_hcd{sfx}'] = bool(blk['is_hcd'][idx])
                if 'temp' in blk:
                    res[f'temp{sfx}'] = blk['temp'][idx]
            rows.append(res)
        return meta, rows


def run_analysis_group(geoms, skewers_lists, modes=("HCD_only",),
                       products=("colden", "tau", "temp"),
                       zero_below_cut=True, engine_kwargs=None):
    """run_analysis_batched for several halos, in ONE read of the snapshot.

    geoms: one SkewerGeometry per halo; skewers_lists: the skewers of each.
    Returns a list of (meta, rows), one per halo, identical to calling
    run_analysis_batched halo by halo (see utils.run_halo_group).
    """
    modes = tuple(LEGACY_FILTER_MODES.get(m, m) for m in tuple(modes))
    los = [g._los_arrays(sk) for g, sk in zip(geoms, skewers_lists)]

    n_sk = sum(len(sk) for sk in skewers_lists)
    print(f"--> [Group of {len(geoms)} halos] extracting {n_sk} skewers "
          f"({', '.join(modes)}) in RAM...", flush=True)

    outs = _get_utils().run_halo_group(
        geoms[0].ctx, [g.halo for g in geoms],
        [a for a, _ in los], [c for _, c in los],
        modes=modes, products=products, zero_below_cut=zero_below_cut,
        engine_kwargs=engine_kwargs)
    return [g._meta_and_rows(out, sk, modes)
            for g, out, sk in zip(geoms, outs, skewers_lists)]


# ======================================================================
# Quantities that are no longer stored, recomputed on demand
# ======================================================================

def impact_code(row):
    """Impact parameter in simulation length units."""
    return row['r_multiplier'] * row['halo_r200']


def impact_kpc(row, meta):
    """Impact parameter in comoving kpc/h."""
    return impact_code(row) * meta['length_unit_kpc']


def axis_name(row):
    return AXIS_NAMES[int(row['axis'])]


def number_density_hi(n_hi_col, meta):
    """n_HI in each pixel, cm^-3, from the stored per-pixel column density."""
    return np.asarray(n_hi_col) / meta['pixel_length_cm']


def total_col_den(log_n_hi):
    """Integrated N_HI in cm^-2 (0 when the sightline is below the HCD cut).

    Replaces the old `total_col_den` column, which was exactly 10**log_n_hi.
    """
    lg = np.asarray(log_n_hi, dtype=float)
    return np.where(lg > 0, 10.0**lg, 0.0)


def categorize(log_n_hi):
    """Absorber regime from log10 N_HI, as a pandas Categorical.

    Replaces `pd.cut(df['total_col_den'], bins=[0, 1.6e17, 1e19, 2e20, 1e21, inf])`:
    log10(1.6e17) = 17.204, so the two are the same cut.
    """
    import pandas as pd
    return pd.cut(log_n_hi, bins=CATEGORY_EDGES, labels=CATEGORY_LABELS, right=False)


# ----------------------------------------------------------------------
# Halo concentration (NFW), on demand
# ----------------------------------------------------------------------
# The only stored primary is halo_vmax (SubhaloVmax of the central subhalo,
# km/s physical).  Everything else is already in the pickle:
#
#     V200^2 = G * (halo_mass * 1e10) / (halo_r200 * length_unit_kpc * a)
#
# halo_mass is in 1e10 Msun/h and halo_r200 in code length (comoving, /h):
# both carry 1/h, so h cancels in M/R and is not needed.  a = 1/(1+z) turns
# the comoving radius physical; Msun/h over kpc/h is Msun/kpc, which is what
# G below expects.
#
# c200 then inverts the exact NFW identity (Klypin, Kravtsov, Bullock &
# Primack 2001, ApJ 554, 903, Table 2):
#
#     (Vmax/V200)^2 = k * c / mu(c),   mu(x) = ln(1+x) - x/(1+x),
#     k = mu(x_max)/x_max = 0.2162,    x_max = 2.16258
#
# (x_max is the NFW rotation-curve peak, root of d/dx[mu(x)/x] = 0; k is the
# "0.216" of the literature.)  The relation is monotonic in c, so it is
# inverted with a precomputed table + np.interp -- vectorised, no scipy.
# It is only defined for Vmax/V200 > 1, i.e. c > 2.163: below that the NFW
# rotation curve would peak outside R200.  Those sightlines get NaN -- do not
# clip; at z=3 (typical c ~ 3-5 plus noise) this tail is real information.

G_KPC_KMS2_PER_MSUN = 4.30091e-6   # G in kpc (km/s)^2 / Msun
NFW_XMAX = 2.16258


def _nfw_mu(x):
    x = np.asarray(x, dtype=float)
    return np.log1p(x) - x / (1.0 + x)


_NFW_K = float(_nfw_mu(NFW_XMAX) / NFW_XMAX)          # ~0.21622
_C_TAB = np.logspace(np.log10(NFW_XMAX * (1.0 + 1e-6)), 3.0, 4096)
_RATIO_TAB = np.sqrt(_NFW_K * _C_TAB / _nfw_mu(_C_TAB))


def c200_from_vmax_ratio(ratio):
    """NFW c200 from Vmax/V200 (scalar or array).  NaN where undefined.

    Undefined means ratio <= 1 (c < 2.163, mapping degenerates) or ratio
    beyond c = 1000, as well as non-finite input (e.g. halo_vmax = NaN for
    halos with no central subhalo).
    """
    r = np.asarray(ratio, dtype=float)
    c = np.interp(r, _RATIO_TAB, _C_TAB, left=np.nan, right=np.nan)
    c = np.where(np.isfinite(r), c, np.nan)
    return float(c) if c.ndim == 0 else c


def halo_v200(halo_mass_code, halo_r200_code, redshift, length_unit_kpc):
    """V200 = sqrt(G M200 / R200) in km/s, from stored quantities only.

    halo_mass_code : Group_M_Crit200, in 1e10 Msun/h (Gadget convention)
    halo_r200_code : Group_R_Crit200, in code length units (comoving, /h)
    redshift       : the per-row 'redshift' column (or meta['redshift'])
    length_unit_kpc: meta['length_unit_kpc'] (code length -> ckpc/h)

    h cancels between mass and radius, see the block comment above.
    """
    a = 1.0 / (1.0 + np.asarray(redshift, dtype=float))
    m = np.asarray(halo_mass_code, dtype=float) * 1e10                  # Msun/h
    r = np.asarray(halo_r200_code, dtype=float) * length_unit_kpc * a   # kpc/h, physical
    with np.errstate(divide='ignore', invalid='ignore'):
        v2 = G_KPC_KMS2_PER_MSUN * m / r
    v2 = np.where(np.isfinite(v2) & (v2 > 0), v2, np.nan)
    v = np.sqrt(v2)
    return float(v) if v.ndim == 0 else v


def add_concentration(df, meta=None):
    """Add halo_v200 [km/s] and halo_c200 (NFW) columns, in place.

    `meta` defaults to `df.attrs`, which is where the pipeline stores the
    snapshot constants.  Mirrors add_moments:

        import pandas as pd, radii_skewers as rs
        df = pd.read_pickle("run0.pkl")
        rs.add_concentration(df)

    halo_c200 is the c200c concentration (critical overdensity, consistent
    with Group_M_Crit200 / Group_R_Crit200) via the Klypin+01 Vmax/V200
    proxy.  NaN encodes: no central subhalo, no Subhalo catalogue, or
    Vmax/V200 <= 1 (c < 2.163) -- keep the NaNs, they are information.
    """
    meta = meta if meta is not None else df.attrs
    if 'halo_vmax' not in df.columns:
        raise KeyError("No 'halo_vmax' column -- was this file written by the "
                       "current pipeline?")
    if 'length_unit_kpc' not in meta:
        raise KeyError("No 'length_unit_kpc' in meta/df.attrs -- was this file "
                       "written by the current pipeline?")
    lukpc = float(meta['length_unit_kpc'])
    if 'redshift' in df.columns:
        z = df['redshift'].to_numpy(dtype=float)
    else:
        z = float(meta['redshift'])
    v200 = halo_v200(df['halo_mass'].to_numpy(dtype=float),
                     df['halo_r200'].to_numpy(dtype=float),
                     z, lukpc)
    vmax = df['halo_vmax'].to_numpy(dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = vmax / v200
    df['halo_v200'] = v200
    df['halo_c200'] = c200_from_vmax_ratio(ratio)
    return df


# ----------------------------------------------------------------------
# Spectral moments
# ----------------------------------------------------------------------

#: Below this equivalent width the profile carries no usable shape information.
EW_FLOOR = 1e-3


def spectral_moments(flux, v_grid, wave_grid=None, lam_rest=1215.67, ew_floor=EW_FLOOR):
    """Equivalent width and the first four velocity moments of the absorption.

    Returns a dict with eq_width [Ang], mean_vel [km/s], mean_wave [Ang],
    std_dev [km/s], skewness, kurtosis.  Identical definitions to the ones that
    used to be computed inline in `run_analysis_batched`.
    """
    v_grid = np.asarray(v_grid, dtype=float)
    if wave_grid is None:
        wave_grid = lam_rest * np.exp(v_grid / C_LIGHT_KMS)
    absorption = 1.0 - np.asarray(flux, dtype=float)
    ew = _trapz(absorption, x=wave_grid)
    out = {'eq_width': float(ew), 'mean_vel': 0.0, 'mean_wave': float(lam_rest),
           'std_dev': 0.0, 'skewness': 0.0, 'kurtosis': 0.0}
    if ew <= ew_floor:
        return out
    norm = _trapz(absorption, x=v_grid)
    if norm <= 0:
        return out
    mean_vel = _trapz(v_grid * absorption, x=v_grid) / norm
    dv = v_grid - mean_vel
    var = _trapz(dv**2 * absorption, x=v_grid) / norm
    sd = float(np.sqrt(var)) if var > 0 else 0.0
    out['mean_vel'] = float(mean_vel)
    out['mean_wave'] = float(lam_rest * np.exp(mean_vel / C_LIGHT_KMS))
    out['std_dev'] = sd
    if sd > 0:
        out['skewness'] = float(_trapz(dv**3 * absorption, x=v_grid) / norm / sd**3)
        out['kurtosis'] = float(_trapz(dv**4 * absorption, x=v_grid) / norm / sd**4)
    return out


def coherent_velocity(n_hi_col, v_grid):
    """N_HI-weighted mean velocity, km/s.  0 if the sightline has no HI."""
    v_grid = np.asarray(v_grid, dtype=float)
    w = np.asarray(n_hi_col, dtype=float)
    denom = _trapz(w, x=v_grid)
    if denom <= 0:
        return 0.0
    return float(_trapz(w * v_grid, x=v_grid) / denom)


def hi_weighted_temperature(temp, n_hi_col):
    """Single N_HI-weighted temperature for the whole sightline, K."""
    w = np.asarray(n_hi_col, dtype=float)
    if w.sum() <= 0:
        return 0.0
    return float(np.sum(w * np.asarray(temp, dtype=float)) / w.sum())


def doppler_b(temp, amu=1.00794):
    """Thermal Doppler parameter b = sqrt(2 k T / m) in km/s, from T in K.

    Handy for comparing the stored HI-weighted temperature against the b
    recovered from a Voigt fit.
    """
    k_b = 1.380649e-16     # erg / K
    m_p = 1.67262192e-24   # g
    return np.sqrt(2.0 * k_b * np.asarray(temp, dtype=float) / (amu * m_p)) / 1e5


def peak_shift(flux, v_grid, tol=0.02):
    """Velocity of the deepest absorption trough closest to v = 0, km/s.

    Same algorithm as the old `align_to_absorption_center`, minus the aligned
    axis (which is just `v_grid - peak_shift`, so storing it was duplication).
    """
    flux = np.asarray(flux, dtype=float)
    v_grid = np.asarray(v_grid, dtype=float)
    threshold = float(np.min(flux)) + tol
    bottom = np.where(flux <= threshold)[0]
    if bottom.size == 0:
        return 0.0
    jumps = np.where(np.diff(bottom) > 1)[0]
    if jumps.size == 0:
        return float(np.mean(v_grid[bottom]))
    starts = np.insert(bottom[jumps + 1], 0, bottom[0])
    ends = np.append(bottom[jumps], bottom[-1])
    best, best_dist = 0.0, np.inf
    for s, e in zip(starts, ends):
        centre = float(np.mean(v_grid[s:e + 1]))
        if abs(centre) < best_dist:
            best_dist, best = abs(centre), centre
    return best


def add_moments(df, meta=None, modes=None):
    """Add the moment columns back to a loaded DataFrame, in place.

    `meta` defaults to `df.attrs`, which is where the pipeline stores the grids.
    The modes are taken verbatim from meta (NOT normalised): old pickles carry
    columns like flux_DLAs and the lookup must match what was written.

        import pandas as pd, radii_skewers as rs
        df = pd.read_pickle("run0.pkl")
        rs.add_moments(df)
    """
    meta = meta if meta is not None else df.attrs
    if 'v_grid' not in meta:
        raise KeyError("No 'v_grid' in meta/df.attrs -- was this file written by the "
                       "current pipeline?")
    v_grid = np.asarray(meta['v_grid'])
    lam_rest = float(meta.get('lam_rest', 1215.67))
    wave_grid = np.asarray(meta.get('wave_grid', lam_rest * np.exp(v_grid / C_LIGHT_KMS)))
    modes = tuple(modes or meta.get('modes', ('HCD_only',)))
    single = (len(modes) == 1)
    for mode in modes:
        sfx = "" if single else f"_{mode}"
        if f'flux{sfx}' not in df.columns:
            continue
        recs = [spectral_moments(f, v_grid, wave_grid, lam_rest) for f in df[f'flux{sfx}']]
        for key in ('eq_width', 'mean_vel', 'mean_wave', 'std_dev', 'skewness', 'kurtosis'):
            df[f'{key}{sfx}'] = [r[key] for r in recs]
        if f'n_hi_col{sfx}' in df.columns:
            df[f'vel_coherent{sfx}'] = [coherent_velocity(n, v_grid)
                                        for n in df[f'n_hi_col{sfx}']]
            if f'temp{sfx}' in df.columns:
                df[f'T_hi{sfx}'] = [hi_weighted_temperature(t, n) for t, n
                                    in zip(df[f'temp{sfx}'], df[f'n_hi_col{sfx}'])]
    return df