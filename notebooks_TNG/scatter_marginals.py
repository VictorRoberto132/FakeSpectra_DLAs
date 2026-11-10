"""
scatter_marginals.py
====================

Publication-quality scatter plot with per-bin summary statistics and
marginal log-PDF projections, of the kind used to display the halo
concentration--mass relation.

Layout::

    +-------------------------------+
    |  top panel: log10 P(x) per bin|
    +-------------------------------+-------+
    |                               |       |
    |  main panel:                  | right |
    |   - greyscale density, all    | panel |
    |   - HPD contour per bin       | log10 |
    |   - statistic +/- spread      |  P(y) |
    |   - flagged points highlighted|       |
    +-------------------------------+-------+

The sample is split into sub-populations by a third property ``z``.  When
the splitting property *is* one of the plotted axes (binning in mass and
plotting mass against concentration, say), pass ``z=x``.

Design notes
------------
* **Speed.**  Densities are estimated by default with a histogram smoothed
  by a Gaussian filter of Scott's-rule width.  This is an O(N)
  approximation to a Gaussian KDE and is several hundred times faster on
  samples of 1e4--1e7 objects, where an exact ``gaussian_kde`` evaluation
  on a plotting grid costs tens of seconds per panel.  Set
  ``density_method='kde'`` to force the exact estimator on small samples.
* **Contours** are highest-density regions: the level drawn genuinely
  encloses the requested fraction of the probability mass, so
  ``contour_frac=0.68`` is a "1 sigma" region rather than an arbitrary
  iso-density line.
* **Marginals** are peak-normalised, so bins containing very different
  numbers of objects remain comparable in shape.
* **usetex safety.**  Legend text is escaped, so the figure compiles with
  ``text.usetex=True``, under which a bare ``%`` or ``#`` is fatal.
* **Vector output.**  Dense point clouds and the density image are
  rasterised; all lines, markers and text remain vector, which keeps PDF
  files small without visible degradation.

Requires only ``numpy``, ``scipy`` and ``matplotlib``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.stats import gaussian_kde

__all__ = [
    "FIG_WIDTH",
    "set_journal_style",
    "scatter_with_marginals",
    "BinStats",
    "MarginalFigure",
]


# ---------------------------------------------------------------------------
# journal geometry and style
# ---------------------------------------------------------------------------

#: Text-block widths in inches, as ``(single_column, double_column)``.
FIG_WIDTH = {
    "aas": (3.35, 7.10),      # ApJ / AJ / ApJS
    "aanda": (3.46, 7.09),    # A&A: 88 mm / 180 mm
    "mnras": (3.31, 6.77),    # MNRAS: 84 mm / 172 mm
    "prd": (3.38, 7.00),      # Phys. Rev. D
}


def set_journal_style(journal="aas", fontsize=8.0, usetex=False, family="serif"):
    """Apply a conservative rcParam set suited to journal figures.

    Follows the conventions of the astronomical literature: serif text
    matching the body font, ticks on all four sides pointing inwards,
    minor ticks enabled, thin lines, and a font size chosen so the figure
    is legible at the size it will actually be printed.

    Parameters
    ----------
    journal : str
        Key into :data:`FIG_WIDTH`.
    fontsize : float
        Base font size in points.  8 pt suits a single-column figure
        reproduced at full width; use 9--10 pt for double-column figures.
    usetex : bool
        Render all text with LaTeX.  Slower, but guarantees the figure
        fonts match the manuscript exactly.
    family : str
        Font family; ``'serif'`` matches most journal body text.

    Returns
    -------
    tuple
        ``(single_column_width, double_column_width)`` in inches.
    """
    mpl.rcParams.update({
        "text.usetex": usetex,
        "font.family": family,
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix" if family == "serif" else "dejavusans",
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize,
        "xtick.labelsize": fontsize - 0.5,
        "ytick.labelsize": fontsize - 0.5,
        "legend.fontsize": fontsize - 1.0,
        "legend.frameon": True,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "0.7",
        "legend.fancybox": False,
        "legend.borderpad": 0.35,
        "legend.labelspacing": 0.30,
        "legend.handletextpad": 0.5,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.minor.width": 0.55,
        "ytick.minor.width": 0.55,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.dpi": 600,
        "figure.dpi": 120,
        "pdf.fonttype": 42,     # embed TrueType: no Type-3 warnings at submission
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })
    return FIG_WIDTH.get(journal, FIG_WIDTH["aas"])


def _tex_escape(text):
    """Escape characters that are fatal under ``text.usetex``.

    Only applied outside maths: the labels built here keep ``%`` and ``#``
    in their text portions and symbols in their maths portions.
    """
    if not mpl.rcParams.get("text.usetex", False):
        return text
    out, i = [], 0
    in_math = False
    while i < len(text):
        ch = text[i]
        if ch == "$":
            in_math = not in_math
            out.append(ch)
        elif not in_math and ch in "%#&_":
            out.append("\\" + ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# input normalisation
# ---------------------------------------------------------------------------

def _as_ranges(z_ranges):
    """Normalise the bin specification to a list of ``(lo, hi)`` pairs.

    Accepts a sequence of ``(lo, hi)`` pairs -- which may overlap or leave
    gaps, occasionally useful -- or a monotonic 1-D array of ``N + 1``
    edges, expanded into ``N`` contiguous bins.
    """
    arr = np.asarray(z_ranges, dtype=float)

    if arr.ndim == 2 and arr.shape[1] == 2:
        pairs = [(float(lo), float(hi)) for lo, hi in arr]
    elif arr.ndim == 1 and arr.size >= 2:
        if np.any(np.diff(arr) <= 0):
            raise ValueError("bin edges must be strictly increasing.")
        pairs = [(float(arr[i]), float(arr[i + 1])) for i in range(arr.size - 1)]
    else:
        raise ValueError(
            "z_ranges must be a sequence of (lo, hi) pairs or a 1-D array of "
            f">= 2 edges; got shape {arr.shape}."
        )

    for lo, hi in pairs:
        if not np.isfinite([lo, hi]).all() or hi <= lo:
            raise ValueError(f"invalid bin ({lo}, {hi}): need finite lo < hi.")
    return pairs


def _resolve_colors(colors, n):
    """Return ``n`` colours from a colormap name, Colormap, or explicit list."""
    if colors is None:
        colors = "viridis"

    if isinstance(colors, str):
        try:
            cmap = mpl.colormaps[colors]
        except (KeyError, AttributeError, TypeError):
            cmap = plt.get_cmap(colors)
        # Trim the extremes: the ends of most sequential maps are near-white
        # or near-black and vanish against a greyscale background.
        return [cmap(t) for t in np.linspace(0.08, 0.92, n)]

    if isinstance(colors, mpl.colors.Colormap):
        return [colors(t) for t in np.linspace(0.08, 0.92, n)]

    colors = list(colors)
    if len(colors) < n:
        raise ValueError(f"need at least {n} colours for {n} bins, got {len(colors)}.")
    return colors[:n]


def _truncate_cmap(cmap, vmin=0.0, vmax=1.0, n=256):
    """A colormap restricted to a sub-range of another.

    Used to keep the densest part of the greyscale background short of
    pure black, so that coloured contours and markers drawn on top of it
    stay legible, while the colourbar still runs cleanly from 0 to 1.
    """
    if isinstance(cmap, str):
        try:
            cmap = mpl.colormaps[cmap]
        except (KeyError, AttributeError, TypeError):
            cmap = plt.get_cmap(cmap)
    return mpl.colors.LinearSegmentedColormap.from_list(
        f"{cmap.name}_trunc", cmap(np.linspace(vmin, vmax, n))
    )


def _en_dash():
    """Range dash that is safe under both mathtext and usetex."""
    return "--" if mpl.rcParams.get("text.usetex", False) else "\u2013"


def _finite_mask(*arrays):
    """Entries finite in every input array."""
    good = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        if a is not None:
            good &= np.isfinite(a)
    return good


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def _weighted_quantile(values, quantiles, weights=None):
    """Weighted quantiles via the empirical CDF.  ``quantiles`` in [0, 1]."""
    values = np.asarray(values, float)
    q = np.atleast_1d(quantiles).astype(float)
    if weights is None:
        return np.percentile(values, 100.0 * q)
    weights = np.asarray(weights, float)
    order = np.argsort(values)
    v, w = values[order], weights[order]
    total = np.sum(w)
    if total <= 0:
        return np.percentile(values, 100.0 * q)
    cdf = (np.cumsum(w) - 0.5 * w) / total
    return np.interp(q, cdf, v)


def _summary(values, weights=None, statistic="median", spread="percentile",
             percentiles=(16.0, 84.0)):
    """Central value and asymmetric spread of a 1-D sample.

    Returns ``(centre, lower_error, upper_error)``, the errors already
    expressed as distances from the centre, ready for ``errorbar``.
    """
    values = np.asarray(values, float)
    ok = np.isfinite(values)
    values = values[ok]
    w = None if weights is None else np.asarray(weights, float)[ok]
    if values.size == 0:
        return np.nan, np.nan, np.nan

    if statistic == "median":
        centre = float(np.atleast_1d(_weighted_quantile(values, 0.5, w))[0])
    else:
        centre = float(np.average(values, weights=w))

    if spread == "percentile":
        lo, hi = np.atleast_1d(
            _weighted_quantile(values, np.asarray(percentiles) / 100.0, w))
    else:
        if w is None:
            s = float(np.std(values))
        else:
            s = float(np.sqrt(np.average((values - centre) ** 2, weights=w)))
        lo, hi = centre - s, centre + s

    return centre, max(centre - float(lo), 0.0), max(float(hi) - centre, 0.0)


def _bootstrap_error(values, weights=None, statistic="median", n_boot=500,
                     percentiles=(16.0, 84.0), rng=None):
    """Uncertainty *on the statistic itself*, by bootstrap resampling.

    A different quantity from the population spread returned by
    :func:`_summary`.  Which of the two belongs on the figure depends on
    the claim being made: the spread describes the scatter of the
    objects, the bootstrap error how well the centre is determined.
    """
    values = np.asarray(values, float)
    ok = np.isfinite(values)
    values = values[ok]
    w = None if weights is None else np.asarray(weights, float)[ok]
    n = values.size
    if n < 3:
        return np.nan, np.nan
    rng = np.random.default_rng() if rng is None else rng

    idx = rng.integers(0, n, size=(n_boot, n))
    if w is None:
        sample = values[idx]
        draws = (np.median(sample, axis=1) if statistic == "median"
                 else np.mean(sample, axis=1))
    else:
        draws = np.array([
            float(np.atleast_1d(_weighted_quantile(values[i], 0.5, w[i]))[0])
            if statistic == "median" else float(np.average(values[i], weights=w[i]))
            for i in idx
        ])

    centre = float(np.median(draws))
    lo, hi = np.percentile(draws, percentiles)
    return max(centre - float(lo), 0.0), max(float(hi) - centre, 0.0)


# ---------------------------------------------------------------------------
# density estimation
# ---------------------------------------------------------------------------

def _scott_bandwidth(values, n_dim, weights=None):
    """Scott's-rule bandwidth in data units."""
    values = np.asarray(values, float)
    if weights is None:
        n_eff = float(values.size)
    else:
        w = np.asarray(weights, float)
        sw2 = np.sum(w ** 2)
        n_eff = (np.sum(w) ** 2 / sw2) if sw2 > 0 else float(values.size)
    n_eff = max(n_eff, 2.0)
    sigma = float(np.std(values))
    if not np.isfinite(sigma) or sigma <= 0:
        return None
    return sigma * n_eff ** (-1.0 / (n_dim + 4))


def _grid_edges(grid):
    """Bin edges bracketing a vector of evenly spaced bin centres."""
    step = grid[1] - grid[0]
    return np.concatenate([grid - 0.5 * step, [grid[-1] + 0.5 * step]])


def _density_1d(values, grid, weights=None, method="auto", bw_scale=1.0,
                min_points=8, kde_max_points=5000):
    """Peak-normalised 1-D density on ``grid``; ``None`` if degenerate.

    Peak normalisation makes the curve <= 1 everywhere, so its base-10
    logarithm is <= 0 and bins containing very different numbers of
    objects can be compared directly in shape.
    """
    values = np.asarray(values, float)
    ok = np.isfinite(values)
    values = values[ok]
    w = None if weights is None else np.asarray(weights, float)[ok]
    if values.size < min_points or np.ptp(values) <= 0:
        return None

    if method == "auto":
        method = "kde" if (values.size <= kde_max_points and w is None) else "hist"

    if method == "kde":
        try:
            kde = gaussian_kde(values, weights=w)
            kde.set_bandwidth(kde.factor * bw_scale)
            dens = kde(grid)
        except (np.linalg.LinAlgError, ValueError):
            return None
    elif method == "hist":
        bw = _scott_bandwidth(values, 1, w)
        if bw is None:
            return None
        counts, _ = np.histogram(values, bins=_grid_edges(grid), weights=w)
        sigma_pix = (bw * bw_scale) / (grid[1] - grid[0])
        dens = gaussian_filter1d(counts.astype(float), max(sigma_pix, 0.6),
                                 mode="nearest")
    else:
        raise ValueError("density_method must be 'auto', 'kde' or 'hist'.")

    peak = float(np.max(dens))
    return dens / peak if peak > 0 else None


def _density_2d(xv, yv, xgrid, ygrid, weights=None, method="auto", bw_scale=1.0,
                min_points=12, kde_max_points=5000):
    """Peak-normalised 2-D density on a regular grid; ``None`` if degenerate.

    The ``'hist'`` branch is a histogram convolved with a Gaussian of
    Scott's-rule width: an O(N) approximation to the exact KDE, which
    costs O(N x M) and becomes prohibitive on large catalogues.
    """
    xv, yv = np.asarray(xv, float), np.asarray(yv, float)
    ok = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[ok], yv[ok]
    w = None if weights is None else np.asarray(weights, float)[ok]
    if xv.size < min_points or np.ptp(xv) <= 0 or np.ptp(yv) <= 0:
        return None

    if method == "auto":
        method = "kde" if (xv.size <= kde_max_points and w is None) else "hist"

    if method == "kde":
        try:
            kde = gaussian_kde(np.vstack([xv, yv]), weights=w)
            kde.set_bandwidth(kde.factor * bw_scale)
            XX, YY = np.meshgrid(xgrid, ygrid)
            dens = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
        except (np.linalg.LinAlgError, ValueError):
            return None
    elif method == "hist":
        bwx = _scott_bandwidth(xv, 2, w)
        bwy = _scott_bandwidth(yv, 2, w)
        if bwx is None or bwy is None:
            return None
        counts, _, _ = np.histogram2d(
            xv, yv, bins=[_grid_edges(xgrid), _grid_edges(ygrid)], weights=w
        )
        sig = (max(bwy * bw_scale / (ygrid[1] - ygrid[0]), 0.6),
               max(bwx * bw_scale / (xgrid[1] - xgrid[0]), 0.6))
        dens = gaussian_filter(counts.T.astype(float), sig, mode="nearest")
    else:
        raise ValueError("density_method must be 'auto', 'kde' or 'hist'.")

    peak = float(np.max(dens))
    return dens / peak if peak > 0 else None


def _hpd_level(density, frac):
    """Density level enclosing ``frac`` of the total probability mass."""
    d = np.sort(density.ravel())[::-1]
    cum = np.cumsum(d)
    if cum[-1] <= 0:
        return None
    cum /= cum[-1]
    idx = int(np.searchsorted(cum, frac))
    return float(d[min(idx, d.size - 1)])


# ---------------------------------------------------------------------------
# result containers
# ---------------------------------------------------------------------------

@dataclass
class BinStats:
    """Summary of one sub-population.

    Returned so the numbers behind the figure can be tabulated without
    recomputing them.
    """
    lo: float
    hi: float
    n: int = 0
    n_flagged: int = 0
    frac_flagged: float = np.nan
    z_rep: float = np.nan
    x_stat: float = np.nan
    x_err: tuple = (np.nan, np.nan)
    y_stat: float = np.nan
    y_err: tuple = (np.nan, np.nan)
    color: object = None

    def as_row(self, fmt="{:.3f}"):
        """A tab-separated row, convenient for pasting into a table."""
        f = fmt.format
        pct = (f(100 * self.frac_flagged)
               if np.isfinite(self.frac_flagged) else "--")
        return "\t".join([
            f(self.lo), f(self.hi), str(self.n), pct,
            f(self.x_stat), f(self.x_err[0]), f(self.x_err[1]),
            f(self.y_stat), f(self.y_err[0]), f(self.y_err[1]),
        ])

    def latex_row(self, fmt="{:.3f}"):
        r"""A LaTeX table row with the statistic as ``$c^{+u}_{-l}$``."""
        f = fmt.format
        pct = (f"{100 * self.frac_flagged:.0f}"
               if np.isfinite(self.frac_flagged) else "--")
        # Thin space rather than a comma: a comma in maths mode is an
        # operator and is typeset with the wrong spacing.
        n_tex = f"{self.n:,d}".replace(",", r"\,")
        return (rf"$[{self.lo:.2f}, {self.hi:.2f})$ & ${n_tex}$ & ${pct}$ & "
                rf"${f(self.y_stat)}^{{+{f(self.y_err[1])}}}"
                rf"_{{-{f(self.y_err[0])}}}$ \\")


@dataclass
class MarginalFigure:
    """Handle to the assembled figure."""
    fig: object
    main: object
    top: object
    right: object
    stats: list = field(default_factory=list)

    def save(self, path, **kwargs):
        """Save the figure; vector formats are preferred for submission."""
        self.fig.savefig(path, **kwargs)
        return path


# ---------------------------------------------------------------------------
# main routine
# ---------------------------------------------------------------------------

def _default_bin_label(lo, hi, z_rep, n_obj, frac_flagged, z_name,
                       fields=("z", "n", "flag"), show_flag_column=True):
    """Build a per-bin legend entry from a selectable set of fields.

    ``fields`` may contain ``'z'`` (representative value), ``'range'``
    (the bin interval), ``'n'`` (number of objects) and ``'flag'``
    (percentage flagged).  Narrow single-column figures usually want
    ``('z',)`` alone: a legend wide enough to cover the data is worse
    than no legend at all.
    """
    parts = []
    if "z" in fields:
        parts.append(rf"${z_name} \approx {z_rep:.3g}$" if np.isfinite(z_rep)
                     else rf"${z_name} \in [{lo:.3g}, {hi:.3g})$")
    if "range" in fields:
        parts.append(rf"${z_name} \in [{lo:.3g}, {hi:.3g})$")
    if "n" in fields:
        parts.append(f"N = {n_obj:,d}")
    # Show the flagged percentage for every bin once any object in the
    # sample is flagged, so that a bin reading "0%" is distinguishable
    # from one where the column simply was not reported.
    if "flag" in fields and show_flag_column and np.isfinite(frac_flagged):
        parts.append(f"{100 * frac_flagged:.0f}%")
    return " | ".join(parts)


def scatter_with_marginals(
    x,
    y,
    z=None,
    z_ranges=None,
    flagged=None,
    weights=None,
    *,
    # --- statistics -------------------------------------------------------
    statistic="median",
    spread="percentile",
    percentiles=(16.0, 84.0),
    errorbar="spread",              # 'spread' | 'statistic' | 'both'
    n_bootstrap=500,
    # --- colours ----------------------------------------------------------
    colors="tab10",
    flagged_color="crimson",
    background_cmap="Greys",
    background_max=0.85,
    # --- density estimation ----------------------------------------------
    density_method="auto",
    bw_scale=1.0,
    grid_size=256,
    kde_max_points=5000,
    # --- what to draw -----------------------------------------------------
    show_background=True,
    background_includes_flagged=True,
    show_contours=True,
    contour_frac=0.68,
    show_points=True,
    n_points=300,
    point_size=1.6,
    point_alpha=0.45,
    show_flagged=True,
    show_marginal_flagged=True,
    show_colorbar=True,
    # --- limits and labels ------------------------------------------------
    xlim=None,
    ylim=None,
    logp_lim=(-3.0, 0.15),
    xlabel=r"$x$",
    ylabel=r"$y$",
    plabel=r"$\log_{10} P$",
    z_name="z",
    bin_label_fmt=None,
    bin_label_fields=("z", "n", "flag"),
    flagged_label="Ill-defined",
    spread_label=None,
    # --- figure -----------------------------------------------------------
    figsize=None,
    width=7.1,
    aspect=0.85,
    panel_frac=0.24,
    legend_bins=True,
    legend_style=True,
    legend_marginal=True,
    legend_bins_loc="center left",
    legend_style_loc="upper right",
    min_points_per_bin=8,
    random_state=None,
):
    r"""Scatter plot with per-bin statistics and marginal log-PDF panels.

    Parameters
    ----------
    x, y : array_like
        Coordinates of the objects.
    z : array_like, optional
        Property used to split the sample.  Defaults to ``x``, covering
        the common case of binning in the quantity plotted horizontally.
    z_ranges : array_like
        A sequence of ``(lo, hi)`` pairs, or a 1-D array of bin edges.
        Bins are applied as ``lo <= z < hi``; the final bin is closed on
        the right, so the most extreme objects are not silently dropped.
    flagged : array_like of bool, optional
        Objects that are ill defined or contaminated.  They are drawn
        separately, reported as a percentage in the per-bin legend, given
        a dashed marginal curve, and **excluded** from the statistics and
        contours.
    weights : array_like, optional
        Per-object weights, propagated through statistics and densities.
    statistic : {'median', 'mean'}
        Central estimator, drawn as the square marker.
    spread : {'percentile', 'std'}
        Definition of the population scatter.
    errorbar : {'spread', 'statistic', 'both'}
        What the error bars represent.  ``'spread'`` shows the scatter of
        the objects; ``'statistic'`` the bootstrap uncertainty on the
        centre, much smaller for large samples; ``'both'`` draws the
        spread as a thin bar and the bootstrap error as a thick one.
        State which is plotted in the caption -- conflating them is a
        common and consequential error.
    colors : str, Colormap or list
        Colormap name, Colormap, or one colour per bin.
    density_method : {'auto', 'hist', 'kde'}
        Density estimator.  ``'auto'`` uses the exact KDE below
        ``kde_max_points`` unweighted objects and the fast smoothed
        histogram above it.
    bw_scale : float
        Multiplies the automatically chosen bandwidth: below 1 sharpens
        the density, above 1 smooths it.
    contour_frac : float
        Probability mass enclosed by the per-bin contour.
    show_points : bool
        Overlay a random representative subsample per bin, coloured by bin.
    width, aspect : float
        Figure width in inches and height/width ratio.  Use
        ``FIG_WIDTH[journal]`` for the correct text-block width.
    panel_frac : float
        Fraction of the figure given to the marginal panels.

    Returns
    -------
    MarginalFigure
        With attributes ``fig``, ``main``, ``top``, ``right`` and
        ``stats`` (a list of :class:`BinStats`), plus a ``save`` method.

    Notes
    -----
    The marginal curves are peak-normalised, so the top panel's vertical
    axis and the right panel's horizontal axis show :math:`\log_{10}` of
    a density scaled to unit maximum, not of a normalised probability
    density.  Say so in the caption.
    """
    # ---------------- input handling ------------------------------------
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError(f"x and y must have equal length ({x.size} vs {y.size}).")

    z = x if z is None else np.asarray(z, dtype=float).ravel()
    if z.size != x.size:
        raise ValueError("z must have the same length as x and y.")

    if flagged is None:
        flagged = np.zeros(x.size, dtype=bool)
    else:
        flagged = np.asarray(flagged).astype(bool).ravel()
        if flagged.size != x.size:
            raise ValueError("flagged must have the same length as x and y.")

    if weights is not None:
        weights = np.asarray(weights, dtype=float).ravel()
        if weights.size != x.size:
            raise ValueError("weights must have the same length as x and y.")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative.")

    if z_ranges is None:
        raise ValueError("z_ranges is required.")
    if statistic not in ("median", "mean"):
        raise ValueError("statistic must be 'median' or 'mean'.")
    if spread not in ("percentile", "std"):
        raise ValueError("spread must be 'percentile' or 'std'.")
    if errorbar not in ("spread", "statistic", "both"):
        raise ValueError("errorbar must be 'spread', 'statistic' or 'both'.")

    ranges = _as_ranges(z_ranges)
    bin_colors = _resolve_colors(colors, len(ranges))
    rng = np.random.default_rng(random_state)

    finite = _finite_mask(x, y, z, weights)
    if not finite.any():
        raise ValueError("no finite (x, y, z) triplets to plot.")

    if spread_label is None:
        spread_label = (
            f"{percentiles[0]:.0f}{_en_dash()}{percentiles[1]:.0f}th pct."
            if spread == "percentile" else r"$1\sigma$"
        )

    # ---------------- figure scaffolding --------------------------------
    if figsize is None:
        figsize = (width, width * aspect)
    fig = plt.figure(figsize=figsize)
    main_frac = 1.0 - panel_frac
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=(main_frac, panel_frac),
        height_ratios=(panel_frac, main_frac),
        wspace=0.035, hspace=0.035,
        left=0.095, right=0.985, bottom=0.095, top=0.985,
    )
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_top.tick_params(labelbottom=False)
    ax_right.tick_params(labelleft=False)

    # Explicit limits also define the density evaluation range, so zooming
    # does not spend grid resolution on empty regions.
    xs, ys = x[finite], y[finite]
    if xlim is None:
        pad = 0.04 * np.ptp(xs)
        xlim = (xs.min() - pad, xs.max() + pad)
    if ylim is None:
        pad = 0.04 * np.ptp(ys)
        ylim = (ys.min() - pad, ys.max() + pad)
    # A constant coordinate gives a zero-width range, which makes the axis
    # transform singular.  Widen it rather than letting matplotlib warn.
    def _nondegenerate(lim, fallback=0.5):
        lo, hi = float(lim[0]), float(lim[1])
        if not np.isfinite([lo, hi]).all() or hi <= lo:
            centre = lo if np.isfinite(lo) else 0.0
            return centre - fallback, centre + fallback
        return lo, hi

    xlim = _nondegenerate(xlim)
    ylim = _nondegenerate(ylim)
    xgrid = np.linspace(*xlim, grid_size)
    ygrid = np.linspace(*ylim, grid_size)

    # Resolve 'auto' once, for the whole figure, rather than per bin.  If
    # each bin chose its own estimator, sparsely and densely populated
    # bins in the same panel would be smoothed by different algorithms and
    # would not be strictly comparable -- a difference in the figure that
    # does not correspond to a difference in the data.
    if density_method == "auto":
        resolved_method = ("kde" if (int(finite.sum()) <= kde_max_points
                                     and weights is None) else "hist")
    else:
        resolved_method = density_method
    dens_kw = dict(method=resolved_method, bw_scale=bw_scale)

    # ---------------- greyscale background ------------------------------
    if show_background:
        bg = finite if background_includes_flagged else (finite & ~flagged)
        dens = _density_2d(x[bg], y[bg], xgrid, ygrid,
                           weights=None if weights is None else weights[bg],
                           **dens_kw)
        if dens is not None:
            im = ax.imshow(
                dens, origin="lower", aspect="auto",
                extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
                cmap=_truncate_cmap(background_cmap, 0.0, background_max),
                vmin=0.0, vmax=1.0,
                zorder=0, interpolation="bilinear", rasterized=True,
            )
            if show_colorbar:
                cax = ax.inset_axes([0.035, 0.930, 0.26, 0.020])
                cb = fig.colorbar(im, cax=cax, orientation="horizontal")
                cb.set_ticks([0.0, 0.5, 1.0])
                cb.outline.set_linewidth(0.5)
                small = mpl.rcParams["legend.fontsize"] - 0.5
                cax.tick_params(labelsize=small, length=1.8, width=0.5, pad=1.2)
                cax.set_title(_tex_escape("density (peak-normalised)"),
                              fontsize=small, pad=2.5)

    # ---------------- flagged objects -----------------------------------
    has_flagged = bool(flagged[finite].any())
    if show_flagged and has_flagged:
        sel = finite & flagged
        ax.plot(x[sel], y[sel], ls="none", marker=".", ms=point_size,
                mec="none", color=flagged_color, alpha=0.55, zorder=2,
                rasterized=True)

    # ---------------- per-bin loop --------------------------------------
    stats = []
    for (lo, hi), color in zip(ranges, bin_colors):
        is_last = (lo, hi) == ranges[-1]
        in_bin = finite & (z >= lo) & ((z <= hi) if is_last else (z < hi))
        good, bad = in_bin & ~flagged, in_bin & flagged
        n_obj, n_bad = int(in_bin.sum()), int(bad.sum())

        entry = BinStats(
            lo=lo, hi=hi, n=n_obj, n_flagged=n_bad,
            frac_flagged=(n_bad / n_obj if n_obj else np.nan),
            z_rep=(float(np.median(z[in_bin])) if n_obj else np.nan),
            color=color,
        )
        if int(good.sum()) < min_points_per_bin:
            stats.append(entry)
            continue

        gx, gy = x[good], y[good]
        gw = None if weights is None else weights[good]

        xc, xlo, xhi = _summary(gx, gw, statistic, spread, percentiles)
        yc, ylo, yhi = _summary(gy, gw, statistic, spread, percentiles)
        entry.x_stat, entry.x_err = xc, (xlo, xhi)
        entry.y_stat, entry.y_err = yc, (ylo, yhi)

        if errorbar in ("statistic", "both"):
            bxl, bxh = _bootstrap_error(gx, gw, statistic, n_bootstrap,
                                        percentiles, rng)
            byl, byh = _bootstrap_error(gy, gw, statistic, n_bootstrap,
                                        percentiles, rng)
        else:
            bxl = bxh = byl = byh = np.nan

        # --- representative individual objects ---------------------------
        if show_points:
            idx = np.flatnonzero(good)
            if idx.size > n_points:
                idx = rng.choice(idx, size=n_points, replace=False)
            ax.plot(x[idx], y[idx], ls="none", marker=".", ms=point_size,
                    mec="none", color=color, alpha=point_alpha, zorder=3,
                    rasterized=True)

        # --- highest-density contour -------------------------------------
        if show_contours:
            d = _density_2d(gx, gy, xgrid, ygrid, weights=gw, **dens_kw)
            if d is not None:
                lvl = _hpd_level(d, contour_frac)
                if lvl is not None and lvl > 0:
                    ax.contour(xgrid, ygrid, d, levels=[lvl], colors=[color],
                               linewidths=1.0, zorder=4)

        # --- statistic marker --------------------------------------------
        if errorbar == "both":
            ax.errorbar(xc, yc, xerr=[[xlo], [xhi]], yerr=[[ylo], [yhi]],
                        fmt="none", ecolor=color, elinewidth=0.9, alpha=0.85,
                        capsize=0, zorder=5)
            ex, ey, elw = [[bxl], [bxh]], [[byl], [byh]], 2.2
        elif errorbar == "statistic":
            ex, ey, elw = [[bxl], [bxh]], [[byl], [byh]], 1.8
        else:
            ex, ey, elw = [[xlo], [xhi]], [[ylo], [yhi]], 1.8
        ax.errorbar(xc, yc, xerr=ex, yerr=ey, marker="s", ms=5.0,
                    mfc=color, mec="0.15", mew=0.6, ecolor=color,
                    elinewidth=elw, capsize=0, zorder=6)

        # --- marginals ----------------------------------------------------
        px = _density_1d(gx, xgrid, gw, min_points=min_points_per_bin, **dens_kw)
        if px is not None:
            with np.errstate(divide="ignore"):
                ax_top.plot(xgrid, np.log10(px), color=color, lw=1.0, zorder=3)
        ax_top.axvline(xc, color=color, lw=1.2, zorder=4)
        ax_top.axvspan(xc - xlo, xc + xhi, color=color, alpha=0.16, lw=0, zorder=1)

        py = _density_1d(gy, ygrid, gw, min_points=min_points_per_bin, **dens_kw)
        if py is not None:
            with np.errstate(divide="ignore"):
                ax_right.plot(np.log10(py), ygrid, color=color, lw=1.0, zorder=3)
        ax_right.axhline(yc, color=color, lw=1.2, zorder=4)
        ax_right.axhspan(yc - ylo, yc + yhi, color=color, alpha=0.16, lw=0, zorder=1)

        # --- dashed marginals for the flagged sub-population -------------
        if show_marginal_flagged and n_bad >= min_points_per_bin:
            bw_ = None if weights is None else weights[bad]
            pxb = _density_1d(x[bad], xgrid, bw_, min_points=min_points_per_bin,
                              **dens_kw)
            if pxb is not None:
                with np.errstate(divide="ignore"):
                    ax_top.plot(xgrid, np.log10(pxb), color=color, lw=0.8,
                                ls=(0, (3, 2)), zorder=2)
            pyb = _density_1d(y[bad], ygrid, bw_, min_points=min_points_per_bin,
                              **dens_kw)
            if pyb is not None:
                with np.errstate(divide="ignore"):
                    ax_right.plot(np.log10(pyb), ygrid, color=color, lw=0.8,
                                  ls=(0, (3, 2)), zorder=2)

        stats.append(entry)

    # ---------------- axes cosmetics ------------------------------------
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax_top.set_ylim(*logp_lim)
    ax_top.set_ylabel(plabel)
    ax_right.set_xlim(*logp_lim)
    ax_right.set_xlabel(plabel)
    # Integer decades, with the tick that collides with the main axes pruned.
    ax_top.yaxis.set_major_locator(
        mpl.ticker.MaxNLocator(4, integer=True, prune="lower"))
    ax_right.xaxis.set_major_locator(
        mpl.ticker.MaxNLocator(3, integer=True, prune="lower"))

    # ---------------- legends -------------------------------------------
    if legend_style:
        handles = [
            Line2D([], [], ls="-", color="0.35", lw=1.0,
                   label=_tex_escape(
                       f"{int(round(100 * contour_frac))}% HPD contour")),
            Line2D([], [], ls="-", marker="s", color="0.45", mfc="0.65",
                   mec="0.15", ms=5, mew=0.6,
                   label=_tex_escape(f"{statistic} $\\pm$ {spread_label}")),
        ]
        if show_flagged and has_flagged:
            handles.append(Line2D([], [], ls="none", marker=".", ms=5,
                                  color=flagged_color,
                                  label=_tex_escape(flagged_label)))
        # add_artist keeps this legend when a second one is attached below
        ax.add_artist(ax.legend(handles=handles, loc=legend_style_loc))

    if legend_bins:
        if bin_label_fmt is None:
            def _label(s):
                return _default_bin_label(s.lo, s.hi, s.z_rep, s.n,
                                          s.frac_flagged, z_name,
                                          fields=bin_label_fields,
                                          show_flag_column=has_flagged)
        else:
            def _label(s):
                return bin_label_fmt(s.lo, s.hi, s.z_rep, s.n,
                                     s.frac_flagged, z_name)
        bin_handles = [
            Patch(facecolor=s.color, edgecolor="0.25", linewidth=0.5,
                  label=_tex_escape(_label(s)))
            for s in stats if s.n > 0
        ]
        if bin_handles:
            ax.legend(handles=bin_handles, loc=legend_bins_loc,
                      handlelength=1.0, handleheight=0.9)

    if legend_marginal:
        top_handles = [
            Line2D([], [], color="0.35", lw=1.0, label=_tex_escape("all objects")),
            Line2D([], [], color="0.35", lw=1.2, label=_tex_escape(statistic)),
            Patch(facecolor="0.55", alpha=0.4, label=_tex_escape(spread_label)),
        ]
        if show_marginal_flagged and has_flagged:
            top_handles.append(Line2D([], [], color="0.35", lw=0.8,
                                      ls=(0, (3, 2)),
                                      label=_tex_escape(flagged_label)))
        ax_top.legend(handles=top_handles, loc="lower right",
                      ncol=len(top_handles), handlelength=1.3,
                      columnspacing=1.0, borderpad=0.25,
                      fontsize=mpl.rcParams["legend.fontsize"] - 0.5)

    return MarginalFigure(fig=fig, main=ax, top=ax_top, right=ax_right,
                          stats=stats)


# ---------------------------------------------------------------------------
# worked example
# ---------------------------------------------------------------------------

def _toy_dataset(n=60000, seed=42):
    """A mock concentration--mass relation with mass-dependent bad fits.

    Built to reproduce the qualitative features of the real measurement: a
    steeply falling mass function, so the lowest-mass bin holds two orders
    of magnitude more objects than the highest; a mildly declining c(M)
    with lognormal scatter; and a contaminated population confined almost
    entirely to the poorly resolved low-mass end and biased high in
    concentration.
    """
    rng = np.random.default_rng(seed)

    logM = 3.10 + rng.exponential(0.42, n)
    logc = 1.13 - 0.115 * (logM - 3.10) + rng.normal(0.0, 0.055, n)

    p_bad = 0.55 * np.exp(-(logM - 3.10) / 0.10)
    bad = rng.random(n) < p_bad
    logc[bad] = 1.05 + 1.55 * (logM[bad] - 3.10) + rng.normal(0.0, 0.012, bad.sum())

    return logM, logc, bad


def _example():
    """Build both a double-column and a single-column version."""
    single, double = set_journal_style(journal="aas", fontsize=8.5)

    logM, logc, bad = _toy_dataset()
    edges = [3.10, 3.35, 3.80, 4.25, 4.65, 5.40]

    config_dict = dict(
        x=logM, y=logc, z=logM, z_ranges=edges, flagged=bad,
        statistic="median", spread="percentile", errorbar="spread",
        colors="plasma", contour_frac=0.68, density_method="auto",
        xlabel=r"$\log_{10} M_{200}\,[10^{10}\,M_{\odot}\,h^{-1}]$",
        ylabel=r"$\log_{10} c$",
        z_name=r"\log_{10} M_{200}",
        flagged_label=r"ill-defined $r_{-2}$",
        xlim=(3.05, 5.20), ylim=(0.80, 1.28), logp_lim=(-3.0, 0.15),
        random_state=0,
    )

    # Single-column version.  The annotations that fit comfortably across
    # two columns will cover the data at half the width, so the legend is
    # reduced to the bin value alone and the busier elements are dropped.
    fig = scatter_with_marginals(
        width=single, aspect=1.05, panel_frac=0.26,
        show_points=True, show_colorbar=False, legend_marginal=False,
        bin_label_fields=("z",), legend_bins_loc="lower left",
        legend_style_loc="upper right", **config_dict)
    fig.save("scatter_marginals_narrow.pdf")
    fig.save("scatter_marginals_narrow.png", dpi=300)

    return fig


if __name__ == "__main__":
    _example()
