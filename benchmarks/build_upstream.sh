#!/usr/bin/env bash
# Build the ORIGINAL (upstream) fake_spectra next to this script, so it can be
# compared with the HCD fork without touching the conda environment.
#
# The source is exported from a tag of the fork repository (the fork keeps
# the upstream tags), and the C extension is compiled in place. Nothing is
# installed: the result is just the folder benchmarks/fake_spectra_upstream/.
#
# Usage (from anywhere, with the skewerjax_fakespectra env active):
#     bash benchmarks/build_upstream.sh                 # tag v2.2.4
#     bash benchmarks/build_upstream.sh v2.2.4 ~/fake_spectra_fork
set -euo pipefail

TAG="${1:-v2.2.4}"                            # upstream release the fork started from
FORK_REPO="${2:-$HOME/fake_spectra_fork}"     # git repo of the fork
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$HERE/fake_spectra_upstream"

# Start from a clean folder, so an old build never survives a tag change
rm -rf "$DEST"
mkdir -p "$DEST"

# Export the tagged source tree (no .git, no local edits)
git -C "$FORK_REPO" archive "$TAG" | tar -x -C "$DEST"
echo "$TAG" > "$DEST/UPSTREAM_TAG"

# cloudy_tables is a git submodule, which `git archive` leaves out. Export it
# at the commit that the tag points to (the fork never changed it).
SUB_SHA="$(git -C "$FORK_REPO" ls-tree "$TAG" fake_spectra/cloudy_tables | awk '{print $3}')"
git -C "$FORK_REPO/fake_spectra/cloudy_tables" archive "$SUB_SHA" \
    | tar -x -C "$DEST/fake_spectra/cloudy_tables"

# Compile _spectra_priv in place (needs gsl-config and a C++ compiler)
cd "$DEST"
python setup.py build_ext --inplace > "$HERE/build_upstream.log" 2>&1
echo "Built upstream fake_spectra $TAG in $DEST (log: $HERE/build_upstream.log)"
