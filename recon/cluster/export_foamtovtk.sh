#!/bin/bash
# Alternative to export_deformed.py: export a reconstructed OpenFOAM case to VTK with
# the native foamToVTK from the OF7 apptainer container. Gives the full 3D internal
# mesh AND the boundary patches (airfoil + flap walls) with moving points per time —
# the most faithful "deformed mesh" for ParaView, at the cost of bigger files.
#
# The case must already be reconstructed (root time dirs [0-9]*), e.g. produced by
# field_run_flap.pbs into /scratch_local/$USER/fldrun_<sim>. foamToVTK reads the
# moving-mesh points from each time dir, so the flap deflection is preserved.
#
# Usage:
#   bash export_foamtovtk.sh <case_dir> [ "<time_ranges>" ]
# Examples:
#   bash export_foamtovtk.sh /scratch_local/$USER/fldrun_sim_Cc_060_test
#   bash export_foamtovtk.sh /scratch_local/$USER/fldrun_sim_Cc_060_test "1.2:1.6"
# time_ranges uses OpenFOAM timeSelector syntax (comma list and/or colon ranges).
# Omit it to export every time dir (can be large — scp only what you need).
set -e

CASE="${1:?usage: export_foamtovtk.sh <case_dir> [time_ranges]}"
TIMES="${2:-}"
CONTAINER="${OF7_SIF:-/work/u10677113/of7.sif}"

[ -d "$CASE" ] || { echo "case dir not found: $CASE" >&2; exit 1; }
[ -f "$CONTAINER" ] || { echo "OF7 container not found: $CONTAINER (set OF7_SIF)" >&2; exit 1; }

NT=$(ls -d "$CASE"/[0-9]* 2>/dev/null | wc -l)
echo "case: $CASE   ($NT numeric time dirs)   container: $CONTAINER"
[ "$NT" -gt 0 ] || { echo "no reconstructed time dirs — run field_run_flap.pbs first" >&2; exit 1; }

TIME_ARG=""
[ -n "$TIMES" ] && TIME_ARG="-time $TIMES"

# foamToVTK inside the OF7 container; bind both work and scratch_local.
apptainer exec --bind /work --bind /scratch_local "$CONTAINER" /bin/bash -c \
    "source /opt/openfoam7/etc/bashrc && cd '$CASE' && foamToVTK -fields '(p U)' $TIME_ARG"

echo
echo "done -> $CASE/VTK"
echo "internal mesh:  $CASE/VTK/*.vtk       (open the series in ParaView)"
echo "wall patches:   $CASE/VTK/<patch>/    (airfoil/flap surfaces)"
echo "scp the VTK/ dir home, then open in ParaView ('Surface With Edges' shows the grid)."
