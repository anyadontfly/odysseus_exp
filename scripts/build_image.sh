#!/bin/bash
set -euo pipefail

DEF_FILE="/projects/bcjw/pyao3/odysseus_exp/trl_dev.def"
SIF_OUT="/projects/bcjw/pyao3/trl-dev.sif"
CACHE_BASE="/projects/bcjw/pyao3"

export APPTAINER_TMPDIR="${CACHE_BASE}/.apptainer_tmp"
export APPTAINER_CACHEDIR="${CACHE_BASE}/.apptainer_cache"

mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

echo "========================================"
echo " TRL dev container build"
echo "========================================"
echo " def file : $DEF_FILE"
echo " output   : $SIF_OUT"
echo " tmp dir  : $APPTAINER_TMPDIR"
echo " cache dir: $APPTAINER_CACHEDIR"
echo " started  : $(date)"
echo "========================================"

apptainer build --fakeroot "$SIF_OUT" "$DEF_FILE"

echo "========================================"
echo " build complete: $(date)"
echo " SIF: $SIF_OUT ($(du -sh "$SIF_OUT" | cut -f1))"
echo "========================================"