#!/usr/bin/env bash
# Copy the minimal public model surface from hp-transfer-gts and rewrite
# package imports gt_transfer -> gnn_transfer.
set -euo pipefail

SRC_ROOT="${1:-$HOME/hp-transfer-gts}"
DST_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$SRC_ROOT/src/gt_transfer"
DST="$DST_ROOT/src/gnn_transfer"

if [[ ! -d "$SRC" ]]; then
  echo "Source not found: $SRC" >&2
  exit 1
fi

mkdir -p "$DST/models/gnns" "$DST/models/encoders"

declare -a FILES=(
  "models/gnns/gcn.py"
  "models/gnns/simple_transfer_gnn.py"
  "models/gnns/standard_gnn.py"
  "models/encoders/standard_encoders.py"
  "models/encoders/deep_node_encoder.py"
  "models/encoders/simple_lappe.py"
)

for f in "${FILES[@]}"; do
  cp "$SRC/$f" "$DST/$f"
  echo "copied $f"
done

find "$DST/models" -name '*.py' -print0 | xargs -0 sed -i 's/gt_transfer/gnn_transfer/g'
echo "rewrote imports under $DST/models"
echo "Done. Review diffs, then commit."
