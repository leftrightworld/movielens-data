#!/bin/bash
# Extension round: degree-alpha LightGCN sweep on the sparse dataset.
# alpha=0.5 is the original model (re-run to record beyond-accuracy metrics
# in its JSON; metrics are identical by seed). MF re-run for the same reason.
set -uo pipefail
cd "$(dirname "$0")"

LG="--batch 65536 --lr 3e-3 --epochs 600"

python3 train.py --dataset vgames --model mf --dim 64 --save-emb
for A in 0.5 0.6 0.7 0.8; do
  python3 train.py --dataset vgames --model lightgcn --dim 64 --layers 3 \
    --alpha $A $LG --save-emb
done

echo "EXTENSION DONE"
