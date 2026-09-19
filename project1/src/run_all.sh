#!/bin/bash
# Full experiment queue for Project 1. Runs sequentially; each run writes
# results/<name>.json. Main results first, then ablations and sweeps.
set -uo pipefail
cd "$(dirname "$0")"

LG="--batch 65536 --lr 3e-3"      # large-batch config validated for LightGCN

# --- 1. Main results (embeddings saved for the analysis notebook) ---
python3 train.py --dataset ml-1m  --model mf       --dim 64 --save-emb
python3 train.py --dataset ml-1m  --model lightgcn --dim 64 --layers 3 $LG --save-emb
python3 train.py --dataset vgames --model mf       --dim 64 --save-emb
python3 train.py --dataset vgames --model lightgcn --dim 64 --layers 3 $LG --save-emb

# --- 2. Ablation: number of propagation layers (LightGCN) ---
for L in 1 2 4; do
  python3 train.py --dataset ml-1m  --model lightgcn --dim 64 --layers $L $LG
  python3 train.py --dataset vgames --model lightgcn --dim 64 --layers $L $LG
done

# --- 3. Parameter sweep: embedding dimension ---
for D in 16 32; do
  python3 train.py --dataset ml-1m  --model mf       --dim $D
  python3 train.py --dataset ml-1m  --model lightgcn --dim $D --layers 3 $LG
  python3 train.py --dataset vgames --model mf       --dim $D
  python3 train.py --dataset vgames --model lightgcn --dim $D --layers 3 $LG
done

echo "ALL RUNS COMPLETE"
