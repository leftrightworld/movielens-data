#!/bin/bash
# Additions round: no-training baselines + temporal-split robustness check.
set -uo pipefail
cd "$(dirname "$0")"

LG="--batch 65536 --lr 3e-3 --epochs 600"

# --- 1. Baselines on the existing random splits ---
python3 baselines.py --dataset ml-1m  --model pop
python3 baselines.py --dataset vgames --model pop
python3 baselines.py --dataset ml-1m  --model itemknn
python3 baselines.py --dataset vgames --model itemknn

# --- 2. Temporal (leave-last-10%) splits ---
python3 data_prep.py --temporal

# --- 3. All four methods on the temporal splits ---
for DS in ml-1m-t vgames-t; do
  python3 baselines.py --dataset $DS --model pop
  python3 baselines.py --dataset $DS --model itemknn
  python3 train.py --dataset $DS --model mf --dim 64
  python3 train.py --dataset $DS --model lightgcn --dim 64 --layers 3 $LG
done

echo "ADDITIONS DONE"
