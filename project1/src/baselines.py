"""No-training baselines evaluated under the exact protocol of train.py.

  Popularity : every user gets the same list, items ranked by train count.
  ItemKNN    : item-item cosine similarity on the binary train matrix,
               score(u, i) = sum of sim(i, j) over the user's train items j,
               truncated to the top-k neighbors per item (default k=100).

Usage:
  python3 baselines.py --dataset ml-1m  --model pop
  python3 baselines.py --dataset ml-1m  --model itemknn
Writes results/<dataset>_<model>.json in the same format as train.py.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from train import load_dataset, group_by_user

HERE = Path(__file__).resolve().parents[1]


@torch.no_grad()
def evaluate_scores(score_fn, train_by_user, eval_by_user, n_users, n_items,
                    device, ks=(10, 20), extra_mask_by_user=None, batch=1024):
    """Same metrics as train.evaluate, but takes score_fn(users)->(b, n_items)."""
    ks = sorted(ks)
    kmax = ks[-1]
    hits = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}
    idcg_cache = np.cumsum(1.0 / np.log2(np.arange(2, kmax + 2)))
    for start in range(0, n_users, batch):
        users = np.arange(start, min(start + batch, n_users))
        scores = score_fn(users).float()
        for row, u in enumerate(users):
            scores[row, train_by_user[u]] = -np.inf
            if extra_mask_by_user is not None and len(extra_mask_by_user[u]):
                scores[row, extra_mask_by_user[u]] = -np.inf
        top = torch.topk(scores, kmax, dim=1).indices.cpu().numpy()
        for row, u in enumerate(users):
            pos = eval_by_user[u]
            if len(pos) == 0:
                continue
            pos_set = set(pos.tolist())
            rel = np.fromiter((i in pos_set for i in top[row]), dtype=bool,
                              count=kmax)
            gains = rel / np.log2(np.arange(2, kmax + 2))
            for k in ks:
                hits[k].append(rel[:k].sum() / len(pos))
                idcg = idcg_cache[min(len(pos), k) - 1]
                ndcgs[k].append(gains[:k].sum() / idcg)
    return {f"{m}@{k}": float(np.mean(v[k]))
            for m, v in (("recall", hits), ("ndcg", ndcgs)) for k in ks}


def build_itemknn_sim(train, n_users, n_items, device, knn=100, chunk=2048):
    """Cosine item-item similarity, top-knn neighbors kept per row."""
    # sparse mm is unsupported on MPS -> run the similarity build on CPU there
    sp_dev = device if device == "cuda" else "cpu"
    counts = np.bincount(train[:, 1], minlength=n_items)
    inv_norm = 1.0 / np.sqrt(np.maximum(counts, 1))
    vals = torch.from_numpy(inv_norm[train[:, 1]]).float()
    idx_t = torch.from_numpy(train[:, [1, 0]].T.copy())      # (item, user)
    n_t = torch.sparse_coo_tensor(idx_t, vals, (n_items, n_users)).coalesce().to(sp_dev)

    # column-normalized dense chunks of N (users x chunk of items)
    order = np.argsort(train[:, 1], kind="stable")
    tr_sorted = train[order]
    vals_sorted = vals.numpy()[order]
    starts = np.searchsorted(tr_sorted[:, 1], np.arange(n_items + 1))

    sim = torch.zeros((n_items, n_items), dtype=torch.float16, device=device)
    for c0 in range(0, n_items, chunk):
        c1 = min(c0 + chunk, n_items)
        lo, hi = starts[c0], starts[c1]
        dense = torch.zeros((n_users, c1 - c0), device=sp_dev)
        rows = torch.from_numpy(tr_sorted[lo:hi, 0]).to(sp_dev)
        cols = torch.from_numpy(tr_sorted[lo:hi, 1] - c0).to(sp_dev)
        dense[rows, cols] = torch.from_numpy(vals_sorted[lo:hi]).float().to(sp_dev)
        sim[:, c0:c1] = torch.sparse.mm(n_t, dense).half().to(device)
    sim.fill_diagonal_(0)

    # keep only the top-knn neighbors per row
    for r0 in range(0, n_items, chunk):
        r1 = min(r0 + chunk, n_items)
        block = sim[r0:r1].float()
        kv, ki = torch.topk(block, min(knn, n_items - 1), dim=1)
        pruned = torch.zeros_like(block)
        pruned.scatter_(1, ki, kv)
        sim[r0:r1] = pruned.half()
    return sim


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True, choices=["pop", "itemknn"])
    p.add_argument("--knn", type=int, default=100)
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    train, valid, test, n_users, n_items = load_dataset(args.dataset)
    train_by_user = group_by_user(train, n_users)
    valid_by_user = group_by_user(valid, n_users)
    test_by_user = group_by_user(test, n_users)
    t0 = time.time()

    if args.model == "pop":
        pop = torch.from_numpy(
            np.bincount(train[:, 1], minlength=n_items).astype(np.float32)).to(device)

        def score_fn(users):
            return pop.unsqueeze(0).repeat(len(users), 1)
    else:
        sim = build_itemknn_sim(train, n_users, n_items, device, knn=args.knn)

        def score_fn(users):
            hist = torch.zeros((len(users), n_items), dtype=torch.float16,
                               device=device)
            for row, u in enumerate(users):
                hist[row, train_by_user[u]] = 1.0
            return hist @ sim

    test_m = evaluate_scores(score_fn, train_by_user, test_by_user,
                             n_users, n_items, device,
                             extra_mask_by_user=valid_by_user)
    name = f"{args.dataset}_{args.model}"
    result = {
        "name": name,
        "config": vars(args),
        "device": device,
        "train_seconds": round(time.time() - t0, 1),
        "best_epoch": 0,            # no training
        "test": test_m,
        "history": [],
    }
    out = HERE / "results" / f"{name}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{name}] TEST {json.dumps(test_m)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
