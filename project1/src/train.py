"""Train MF-BPR or LightGCN on a preprocessed dataset and log results.

Usage:
  python3 src/train.py --dataset ml-1m --model mf       --dim 64
  python3 src/train.py --dataset ml-1m --model lightgcn --dim 64 --layers 3

Protocol: BPR training with 1 uniformly sampled negative per positive,
full-ranking evaluation (all items the user has not interacted with in
train), Recall@K and NDCG@K, early stopping on validation NDCG@10.
Results go to results/<dataset>_<model>_<tag>.json; the best embeddings
are saved alongside for the analysis notebook.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from models import MFBPR, LightGCN, bpr_loss

HERE = Path(__file__).resolve().parents[1]


def load_dataset(name):
    z = np.load(HERE / "data" / f"{name}.npz")
    return (z["train"], z["valid"], z["test"],
            int(z["n_users"]), int(z["n_items"]))


def group_by_user(pairs, n_users):
    """list of np arrays: items per user."""
    out = [[] for _ in range(n_users)]
    for u, i in pairs:
        out[u].append(i)
    return [np.asarray(x, dtype=np.int64) for x in out]


@torch.no_grad()
def evaluate(model, train_by_user, eval_by_user, n_users, n_items, device,
             ks=(10, 20), extra_mask_by_user=None, batch=1024):
    """Full-ranking Recall@K / NDCG@K, averaged over users with eval items."""
    model.eval()
    user_e, item_e = model.final_embeddings()
    ks = sorted(ks)
    kmax = ks[-1]
    hits = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}
    topk_all = np.zeros((n_users, kmax), dtype=np.int64)

    idcg_cache = np.cumsum(1.0 / np.log2(np.arange(2, kmax + 2)))
    for start in range(0, n_users, batch):
        users = np.arange(start, min(start + batch, n_users))
        scores = user_e[users] @ item_e.T
        for row, u in enumerate(users):
            scores[row, train_by_user[u]] = -np.inf
            if extra_mask_by_user is not None and len(extra_mask_by_user[u]):
                scores[row, extra_mask_by_user[u]] = -np.inf
        top = torch.topk(scores, kmax, dim=1).indices.cpu().numpy()
        topk_all[users] = top
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
    metrics = {}
    for k in ks:
        metrics[f"recall@{k}"] = float(np.mean(hits[k]))
        metrics[f"ndcg@{k}"] = float(np.mean(ndcgs[k]))
    return metrics, topk_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True, choices=["mf", "lightgcn"])
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--reg", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=6)   # 6 evals = 30 epochs
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="")
    p.add_argument("--save-emb", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")

    train, valid, test, n_users, n_items = load_dataset(args.dataset)
    train_by_user = group_by_user(train, n_users)
    valid_by_user = group_by_user(valid, n_users)
    test_by_user = group_by_user(test, n_users)

    if args.model == "mf":
        model = MFBPR(n_users, n_items, args.dim)
        name = f"{args.dataset}_mf_d{args.dim}"
    else:
        model = LightGCN(n_users, n_items, args.dim, args.layers,
                         torch.from_numpy(train))
        name = f"{args.dataset}_lightgcn_d{args.dim}_l{args.layers}"
    if args.tag:
        name += f"_{args.tag}"
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    tr_u = torch.from_numpy(train[:, 0]).to(device)
    tr_i = torch.from_numpy(train[:, 1]).to(device)
    n_train = len(train)

    best = {"ndcg@10": -1}
    best_state = None
    history = []
    epochs_since_best = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.from_numpy(rng.permutation(n_train)).to(device)
        negs = torch.from_numpy(
            rng.integers(0, n_items, n_train)).to(device)
        total = 0.0
        for s in range(0, n_train, args.batch):
            idx = perm[s:s + args.batch]
            u, i, j = tr_u[idx], tr_i[idx], negs[idx]
            if args.model == "lightgcn":
                user_all, item_all = model.propagate()
                ue, pe, ne = user_all[u], item_all[i], item_all[j]
            else:
                ue, pe, ne = model.user_emb(u), model.item_emb(i), model.item_emb(j)
            loss = bpr_loss(ue, pe, ne, model.ego_embeddings(u, i, j), args.reg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)

        if epoch % args.eval_every == 0:
            m, _ = evaluate(model, train_by_user, valid_by_user,
                            n_users, n_items, device)
            history.append({"epoch": epoch, "loss": total / n_train, **m})
            flag = ""
            if m["ndcg@10"] > best["ndcg@10"]:
                best = m
                best["epoch"] = epoch
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                epochs_since_best = 0
                flag = " *"
            else:
                epochs_since_best += 1
            print(f"[{name}] epoch {epoch:3d} loss {total/n_train:.4f} "
                  f"valid ndcg@10 {m['ndcg@10']:.4f} recall@20 {m['recall@20']:.4f}"
                  f"{flag}", flush=True)
            if epochs_since_best >= args.patience:
                print(f"[{name}] early stop at epoch {epoch}", flush=True)
                break

    model.load_state_dict(best_state)
    test_m, topk = evaluate(model, train_by_user, test_by_user,
                            n_users, n_items, device,
                            extra_mask_by_user=valid_by_user)
    result = {
        "name": name,
        "config": vars(args),
        "device": device,
        "train_seconds": round(time.time() - t0, 1),
        "best_epoch": best["epoch"],
        "valid": {k: v for k, v in best.items() if k != "epoch"},
        "test": test_m,
        "history": history,
    }
    out = HERE / "results" / f"{name}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    if args.save_emb:
        ue, ie = model.final_embeddings()
        np.savez_compressed(HERE / "results" / f"{name}_emb.npz",
                            user=ue.detach().cpu().numpy(),
                            item=ie.detach().cpu().numpy(),
                            topk=topk)
    print(f"[{name}] TEST {json.dumps(test_m)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
