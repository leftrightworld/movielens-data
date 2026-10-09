"""Preprocess MovieLens-1M and Amazon Video Games into implicit-feedback splits.

Protocol (standard for implicit top-K recommendation, cf. LightGCN / NGCF):
  1. Every observed (user, item) interaction is a positive signal (implicit feedback).
  2. Duplicate (user, item) pairs are collapsed to one interaction.
  3. Iterative 5-core filtering: keep users and items with >= 5 interactions.
  4. Per-user split into train / valid / test = 80% / 10% / 10%.
     Default: random (seed 42).  --temporal: chronological per user — the last
     10% of each user's interactions become the test set, the previous 10% the
     validation set (strictly "predict the future from the past").

Outputs (per dataset, under data/):
  <name>.npz   : train / valid / test arrays of (user_idx, item_idx), n_users, n_items
  <name>_stats.json : dataset statistics used in the report
Temporal variants are suffixed "-t" (ml-1m-t, vgames-t).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]   # repo root (movielens-data/)
OUT = Path(__file__).resolve().parents[1] / "data"
SEED = 42


def five_core(df: pd.DataFrame) -> pd.DataFrame:
    """Iteratively drop users/items with fewer than 5 interactions."""
    while True:
        uc = df["user"].map(df["user"].value_counts())
        ic = df["item"].map(df["item"].value_counts())
        keep = (uc >= 5) & (ic >= 5)
        if keep.all():
            return df
        df = df[keep]


def split_and_save(df: pd.DataFrame, name: str, temporal: bool = False) -> dict:
    df = df.drop_duplicates(subset=["user", "item"]).copy()
    n_raw = len(df)
    df = five_core(df)

    user_ids = {u: k for k, u in enumerate(sorted(df["user"].unique()))}
    item_ids = {i: k for k, i in enumerate(sorted(df["item"].unique()))}
    df["u"] = df["user"].map(user_ids).astype(np.int64)
    df["i"] = df["item"].map(item_ids).astype(np.int64)
    if temporal:
        df = df.sort_values("ts", kind="stable")

    rng = np.random.default_rng(SEED)
    train, valid, test = [], [], []
    for _, grp in df.groupby("u", sort=False):
        arr = grp[["u", "i"]].to_numpy()
        n = len(arr)
        n_te = max(1, int(round(n * 0.1)))
        n_va = max(1, int(round(n * 0.1)))
        if temporal:
            # arr is in chronological order: oldest -> train, newest -> test
            test.append(arr[n - n_te:])
            valid.append(arr[n - n_te - n_va:n - n_te])
            train.append(arr[:n - n_te - n_va])
        else:
            rng.shuffle(arr)
            test.append(arr[:n_te])
            valid.append(arr[n_te:n_te + n_va])
            train.append(arr[n_te + n_va:])
    train, valid, test = (np.concatenate(x) for x in (train, valid, test))

    n_users, n_items = len(user_ids), len(item_ids)
    np.savez_compressed(
        OUT / f"{name}.npz",
        train=train, valid=valid, test=test,
        n_users=n_users, n_items=n_items,
    )
    stats = {
        "dataset": name,
        "split_protocol": "temporal (per-user leave-last-10%)" if temporal
                          else "random per-user 80/10/10 (seed 42)",
        "interactions_after_dedup": int(n_raw),
        "interactions_after_5core": int(len(df)),
        "n_users": n_users,
        "n_items": n_items,
        "density_pct": round(100 * len(df) / (n_users * n_items), 4),
        "avg_interactions_per_user": round(len(df) / n_users, 2),
        "avg_interactions_per_item": round(len(df) / n_items, 2),
        "split": {"train": int(len(train)), "valid": int(len(valid)), "test": int(len(test))},
    }
    with open(OUT / f"{name}_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def load_raw(temporal: bool):
    ml_cols = ["user_id", "movie_id"] + (["timestamp"] if temporal else [])
    ml = pd.read_csv(ROOT / "movielens-1m" / "ratings.csv", usecols=ml_cols)
    ml.columns = ["user", "item"] + (["ts"] if temporal else [])

    vg_cols = ["user_id", "parent_asin"] + (["timestamp"] if temporal else [])
    vg = pd.concat([
        pd.read_csv(ROOT / "amazon-vgames-2023" / f"ratings_part{i}.csv.gz",
                    usecols=vg_cols)
        for i in (1, 2)
    ], ignore_index=True)
    vg = vg[vg_cols]
    vg.columns = ["user", "item"] + (["ts"] if temporal else [])
    return ml, vg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--temporal", action="store_true",
                   help="build chronological splits (ml-1m-t, vgames-t) "
                        "instead of the random ones")
    args = p.parse_args()

    OUT.mkdir(exist_ok=True)
    suffix = "-t" if args.temporal else ""
    ml, vg = load_raw(args.temporal)
    print(json.dumps(split_and_save(ml, f"ml-1m{suffix}", args.temporal), indent=2))
    print(json.dumps(split_and_save(vg, f"vgames{suffix}", args.temporal), indent=2))


if __name__ == "__main__":
    main()
