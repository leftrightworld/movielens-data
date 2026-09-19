"""Preprocess MovieLens-1M and Amazon All_Beauty into implicit-feedback splits.

Protocol (standard for implicit top-K recommendation, cf. LightGCN / NGCF):
  1. Every observed (user, item) interaction is a positive signal (implicit feedback).
  2. Duplicate (user, item) pairs are collapsed to one interaction.
  3. Iterative 5-core filtering: keep users and items with >= 5 interactions.
  4. Per-user random split into train / valid / test = 80% / 10% / 10%.

Outputs (per dataset, under data/):
  <name>.npz   : train / valid / test arrays of (user_idx, item_idx), n_users, n_items
  <name>_stats.json : dataset statistics used in the report
"""
import json
import sys
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


def split_and_save(df: pd.DataFrame, name: str) -> dict:
    df = df.drop_duplicates(subset=["user", "item"]).copy()
    n_raw = len(df)
    df = five_core(df)

    user_ids = {u: k for k, u in enumerate(sorted(df["user"].unique()))}
    item_ids = {i: k for k, i in enumerate(sorted(df["item"].unique()))}
    df["u"] = df["user"].map(user_ids).astype(np.int64)
    df["i"] = df["item"].map(item_ids).astype(np.int64)

    rng = np.random.default_rng(SEED)
    train, valid, test = [], [], []
    for _, grp in df.groupby("u", sort=False):
        arr = grp[["u", "i"]].to_numpy()
        rng.shuffle(arr)
        n = len(arr)
        n_te = max(1, int(round(n * 0.1)))
        n_va = max(1, int(round(n * 0.1)))
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


def main():
    OUT.mkdir(exist_ok=True)

    ml = pd.read_csv(ROOT / "ratings.csv", usecols=["user_id", "movie_id"])
    ml.columns = ["user", "item"]
    print(json.dumps(split_and_save(ml, "ml-1m"), indent=2))

    bt = pd.read_csv(ROOT / "amazon-beauty-2023" / "reviews.csv.gz",
                     usecols=["user_id", "parent_asin"])
    bt.columns = ["user", "item"]
    print(json.dumps(split_and_save(bt, "beauty"), indent=2))

    # Amazon Video_Games 2023: (user, parent_asin) pairs pre-extracted from the
    # raw Video_Games.jsonl (see notebook Section 2 for the download source).
    vg_path = OUT / "video_games_pairs.parquet"
    if vg_path.exists():
        vg = pd.read_parquet(vg_path)
        print(json.dumps(split_and_save(vg, "vgames"), indent=2))


if __name__ == "__main__":
    sys.exit(main())
