"""Build the companion notebook (project1_report.ipynb) for Project 1.

The notebook is the deliverable shown to the professor: dataset introduction,
preprocessing protocol, method descriptions, experiment results and analysis.
All analysis cells read precomputed logs from results/ so the notebook
executes in minutes; full training is reproduced via src/run_all.sh.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
C = []

C.append(md(r"""# Matrix Factorization vs. LightGCN for Implicit-Feedback Recommendation

**Project 1 — Technical Review (code companion notebook)**

This notebook accompanies our report. We review and compare two recommender-system
methods on two datasets with very different interaction densities:

| | Method 1 | Method 2 |
|---|---|---|
| Model | **MF-BPR** — Matrix Factorization trained with the BPR pairwise ranking loss (Rendle et al., UAI 2009) | **LightGCN** — simplified graph convolution over the user–item bipartite graph (He et al., SIGIR 2020) |
| Signal used | Direct user–item interactions only (1st-order) | Multi-hop neighborhood signal propagated on the interaction graph (higher-order) |

| | Dataset 1 | Dataset 2 |
|---|---|---|
| Name | **MovieLens-1M** | **Amazon Reviews 2023 — Video Games** |
| Nature | Dense movie ratings | Sparse e-commerce reviews |

Both models are implemented **from scratch in PyTorch** (`src/models.py`, ~100 lines total).
The full experiment queue runs in ≈ 1.5 h on a single NVIDIA L4 GPU, and also runs
unmodified on an Apple-Silicon laptop (see Section 7 on reproducibility).
All numbers, figures and case studies in the report are produced by this code —
nothing is copied from published papers.
"""))

C.append(md("## 1. Setup"))
C.append(code(r"""import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()            # project1/
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})

def load_result(name):
    with open(RESULTS / f"{name}.json") as f:
        return json.load(f)
"""))

C.append(md(r"""## 2. Datasets

### 2.1 MovieLens-1M

MovieLens-1M is the classic movie-rating benchmark collected by GroupLens Research:
**1,000,209 ratings** (1–5 stars) from **6,040 users** on **~3,900 movies** (2000–2003).
Every user has rated at least 20 movies, which makes it a *dense* collaborative-filtering
dataset. We treat every observed rating as an implicit positive interaction
(the standard protocol for top-K ranking, cf. the LightGCN paper).

*Files used:* `ratings.csv` (converted from the official `ratings.dat`).
Download: <https://grouplens.org/datasets/movielens/1m/>"""))

C.append(code(r"""ratings = pd.read_csv(ROOT.parent / "ratings.csv")
display(ratings.head(3))
print(f"{len(ratings):,} ratings | {ratings.user_id.nunique():,} users | "
      f"{ratings.movie_id.nunique():,} movies")
"""))

C.append(md(r"""### 2.2 Amazon Reviews 2023 — why *Video Games* (a negative result worth reporting)

Our second dataset comes from **Amazon Reviews 2023** (McAuley Lab, UCSD;
<https://amazon-reviews-2023.github.io/>). We initially selected the *All_Beauty*
category (701K reviews). However, an interaction-graph analysis shows that this
category **cannot support collaborative-filtering evaluation at all**: most users
wrote exactly one review, so the standard 5-core filter collapses the dataset to
almost nothing — and even a 2-core leaves users with too few interactions to split
into train/validation/test."""))

C.append(code(r"""beauty = pd.read_csv(ROOT.parent / "amazon-beauty-2023" / "reviews.csv.gz",
                     usecols=["user_id", "parent_asin"])
beauty.columns = ["user", "item"]
beauty = beauty.drop_duplicates()

def k_core(df, k):
    while True:
        uc = df["user"].map(df["user"].value_counts())
        ic = df["item"].map(df["item"].value_counts())
        keep = (uc >= k) & (ic >= k)
        if keep.all():
            return df
        df = df[keep]

rows = [{"filter": "none (dedup)", "interactions": len(beauty),
         "users": beauty.user.nunique(), "items": beauty.item.nunique()}]
for k in (2, 3, 5):
    d = k_core(beauty.copy(), k)
    rows.append({"filter": f"{k}-core", "interactions": len(d),
                 "users": d.user.nunique(), "items": d.item.nunique()})
display(pd.DataFrame(rows).set_index("filter"))
print("All_Beauty collapses under k-core filtering -> unusable for CF evaluation.")
"""))

C.append(md(r"""This is itself a finding we discuss in the report: **real long-tail e-commerce
feedback is qualitatively different from academic benchmarks** — the majority of
users are one-shot reviewers with no collaborative signal.

We therefore switched to the **Video_Games** category (4.6M raw reviews), where
repeat purchasing is common. After 5-core filtering it keeps a healthy graph of
**814,586 interactions** — the same order of magnitude as MovieLens-1M but **144×
sparser**, giving us exactly the density contrast the comparison needs."""))

C.append(md("### 2.3 The two datasets side by side"))
C.append(code(r"""stats = pd.DataFrame([json.load(open(DATA / "ml-1m_stats.json")),
                      json.load(open(DATA / "vgames_stats.json"))])
stats["split"] = stats["split"].apply(lambda s: f"{s['train']:,}/{s['valid']:,}/{s['test']:,}")
stats = stats.rename(columns={"split": "train/valid/test"}).set_index("dataset")
display(stats.T)
"""))

C.append(code(r"""fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for name, label in (("ml-1m", "MovieLens-1M"), ("vgames", "Amazon Video Games")):
    z = np.load(DATA / f"{name}.npz")
    allpairs = np.concatenate([z["train"], z["valid"], z["test"]])
    ucounts = np.bincount(allpairs[:, 0])
    icounts = np.bincount(allpairs[:, 1])
    for ax, cnt, what in ((axes[0], ucounts, "user"), (axes[1], icounts, "item")):
        v = np.sort(cnt)[::-1]
        ax.plot(np.arange(1, len(v) + 1), v, label=f"{label}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"{what} rank"); ax.set_ylabel("# interactions")
axes[0].set_title("Interactions per user (log-log)")
axes[1].set_title("Interactions per item (log-log)")
axes[0].legend(); axes[1].legend()
plt.tight_layout(); plt.savefig(FIGS / "longtail.png", bbox_inches="tight"); plt.show()
"""))

C.append(md(r"""**Reading the figure:** both datasets are long-tailed, but the curves differ sharply.
A MovieLens user has 165 interactions on average, a Video-Games user only 8.6.
This is the core experimental variable of our study: *how much does the graph-based
method gain over plain MF as the signal per user shrinks?*

## 3. Preprocessing protocol

Implemented in `src/data_prep.py` (run once: `python3 src/data_prep.py`):

1. **Implicit feedback** — every observed (user, item) pair is a positive; duplicates collapsed.
2. **5-core filtering** — iteratively keep users/items with ≥ 5 interactions.
3. **Per-user random split** — 80% train / 10% validation / 10% test (seed 42),
   so every user appears in all three sets.
4. **Evaluation** — *full ranking*: for each user we score **all** items they have not
   interacted with in train, and measure Recall@K and NDCG@K (K = 10, 20) on the held-out
   items. No negative-sample evaluation tricks, which are known to bias comparisons.

## 4. Methods

### 4.1 Method 1: MF-BPR

Matrix Factorization represents each user $u$ and item $i$ as $d$-dimensional embeddings
$\mathbf{e}_u, \mathbf{e}_i$ and scores a pair by the dot product
$\hat{y}_{ui} = \mathbf{e}_u^\top \mathbf{e}_i$.
We train it with the **BPR loss** — for each observed pair $(u,i)$ and a sampled
unobserved item $j$:

$$\mathcal{L}_{BPR} = -\ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \lambda\lVert\Theta\rVert^2$$

i.e. the model learns to rank an interacted item above a random non-interacted one.

### 4.2 Method 2: LightGCN

LightGCN keeps the same embedding tables and the same BPR loss, but replaces the final
embedding with a **propagation over the user–item bipartite graph**. With
$\mathbf{e}^{(0)}$ the free embeddings and $\mathcal{N}_u$ the items of user $u$:

$$\mathbf{e}_u^{(k+1)} = \sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{|\mathcal{N}_u||\mathcal{N}_i|}} \mathbf{e}_i^{(k)}, \qquad
\mathbf{e}_u = \frac{1}{L+1}\sum_{k=0}^{L} \mathbf{e}_u^{(k)}$$

(and symmetrically for items). There are **no feature transforms and no nonlinearities** —
LightGCN is deliberately a *simplification* of standard GCNs, which its authors showed to
work better for recommendation. Layer $k$ mixes in $k$-hop neighbors: $L=2$ already reaches
"users who liked the items I liked".

**The only difference between our two methods is this propagation step**, so any
performance gap can be attributed to the higher-order graph signal — a clean ablation
by construction.

### 4.3 Implementation notes

* Both models: `src/models.py`, from-scratch PyTorch, ~100 lines.
* Propagation uses gather + `index_add` on the edge list, which runs on Apple-Silicon
  **MPS**; `torch.sparse` matmul does not.
* LightGCN propagates the full graph every optimization step, so we use a large batch
  (65,536) — 9× fewer propagations per epoch, 110 s → 12 s per epoch on M2, with
  identical convergence in validation metrics.
* Early stopping on validation NDCG@10 (patience = 30 epochs). Budget: max 300
  epochs for MF (always early-stops well before), **600 epochs for LightGCN** —
  at a 300-epoch cap several LightGCN configs were still improving, which
  initially *reversed* the layer-ablation ordering (see Section 5.3). All
  LightGCN configs reported here use the fair 600-epoch budget.

## 5. Experiments

All runs: `bash src/run_all.sh` (≈ 1.5 h on one NVIDIA L4 GPU; also runs
overnight on a MacBook Air M2 — we verified the MF results are identical
across the two machines).
Hyperparameters: Adam, lr $10^{-3}$ (MF) / $3\times10^{-3}$ (LightGCN, large-batch),
$L_2$ reg $10^{-4}$, embedding dim 64 unless stated.

### 5.1 Main results"""))

C.append(code(r"""runs = {
    ("MovieLens-1M", "MF-BPR"): "ml-1m_mf_d64",
    ("MovieLens-1M", "LightGCN (3 layers)"): "ml-1m_lightgcn_d64_l3",
    ("Video Games", "MF-BPR"): "vgames_mf_d64",
    ("Video Games", "LightGCN (3 layers)"): "vgames_lightgcn_d64_l3",
}
rows = []
for (ds, model), name in runs.items():
    r = load_result(name)
    rows.append({"dataset": ds, "model": model, **r["test"],
                 "best_epoch": r["best_epoch"],
                 "train_min": round(r["train_seconds"] / 60, 1)})
main = pd.DataFrame(rows).set_index(["dataset", "model"])
display(main.round(4))

for ds in ("MovieLens-1M", "Video Games"):
    mf = main.loc[(ds, "MF-BPR")]
    lg = main.loc[(ds, "LightGCN (3 layers)")]
    for m in ("recall@20", "ndcg@10"):
        gain = 100 * (lg[m] - mf[m]) / mf[m]
        print(f"{ds:14s} {m:10s} MF {mf[m]:.4f} -> LightGCN {lg[m]:.4f} ({gain:+.1f}%)")
"""))

C.append(md("### 5.2 Convergence behaviour"))
C.append(code(r"""fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)
for ax, ds, title in ((axes[0], "ml-1m", "MovieLens-1M"),
                      (axes[1], "vgames", "Amazon Video Games")):
    for model, label in (("mf_d64", "MF-BPR"), ("lightgcn_d64_l3", "LightGCN L=3")):
        h = load_result(f"{ds}_{model}")["history"]
        ax.plot([e["epoch"] for e in h], [e["ndcg@10"] for e in h], label=label)
    ax.set_title(title); ax.set_xlabel("epoch"); ax.set_ylabel("valid NDCG@10")
    ax.legend()
plt.tight_layout(); plt.savefig(FIGS / "convergence.png", bbox_inches="tight"); plt.show()
"""))

C.append(md(r"""### 5.3 Ablation: number of propagation layers

The layer count $L$ is LightGCN's key component — it controls how many hops of
collaborative signal are mixed into each embedding. $L=0$ would reduce LightGCN
exactly to MF, so this ablation directly measures the value of graph propagation.

**A methodological lesson we hit here:** with a 300-epoch budget, the curve
appeared to *peak* at $L=2$–$3$ and drop at $L=4$ — the classic "over-smoothing"
picture. Doubling the budget to 600 epochs removed the drop entirely: deeper
propagation simply converges more slowly (best epoch grows from 110 at $L=1$ to
~565 at $L=4$), so an unfair training budget masquerades as over-smoothing.
Under the fair 600-epoch budget used below, NDCG@10 increases monotonically up
to $L=4$ on MovieLens-1M and saturates at $L=3$–$4$ on Video Games."""))

C.append(code(r"""fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, ds, title in ((axes[0], "ml-1m", "MovieLens-1M"),
                      (axes[1], "vgames", "Amazon Video Games")):
    xs, ys = [], []
    for L in (1, 2, 3, 4):
        r = load_result(f"{ds}_lightgcn_d64_l{L}")
        xs.append(L); ys.append(r["test"]["ndcg@10"])
    mf = load_result(f"{ds}_mf_d64")["test"]["ndcg@10"]
    ax.plot(xs, ys, "o-", label="LightGCN")
    ax.axhline(mf, color="crimson", ls="--", label="MF-BPR (no propagation)")
    ax.set_title(title); ax.set_xlabel("propagation layers L"); ax.set_ylabel("test NDCG@10")
    ax.set_xticks(xs); ax.legend()
plt.tight_layout(); plt.savefig(FIGS / "ablation_layers.png", bbox_inches="tight"); plt.show()
"""))

C.append(md("### 5.4 Parameter study: embedding dimension"))
C.append(code(r"""fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for ax, ds, title in ((axes[0], "ml-1m", "MovieLens-1M"),
                      (axes[1], "vgames", "Amazon Video Games")):
    for model, label in (("mf_d{d}", "MF-BPR"), ("lightgcn_d{d}_l3", "LightGCN L=3")):
        dims, ys = [16, 32, 64], []
        for d in dims:
            ys.append(load_result(f"{ds}_{model.format(d=d)}")["test"]["ndcg@10"])
        ax.plot(dims, ys, "o-", label=label)
    ax.set_title(title); ax.set_xlabel("embedding dim"); ax.set_ylabel("test NDCG@10")
    ax.set_xscale("log", base=2); ax.set_xticks([16, 32, 64], [16, 32, 64]); ax.legend()
plt.tight_layout(); plt.savefig(FIGS / "dim_sweep.png", bbox_inches="tight"); plt.show()
"""))

C.append(md(r"""### 5.5 Who benefits from the graph? Analysis by user activity

We bucket test users by their number of *training* interactions and compare
per-bucket Recall@20. Hypothesis: propagation mainly helps **low-activity (cold)
users**, whose own history is too short for MF to position them well — the graph
lets them borrow their neighbors' signal."""))

C.append(code(r"""def per_user_recall(topk, test_by_user, k=20):
    out = np.full(len(test_by_user), np.nan)
    for u, pos in enumerate(test_by_user):
        if len(pos):
            out[u] = len(set(topk[u, :k]) & set(pos)) / len(pos)
    return out

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, ds, title in ((axes[0], "ml-1m", "MovieLens-1M"),
                      (axes[1], "vgames", "Amazon Video Games")):
    z = np.load(DATA / f"{ds}.npz")
    n_users = int(z["n_users"])
    train_deg = np.bincount(z["train"][:, 0], minlength=n_users)
    test_by_user = [[] for _ in range(n_users)]
    for u, i in z["test"]:
        test_by_user[u].append(i)
    recs = {}
    for model, label in (("mf_d64", "MF-BPR"), ("lightgcn_d64_l3", "LightGCN L=3")):
        topk = np.load(RESULTS / f"{ds}_{model}_emb.npz")["topk"]
        recs[label] = per_user_recall(topk, test_by_user)
    # Quantile-based degree bins; np.unique guards against degenerate
    # quantiles on Video Games, where most users have only 3-7 interactions.
    edges = np.unique(np.quantile(train_deg, [0, .25, .5, .75, 1.0]).astype(int))
    bins = [(edges[j], edges[j + 1]) for j in range(len(edges) - 1)]
    labels_q = [f"{lo}-{hi - 1}" if j < len(bins) - 1 else f"{lo}+"
                for j, (lo, hi) in enumerate(bins)]
    x = np.arange(len(bins))
    for off, (label, r) in zip((-0.17, 0.17), recs.items()):
        means = []
        for j, (lo, hi) in enumerate(bins):
            mask = (train_deg >= lo) & ((train_deg < hi) | (j == len(bins) - 1))
            means.append(np.nanmean(r[mask]))
        ax.bar(x + off, means, width=0.34, label=label)
    ax.set_xticks(x, labels_q); ax.set_ylabel("Recall@20")
    ax.set_xlabel("user activity quartile (train interactions)")
    ax.set_title(title); ax.legend()
plt.tight_layout(); plt.savefig(FIGS / "activity_buckets.png", bbox_inches="tight"); plt.show()
"""))

C.append(md(r"""### 5.6 Popularity bias and catalog coverage

A model can score well by only recommending bestsellers. We measure (a) the average
training popularity of recommended items and (b) *catalog coverage* — the fraction of
the catalog that ever appears in a top-10 list."""))

C.append(code(r"""rows = []
for ds, title in (("ml-1m", "MovieLens-1M"), ("vgames", "Video Games")):
    z = np.load(DATA / f"{ds}.npz")
    item_pop = np.bincount(z["train"][:, 1], minlength=int(z["n_items"]))
    for model, label in (("mf_d64", "MF-BPR"), ("lightgcn_d64_l3", "LightGCN L=3")):
        topk = np.load(RESULTS / f"{ds}_{model}_emb.npz")["topk"][:, :10]
        rows.append({
            "dataset": title, "model": label,
            "mean pop. of recommended items": round(float(item_pop[topk].mean()), 1),
            "catalog coverage@10 (%)": round(100 * len(np.unique(topk)) / len(item_pop), 1),
        })
    rows.append({"dataset": title, "model": "(train set avg item pop.)",
                 "mean pop. of recommended items": round(float(item_pop[z['train'][:,1]].mean()), 1),
                 "catalog coverage@10 (%)": np.nan})
display(pd.DataFrame(rows).set_index(["dataset", "model"]))
"""))

C.append(md(r"""### 5.7 Case study (MovieLens-1M)

We inspect one low-activity and one high-activity user: their favourite training
genres and the top-10 lists from each model. This makes the aggregate numbers
concrete and surfaces failure modes."""))

C.append(code(r"""movies = pd.read_csv(ROOT.parent / "movies.csv")
z = np.load(DATA / "ml-1m.npz")
n_users = int(z["n_users"])
train_deg = np.bincount(z["train"][:, 0], minlength=n_users)

# Reconstruct the item-id mapping used by data_prep (sorted raw ids).
ml = pd.read_csv(ROOT.parent / "ratings.csv", usecols=["user_id", "movie_id"])
ml.columns = ["user", "item"]
ml = ml.drop_duplicates()
# apply same 5-core as data_prep
def k_core5(df):
    while True:
        uc = df["user"].map(df["user"].value_counts())
        ic = df["item"].map(df["item"].value_counts())
        keep = (uc >= 5) & (ic >= 5)
        if keep.all():
            return df
        df = df[keep]
ml = k_core5(ml)
item_raw = np.array(sorted(ml["item"].unique()))
title_of = movies.set_index("movie_id")["title"]
genre_of = movies.set_index("movie_id")["genres"]

test_by_user = [[] for _ in range(n_users)]
for u, i in z["test"]:
    test_by_user[u].append(i)
train_by_user = [[] for _ in range(n_users)]
for u, i in z["train"]:
    train_by_user[u].append(i)

topk = {m: np.load(RESULTS / f"ml-1m_{m}_emb.npz")["topk"] for m in ("mf_d64", "lightgcn_d64_l3")}

rng = np.random.default_rng(0)
cold = rng.choice(np.where(train_deg <= np.quantile(train_deg, 0.1))[0])
heavy = rng.choice(np.where(train_deg >= np.quantile(train_deg, 0.95))[0])

for u, kind in ((cold, "LOW-activity"), ((heavy), "HIGH-activity")):
    print(f"=== {kind} user (internal id {u}, {train_deg[u]} train interactions) ===")
    hist_genres = pd.Series([g for i in train_by_user[u]
                             for g in genre_of[item_raw[i]].split("|")]).value_counts()
    print("train-history genres:", dict(hist_genres.head(5)))
    test_set = set(test_by_user[u])
    for m, label in (("mf_d64", "MF-BPR"), ("lightgcn_d64_l3", "LightGCN")):
        print(f"-- {label} top-10 --")
        for i in topk[m][u, :10]:
            hit = "  <-- HELD-OUT HIT" if i in test_set else ""
            print(f"   {title_of[item_raw[i]]}{hit}")
    print()
"""))

C.append(md(r"""## 6. Conclusions

1. **Higher-order graph signal helps, and helps most where data is sparse.**
   LightGCN beats MF-BPR on both datasets, but the margin differs by an order of
   magnitude: **+6.1%** NDCG@10 on dense MovieLens-1M (0.246 → 0.261) versus
   **+43.6%** on the 144×-sparser Video Games (0.0344 → 0.0494); Recall@20 gains
   are +10.7% and +39.2% respectively (Section 5.1).
2. **The gain is concentrated on low-activity users.** On MovieLens-1M the
   improvement falls monotonically with user activity: +14.6% Recall@20 for the
   coldest quartile (≤ 36 train interactions) down to +3.9% for the most active
   quartile (Section 5.5). Propagation lets cold users borrow their neighbors'
   signal; heavy users already give MF enough direct evidence.
3. **How LightGCN wins differs by regime — and is not free.** On dense data it
   *reduces* popularity bias (mean popularity of recommended items 1150 → 994)
   and *raises* catalog coverage (51% → 61%). On sparse data the opposite
   happens: propagation concentrates mass on bestsellers (mean popularity
   373 → 418) and coverage *drops* from 62% to 41% (Section 5.6). Part of the
   +43.6% on Video Games is therefore a popularity bet — a real weakness if the
   product goal values discovery, and our candidate direction for the "improve
   the method" extension.
4. **Layer ablation** (Section 5.3): even L=1 clearly beats MF on both datasets
   (+26% NDCG@10 on Video Games). Under a *fair* 600-epoch budget we observe **no
   over-smoothing up to L=4**: accuracy grows monotonically on MovieLens-1M and
   saturates at L=3–4 on Video Games. The "peak at shallow L" we saw first was an
   artifact of a 300-epoch cap — deeper models converge more slowly (best epoch
   110 at L=1 vs ~565 at L=4), a caveat worth remembering when reading ablation
   tables in the literature.
   Also noteworthy from the dimension sweep (Section 5.4): **LightGCN at d=16 —
   a quarter of the parameters — still beats MF at d=64 on both datasets**, i.e.
   the graph prior buys more than extra capacity does.
5. **Convergence/cost trade-off**: MF reaches its best validation score by epoch
   45–125; LightGCN needs 450–565 epochs *and* a full-graph propagation every
   optimization step. With similar parameter counts, LightGCN buys its accuracy
   with ~10× the training compute (Section 5.2).
6. **Dataset finding**: the raw Amazon *All_Beauty* 2023 category is unusable for
   CF evaluation — 5-core filtering collapses 693K interactions to 2.5K because
   most users are one-shot reviewers (Section 2.2). Benchmark choice presupposes
   a minimum of collaborative signal; we consider this negative result as
   informative as the main comparison.

## 7. Reproducibility

```
project1/
├── src/data_prep.py        # preprocessing (Section 3)
├── src/models.py           # MF-BPR + LightGCN, from scratch (Section 4)
├── src/train.py            # training / evaluation CLI
├── src/run_all.sh          # full experiment queue (~6 h on M2)
├── data/                   # processed splits (npz) + stats
├── results/                # one JSON log + embeddings per run
└── figures/                # all figures saved by this notebook
```

* Raw data: MovieLens-1M from grouplens.org; Amazon Reviews 2023 (Video_Games,
  All_Beauty) from the McAuley-Lab HuggingFace repository. Place the converted
  CSVs as described in Section 2 (paths are relative; no network access needed
  to run this notebook).
* Environment: Python 3.9+, `torch`, `pandas`, `numpy`, `matplotlib`. Seeds fixed (42).
* Hardware: experiments were run on an NVIDIA L4 (AWS g6.xlarge); the pipeline also
  runs on a MacBook Air M2 (16 GB, MPS) with no code changes — the MF run produced
  **identical test metrics on both machines**, since all sampling uses a seeded
  NumPy generator on the CPU.
"""))

nb.cells = C
path = "project1_report.ipynb"
with open(path, "w") as f:
    nbf.write(nb, f)
print("wrote", path, len(C), "cells")
