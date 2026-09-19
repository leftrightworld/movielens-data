# movielens-data

Datasets and course work for **Project 1 — Technical Review** (recommender
systems: MF-BPR vs. LightGCN).

## Repository map

```
movielens-data/
├── movielens-1m/            MovieLens-1M as analysis-ready CSVs
├── amazon-vgames-2023/      Amazon Reviews 2023 "Video Games" as CSVs
├── project1/                the course project (code, experiments, report)
│   ├── project1_report.ipynb   code-companion notebook (executed, with figures)
│   ├── report/                 LaTeX report + compiled report.pdf (7 pages)
│   ├── src/                    all source code (models, training, data prep)
│   ├── data/                   preprocessed splits (npz) + statistics
│   ├── results/                one JSON log per experiment + saved embeddings
│   └── figures/                all figures used by the notebook and report
└── project1-technical-review.pdf   the assignment brief
```

## Where to start

* **Read the findings** → `project1/report/report.pdf`
* **Read / run the code** → `project1/project1_report.ipynb` (see `project1/README.md`)
* **Just want the datasets** → the two dataset folders below; each has its own
  README with column descriptions and a Colab-ready loading snippet.

## Datasets at a glance

| Folder | Source | Size | Notes |
|--------|--------|------|-------|
| `movielens-1m/` | [GroupLens](https://grouplens.org/datasets/movielens/1m/) | 1.0M ratings, 6,040 users, ~3.9K movies | dense benchmark |
| `amazon-vgames-2023/` | [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab) | 4.62M ratings, 137K products | sparse e-commerce; ratings split in two files for GitHub's 100 MB limit |

Data usage is subject to the original licenses (GroupLens; McAuley Lab) —
research / non-commercial, cite the sources.
