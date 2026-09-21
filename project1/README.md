# Project 1 — Technical Review: MF-BPR vs. LightGCN

We compare **matrix factorization (MF-BPR)** against **graph-based
collaborative filtering (LightGCN)** on two datasets that differ mainly in
density — MovieLens-1M (dense) and Amazon Video Games 2023 (144× sparser) —
to answer: *when, and for whom, does graph propagation help recommendation?*

**Read this first:** `report/report.pdf` (7 pages — methods, all results,
analysis, conclusions). The notebook `project1_report.ipynb` is the code
companion: every table and figure in the report is produced there.

## Folder structure

```
project1/
├── DEVLOG.md               dated dev log (in Chinese, for teammates)
├── project1_report.ipynb   code-companion notebook (kept executed — open on
│                           GitHub to see all outputs without running anything)
├── report/
│   ├── report.pdf          the report (compiled)
│   ├── report.tex          LaTeX source
│   └── gen_tables.py       generates the report's tables from results/ JSONs
├── src/
│   ├── data_prep.py        raw CSVs → implicit feedback, 5-core, 80/10/10 split
│   ├── models.py           MF-BPR + LightGCN, from scratch (~100 lines)
│   ├── train.py            training / evaluation CLI (one JSON log per run)
│   └── run_all.sh          the full experiment queue (18+ runs)
├── data/                   preprocessed splits (ml-1m.npz, vgames.npz) + stats
├── results/                per-run logs (*.json) and, for the four main runs,
│                           saved embeddings + top-K lists (*_emb.npz)
├── figures/                all figures (produced by the notebook)
└── archive/                superseded artifacts
```

## Reproduce everything

```bash
pip install torch pandas numpy matplotlib        # any recent versions
python3 src/data_prep.py                         # ~2 min: rebuilds data/*.npz
bash src/run_all.sh                              # ~1.5 h on one NVIDIA L4 GPU
                                                 # (also runs on Apple Silicon / CPU)
cd report && python3 gen_tables.py && pdflatex report.tex   # rebuild the report
```

Everything is seeded (42): `data_prep.py` reproduces the committed `data/*.npz`
byte-for-byte, and re-running the notebook reproduces the committed figures.
The MF run yields identical metrics on Apple M2 (MPS) and NVIDIA L4 (CUDA).

## Headline results (test NDCG@10, d=64)

| Dataset | MF-BPR | LightGCN (L=3) | gain |
|---------|--------|----------------|------|
| MovieLens-1M | 0.2456 | 0.2605 | **+6.1%** |
| Video Games | 0.0344 | 0.0494 | **+43.5%** |

See the report for the per-user, popularity, ablation, and convergence
analyses behind these numbers.
