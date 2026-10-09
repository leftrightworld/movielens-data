"""Generate LaTeX table fragments for the report from results/ and data/.

Run from project1/report/:  python3 gen_tables.py
Writes: tab_datasets.tex, tab_main.tex, tab_activity.tex, tab_popularity.tex
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P = HERE.parent
RESULTS, DATA = P / "results", P / "data"


def res(name):
    return json.load(open(RESULTS / f"{name}.json"))


# ---------- datasets ----------
rows = []
for ds in ("ml-1m", "vgames"):
    s = json.load(open(DATA / f"{ds}_stats.json"))
    rows.append(s)
with open(HERE / "tab_datasets.tex", "w") as f:
    f.write("\\begin{tabular}{lrr}\n\\toprule\n")
    f.write(" & MovieLens-1M & Video Games \\\\\n\\midrule\n")
    for key, label, fmt in [
        ("interactions_after_5core", "Interactions (5-core)", "{:,}"),
        ("n_users", "Users", "{:,}"),
        ("n_items", "Items", "{:,}"),
        ("density_pct", "Density (\\%)", "{:.4f}"),
        ("avg_interactions_per_user", "Avg.\\ interactions / user", "{:.1f}"),
        ("avg_interactions_per_item", "Avg.\\ interactions / item", "{:.1f}"),
    ]:
        f.write(f"{label} & {fmt.format(rows[0][key])} & {fmt.format(rows[1][key])} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------- main results ----------
METHODS = [("Popularity", "{ds}_pop"), ("ItemKNN ($k{=}100$)", "{ds}_itemknn"),
           ("MF-BPR", "{ds}_mf_d64"), ("LightGCN ($L{=}3$)", "{ds}_lightgcn_d64_l3")]

with open(HERE / "tab_main.tex", "w") as f:
    f.write("\\begin{tabular}{llrrrrr}\n\\toprule\n")
    f.write("Dataset & Model & R@10 & N@10 & R@20 & N@20 & Best ep. \\\\\n\\midrule\n")
    for di, (ds, label) in enumerate((("ml-1m", "MovieLens-1M"),
                                      ("vgames", "Video Games"))):
        for mi, (model, pat) in enumerate(METHODS):
            r = res(pat.format(ds=ds))
            t = r["test"]
            ep = r["best_epoch"] if r["best_epoch"] else "--"
            f.write(f"{label if mi == 0 else ''} & {model} & "
                    f"{t['recall@10']:.4f} & {t['ndcg@10']:.4f} & "
                    f"{t['recall@20']:.4f} & {t['ndcg@20']:.4f} & {ep} \\\\\n")
        lg = res(f"{ds}_lightgcn_d64_l3")["test"]
        mf = res(f"{ds}_mf_d64")["test"]
        gains = " & ".join(
            f"{100*(lg[m]-mf[m])/mf[m]:+.1f}\\%"
            for m in ("recall@10", "ndcg@10", "recall@20", "ndcg@20"))
        f.write(f" & \\emph{{LightGCN vs.\\ MF}} & {gains} & \\\\\n")
        if di == 0:
            f.write("\\midrule\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------- temporal-split robustness ----------
with open(HERE / "tab_temporal.tex", "w") as f:
    f.write("\\begin{tabular}{llrrrr}\n\\toprule\n")
    f.write(" & & \\multicolumn{2}{c}{Random split} & "
            "\\multicolumn{2}{c}{Temporal split} \\\\\n")
    f.write("Dataset & Model & N@10 & R@20 & N@10 & R@20 \\\\\n\\midrule\n")
    for di, (ds, label) in enumerate((("ml-1m", "MovieLens-1M"),
                                      ("vgames", "Video Games"))):
        for mi, (model, pat) in enumerate(METHODS):
            rr = res(pat.format(ds=ds))["test"]
            rt = res(pat.format(ds=f"{ds}-t"))["test"]
            f.write(f"{label if mi == 0 else ''} & {model} & "
                    f"{rr['ndcg@10']:.4f} & {rr['recall@20']:.4f} & "
                    f"{rt['ndcg@10']:.4f} & {rt['recall@20']:.4f} \\\\\n")
        g = {}
        for proto, suffix in (("rand", ""), ("temp", "-t")):
            lg = res(f"{ds}{suffix}_lightgcn_d64_l3")["test"]
            mf = res(f"{ds}{suffix}_mf_d64")["test"]
            g[proto] = {m: 100 * (lg[m] - mf[m]) / mf[m]
                        for m in ("ndcg@10", "recall@20")}
        f.write(f" & \\emph{{LightGCN vs.\\ MF}} & "
                f"{g['rand']['ndcg@10']:+.1f}\\% & {g['rand']['recall@20']:+.1f}\\% & "
                f"{g['temp']['ndcg@10']:+.1f}\\% & {g['temp']['recall@20']:+.1f}\\% \\\\\n")
        if di == 0:
            f.write("\\midrule\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------- activity buckets ----------
def bucket_table():
    lines = []
    for ds, label in (("ml-1m", "MovieLens-1M"), ("vgames", "Video Games")):
        z = np.load(DATA / f"{ds}.npz")
        n_users = int(z["n_users"])
        deg = np.bincount(z["train"][:, 0], minlength=n_users)
        test_by_user = [[] for _ in range(n_users)]
        for u, i in z["test"]:
            test_by_user[u].append(i)
        per = {}
        for m in ("mf_d64", "lightgcn_d64_l3"):
            topk = np.load(RESULTS / f"{ds}_{m}_emb.npz")["topk"]
            out = np.full(n_users, np.nan)
            for u, pos in enumerate(test_by_user):
                if pos:
                    out[u] = len(set(topk[u, :20]) & set(pos)) / len(pos)
            per[m] = out
        edges = np.unique(np.quantile(deg, [0, .25, .5, .75, 1.0]).astype(int))
        bins = [(edges[j], edges[j + 1]) for j in range(len(edges) - 1)]
        for j, (lo, hi) in enumerate(bins):
            mask = (deg >= lo) & ((deg < hi) | (j == len(bins) - 1))
            mf = np.nanmean(per["mf_d64"][mask])
            lg = np.nanmean(per["lightgcn_d64_l3"][mask])
            blabel = f"{lo}--{hi-1}" if j < len(bins) - 1 else f"$\\geq${lo}"
            lines.append(f"{label if j == 0 else ''} & {blabel} & "
                         f"{mf:.4f} & {lg:.4f} & {100*(lg-mf)/mf:+.1f}\\% \\\\")
        lines.append("\\midrule" if ds == "ml-1m" else "")
    return "\n".join(x for x in lines if x)

with open(HERE / "tab_activity.tex", "w") as f:
    f.write("\\begin{tabular}{llrrr}\n\\toprule\n")
    f.write("Dataset & Train inter. & MF-BPR & LightGCN & Gain \\\\\n\\midrule\n")
    f.write(bucket_table() + "\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------- popularity / coverage ----------
with open(HERE / "tab_popularity.tex", "w") as f:
    f.write("\\begin{tabular}{llrr}\n\\toprule\n")
    f.write("Dataset & Model & Mean popularity & Coverage@10 \\\\\n\\midrule\n")
    for ds, label in (("ml-1m", "MovieLens-1M"), ("vgames", "Video Games")):
        z = np.load(DATA / f"{ds}.npz")
        pop = np.bincount(z["train"][:, 1], minlength=int(z["n_items"]))
        for m, ml in (("mf_d64", "MF-BPR"), ("lightgcn_d64_l3", "LightGCN")):
            topk = np.load(RESULTS / f"{ds}_{m}_emb.npz")["topk"][:, :10]
            f.write(f"{label if m == 'mf_d64' else ''} & {ml} & "
                    f"{pop[topk].mean():.0f} & "
                    f"{100*len(np.unique(topk))/len(pop):.1f}\\% \\\\\n")
        if ds == "ml-1m":
            f.write("\\midrule\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

print("wrote tab_datasets / tab_main / tab_activity / tab_popularity")
