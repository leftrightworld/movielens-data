"""Convert the raw Amazon Reviews 2023 Video_Games jsonl files into the
gzipped CSVs in this folder.

Usage:
  1. Download from https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
       raw/review_categories/Video_Games.jsonl        (~2.7 GB)
       raw/meta_categories/meta_Video_Games.jsonl     (~0.4 GB)
  2. python3 convert.py /path/to/downloads

Produces ratings_part1.csv.gz, ratings_part2.csv.gz, products.csv.gz.
Ratings are split into two halves to stay under GitHub's 100 MB file limit;
review text and image/video URLs are intentionally omitted.
"""
import csv
import gzip
import json
import sys
from pathlib import Path

RATING_COLS = ["rating", "asin", "parent_asin", "user_id", "timestamp",
               "verified_purchase", "helpful_vote"]
PRODUCT_COLS = ["parent_asin", "title", "main_category", "average_rating",
                "rating_number", "price", "store", "features", "description",
                "num_images", "num_videos", "details"]


def main(src: Path, out: Path):
    rows = []
    with open(src / "Video_Games.jsonl") as fin:
        for line in fin:
            r = json.loads(line)
            rows.append([r.get("rating"), r.get("asin"), r.get("parent_asin"),
                         r.get("user_id"), r.get("timestamp"),
                         r.get("verified_purchase"), r.get("helpful_vote")])
    half = (len(rows) + 1) // 2
    for part, chunk in ((1, rows[:half]), (2, rows[half:])):
        with gzip.open(out / f"ratings_part{part}.csv.gz", "wt", newline="") as f:
            w = csv.writer(f)
            w.writerow(RATING_COLS)
            w.writerows(chunk)
    print(f"ratings: {len(rows):,} rows -> 2 parts")

    m = 0
    with open(src / "meta_Video_Games.jsonl") as fin, \
         gzip.open(out / "products.csv.gz", "wt", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(PRODUCT_COLS)
        for line in fin:
            p = json.loads(line)
            w.writerow([p.get("parent_asin"), p.get("title", ""),
                        p.get("main_category"), p.get("average_rating"),
                        p.get("rating_number"), p.get("price"), p.get("store"),
                        " | ".join(p.get("features") or []),
                        " | ".join(p.get("description") or []),
                        len(p.get("images") or []), len(p.get("videos") or []),
                        json.dumps(p.get("details") or {}, ensure_ascii=False)])
            m += 1
    print(f"products: {m:,}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(__file__).resolve().parent)
