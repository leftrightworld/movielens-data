# Amazon Reviews 2023 — Video Games (CSV)

Converted from [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/)
(McAuley Lab, UCSD), category **Video_Games**. Ratings are split into two
parts to stay under GitHub's 100 MB file limit; review text and image/video
URLs are omitted (get them from the official source if needed). Files are
gzipped CSV — pandas reads them directly.

| File | Rows | Columns |
|------|------|---------|
| ratings_part1.csv.gz | 2,312,308 | rating, asin, parent_asin, user_id, timestamp (ms), verified_purchase, helpful_vote |
| ratings_part2.csv.gz | 2,312,307 | (same schema) |
| products.csv.gz | 137,269 | parent_asin, title, main_category, average_rating, rating_number, price, store, features, description, num_images, num_videos, details (JSON) |

Join key: `parent_asin` (groups product variants). Convert timestamps with
`pd.to_datetime(df.timestamp, unit="ms")`.

## Load in Google Colab

```python
import pandas as pd

base = "https://raw.githubusercontent.com/leftrightworld/movielens-data/main/amazon-vgames-2023/"
ratings = pd.concat([pd.read_csv(base + f"ratings_part{i}.csv.gz") for i in (1, 2)],
                    ignore_index=True)
products = pd.read_csv(base + "products.csv.gz")
```

Data usage subject to the original dataset license (cite McAuley Lab,
Amazon Reviews 2023: Hou et al., arXiv:2403.03952).

## Provenance

These CSVs were generated from the raw McAuley-Lab jsonl files by `convert.py` in this folder — see its docstring for usage.
