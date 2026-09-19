# MovieLens 1M — CSV for analysis

Converted from the official [MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/) (GroupLens Research).

| File | Rows | Columns |
|------|------|---------|
| ratings.csv | 1,000,209 | user_id, movie_id, rating, timestamp, datetime |
| movies.csv | 3,883 | movie_id, title, genres, year |
| users.csv | 6,040 | user_id, gender, age, occupation, zip_code, age_group, occupation_name |

## Load in Google Colab

```python
import pandas as pd

base = "https://raw.githubusercontent.com/leftrightworld/movielens-data/main/"
ratings = pd.read_csv(base + "ratings.csv")
movies  = pd.read_csv(base + "movies.csv")
users   = pd.read_csv(base + "users.csv")
```

Data usage subject to the original GroupLens license (research/non-commercial, cite the MovieLens paper).

---

# Amazon Reviews 2023 — Video Games (CSV)

Converted from [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab, UCSD), category **Video_Games**. Ratings are split into two parts to stay under GitHub's 100 MB file limit; review text and image/video URLs are omitted (the full raw data is on the official site). Files are gzipped CSV — pandas reads them directly.

| File | Rows | Columns |
|------|------|---------|
| amazon-vgames-2023/ratings_part1.csv.gz | 2,312,308 | rating, asin, parent_asin, user_id, timestamp (ms), verified_purchase, helpful_vote |
| amazon-vgames-2023/ratings_part2.csv.gz | 2,312,307 | (same schema) |
| amazon-vgames-2023/products.csv.gz | 137,269 | parent_asin, title, main_category, average_rating, rating_number, price, store, features, description, num_images, num_videos, details (JSON) |

Join key: `parent_asin`. Convert timestamp with `pd.to_datetime(df.timestamp, unit="ms")`.

## Load in Google Colab

```python
import pandas as pd

base = "https://raw.githubusercontent.com/leftrightworld/movielens-data/main/amazon-vgames-2023/"
ratings = pd.concat([pd.read_csv(base + f"ratings_part{i}.csv.gz") for i in (1, 2)],
                    ignore_index=True)
products = pd.read_csv(base + "products.csv.gz")
```

Data usage subject to the original dataset license (cite McAuley Lab, Amazon Reviews 2023).

---

# Project 1 — Technical Review (course project)

See `project1/`: MF-BPR vs LightGCN on MovieLens-1M and Amazon Video Games 2023,
implemented from scratch in PyTorch. Entry point: `project1/project1_report.ipynb`
(fully executed, all figures included). Reproduce: `python3 project1/src/data_prep.py`
then `bash project1/src/run_all.sh`.
