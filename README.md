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

# Amazon Reviews 2023 — All_Beauty (CSV)

Converted from [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab, UCSD), category **All_Beauty**. Image/video URL fields were dropped (kept as counts) to fit GitHub; files are gzipped CSV — pandas reads them directly.

| File | Rows | Columns |
|------|------|---------|
| amazon-beauty-2023/reviews.csv.gz | 701,528 | rating, title, text, asin, parent_asin, user_id, timestamp (ms), verified_purchase, helpful_vote, num_images |
| amazon-beauty-2023/products.csv.gz | 112,590 | parent_asin, title, main_category, average_rating, rating_number, price, store, features, description, num_images, num_videos, details (JSON) |

Join key: `parent_asin`. Convert timestamp with `pd.to_datetime(df.timestamp, unit="ms")`.

## Load in Google Colab

```python
import pandas as pd

base = "https://raw.githubusercontent.com/leftrightworld/movielens-data/main/amazon-beauty-2023/"
reviews  = pd.read_csv(base + "reviews.csv.gz")
products = pd.read_csv(base + "products.csv.gz")
```

Data usage subject to the original dataset license (cite McAuley Lab, Amazon Reviews 2023).

---

# Project 1 — Technical Review (course project)

See `project1/`: MF-BPR vs LightGCN on MovieLens-1M and Amazon Video Games 2023,
implemented from scratch in PyTorch. Entry point: `project1/project1_report.ipynb`
(fully executed, all figures included). Reproduce: `python3 project1/src/data_prep.py`
then `bash project1/src/run_all.sh`.
