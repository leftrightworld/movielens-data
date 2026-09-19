# MovieLens 1M — CSV for analysis

Converted from the official [MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/)
(GroupLens Research). The original `.dat` files use `::` separators and coded
fields; these CSVs are ready for pandas.

| File | Rows | Columns |
|------|------|---------|
| ratings.csv | 1,000,209 | user_id, movie_id, rating (1–5), timestamp, datetime |
| movies.csv | 3,883 | movie_id, title, genres (`\|`-separated), year |
| users.csv | 6,040 | user_id, gender, age, occupation, zip_code, age_group, occupation_name |

`age_group` and `occupation_name` are human-readable decodings of the official
integer codes (e.g. `15 → scientist`). Join keys: `user_id`, `movie_id`.

## Load in Google Colab

```python
import pandas as pd

base = "https://raw.githubusercontent.com/leftrightworld/movielens-data/main/movielens-1m/"
ratings = pd.read_csv(base + "ratings.csv")
movies  = pd.read_csv(base + "movies.csv")
users   = pd.read_csv(base + "users.csv")
```

Data usage subject to the original GroupLens license (research/non-commercial,
cite the MovieLens paper: Harper & Konstan, ACM TiiS 2015).
