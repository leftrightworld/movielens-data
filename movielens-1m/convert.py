"""Convert the official MovieLens-1M .dat files into the CSVs in this folder.

Usage:
  1. Download and unzip https://files.grouplens.org/datasets/movielens/ml-1m.zip
  2. python3 convert.py /path/to/ml-1m

Produces ratings.csv, movies.csv, users.csv (the files committed here).
"""
import sys
from pathlib import Path

import pandas as pd

AGE_MAP = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44",
           45: "45-49", 50: "50-55", 56: "56+"}
OCC_MAP = {0: "other", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
           4: "college/grad student", 5: "customer service", 6: "doctor/health care",
           7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student",
           11: "lawyer", 12: "programmer", 13: "retired", 14: "sales/marketing",
           15: "scientist", 16: "self-employed", 17: "technician/engineer",
           18: "tradesman/craftsman", 19: "unemployed", 20: "writer"}


def main(src: Path, out: Path):
    ratings = pd.read_csv(src / "ratings.dat", sep="::", engine="python",
                          names=["user_id", "movie_id", "rating", "timestamp"],
                          encoding="latin-1")
    ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")
    ratings.to_csv(out / "ratings.csv", index=False)

    movies = pd.read_csv(src / "movies.dat", sep="::", engine="python",
                         names=["movie_id", "title", "genres"],
                         encoding="latin-1")
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)\s*$").astype("Int64")
    movies.to_csv(out / "movies.csv", index=False)

    users = pd.read_csv(src / "users.dat", sep="::", engine="python",
                        names=["user_id", "gender", "age", "occupation", "zip_code"],
                        encoding="latin-1", dtype={"zip_code": str})
    users["age_group"] = users["age"].map(AGE_MAP)
    users["occupation_name"] = users["occupation"].map(OCC_MAP)
    users.to_csv(out / "users.csv", index=False)

    print(f"ratings {len(ratings):,} | movies {len(movies):,} | users {len(users):,}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(__file__).resolve().parent)
