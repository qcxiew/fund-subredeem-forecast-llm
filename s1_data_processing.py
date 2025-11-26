# data_processing.py
import pandas as pd
from datetime import timedelta
 
# —— Data partitioning ——
def split_data():
    # 1. Redemption sequence: Force the fund_code to be read as a string, retaining leading zeros.
    df = pd.read_csv(
        "fund_apply_redeem_series.csv",
        parse_dates=["transaction_date"],
        dtype={"fund_code": str}
    )

    df["fund_code"] = df["fund_code"].str.zfill(6)  # If the original file lengths are inconsistent, the width can be adjusted as needed.
    df["date"] = df.transaction_date.dt.date
    last_date = df.transaction_date.max().date()
    split_date = last_date - timedelta(days=15)
    train_s = df[df.date <= split_date]
    test_s  = df[df.date >  split_date]
    train_s.to_csv("train_series.csv", index=False)
    test_s.to_csv("test_series.csv",  index=False)
 
    # 2. News segmentation: Retain the original format; no need to process fund_code.
    all_news = pd.read_json(
        "fund_news.jsonl",
        lines=True,
        convert_dates=["datetime"]
    )

    all_news["date"] = all_news.datetime.dt.date
    all_news[all_news.date <= split_date]    \
        .to_json("train_news.jsonl", orient="records", lines=True,  force_ascii=False )

    all_news[all_news.date >  split_date]    \
        .to_json("test_news.jsonl",  orient="records", lines=True, force_ascii=False )


def merge_series():
    df_train = pd.read_csv("train_series.csv", parse_dates=["transaction_date"], dtype={"fund_code":str})
    df_test  = pd.read_csv("test_series.csv",  parse_dates=["transaction_date"], dtype={"fund_code":str})
    df_all   = pd.concat([df_train, df_test], ignore_index=True)
    # If leading zeros need to be retained
    df_all["fund_code"] = df_all["fund_code"].str.zfill(6)
    df_all = df_all.sort_values(["fund_code","transaction_date"])
    df_all.to_csv("processed_series.csv", index=False)
    print("✔ generated processed_series.csv")

if __name__ == "__main__":
    split_data()
    print("Data partitioning is complete; leading zeros in fund_code have been preserved.")
    merge_series()    # Added: Generate processed_series.csv
