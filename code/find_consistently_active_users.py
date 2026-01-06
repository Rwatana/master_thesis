#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find users who are consistently active across ALL months in a window.

Main definition ("active"):
  - For every month in the window:
      * post_count >= min_posts
      * active_days >= min_active_days   (days with at least 1 post)

Optional (interaction / edge-like activity):
  - For every month in the window:
      * unique_hashtags >= min_unique_hashtags  (from hashtags_2017.csv)
      * unique_mentions >= min_unique_mentions  (from mentions_2017.csv)

Outputs:
  - consistently_active_users_ALLMONTHS.csv (the final list)
  - per_user_month_activity.csv (monthly stats for selected users)
  - per_user_summary.csv (summary stats for selected users)

Typical usage:
  python find_consistently_active_users.py \
    --posts_csv dataset_A_active_all.csv \
    --end_date 2017-12-31 --num_months 12 \
    --min_posts 5 --min_active_days 3 \
    --topk 200

If you want to add hashtag/mention thresholds:
  python find_consistently_active_users.py \
    --end_date 2017-12-31 --num_months 12 \
    --min_posts 5 --min_active_days 3 \
    --use_hashtags --min_unique_hashtags 3 \
    --use_mentions --min_unique_mentions 2
"""

import argparse
import os
import sys
from dataclasses import dataclass

import pandas as pd


def month_window(end_date: pd.Timestamp, num_months: int):
    """Return (start_ts, end_ts) covering full months up to end_date's month."""
    end_month_end = end_date.to_period("M").end_time
    start_month_start = (end_date - pd.DateOffset(months=num_months - 1)).to_period("M").start_time
    return start_month_start, end_month_end


def ensure_cols(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in dataframe. Existing={list(df.columns)}")


def load_posts_activity(posts_csv: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    """
    Return per-(username, month) metrics:
      - post_count
      - active_days
    """
    usecols = ["username", "datetime"]
    df = pd.read_csv(posts_csv, usecols=usecols, parse_dates=["datetime"], low_memory=False)
    ensure_cols(df, usecols)

    df["username"] = df["username"].astype(str).str.strip()
    df = df.dropna(subset=["username", "datetime"])

    df = df[(df["datetime"] >= start_ts) & (df["datetime"] <= end_ts)].copy()
    df["month"] = df["datetime"].dt.to_period("M").astype(str)
    df["date"] = df["datetime"].dt.date

    agg = df.groupby(["username", "month"], as_index=False).agg(
        post_count=("datetime", "size"),
        active_days=("date", "nunique"),
    )
    return agg


def load_unique_targets_per_month(path: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp,
                                  src_col: str, tgt_col: str, ts_col: str, label: str) -> pd.DataFrame:
    """
    Generic loader for edge-like sources (hashtags_2017.csv / mentions_2017.csv)
    Returns per-(username, month): unique_{label}
    """
    df = pd.read_csv(path, low_memory=False)
    # normalize
    if src_col not in df.columns or tgt_col not in df.columns or ts_col not in df.columns:
        raise ValueError(f"{path} must have columns: {src_col}, {tgt_col}, {ts_col}. Got={list(df.columns)}")

    df = df[[src_col, tgt_col, ts_col]].copy()
    df.rename(columns={src_col: "username", tgt_col: "target"}, inplace=True)

    # timestamps are usually unix seconds in your pipeline
    df["datetime"] = pd.to_datetime(df[ts_col], unit="s", errors="coerce")
    df["username"] = df["username"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()
    df = df.dropna(subset=["username", "target", "datetime"])

    df = df[(df["datetime"] >= start_ts) & (df["datetime"] <= end_ts)].copy()
    df["month"] = df["datetime"].dt.to_period("M").astype(str)

    out = df.groupby(["username", "month"], as_index=False).agg(**{
        f"unique_{label}": ("target", "nunique")
    })
    return out


def build_full_month_table(activity: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    """
    Ensure every user has explicit rows for all months (missing -> 0),
    so the 'all months active' logic is correct.
    """
    users = activity["username"].unique()
    base = pd.MultiIndex.from_product([users, months], names=["username", "month"]).to_frame(index=False)
    merged = base.merge(activity, on=["username", "month"], how="left")
    for c in merged.columns:
        if c not in ["username", "month"]:
            merged[c] = merged[c].fillna(0)
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts_csv", type=str, default="dataset_A_active_all.csv")
    ap.add_argument("--hashtags_csv", type=str, default="hashtags_2017.csv")
    ap.add_argument("--mentions_csv", type=str, default="mentions_2017.csv")

    ap.add_argument("--end_date", type=str, default="2017-12-31")
    ap.add_argument("--num_months", type=int, default=12)

    ap.add_argument("--min_posts", type=int, default=5)
    ap.add_argument("--min_active_days", type=int, default=3)

    ap.add_argument("--use_hashtags", action="store_true")
    ap.add_argument("--min_unique_hashtags", type=int, default=0)

    ap.add_argument("--use_mentions", action="store_true")
    ap.add_argument("--min_unique_mentions", type=int, default=0)

    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--outdir", type=str, default="consistently_active_out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    end_date = pd.to_datetime(args.end_date)
    start_ts, end_ts = month_window(end_date, int(args.num_months))
    months = pd.period_range(start=start_ts.to_period("M"), end=end_ts.to_period("M"), freq="M").astype(str).tolist()

    print(f"[Window] {months[0]} .. {months[-1]}  (num_months={len(months)})")
    print(f"[Range ] {start_ts} .. {end_ts}")

    if not os.path.exists(args.posts_csv):
        print(f"[Error] missing posts_csv: {args.posts_csv}")
        return 1

    # 1) posts activity
    posts_agg = load_posts_activity(args.posts_csv, start_ts, end_ts)
    full = build_full_month_table(posts_agg, months)

    # 2) optional hashtag / mention activity
    if args.use_hashtags:
        if not os.path.exists(args.hashtags_csv):
            print(f"[Error] missing hashtags_csv: {args.hashtags_csv}")
            return 1
        h = load_unique_targets_per_month(
            args.hashtags_csv, start_ts, end_ts,
            src_col="source", tgt_col="target", ts_col="timestamp", label="hashtags"
        )
        full = full.merge(h, on=["username", "month"], how="left")
        full["unique_hashtags"] = full["unique_hashtags"].fillna(0)

    if args.use_mentions:
        if not os.path.exists(args.mentions_csv):
            print(f"[Error] missing mentions_csv: {args.mentions_csv}")
            return 1
        m = load_unique_targets_per_month(
            args.mentions_csv, start_ts, end_ts,
            src_col="source", tgt_col="target", ts_col="timestamp", label="mentions"
        )
        full = full.merge(m, on=["username", "month"], how="left")
        full["unique_mentions"] = full["unique_mentions"].fillna(0)

    # 3) all-months constraints
    cond = (full["post_count"] >= int(args.min_posts)) & (full["active_days"] >= int(args.min_active_days))

    if args.use_hashtags and int(args.min_unique_hashtags) > 0:
        cond = cond & (full["unique_hashtags"] >= int(args.min_unique_hashtags))
    if args.use_mentions and int(args.min_unique_mentions) > 0:
        cond = cond & (full["unique_mentions"] >= int(args.min_unique_mentions))

    # For each user, must satisfy condition in ALL months
    per_user_ok = full.assign(ok=cond).groupby("username", as_index=False).agg(
        ok_months=("ok", "sum"),
        total_months=("month", "nunique"),
        total_posts=("post_count", "sum"),
        min_posts=("post_count", "min"),
        mean_posts=("post_count", "mean"),
        min_active_days=("active_days", "min"),
        mean_active_days=("active_days", "mean"),
        max_posts=("post_count", "max"),
    )

    # optional summary cols
    if args.use_hashtags:
        extra = full.groupby("username", as_index=False).agg(
            min_unique_hashtags=("unique_hashtags", "min"),
            mean_unique_hashtags=("unique_hashtags", "mean"),
            max_unique_hashtags=("unique_hashtags", "max"),
        )
        per_user_ok = per_user_ok.merge(extra, on="username", how="left")

    if args.use_mentions:
        extra = full.groupby("username", as_index=False).agg(
            min_unique_mentions=("unique_mentions", "min"),
            mean_unique_mentions=("unique_mentions", "mean"),
            max_unique_mentions=("unique_mentions", "max"),
        )
        per_user_ok = per_user_ok.merge(extra, on="username", how="left")

    # filter: must be ok for all months
    keep = per_user_ok[(per_user_ok["ok_months"] == per_user_ok["total_months"]) & (per_user_ok["total_months"] == len(months))].copy()
    keep = keep.sort_values(["total_posts", "mean_posts"], ascending=False).reset_index(drop=True)

    print(f"\n[Result] users active in ALL months = {len(keep):,}")

    # 4) save outputs
    out_users = os.path.join(args.outdir, "consistently_active_users_ALLMONTHS.csv")
    keep.to_csv(out_users, index=False)
    print(f"[Write] {out_users}")

    # save per-user-month activity only for selected users (topk)
    topk = min(int(args.topk), len(keep))
    selected_users = set(keep.head(topk)["username"].tolist())
    per_user_month = full[full["username"].isin(selected_users)].copy()
    out_month = os.path.join(args.outdir, "per_user_month_activity.csv")
    per_user_month.to_csv(out_month, index=False)
    print(f"[Write] {out_month}")

    out_sum = os.path.join(args.outdir, "per_user_summary.csv")
    keep.head(topk).to_csv(out_sum, index=False)
    print(f"[Write] {out_sum}")

    # show top
    if topk > 0:
        print("\n===== TOP users (by total_posts) =====")
        print(keep.head(min(30, topk)).to_string(index=False))

    print("\n[Hint] Next step: pick a username from the list and run your edge-rich ranking / XAI on it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
