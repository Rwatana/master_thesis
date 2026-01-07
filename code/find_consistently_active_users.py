# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Find users who are consistently active across ALL months in a window.

# Main definition ("active"):
#   - For every month in the window:
#       * post_count >= min_posts
#       * active_days >= min_active_days   (days with at least 1 post)

# Optional (interaction / edge-like activity):
#   - For every month in the window:
#       * unique_hashtags >= min_unique_hashtags  (from hashtags_2017.csv)
#       * unique_mentions >= min_unique_mentions  (from mentions_2017.csv)

# Outputs:
#   - consistently_active_users_ALLMONTHS.csv (the final list)
#   - per_user_month_activity.csv (monthly stats for selected users)
#   - per_user_summary.csv (summary stats for selected users)

# Typical usage:
#   python find_consistently_active_users.py \
#     --posts_csv dataset_A_active_all.csv \
#     --end_date 2017-12-31 --num_months 12 \
#     --min_posts 5 --min_active_days 3 \
#     --topk 200

# If you want to add hashtag/mention thresholds:
#   python find_consistently_active_users.py \
#     --end_date 2017-12-31 --num_months 12 \
#     --min_posts 5 --min_active_days 3 \
#     --use_hashtags --min_unique_hashtags 3 \
#     --use_mentions --min_unique_mentions 2
# """

# import argparse
# import os
# import sys
# from dataclasses import dataclass

# import pandas as pd


# def month_window(end_date: pd.Timestamp, num_months: int):
#     """Return (start_ts, end_ts) covering full months up to end_date's month."""
#     end_month_end = end_date.to_period("M").end_time
#     start_month_start = (end_date - pd.DateOffset(months=num_months - 1)).to_period("M").start_time
#     return start_month_start, end_month_end


# def ensure_cols(df: pd.DataFrame, cols):
#     missing = [c for c in cols if c not in df.columns]
#     if missing:
#         raise ValueError(f"Missing columns {missing} in dataframe. Existing={list(df.columns)}")


# def load_posts_activity(posts_csv: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
#     """
#     Return per-(username, month) metrics:
#       - post_count
#       - active_days
#     """
#     usecols = ["username", "datetime"]
#     df = pd.read_csv(posts_csv, usecols=usecols, parse_dates=["datetime"], low_memory=False)
#     ensure_cols(df, usecols)

#     df["username"] = df["username"].astype(str).str.strip()
#     df = df.dropna(subset=["username", "datetime"])

#     df = df[(df["datetime"] >= start_ts) & (df["datetime"] <= end_ts)].copy()
#     df["month"] = df["datetime"].dt.to_period("M").astype(str)
#     df["date"] = df["datetime"].dt.date

#     agg = df.groupby(["username", "month"], as_index=False).agg(
#         post_count=("datetime", "size"),
#         active_days=("date", "nunique"),
#     )
#     return agg


# def load_unique_targets_per_month(path: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp,
#                                   src_col: str, tgt_col: str, ts_col: str, label: str) -> pd.DataFrame:
#     """
#     Generic loader for edge-like sources (hashtags_2017.csv / mentions_2017.csv)
#     Returns per-(username, month): unique_{label}
#     """
#     df = pd.read_csv(path, low_memory=False)
#     # normalize
#     if src_col not in df.columns or tgt_col not in df.columns or ts_col not in df.columns:
#         raise ValueError(f"{path} must have columns: {src_col}, {tgt_col}, {ts_col}. Got={list(df.columns)}")

#     df = df[[src_col, tgt_col, ts_col]].copy()
#     df.rename(columns={src_col: "username", tgt_col: "target"}, inplace=True)

#     # timestamps are usually unix seconds in your pipeline
#     df["datetime"] = pd.to_datetime(df[ts_col], unit="s", errors="coerce")
#     df["username"] = df["username"].astype(str).str.strip()
#     df["target"] = df["target"].astype(str).str.strip()
#     df = df.dropna(subset=["username", "target", "datetime"])

#     df = df[(df["datetime"] >= start_ts) & (df["datetime"] <= end_ts)].copy()
#     df["month"] = df["datetime"].dt.to_period("M").astype(str)

#     out = df.groupby(["username", "month"], as_index=False).agg(**{
#         f"unique_{label}": ("target", "nunique")
#     })
#     return out


# def build_full_month_table(activity: pd.DataFrame, months: list[str]) -> pd.DataFrame:
#     """
#     Ensure every user has explicit rows for all months (missing -> 0),
#     so the 'all months active' logic is correct.
#     """
#     users = activity["username"].unique()
#     base = pd.MultiIndex.from_product([users, months], names=["username", "month"]).to_frame(index=False)
#     merged = base.merge(activity, on=["username", "month"], how="left")
#     for c in merged.columns:
#         if c not in ["username", "month"]:
#             merged[c] = merged[c].fillna(0)
#     return merged


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--posts_csv", type=str, default="dataset_A_active_all.csv")
#     ap.add_argument("--hashtags_csv", type=str, default="hashtags_2017.csv")
#     ap.add_argument("--mentions_csv", type=str, default="mentions_2017.csv")

#     ap.add_argument("--end_date", type=str, default="2017-12-31")
#     ap.add_argument("--num_months", type=int, default=12)

#     ap.add_argument("--min_posts", type=int, default=5)
#     ap.add_argument("--min_active_days", type=int, default=3)

#     ap.add_argument("--use_hashtags", action="store_true")
#     ap.add_argument("--min_unique_hashtags", type=int, default=0)

#     ap.add_argument("--use_mentions", action="store_true")
#     ap.add_argument("--min_unique_mentions", type=int, default=0)

#     ap.add_argument("--topk", type=int, default=200)
#     ap.add_argument("--outdir", type=str, default="consistently_active_out")
#     args = ap.parse_args()

#     os.makedirs(args.outdir, exist_ok=True)

#     end_date = pd.to_datetime(args.end_date)
#     start_ts, end_ts = month_window(end_date, int(args.num_months))
#     months = pd.period_range(start=start_ts.to_period("M"), end=end_ts.to_period("M"), freq="M").astype(str).tolist()

#     print(f"[Window] {months[0]} .. {months[-1]}  (num_months={len(months)})")
#     print(f"[Range ] {start_ts} .. {end_ts}")

#     if not os.path.exists(args.posts_csv):
#         print(f"[Error] missing posts_csv: {args.posts_csv}")
#         return 1

#     # 1) posts activity
#     posts_agg = load_posts_activity(args.posts_csv, start_ts, end_ts)
#     full = build_full_month_table(posts_agg, months)

#     # 2) optional hashtag / mention activity
#     if args.use_hashtags:
#         if not os.path.exists(args.hashtags_csv):
#             print(f"[Error] missing hashtags_csv: {args.hashtags_csv}")
#             return 1
#         h = load_unique_targets_per_month(
#             args.hashtags_csv, start_ts, end_ts,
#             src_col="source", tgt_col="target", ts_col="timestamp", label="hashtags"
#         )
#         full = full.merge(h, on=["username", "month"], how="left")
#         full["unique_hashtags"] = full["unique_hashtags"].fillna(0)

#     if args.use_mentions:
#         if not os.path.exists(args.mentions_csv):
#             print(f"[Error] missing mentions_csv: {args.mentions_csv}")
#             return 1
#         m = load_unique_targets_per_month(
#             args.mentions_csv, start_ts, end_ts,
#             src_col="source", tgt_col="target", ts_col="timestamp", label="mentions"
#         )
#         full = full.merge(m, on=["username", "month"], how="left")
#         full["unique_mentions"] = full["unique_mentions"].fillna(0)

#     # 3) all-months constraints
#     cond = (full["post_count"] >= int(args.min_posts)) & (full["active_days"] >= int(args.min_active_days))

#     if args.use_hashtags and int(args.min_unique_hashtags) > 0:
#         cond = cond & (full["unique_hashtags"] >= int(args.min_unique_hashtags))
#     if args.use_mentions and int(args.min_unique_mentions) > 0:
#         cond = cond & (full["unique_mentions"] >= int(args.min_unique_mentions))

#     # For each user, must satisfy condition in ALL months
#     per_user_ok = full.assign(ok=cond).groupby("username", as_index=False).agg(
#         ok_months=("ok", "sum"),
#         total_months=("month", "nunique"),
#         total_posts=("post_count", "sum"),
#         min_posts=("post_count", "min"),
#         mean_posts=("post_count", "mean"),
#         min_active_days=("active_days", "min"),
#         mean_active_days=("active_days", "mean"),
#         max_posts=("post_count", "max"),
#     )

#     # optional summary cols
#     if args.use_hashtags:
#         extra = full.groupby("username", as_index=False).agg(
#             min_unique_hashtags=("unique_hashtags", "min"),
#             mean_unique_hashtags=("unique_hashtags", "mean"),
#             max_unique_hashtags=("unique_hashtags", "max"),
#         )
#         per_user_ok = per_user_ok.merge(extra, on="username", how="left")

#     if args.use_mentions:
#         extra = full.groupby("username", as_index=False).agg(
#             min_unique_mentions=("unique_mentions", "min"),
#             mean_unique_mentions=("unique_mentions", "mean"),
#             max_unique_mentions=("unique_mentions", "max"),
#         )
#         per_user_ok = per_user_ok.merge(extra, on="username", how="left")

#     # filter: must be ok for all months
#     keep = per_user_ok[(per_user_ok["ok_months"] == per_user_ok["total_months"]) & (per_user_ok["total_months"] == len(months))].copy()
#     keep = keep.sort_values(["total_posts", "mean_posts"], ascending=False).reset_index(drop=True)

#     print(f"\n[Result] users active in ALL months = {len(keep):,}")

#     # 4) save outputs
#     out_users = os.path.join(args.outdir, "consistently_active_users_ALLMONTHS.csv")
#     keep.to_csv(out_users, index=False)
#     print(f"[Write] {out_users}")

#     # save per-user-month activity only for selected users (topk)
#     topk = min(int(args.topk), len(keep))
#     selected_users = set(keep.head(topk)["username"].tolist())
#     per_user_month = full[full["username"].isin(selected_users)].copy()
#     out_month = os.path.join(args.outdir, "per_user_month_activity.csv")
#     per_user_month.to_csv(out_month, index=False)
#     print(f"[Write] {out_month}")

#     out_sum = os.path.join(args.outdir, "per_user_summary.csv")
#     keep.head(topk).to_csv(out_sum, index=False)
#     print(f"[Write] {out_sum}")

#     # show top
#     if topk > 0:
#         print("\n===== TOP users (by total_posts) =====")
#         print(keep.head(min(30, topk)).to_string(index=False))

#     print("\n[Hint] Next step: pick a username from the list and run your edge-rich ranking / XAI on it.")
#     return 0


# if __name__ == "__main__":
#     raise SystemExit(main())


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import argparse
# import os
# import pandas as pd


# def month_window(end_date: pd.Timestamp, num_months: int):
#     end_month_end = end_date.to_period("M").end_time
#     start_month_start = (end_date - pd.DateOffset(months=num_months - 1)).to_period("M").start_time
#     return start_month_start, end_month_end


# def ensure_cols(df: pd.DataFrame, cols):
#     missing = [c for c in cols if c not in df.columns]
#     if missing:
#         raise ValueError(f"Missing columns {missing} in dataframe. Existing={list(df.columns)}")


# def load_posts_activity(posts_csv: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
#     usecols = ["username", "datetime"]
#     df = pd.read_csv(posts_csv, usecols=usecols, parse_dates=["datetime"], low_memory=False)
#     ensure_cols(df, usecols)

#     df["username"] = df["username"].astype(str).str.strip()
#     df = df.dropna(subset=["username", "datetime"])

#     df = df[(df["datetime"] >= start_ts) & (df["datetime"] <= end_ts)].copy()
#     df["month"] = df["datetime"].dt.to_period("M").astype(str)
#     df["date"] = df["datetime"].dt.date

#     agg = df.groupby(["username", "month"], as_index=False).agg(
#         post_count=("datetime", "size"),
#         active_days=("date", "nunique"),
#     )
#     return agg


# def load_edge_events_per_month(
#     path: str,
#     start_ts: pd.Timestamp,
#     end_ts: pd.Timestamp,
#     src_col: str = "source",
#     tgt_col: str = "target",
#     ts_col: str = "timestamp",
#     label: str = "edge",
# ) -> pd.DataFrame:
#     """
#     Returns per-(username, month):
#       - edge_events_{label} : number of rows (events)
#       - unique_edges_{label}: nunique of targets
#     """
#     usecols = [src_col, tgt_col, ts_col]
#     df = pd.read_csv(path, usecols=usecols, low_memory=False)

#     if src_col not in df.columns or tgt_col not in df.columns or ts_col not in df.columns:
#         raise ValueError(f"{path} must have columns: {src_col}, {tgt_col}, {ts_col}. Got={list(df.columns)}")

#     df = df.rename(columns={src_col: "username", tgt_col: "target"})[["username", "target", ts_col]].copy()

#     # timestamps are usually unix seconds
#     df["datetime"] = pd.to_datetime(df[ts_col], unit="s", errors="coerce")
#     df["username"] = df["username"].astype(str).str.strip()
#     df["target"] = df["target"].astype(str).str.strip()

#     # distinguish sources to avoid collisions (e.g., hashtag "#foo" vs mention "#foo" by accident)
#     df["target"] = df["target"].map(lambda x: f"{label}:{x}")

#     df = df.dropna(subset=["username", "target", "datetime"])
#     df = df[(df["datetime"] >= start_ts) & (df["datetime"] <= end_ts)].copy()
#     df["month"] = df["datetime"].dt.to_period("M").astype(str)

#     out = df.groupby(["username", "month"], as_index=False).agg(
#         **{
#             f"edge_events_{label}": ("target", "size"),
#             f"unique_edges_{label}": ("target", "nunique"),
#         }
#     )
#     return out


# def build_full_month_table(users: pd.Series, months: list[str]) -> pd.DataFrame:
#     base = pd.MultiIndex.from_product([users.unique(), months], names=["username", "month"]).to_frame(index=False)
#     return base


# def main():
#     ap = argparse.ArgumentParser()

#     ap.add_argument("--posts_csv", type=str, default="dataset_A_active_all.csv")
#     ap.add_argument("--hashtags_csv", type=str, default="hashtags_2017.csv")
#     ap.add_argument("--mentions_csv", type=str, default="mentions_2017.csv")

#     ap.add_argument("--end_date", type=str, default="2017-12-31")
#     ap.add_argument("--num_months", type=int, default=12)

#     # post activity constraints
#     ap.add_argument("--min_posts", type=int, default=5)
#     ap.add_argument("--min_active_days", type=int, default=3)

#     # edge constraints (UNION of hashtag+mention)
#     ap.add_argument("--use_hashtags", action="store_true")
#     ap.add_argument("--use_mentions", action="store_true")
#     ap.add_argument("--min_unique_edges", type=int, default=10)   # “しっかり”の主軸
#     ap.add_argument("--min_edge_events", type=int, default=30)    # 出現回数（行数）でも下限をつけたい場合

#     ap.add_argument("--topk", type=int, default=200)
#     ap.add_argument("--outdir", type=str, default="consistently_active_out")
#     args = ap.parse_args()

#     os.makedirs(args.outdir, exist_ok=True)

#     end_date = pd.to_datetime(args.end_date)
#     start_ts, end_ts = month_window(end_date, int(args.num_months))
#     months = pd.period_range(start=start_ts.to_period("M"), end=end_ts.to_period("M"), freq="M").astype(str).tolist()

#     print(f"[Window] {months[0]} .. {months[-1]}  (num_months={len(months)})")
#     print(f"[Range ] {start_ts} .. {end_ts}")

#     # --- posts ---
#     if not os.path.exists(args.posts_csv):
#         raise FileNotFoundError(f"missing posts_csv: {args.posts_csv}")
#     posts_agg = load_posts_activity(args.posts_csv, start_ts, end_ts)

#     # user universe = users who appear in posts within window
#     users = posts_agg["username"].astype(str).str.strip()
#     full = build_full_month_table(users, months)

#     full = full.merge(posts_agg, on=["username", "month"], how="left")
#     full["post_count"] = full["post_count"].fillna(0).astype(int)
#     full["active_days"] = full["active_days"].fillna(0).astype(int)

#     # --- edges: union ---
#     edge_parts = []
#     if args.use_hashtags:
#         if not os.path.exists(args.hashtags_csv):
#             raise FileNotFoundError(f"missing hashtags_csv: {args.hashtags_csv}")
#         edge_parts.append(load_edge_events_per_month(args.hashtags_csv, start_ts, end_ts, label="hashtag"))

#     if args.use_mentions:
#         if not os.path.exists(args.mentions_csv):
#             raise FileNotFoundError(f"missing mentions_csv: {args.mentions_csv}")
#         edge_parts.append(load_edge_events_per_month(args.mentions_csv, start_ts, end_ts, label="mention"))

#     if edge_parts:
#         edges = edge_parts[0]
#         for e in edge_parts[1:]:
#             edges = edges.merge(e, on=["username", "month"], how="outer")

#         # fill missing
#         for c in edges.columns:
#             if c not in ["username", "month"]:
#                 edges[c] = edges[c].fillna(0)

#         full = full.merge(edges, on=["username", "month"], how="left")
#         for c in full.columns:
#             if c not in ["username", "month", "post_count", "active_days"]:
#                 full[c] = full[c].fillna(0)

#         # UNION metrics
#         # unique_edges = unique_hashtag + unique_mention (targets were label-prefixed so sum is ok)
#         unique_cols = [c for c in full.columns if c.startswith("unique_edges_")]
#         event_cols = [c for c in full.columns if c.startswith("edge_events_")]

#         full["unique_edges"] = full[unique_cols].sum(axis=1) if unique_cols else 0
#         full["edge_events"] = full[event_cols].sum(axis=1) if event_cols else 0
#     else:
#         # if you forget to enable, treat as 0 edges
#         full["unique_edges"] = 0
#         full["edge_events"] = 0

#     # --- all-month constraints ---
#     cond = (full["post_count"] >= int(args.min_posts)) & (full["active_days"] >= int(args.min_active_days))

#     # edge-rich: must have enough edges in EVERY month
#     if edge_parts:
#         cond = cond & (full["unique_edges"] >= int(args.min_unique_edges)) & (full["edge_events"] >= int(args.min_edge_events))
#     else:
#         print("[Warn] edge sources are disabled. Use --use_hashtags and/or --use_mentions")

#     per_user_ok = full.assign(ok=cond).groupby("username", as_index=False).agg(
#         ok_months=("ok", "sum"),
#         total_months=("month", "nunique"),
#         total_posts=("post_count", "sum"),
#         min_posts=("post_count", "min"),
#         mean_posts=("post_count", "mean"),
#         min_active_days=("active_days", "min"),
#         mean_active_days=("active_days", "mean"),
#         min_unique_edges=("unique_edges", "min"),
#         mean_unique_edges=("unique_edges", "mean"),
#         min_edge_events=("edge_events", "min"),
#         mean_edge_events=("edge_events", "mean"),
#     )

#     keep = per_user_ok[
#         (per_user_ok["ok_months"] == per_user_ok["total_months"])
#         & (per_user_ok["total_months"] == len(months))
#     ].copy()

#     keep = keep.sort_values(
#         ["min_unique_edges", "mean_unique_edges", "total_posts"],
#         ascending=False
#     ).reset_index(drop=True)

#     print(f"\n[Result] users edge-rich in ALL months = {len(keep):,}")

#     # --- save outputs ---
#     out_users = os.path.join(args.outdir, "consistently_edge_rich_users_ALLMONTHS.csv")
#     keep.to_csv(out_users, index=False)
#     print(f"[Write] {out_users}")

#     topk = min(int(args.topk), len(keep))
#     selected_users = set(keep.head(topk)["username"].tolist())
#     per_user_month = full[full["username"].isin(selected_users)].copy()

#     out_month = os.path.join(args.outdir, "per_user_month_activity_edge_rich.csv")
#     per_user_month.to_csv(out_month, index=False)
#     print(f"[Write] {out_month}")

#     out_sum = os.path.join(args.outdir, "per_user_summary_edge_rich.csv")
#     keep.head(topk).to_csv(out_sum, index=False)
#     print(f"[Write] {out_sum}")

#     if topk > 0:
#         print("\n===== TOP users (edge-rich; by min_unique_edges then mean_unique_edges) =====")
#         print(keep.head(min(30, topk)).to_string(index=False))

#     return 0


# if __name__ == "__main__":
#     raise SystemExit(main())




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find users whose NON-SELF incident edges exist in EVERY month of the window.

Interpretation (matches your prepare_graph_data):
- incident non-self edges for user u exist in a month
  <=> user u has at least one hashtag/mention event in that month
      (because you build edges u -> hashtag / u -> mention and add self-loops separately)

Outputs:
- consistently_incident_edge_users_ALLMONTHS.csv
- per_user_month_activity_incident_edges.csv
- per_user_summary_incident_edges.csv
"""

import argparse
import os
import pandas as pd


def month_window(end_date: pd.Timestamp, num_months: int):
    end_month_end = end_date.to_period("M").end_time
    start_month_start = (end_date - pd.DateOffset(months=num_months - 1)).to_period("M").start_time
    return start_month_start, end_month_end


def ensure_cols(df: pd.DataFrame, cols, name="dataframe"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {name}. Existing={list(df.columns)}")


def load_posts_activity(posts_csv: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    usecols = ["username", "datetime"]
    df = pd.read_csv(posts_csv, usecols=usecols, parse_dates=["datetime"], low_memory=False)
    ensure_cols(df, usecols, name="posts_csv")

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


def _load_edge_file(path: str, label: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp,
                    src_col="source", tgt_col="target", ts_col="timestamp") -> pd.DataFrame:
    usecols = [src_col, tgt_col, ts_col]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    ensure_cols(df, usecols, name=path)

    df = df.rename(columns={src_col: "username", tgt_col: "target"})[["username", "target", ts_col]].copy()

    df["datetime"] = pd.to_datetime(df[ts_col], unit="s", errors="coerce")
    df["username"] = df["username"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()

    # label-prefix to avoid collisions between hashtag/mention
    df["target"] = df["target"].map(lambda x: f"{label}:{x}")

    df = df.dropna(subset=["username", "target", "datetime"])
    df = df[(df["datetime"] >= start_ts) & (df["datetime"] <= end_ts)].copy()
    df["month"] = df["datetime"].dt.to_period("M").astype(str)

    return df[["username", "month", "target"]]


def load_incident_edges_union(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    use_hashtags: bool,
    hashtags_csv: str,
    use_mentions: bool,
    mentions_csv: str,
) -> pd.DataFrame:
    """
    Returns per-(username, month):
      - incident_events: number of edge events (rows)
      - incident_unique_neighbors: nunique of labeled targets
    """
    parts = []
    if use_hashtags:
        parts.append(_load_edge_file(hashtags_csv, "hashtag", start_ts, end_ts))
    if use_mentions:
        parts.append(_load_edge_file(mentions_csv, "mention", start_ts, end_ts))

    if not parts:
        # disabled -> empty
        return pd.DataFrame(columns=["username", "month", "incident_events", "incident_unique_neighbors"])

    all_edges = pd.concat(parts, axis=0, ignore_index=True)

    agg = all_edges.groupby(["username", "month"], as_index=False).agg(
        incident_events=("target", "size"),
        incident_unique_neighbors=("target", "nunique"),
    )
    return agg


def build_full_month_table(users: pd.Series, months: list[str]) -> pd.DataFrame:
    return pd.MultiIndex.from_product([users.unique(), months], names=["username", "month"]).to_frame(index=False)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--posts_csv", type=str, default="dataset_A_active_all.csv")
    ap.add_argument("--hashtags_csv", type=str, default="hashtags_2017.csv")
    ap.add_argument("--mentions_csv", type=str, default="mentions_2017.csv")

    ap.add_argument("--end_date", type=str, default="2017-12-31")
    ap.add_argument("--num_months", type=int, default=12)

    # post constraints (optional but usually desired)
    ap.add_argument("--min_posts", type=int, default=5)
    ap.add_argument("--min_active_days", type=int, default=3)

    # edge sources
    ap.add_argument("--use_hashtags", action="store_true")
    ap.add_argument("--use_mentions", action="store_true")

    # core requirement: non-self incident edges must exist every month
    ap.add_argument("--min_incident_events", type=int, default=1)
    ap.add_argument("--min_unique_incident_neighbors", type=int, default=1)

    # to align with your pipeline (often you keep users who appear in end month)
    ap.add_argument("--restrict_to_end_month_posters", action="store_true")

    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--outdir", type=str, default="consistently_incident_edges_out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    end_date = pd.to_datetime(args.end_date)
    start_ts, end_ts = month_window(end_date, int(args.num_months))
    months = pd.period_range(start=start_ts.to_period("M"), end=end_ts.to_period("M"), freq="M").astype(str).tolist()
    end_month = str(end_date.to_period("M"))

    print(f"[Window] {months[0]} .. {months[-1]}  (num_months={len(months)})")
    print(f"[Range ] {start_ts} .. {end_ts}")
    print(f"[EndMo ] {end_month}")

    # --- posts ---
    if not os.path.exists(args.posts_csv):
        raise FileNotFoundError(f"missing posts_csv: {args.posts_csv}")

    posts_agg = load_posts_activity(args.posts_csv, start_ts, end_ts)

    if args.restrict_to_end_month_posters:
        end_users = set(posts_agg.loc[posts_agg["month"] == end_month, "username"].astype(str))
        posts_agg = posts_agg[posts_agg["username"].isin(end_users)].copy()
        print(f"[Filter] restrict_to_end_month_posters: users={len(end_users):,}")

    users = posts_agg["username"].astype(str).str.strip()
    full = build_full_month_table(users, months)

    full = full.merge(posts_agg, on=["username", "month"], how="left")
    full["post_count"] = full["post_count"].fillna(0).astype(int)
    full["active_days"] = full["active_days"].fillna(0).astype(int)

    # --- incident edges (union) ---
    if not (args.use_hashtags or args.use_mentions):
        print("[Warn] edge sources disabled. Enable --use_hashtags and/or --use_mentions.")
        edges_agg = pd.DataFrame(columns=["username", "month", "incident_events", "incident_unique_neighbors"])
    else:
        if args.use_hashtags and not os.path.exists(args.hashtags_csv):
            raise FileNotFoundError(f"missing hashtags_csv: {args.hashtags_csv}")
        if args.use_mentions and not os.path.exists(args.mentions_csv):
            raise FileNotFoundError(f"missing mentions_csv: {args.mentions_csv}")

        edges_agg = load_incident_edges_union(
            start_ts, end_ts,
            use_hashtags=args.use_hashtags, hashtags_csv=args.hashtags_csv,
            use_mentions=args.use_mentions, mentions_csv=args.mentions_csv,
        )

        if args.restrict_to_end_month_posters:
            edges_agg = edges_agg[edges_agg["username"].isin(set(full["username"].unique()))].copy()

    full = full.merge(edges_agg, on=["username", "month"], how="left")
    full["incident_events"] = full["incident_events"].fillna(0).astype(int)
    full["incident_unique_neighbors"] = full["incident_unique_neighbors"].fillna(0).astype(int)

    # --- constraints per month ---
    cond_posts = (full["post_count"] >= int(args.min_posts)) & (full["active_days"] >= int(args.min_active_days))

    # 핵심: self-loop以外のincident edgeがあること
    cond_incident = (full["incident_events"] >= int(args.min_incident_events)) & (
        full["incident_unique_neighbors"] >= int(args.min_unique_incident_neighbors)
    )

    cond = cond_posts & cond_incident

    # For each user, must satisfy in ALL months
    per_user = full.assign(ok=cond, ok_posts=cond_posts, ok_inc=cond_incident).groupby("username", as_index=False).agg(
        ok_months=("ok", "sum"),
        ok_posts_months=("ok_posts", "sum"),
        ok_incident_months=("ok_inc", "sum"),
        total_months=("month", "nunique"),

        total_posts=("post_count", "sum"),
        min_posts=("post_count", "min"),
        mean_posts=("post_count", "mean"),
        min_active_days=("active_days", "min"),
        mean_active_days=("active_days", "mean"),

        min_incident_events=("incident_events", "min"),
        mean_incident_events=("incident_events", "mean"),
        min_unique_incident_neighbors=("incident_unique_neighbors", "min"),
        mean_unique_incident_neighbors=("incident_unique_neighbors", "mean"),
    )

    keep = per_user[
        (per_user["ok_months"] == per_user["total_months"])
        & (per_user["total_months"] == len(months))
    ].copy()

    keep = keep.sort_values(
        ["min_incident_events", "min_unique_incident_neighbors", "total_posts"],
        ascending=False
    ).reset_index(drop=True)

    print(f"\n[Result] users with NON-SELF incident edges in ALL months = {len(keep):,}")

    # --- save outputs ---
    out_users = os.path.join(args.outdir, "consistently_incident_edge_users_ALLMONTHS.csv")
    keep.to_csv(out_users, index=False)
    print(f"[Write] {out_users}")

    topk = min(int(args.topk), len(keep))
    selected = set(keep.head(topk)["username"].tolist())
    per_user_month = full[full["username"].isin(selected)].copy()

    out_month = os.path.join(args.outdir, "per_user_month_activity_incident_edges.csv")
    per_user_month.to_csv(out_month, index=False)
    print(f"[Write] {out_month}")

    out_sum = os.path.join(args.outdir, "per_user_summary_incident_edges.csv")
    keep.head(topk).to_csv(out_sum, index=False)
    print(f"[Write] {out_sum}")

    if topk > 0:
        print("\n===== TOP users (by min_incident_events, then min_unique_incident_neighbors) =====")
        print(keep.head(min(30, topk)).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
