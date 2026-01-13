import pandas as pd

POSTS = "dataset_A_active_all.csv"
HASH  = "hashtags_2017.csv"
MENT  = "mentions_2017.csv"

# posts master
posts = pd.read_csv(POSTS, usecols=["post_id", "username", "timestamp"])
posts = posts.dropna(subset=["post_id","username","timestamp"]).copy()

# timestamp を数値に寄せる（文字でも数値でも吸収）
posts["timestamp"] = pd.to_numeric(posts["timestamp"], errors="coerce")
posts = posts.dropna(subset=["timestamp"]).copy()
posts["timestamp"] = posts["timestamp"].astype("int64")

# edge csv
def attach_post_id(edge_path: str, out_path: str):
    df = pd.read_csv(edge_path)
    df = df.rename(columns={"source":"username", "target":"target"})  # username に寄せる

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["username","target","timestamp"]).copy()
    df["timestamp"] = df["timestamp"].astype("int64")

    # JOIN（同一timestampに複数投稿があると複数マッチする可能性あり → 後で例として上位Nだけ取ればOK）
    merged = df.merge(posts, on=["username","timestamp"], how="left")

    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("wrote:", out_path, "rows=", len(merged), "post_id non-null=", merged["post_id"].notna().sum())

attach_post_id(HASH, "hashtags_2017_with_postid.csv")
attach_post_id(MENT, "mentions_2017_with_postid.csv")
