import pandas as pd
import os

# --- 設定 ---
# 元データのファイルパス
ORIGINAL_POSTS_FILE = 'preprocessed_posts_with_metadata.csv'
ORIGINAL_HASHTAGS_FILE = 'output_hashtags_all_parallel.csv'
ORIGINAL_MENTIONS_FILE = 'output_mentions_all_parallel.csv'

# ✅ 変更点: 抽出後のファイルパスを変更
POSTS_2017_FILE = 'posts_2017.csv'
HASHTAGS_2017_FILE = 'hashtags_2017.csv'
MENTIONS_2017_FILE = 'mentions_2017.csv'

# ✅ 変更点: 抽出対象の年を指定
TARGET_YEAR = 2017

def filter_data_for_2017():
    """元データから2017年のデータのみを抽出し,新しいファイルとして保存する"""
    print(f"--- Starting Data Filtering Process for Year {TARGET_YEAR} ---")

    # 1. 投稿データ (posts) のフィルタリング
    print(f"\n1. Filtering {ORIGINAL_POSTS_FILE}...")
    if os.path.exists(ORIGINAL_POSTS_FILE):
        df_posts = pd.read_csv(ORIGINAL_POSTS_FILE, parse_dates=['datetime'], low_memory=False)
        
        # ✅ 変更点: 指定した年でデータを抽出
        df_2017 = df_posts[df_posts['datetime'].dt.year == TARGET_YEAR]
        
        df_2017.to_csv(POSTS_2017_FILE, index=False)
        print(f"  - Filtered data for {TARGET_YEAR} saved to {POSTS_2017_FILE} (Rows: {len(df_2017)})")
    else:
        print(f"  - ERROR: File not found.")

    # 2. ハッシュタグデータ (hashtags) のフィルタリング
    print(f"\n2. Filtering {ORIGINAL_HASHTAGS_FILE}...")
    if os.path.exists(ORIGINAL_HASHTAGS_FILE):
        df_hashtags = pd.read_csv(ORIGINAL_HASHTAGS_FILE, header=0, low_memory=False)
        df_hashtags['datetime'] = pd.to_datetime(df_hashtags['timestamp'], unit='s')
        
        # ✅ 変更点: 指定した年でデータを抽出
        df_2017 = df_hashtags[df_hashtags['datetime'].dt.year == TARGET_YEAR]
        
        df_2017.to_csv(HASHTAGS_2017_FILE, index=False)
        print(f"  - Filtered data for {TARGET_YEAR} saved to {HASHTAGS_2017_FILE} (Rows: {len(df_2017)})")
    else:
        print(f"  - ERROR: File not found.")

    # 3. メンションデータ (mentions) のフィルタリング
    print(f"\n3. Filtering {ORIGINAL_MENTIONS_FILE}...")
    if os.path.exists(ORIGINAL_MENTIONS_FILE):
        df_mentions = pd.read_csv(ORIGINAL_MENTIONS_FILE, header=0, low_memory=False)
        df_mentions['datetime'] = pd.to_datetime(df_mentions['timestamp'], unit='s')
        
        # ✅ 変更点: 指定した年でデータを抽出
        df_2017 = df_mentions[df_mentions['datetime'].dt.year == TARGET_YEAR]
        
        df_2017.to_csv(MENTIONS_2017_FILE, index=False)
        print(f"  - Filtered data for {TARGET_YEAR} saved to {MENTIONS_2017_FILE} (Rows: {len(df_2017)})")
    else:
        print(f"  - ERROR: File not found.")
        
    print("\n--- Data Filtering Complete! ---")

if __name__ == '__main__':
    filter_data_for_2017()