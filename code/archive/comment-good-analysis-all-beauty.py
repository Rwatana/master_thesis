import pandas as pd
import os
import json
from datetime import datetime
from tqdm import tqdm
import concurrent.futures
import time

# --- 定数定義 ---
INFLUENCERS_FILE = 'influencers.txt'
INFO_DIR = 'posts_info/unzipped_data_7z/info/'
# ▼▼▼ 出力ファイル名を指定 ▼▼▼
OUTPUT_CSV_FILE = 'output_beauty_category.csv'

# --- 関数定義 ---
# (process_all.pyと全く同じなので、ここでは省略せずに再度記載します)

def load_influencer_data(filepath):
    """influencers.txtを読み込み、DataFrameを返す。"""
    try:
        df = pd.read_csv(filepath, sep='\t', skiprows=[1])
        return df
    except FileNotFoundError:
        print(f"エラー: `{filepath}` が見つかりません。")
        return pd.DataFrame()

def process_single_influencer(username):
    """一人のインフルエンサーに関する全ての投稿ファイルを処理する関数（並列処理のワーカー）。"""
    local_posts = []
    try:
        all_files = os.listdir(INFO_DIR)
        user_post_files = [f for f in all_files if f.startswith(f"{username}-") and f.endswith('.info')]
        for filename in user_post_files:
            filepath = os.path.join(INFO_DIR, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            local_posts.append({
                'username': username,
                'datetime': datetime.fromtimestamp(data.get('taken_at_timestamp', 0)),
                'likes': data.get('edge_media_preview_like', {}).get('count', 0),
                'comments': data.get('edge_media_to_parent_comment', {}).get('count', 0),
            })
        return local_posts
    except Exception as e:
        print(f"警告: {username}の処理中にエラーが発生しました: {e}")
        return []

def load_all_posts_data_parallel(influencer_list):
    """ThreadPoolExecutorを使って、投稿データを並列で読み込む。"""
    all_posts = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=None) as executor:
        results = list(tqdm(executor.map(process_single_influencer, influencer_list), total=len(influencer_list), desc="インフルエンサーを処理中"))
    for post_list in results:
        all_posts.extend(post_list)
    if not all_posts:
        return pd.DataFrame()
    df = pd.DataFrame(all_posts)
    return df.sort_values('datetime', ascending=False).reset_index(drop=True)

# --- メイン処理 ---
if __name__ == "__main__":
    start_time = time.time()
    print("--- Beautyカテゴリの処理を開始します ---")

    # 1. 全インフルエンサーの情報をDataFrameとして読み込む
    df_influencers = load_influencer_data(INFLUENCERS_FILE)

    if not df_influencers.empty:
        # 2. 'beauty'カテゴリのインフルエンサーに絞り込む
        df_beauty = df_influencers[df_influencers['Category'].str.lower() == 'beauty'].copy()
        
        if not df_beauty.empty:
            beauty_influencer_list = sorted(df_beauty['Username'].unique())
            print(f"対象インフルエンサー: {len(beauty_influencer_list)} 人")

            # 3. 並列処理で対象インフルエンサーの投稿データをDataFrameに集約
            df_all_posts = load_all_posts_data_parallel(beauty_influencer_list)

            if not df_all_posts.empty:
                # 4. CSVファイルに保存
                columns_to_save = ['datetime', 'username', 'likes', 'comments']
                df_all_posts[columns_to_save].to_csv(OUTPUT_CSV_FILE, index=False)

                end_time = time.time()
                print("\n--- 処理完了 ---")
                print(f"✅ 合計 {len(df_all_posts)} 件の投稿データを '{OUTPUT_CSV_FILE}' に保存しました。")
                print(f"処理時間: {end_time - start_time:.2f} 秒")
            else:
                print("処理対象の投稿データが見つかりませんでした。")
        else:
            print("Beautyカテゴリのインフルエンサーが見つかりませんでした。")
    else:
        print("インフルエンサー情報が取得できませんでした。")
