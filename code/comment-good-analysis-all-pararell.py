import pandas as pd
import os
import json
from datetime import datetime
from tqdm import tqdm
import concurrent.futures
import time
import multiprocessing

# --- 定数定義 ---
INFLUENCERS_FILE = 'influencers.txt'
INFO_DIR = 'posts_info/unzipped_data_7z/info/'
OUTPUT_DIR = 'category_outputs' # 出力ファイルをまとめるディレクトリ

# --- 関数定義 ---

def process_single_influencer(username):
    """一人のインフルエンサーに関する全ての投稿ファイルを処理する（インフルエンサー並列処理のワーカー）。"""
    local_posts = []
    try:
        # この関数内ではINFO_DIRのファイルリストを都度取得する
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
    except Exception:
        # エラーが発生した場合は空のリストを返す
        return []

def load_all_posts_data_parallel(influencer_list):
    """ThreadPoolExecutorを使って、特定カテゴリの投稿データを並列で読み込む。"""
    all_posts = []
    # os.cpu_count() or 8 などの上限を設定して、スレッドが多すぎないように調整
    max_threads = min(32, os.cpu_count() + 4)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        # tqdmのdescを動的に変更できないため、ここではシンプルなmapを使用
        results = executor.map(process_single_influencer, influencer_list)
    for post_list in results:
        all_posts.extend(post_list)
    if not all_posts:
        return pd.DataFrame()
    return pd.DataFrame(all_posts)

def process_and_save_category(category_info):
    """
    単一のカテゴリに関する全処理を行う関数（カテゴリ並列処理のワーカー）。
    """
    category_name, df_category_influencers = category_info
    
    influencer_list = sorted(df_category_influencers['Username'].unique())
    
    # 3. 並列処理で対象インフルエンサーの投稿データをDataFrameに集約
    df_all_posts = load_all_posts_data_parallel(influencer_list)
    
    if not df_all_posts.empty:
        # 4. カテゴリごとの命名規則でCSVファイルに保存
        output_filename = f"output_{category_name.lower()}_category.csv"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        columns_to_save = ['datetime', 'username', 'likes', 'comments']
        df_sorted = df_all_posts.sort_values('datetime', ascending=False).reset_index(drop=True)
        df_sorted[columns_to_save].to_csv(output_path, index=False)
        
        return f"✅ カテゴリ '{category_name}': {len(df_all_posts)} 件の投稿を '{output_path}' に保存しました。"
    else:
        return f"⚠️ カテゴリ '{category_name}': 処理対象の投稿データが見つかりませんでした。"

# --- メイン処理 ---
if __name__ == "__main__":
    multiprocessing.freeze_support() # Windows/macOSで安全に実行するために必要
    start_time = time.time()
    
    # 出力ディレクトリを作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"--- 全カテゴリの処理を開始します ---")

    # 1. 全インフルエンサーの情報をDataFrameとして読み込む
    try:
        df_influencers = pd.read_csv(INFLUENCERS_FILE, sep='\t', skiprows=[1])
    except FileNotFoundError:
        print(f"エラー: `{INFLUENCERS_FILE}` が見つかりません。")
        df_influencers = pd.DataFrame()

    if not df_influencers.empty:
        # 2. カテゴリごとにインフルエンサーのリストを作成
        all_categories = df_influencers['Category'].unique()
        category_groups = [(cat, df_influencers[df_influencers['Category'] == cat]) for cat in all_categories]
        
        print(f"対象カテゴリ: {len(all_categories)} 件 ({', '.join(all_categories)})")
        
        # カテゴリごとに並列処理を実行
        num_processes = min(len(all_categories), os.cpu_count())
        print(f"カテゴリ処理のために {num_processes} 個のプロセスを開始します...")

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            # tqdmを使ってカテゴリ処理の進捗を表示
            results = list(tqdm(executor.map(process_and_save_category, category_groups), total=len(category_groups), desc="カテゴリを処理中"))
        
        print("\n--- 処理結果 ---")
        for res in results:
            print(res)
    else:
        print("インフルエンサー情報が取得できませんでした。")

    end_time = time.time()
    print("\n--- 全ての処理が完了しました ---")
    print(f"合計処理時間: {end_time - start_time:.2f} 秒")

