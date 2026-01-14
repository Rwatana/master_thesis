import pandas as pd
import os
from tqdm import tqdm
import concurrent.futures
import time
import multiprocessing

# --- 定数定義 ---
PREPROCESSED_FILE = 'preprocessed_posts_with_metadata.csv'
INFLUENCERS_FILE = 'influencers.txt'
OUTPUT_DIR = 'processed_by_category'

# グローバル変数としてデータフレームを定義（子プロセスからアクセスするため）
df_merged = None

def save_category_file(category):
    """
    単一のカテゴリのデータフレームをフィルタリングし,CSVとして保存するワーカー関数。
    """
    try:
        # グローバルなdf_mergedから該当カテゴリのデータを抽出
        df_category = df_merged[df_merged['Category'] == category]
        
        if not df_category.empty:
            safe_category_name = str(category).lower().replace(' ', '_')
            output_filename = f"processed_aurora_{safe_category_name}.csv"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            df_category.to_csv(output_path, index=False)
            return f"✅ カテゴリ '{category}': {len(df_category)} 件を保存しました。"
        else:
            return f"⚠️ カテゴリ '{category}': データがありませんでした。"
    except Exception as e:
        return f"❌ カテゴリ '{category}': エラーが発生しました - {e}"

def main():
    """
    メインの処理を実行する関数。
    """
    global df_merged # グローバル変数を更新することを明示

    print("--- ファイルの分割処理を開始します ---")

    # --- 1. データの読み込み ---
    try:
        print(f"'{PREPROCESSED_FILE}'を読み込んでいます...")
        df_posts = pd.read_csv(PREPROCESSED_FILE)
    except FileNotFoundError:
        print(f"エラー: '{PREPROCESSED_FILE}' が見つかりません。先に preprocess_data.py を実行してください。")
        return

    try:
        print(f"'{INFLUENCERS_FILE}'を読み込んでいます...")
        df_influencers = pd.read_csv(INFLUENCERS_FILE, sep='\t', skiprows=[1])
    except FileNotFoundError:
        print(f"エラー: '{INFLUENCERS_FILE}' が見つかりません。")
        return
        
    # --- 2. カテゴリ情報の結合 ---
    print("投稿データにカテゴリ情報を結合しています...")
    df_influencer_categories = df_influencers[['Username', 'Category']].copy()
    df_merged = pd.merge(df_posts, df_influencer_categories, left_on='username', right_on='Username', how='left')
    df_merged.dropna(subset=['Category'], inplace=True)

    # --- 3. カテゴリごとに並列処理で分割・保存 ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_categories = df_merged['Category'].unique()
    
    print(f"見つかったカテゴリ: {len(all_categories)}件")
    num_processes = min(len(all_categories), os.cpu_count())
    print(f"カテゴリ処理のために {num_processes} 個のプロセスを開始します...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(tqdm(executor.map(save_category_file, all_categories), total=len(all_categories), desc="カテゴリごとにファイルを保存中"))

    print("\n--- 処理結果 ---")
    for res in results:
        print(res)

if __name__ == '__main__':
    multiprocessing.freeze_support() # Windows/macOSで安全に実行するために必要
    start_time = time.time()
    main()
    end_time = time.time()
    print("\n--- 全てのファイルの分割が完了しました ---")
    print(f"合計処理時間: {end_time - start_time:.2f} 秒")