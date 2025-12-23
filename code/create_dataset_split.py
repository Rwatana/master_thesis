import pandas as pd
import os
from tqdm import tqdm
import time

# --- 定数定義 ---
# 入力ファイル
PREPROCESSED_FILE = 'preprocessed_posts_with_metadata_data_check_clean.csv'
HASHTAGS_FILE = 'output_hashtags_all_parallel.csv'
MENTIONS_FILE = 'output_mentions_all_parallel.csv'
INFLUENCERS_FILE = 'influencers.txt'

# ▼▼▼ 修正点: 対象年と出力ディレクトリ名を変更 ▼▼▼
YEAR_TO_PROCESS = 2017
OUTPUT_DIR = f'dataset_{YEAR_TO_PROCESS}'
# ▲▲▲ 修正点 ▲▲▲

TRAIN_DIR = os.path.join(OUTPUT_DIR, 'training')
TEST_DIR = os.path.join(OUTPUT_DIR, 'testing')

def create_dataset_split():
    """
    指定された年のデータセットを訓練用とテスト用に分割する。
    """
    print(f"--- {YEAR_TO_PROCESS}年のデータセット分割処理を開始します ---")
    start_time = time.time()

    # --- 1. 出力ディレクトリの作成 ---
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    print(f"出力先ディレクトリ '{TRAIN_DIR}' と '{TEST_DIR}' を準備しました。")

    # --- 2. 投稿データの読み込みとフィルタリング ---
    try:
        print(f"'{PREPROCESSED_FILE}' を読み込んでいます...")
        df_posts = pd.read_csv(PREPROCESSED_FILE, parse_dates=['datetime'], low_memory=False)
    except FileNotFoundError:
        print(f"エラー: '{PREPROCESSED_FILE}' が見つかりません。")
        return

    # ▼▼▼ 修正点: 対象年でフィルタリング ▼▼▼
    print(f"{YEAR_TO_PROCESS}年の投稿データにフィルタリングしています...")
    df_posts_filtered = df_posts[df_posts['datetime'].dt.year == YEAR_TO_PROCESS].copy()
    # ▲▲▲ 修正点 ▲▲▲
    
    # 訓練用データ (1月-11月)
    df_train_posts = df_posts_filtered[df_posts_filtered['datetime'].dt.month < 12]
    # テスト用データ (12月)
    df_test_posts = df_posts_filtered[df_posts_filtered['datetime'].dt.month == 12]

    print(f"訓練用投稿数: {len(df_train_posts)} 件 | テスト用投稿数: {len(df_test_posts)} 件")
    df_train_posts.to_csv(os.path.join(TRAIN_DIR, 'posts.csv'), index=False)
    df_test_posts.to_csv(os.path.join(TEST_DIR, 'posts.csv'), index=False)
    print("投稿データの分割・保存が完了しました。")

    # --- 3. ハッシュタグとメンションデータの分割 ---
    files_to_process = {
        'hashtags': HASHTAGS_FILE,
        'mentions': MENTIONS_FILE
    }
    for name, filepath in files_to_process.items():
        try:
            print(f"'{filepath}' を読み込んで分割しています...")
            df = pd.read_csv(filepath, header=0)
            df.rename(columns={'source': 'username', 'target': name[:-1]}, inplace=True)
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
            df.dropna(subset=['datetime'], inplace=True)

            # ▼▼▼ 修正点: 対象年でフィルタリング ▼▼▼
            df_filtered = df[df['datetime'].dt.year == YEAR_TO_PROCESS]
            # ▲▲▲ 修正点 ▲▲▲
            
            df_train = df_filtered[df_filtered['datetime'].dt.month < 12]
            df_test = df_filtered[df_filtered['datetime'].dt.month == 12]

            df_train.to_csv(os.path.join(TRAIN_DIR, f'{name}.csv'), index=False)
            df_test.to_csv(os.path.join(TEST_DIR, f'{name}.csv'), index=False)
            print(f"{name}データの分割・保存が完了しました。")
        except FileNotFoundError:
            print(f"警告: '{filepath}' が見つかりませんでした。スキップします。")
            continue

    # --- 4. インフルエンサーデータのフィルタリング ---
    try:
        print(f"'{INFLUENCERS_FILE}' を読み込んでいます...")
        df_influencers = pd.read_csv(INFLUENCERS_FILE, sep='\t', skiprows=[1])
        df_influencers.columns = ['Username', 'Category', '#Followers', '#Followees', '#Posts']
        
        # ▼▼▼ 修正点: 対象年に投稿した実績のあるインフルエンサーのみに絞り込む ▼▼▼
        active_users = set(df_posts_filtered['username'].unique())
        df_influencers_filtered = df_influencers[df_influencers['Username'].isin(active_users)]
        
        print(f"{YEAR_TO_PROCESS}年にアクティブなインフルエンサー: {len(df_influencers_filtered)} 人")
        # ▲▲▲ 修正点 ▲▲▲
        
        df_influencers_filtered.to_csv(os.path.join(TRAIN_DIR, 'influencers.csv'), index=False)
        df_influencers_filtered.to_csv(os.path.join(TEST_DIR, 'influencers.csv'), index=False)
        print("インフルエンサーデータの保存が完了しました。")
    except FileNotFoundError:
        print(f"警告: '{INFLUENCERS_FILE}' が見つかりませんでした。スキップします。")

    end_time = time.time()
    print("\n--- 全ての処理が完了しました ---")
    print(f"合計処理時間: {end_time - start_time:.2f} 秒")

if __name__ == '__main__':
    create_dataset_split()
