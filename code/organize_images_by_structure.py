import pandas as pd
import os
import shutil
from tqdm import tqdm
import concurrent.futures
import time

# --- 定数定義 ---
IMAGE_SOURCE_DIR = 'posts_image/unzipped_data_7z/image/'
INFLUENCERS_FILE = 'influencers.txt'
ORGANIZED_OUTPUT_DIR = 'organized_images' # 整理後の画像ファイル出力先

# --- グローバル変数 (並列処理用) ---
username_to_category = {}

def move_image_worker(filename):
    """
    単一の画像ファイルを適切なカテゴリ/ユーザー名のディレクトリに移動するワーカー関数。
    """
    try:
        if '-' not in filename:
            return
        username = filename.split('-', 1)[0]
        category = username_to_category.get(username, '_unknown_category')
        
        target_dir = os.path.join(ORGANIZED_OUTPUT_DIR, category, username)
        os.makedirs(target_dir, exist_ok=True)
        
        source_path = os.path.join(IMAGE_SOURCE_DIR, filename)
        target_path = os.path.join(target_dir, filename)
        
        shutil.move(source_path, target_path)
        
    except Exception as e:
        print(f"エラー: ファイル '{filename}' の処理中に問題が発生しました: {e}")

def organize_images_in_background():
    """
    メインの処理を実行する関数。
    """
    print("--- 画像ファイルの階層整理を開始します ---")
    start_time = time.time()
    
    global username_to_category

    try:
        print(f"'{INFLUENCERS_FILE}' を読み込んでいます...")
        df_influencers = pd.read_csv(INFLUENCERS_FILE, sep='\t', skiprows=[1])
        df_influencers.columns = ['Username', 'Category', '#Followers', '#Followees', '#Posts']
        username_to_category = pd.Series(df_influencers.Category.values, index=df_influencers.Username).to_dict()
        print(f"{len(username_to_category)} 人のインフルエンサーのカテゴリ情報を読み込みました。")
    except FileNotFoundError:
        print(f"エラー: '{INFLUENCERS_FILE}' が見つかりません。")
        return

    try:
        all_files = [f for f in os.listdir(IMAGE_SOURCE_DIR) if f.endswith('.jpg')]
        print(f"'{IMAGE_SOURCE_DIR}' から {len(all_files):,} 個の.jpgファイルを検出しました。")
    except FileNotFoundError:
        print(f"エラー: ディレクトリ '{IMAGE_SOURCE_DIR}' が見つかりません。")
        return

    # ▼▼▼ 修正点: max_workersを明示的に高い値に設定 ▼▼▼
    # PCのCPUコア数の10倍のスレッドを生成し,I/Oの待機時間を最大限に活用します。
    # ディスク性能がボトルネックの場合は,これ以上数値を上げても効果は限定的です。
    num_threads = (os.cpu_count() or 1) * 10
    print(f"{num_threads}個のスレッドを使用して並列処理を開始します...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    # ▲▲▲ 修正点 ▲▲▲
        list(tqdm(executor.map(move_image_worker, all_files), total=len(all_files), desc="画像ファイルを整理中"))

    end_time = time.time()
    print("\n--- 全てのファイルの整理が完了しました ---")
    print(f"ファイルは '{ORGANIZED_OUTPUT_DIR}' ディレクトリ内に 'カテゴリ名/ユーザー名/' の形式で保存されました。")
    print(f"合計処理時間: {end_time - start_time:.2f} 秒")


if __name__ == '__main__':
    organize_images_in_background()
