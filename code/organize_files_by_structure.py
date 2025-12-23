import pandas as pd
import os
import shutil
from tqdm import tqdm
import concurrent.futures
import time

# --- 定数定義 ---
INFO_DIR = 'posts_info/unzipped_data_7z/info/'
INFLUENCERS_FILE = 'influencers.txt'
ORGANIZED_OUTPUT_DIR = 'organized_posts' # 整理後のファイル出力先

# --- グローバル変数 (並列処理用) ---
# 高速なルックアップのために辞書として保持
username_to_category = {}

def move_file_worker(filename):
    """
    単一の.infoファイルを適切なカテゴリ/ユーザー名のディレクトリに移動するワーカー関数。
    """
    try:
        # ファイル名からユーザー名を抽出 (例: 'username-postid.info' -> 'username')
        if '-' not in filename:
            return # 不正なファイル名はスキップ
        username = filename.split('-', 1)[0]
        
        # 辞書からカテゴリを取得。見つからない場合は '_unknown' カテゴリに入れる
        category = username_to_category.get(username, '_unknown_category')
        
        # ターゲットディレクトリのパスを作成
        target_dir = os.path.join(ORGANIZED_OUTPUT_DIR, category, username)
        
        # ターゲットディレクトリが存在しない場合は作成
        # exist_ok=True は、ディレクトリが既に存在してもエラーにしないオプション
        os.makedirs(target_dir, exist_ok=True)
        
        # ファイルの移動元と移動先のフルパスを定義
        source_path = os.path.join(INFO_DIR, filename)
        target_path = os.path.join(target_dir, filename)
        
        # ファイルを移動
        shutil.move(source_path, target_path)
        
    except Exception as e:
        # エラーが発生した場合、どのファイルで問題が起きたかを表示
        print(f"エラー: ファイル '{filename}' の処理中に問題が発生しました: {e}")

def organize_files_in_background():
    """
    メインの処理を実行する関数。
    """
    print("--- 投稿ファイルの階層整理を開始します ---")
    start_time = time.time()
    
    global username_to_category

    # --- 1. influencers.txtからユーザーとカテゴリのマッピングを作成 ---
    try:
        print(f"'{INFLUENCERS_FILE}' を読み込んでいます...")
        df_influencers = pd.read_csv(INFLUENCERS_FILE, sep='\t', skiprows=[1])
        df_influencers.columns = ['Username', 'Category', '#Followers', '#Followees', '#Posts']
        # 辞書を作成: {'username': 'Category'}
        username_to_category = pd.Series(df_influencers.Category.values, index=df_influencers.Username).to_dict()
        print(f"{len(username_to_category)} 人のインフルエンサーのカテゴリ情報を読み込みました。")
    except FileNotFoundError:
        print(f"エラー: '{INFLUENCERS_FILE}' が見つかりません。")
        return

    # --- 2. 移動対象のファイルリストを取得 ---
    try:
        all_files = [f for f in os.listdir(INFO_DIR) if f.endswith('.info')]
        print(f"'{INFO_DIR}' から {len(all_files):,} 個の.infoファイルを検出しました。")
    except FileNotFoundError:
        print(f"エラー: ディレクトリ '{INFO_DIR}' が見つかりません。")
        return

    # --- 3. 並列処理でファイルの移動を実行 ---
    # ファイル移動はI/Oバウンドな処理なので、ThreadPoolExecutorが適しています
    # max_workers=None で利用可能な最大限のスレッドを使用します
    with concurrent.futures.ThreadPoolExecutor(max_workers=480) as executor:
        # tqdmを使って進捗バーを表示
        list(tqdm(executor.map(move_file_worker, all_files), total=len(all_files), desc="ファイルを整理中"))

    end_time = time.time()
    print("\n--- 全てのファイルの整理が完了しました ---")
    print(f"ファイルは '{ORGANIZED_OUTPUT_DIR}' ディレクトリ内に 'カテゴリ名/ユーザー名/' の形式で保存されました。")
    print(f"合計処理時間: {end_time - start_time:.2f} 秒")


if __name__ == '__main__':
    organize_files_in_background()
