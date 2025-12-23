import os
import json
from datetime import datetime
from tqdm import tqdm
import concurrent.futures
import multiprocessing
import time

# --- 定数定義 ---
INFO_DIR = 'posts_info/unzipped_data_7z/info/'
YEAR_TO_COUNT = 2017

def check_file_year(filename):
    """
    単一の.infoファイルを読み込み、投稿年が指定された年と一致するかどうかをチェックする。
    一致する場合は1を、それ以外は0を返す。
    """
    try:
        filepath = os.path.join(INFO_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            # タイムスタンプキーだけを効率的に読み込む（ファイル全体をパースしない）
            # ただし、単純なjson.load()でも十分高速なため、ここでは可読性を優先
            data = json.load(f)
        
        timestamp = data.get('taken_at_timestamp')
        if timestamp is not None:
            post_year = datetime.fromtimestamp(timestamp).year
            if post_year == YEAR_TO_COUNT:
                return 1 # Match found
    except (json.JSONDecodeError, KeyError, TypeError, FileNotFoundError):
        # 壊れたファイルやキーがない場合は無視
        pass
    return 0 # No match or error

def count_posts_in_year_parallel():
    """
    並列処理を使用して、INFO_DIR内のファイルの投稿年をチェックし、
    指定された年の投稿件数を集計する。
    """
    print(f"--- {YEAR_TO_COUNT}年の投稿件数の集計を開始します (並列処理) ---")
    start_time = time.time()

    # --- 1. ファイルリストの取得 ---
    try:
        all_files = [f for f in os.listdir(INFO_DIR) if f.endswith('.info')]
        print(f"ディレクトリ '{INFO_DIR}' から {len(all_files):,} 個の.infoファイルを検出しました。")
    except FileNotFoundError:
        print(f"エラー: ディレクトリ '{INFO_DIR}' が見つかりません。")
        return

    # --- 2. 並列処理で各ファイルをチェック ---
    post_count = 0
    # 利用可能なCPUコア数を取得
    num_processes = multiprocessing.cpu_count()
    print(f"{num_processes}個のコアを使用して並列処理を開始します...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        # executor.mapは各ファイルに関数を適用し、結果(0 or 1)を返すイテレータを生成
        results_iterator = executor.map(check_file_year, all_files)
        
        # tqdmで進捗を表示しながら、戻り値(1)の合計を計算
        post_count = sum(tqdm(results_iterator, total=len(all_files), desc=f"{YEAR_TO_COUNT}年のファイルを探索中"))

    # --- 3. 結果の表示 ---
    end_time = time.time()
    print("\n--- 集計完了 ---")
    print(f"✅ {YEAR_TO_COUNT}年1月1日から{YEAR_TO_COUNT}年12月31日までの投稿件数: {post_count:,} 件")
    print(f"合計処理時間: {end_time - start_time:.2f} 秒")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    count_posts_in_year_parallel()
