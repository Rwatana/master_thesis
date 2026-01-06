import pandas as pd
import os
import re
from tqdm import tqdm
import concurrent.futures
import multiprocessing

# --- 定数定義 ---
INPUT_FILE = 'preprocessed_posts_with_metadata_data_check.csv'
OUTPUT_FILE = 'preprocessed_posts_with_metadata_data_check_clean.csv'

def clean_record_worker(record):
    """
    単一のレコード文字列をクリーニングするワーカー関数。
    (先頭と末尾の空白を削除)
    """
    return record.strip()

def clean_csv_file_parallel():
    """
    並列処理を使用して、キャプション内に改行が含まれるCSVファイルを
    1行1レコードのクリーンな形式に整形して新しいファイルに保存する。
    """
    print(f"--- ファイル '{INPUT_FILE}' のクリーニングを開始します ---")

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"エラー: '{INPUT_FILE}' が見つかりません。")
        return

    # --- Step 1 & 2 (シーケンシャル処理): レコード境界を特定し、内部の改行を置換 ---
    # この部分は安全に並列化するのが難しく、通常は十分に高速です。
    print("レコード境界を特定し、内部の改行を置換しています...")
    pattern = re.compile(r'(\n[a-zA-Z0-9_.-]+,\d{4}-\d{2}-\d{2})')
    content_with_delimiters = pattern.sub(r' ||NEW_RECORD|| \1', content)
    content_cleaned_newlines = content_with_delimiters.replace('\n', ' ')
    
    # --- Step 3 (シーケンシャル処理): レコードのリストに分割 ---
    records = content_cleaned_newlines.split(' ||NEW_RECORD|| ')
    print(f"{len(records)}件のレコードを検出しました。並列処理でクリーニングします...")

    # --- Step 4 (並列処理): 各レコードをクリーニング ---
    cleaned_records = []
    # ProcessPoolExecutorを使用してCPUバウンドなタスクを処理
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # executor.mapを使って各レコードにワーカー関数を適用
        # tqdmで進捗を表示
        results_iterator = executor.map(clean_record_worker, records)
        cleaned_records = list(tqdm(results_iterator, total=len(records), desc="レコードを並列処理中"))

    # --- Step 5 (シーケンシャル処理): クリーンなレコードを新しいファイルに書き込み ---
    print(f"'{OUTPUT_FILE}' にクリーンなデータを書き込んでいます...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # 処理中に生成された可能性のある空のレコードを除外
        for record in tqdm(cleaned_records, desc="ファイルに書き込み中"):
            if record:
                f.write(record + '\n')

    print("\n--- クリーニング完了 ---")
    # 最初のヘッダー行も1レコードとしてカウントされるため、-1 は不要
    print(f"✅ 約 {len(cleaned_records)} 件のレコードを '{OUTPUT_FILE}' に保存しました。")
    print("今後はこのクリーンなファイルを読み込むようにしてください。")


if __name__ == '__main__':
    # Windows/macOSで実行可能ファイルを作成する際の互換性のために freeze_support() を追加
    multiprocessing.freeze_support()
    clean_csv_file_parallel()

