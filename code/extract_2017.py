import pandas as pd
import os
from datetime import datetime, timezone

# --- 設定 ---
# 2017年の開始と終了のUnixタイムスタンプ（UTC）を計算
# 2017-01-01 00:00:00 UTC
START_2017 = datetime(2017, 1, 1, tzinfo=timezone.utc).timestamp()
# 2018-01-01 00:00:00 UTC (これより前を2017年とする)
START_2018 = datetime(2018, 1, 1, tzinfo=timezone.utc).timestamp()

# 処理するファイルのリスト (入力ファイル名, 出力ファイル名)
FILES_TO_PROCESS = [
    ('output_hashtags_all_parallel.csv', 'hashtags_2017.csv'),
    ('output_mentions_all_parallel.csv', 'mentions_2017.csv')
]

def filter_csv_by_year(input_file, output_file):
    print(f"Processing: {input_file} -> {output_file}")
    
    # 出力ファイルが既に存在する場合は削除（初期化）
    if os.path.exists(output_file):
        os.remove(output_file)

    chunk_size = 100000  # 10万行ずつ読み込む（メモリ節約）
    first_chunk = True
    total_saved = 0

    try:
        # ファイルを少しずつ読み込む
        with pd.read_csv(input_file, chunksize=chunk_size) as reader:
            for chunk in reader:
                # タイムスタンプが数値であることを保証（エラーデータ対策）
                chunk['timestamp'] = pd.to_numeric(chunk['timestamp'], errors='coerce')
                
                # 2017年のデータのみ抽出 (Start <= timestamp < End)
                df_2017 = chunk[
                    (chunk['timestamp'] >= START_2017) & 
                    (chunk['timestamp'] < START_2018)
                ]

                if not df_2017.empty:
                    # 最初のチャンクはヘッダー付きで書き込み,以降はヘッダーなしで追記
                    mode = 'w' if first_chunk else 'a'
                    header = first_chunk
                    
                    df_2017.to_csv(output_file, mode=mode, header=header, index=False)
                    
                    total_saved += len(df_2017)
                    first_chunk = False

        print(f"✅ 完了: {total_saved} 行を {output_file} に保存しました。\n")

    except FileNotFoundError:
        print(f"❌ エラー: ファイル {input_file} が見つかりません。\n")
    except Exception as e:
        print(f"❌ エラー: {e}\n")

if __name__ == "__main__":
    for input_f, output_f in FILES_TO_PROCESS:
        filter_csv_by_year(input_f, output_f)