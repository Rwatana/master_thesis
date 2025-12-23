import pandas as pd

# ファイルのパスを指定してください
file_path = 'preprocessed_posts_with_metadata_data_check_clean.csv'

try:
    print(f"'{file_path}' を読み込んでいます...")
    # datetimeカラムを日付として解釈するように parse_dates を追加
    df = pd.read_csv(file_path, low_memory=False, parse_dates=['datetime'])
    
    # データフレームの総レコード数を取得
    record_count = len(df)
    
    print("\n--- 全体概要 ---")
    print(f"✅ 総レコード数（行数）: {record_count:,}")
    # print("\nデータの先頭5行:")
    # print(df.head())
    # print("\nカラム情報:")
    # print(df.info())
    print("----------------")

    # 'datetime'カラムが存在するか確認
    if 'datetime' in df.columns:
        # 年ごとに投稿数を集計
        yearly_counts = df['datetime'].dt.year.value_counts().sort_index()
        
        print("\n--- 🗓️ 年ごとの投稿数 ---")
        print(yearly_counts.to_string())
        print("--------------------------")
    else:
        print("\n警告: 'datetime' カラムが見つかりませんでした。年ごとの集計はスキップします。")


except FileNotFoundError:
    print(f"エラー: ファイル '{file_path}' が見つかりませんでした。パスを確認してください。")
except Exception as e:
    print(f"エラーが発生しました: {e}")