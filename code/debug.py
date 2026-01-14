import pandas as pd

# ファイル名の指定
file_path = 'dataset_A_active_all_updated.csv'

try:
    # CSVファイルを読み込む
    df = pd.read_csv(file_path)

    # --- 方法1: 表示制限を解除して表示する ---
    # 列の最大表示数を無制限(None)に設定
    pd.set_option('display.max_columns', None)
    # 1行の表示幅を広げる（カラムが多い場合に折り返されないようにする）
    pd.set_option('display.width', 1000)

    print("--- All Columns (First 5 Rows) ---")
    print(df.head())
    print("\n" + "="*50 + "\n")

    # --- 方法2: 転置して表示（おすすめ） ---
    # カラム数が多い場合,縦に並べた方が確認しやすいです
    print("--- Transposed View (First 5 Rows) ---")
    print(df.head().T)

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")