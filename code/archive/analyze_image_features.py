import pandas as pd
import numpy as np

# --- 設定 ---
# 読み込むCSVファイル名。GPU版またはCPU版のスクリプトで生成したファイル名を指定してください。
INPUT_FILE = 'image_object_features_gpu.csv'
INPUT_FILE_CPU_FALLBACK = 'image_object_features_cpu.csv'

# 表示する上位N件
TOP_N = 20

def main():
    print(f"--- 📊 画像オブジェクト分析レポート ---")
    
    # --- 1. CSVファイルの読み込み ---
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"読み込み成功: {INPUT_FILE}\n")
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {INPUT_FILE}")
        try:
            print(f"フォールバックを試行: {INPUT_FILE_CPU_FALLBACK}")
            df = pd.read_csv(INPUT_FILE_CPU_FALLBACK)
            print(f"読み込み成功: {INPUT_FILE_CPU_FALLBACK}\n")
        except FileNotFoundError:
            print(f"エラー: {INPUT_FILE} も {INPUT_FILE_CPU_FALLBACK} も見つかりませんでした。")
            print("ファイル名が正しいか,スクリプトが正しい場所で実行されているか確認してください。")
            return
    except Exception as e:
        print(f"CSVの読み込み中に予期せぬエラーが発生しました: {e}")
        return

    # --- 2. 全ユニークオブジェクトの集計 ---
    print("--- 1. 全ユニークオブジェクトの集計 ---")
    
    # 'all_objects_detected' 列 (e.g., "person,person,cat") を処理
    # NaN (検出ゼロ) を除外
    all_objects_series = df['all_objects_detected'].dropna()
    
    # コンマで分割し,全てのオブジェクトを単一のリスト（all_objects_list）に平坦化
    all_objects_list = []
    for item_list_str in all_objects_series:
        all_objects_list.extend(item_list_str.split(','))
        
    # 空文字列（もしあれば）を除去
    all_objects_list = [obj for obj in all_objects_list if obj]
    
    if not all_objects_list:
        print("エラー: 検出されたオブジェクトがCSV内に見つかりませんでした。")
        return

    # ユニークなオブジェクトのセットを作成
    unique_objects = sorted(list(set(all_objects_list)))
    
    print(f"✅ 合計 {len(unique_objects)} 種類のユニークなオブジェクトが検出されました。")
    print("\n[検出された全オブジェクト リスト]")
    # 10個ずつ区切って表示
    for i in range(0, len(unique_objects), 10):
        print("  " + ", ".join(unique_objects[i:i+10]))

    # --- 3. 全オブジェクトの出現頻度 ---
    print(f"\n\n--- 2. 全オブジェクトの出現頻度 Top {TOP_N} ---")
    print("(画像内に複数写っていても全てカウント)")
    
    all_objects_freq = pd.Series(all_objects_list).value_counts()
    print(all_objects_freq.head(TOP_N).to_string())

    # --- 4. 'first_object' (最優先オブジェクト) の出現頻度 ---
    print(f"\n\n--- 3. 'First Object' (最優先オブジェクト) の出現頻度 Top {TOP_N} ---")
    print("(各画像の「主要な」オブジェクトの傾向)")
    
    first_object_freq = df['first_object'].value_counts()
    print(first_object_freq.head(TOP_N).to_string())

    # --- 5. カテゴリ別の 'first_object' 出現頻度 ---
    print(f"\n\n--- 4. インフルエンサーカテゴリ別の Top 10 'First Object' ---")
    print("(カテゴリごとの主要な被写体の傾向)")
    
    # 'user_category' 列でグループ化
    try:
        grouped = df.groupby('user_category')
        
        for category, group_df in grouped:
            print(f"\n[🏠 カテゴリ: {category}]")
            cat_first_freq = group_df['first_object'].value_counts()
            
            if cat_first_freq.empty:
                print("  このカテゴリの画像には検出されたオブジェクトがありません。")
            else:
                # グラフが見やすいようにインデント
                print(cat_first_freq.head(10).to_string(header=False).replace('\n', '\n  '))
                
    except KeyError:
        print("\n'user_category' 列が見つかりませんでした。カテゴリ別分析をスキップします。")

    print("\n\n--- 分析完了 ---")

if __name__ == "__main__":
    main()