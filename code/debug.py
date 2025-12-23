import pandas as pd
import numpy as np

# --- 定数定義 ---
INFLUENCERS_FILE = 'influencers.txt'

def run_debug():
    """
    Nanoカテゴリの分類が正しく行われているかを確認するためのデバッグ処理。
    """
    print("--- Nanoカテゴリ分類のデバッグを開始します ---")

    # --- Step 1: `influencers.txt` の読み込み ---
    try:
        # User Explorerページと同じロジックで読み込み
        df = pd.read_csv(INFLUENCERS_FILE, sep='\t')
        df.columns = ['Username', 'Category', 'followers', 'Followees', 'Posts']
        print(f"✅ ステップ1: '{INFLUENCERS_FILE}' の読み込みに成功しました。")
        print("最初の5行:")
        print(df.head())
        print("-" * 30)
    except FileNotFoundError:
        print(f"❌ エラー: '{INFLUENCERS_FILE}' が見つかりません。")
        return
    except Exception as e:
        print(f"❌ エラー: ファイル読み込み中に問題が発生しました: {e}")
        return

    # --- Step 2: 'followers' 列を数値に変換 ---
    print("ステップ2: 'followers' 列を数値に変換します...")
    # 'coerce'を指定することで、数値に変換できない値はNaN（Not a Number）になります
    df['followers_numeric'] = pd.to_numeric(df['followers'], errors='coerce')
    
    # 変換に失敗した行があるか確認
    failed_conversions = df[df['followers_numeric'].isna() & df['followers'].notna()]
    if not failed_conversions.empty:
        print(f"⚠️ 警告: {len(failed_conversions)}件のフォロワー数を数値に変換できませんでした。例:")
        print(failed_conversions.head())
    else:
        print("✅ 全てのフォロワー数を数値に正常に変換できました。")
    print("-" * 30)

    # --- Step 3: Nanoカテゴリの範囲（1,000-10,000）に該当するユーザーを抽出 ---
    print("ステップ3: フォロワー数が 1,000人以上 10,000人未満のユーザーを抽出します...")
    # dropna()で数値に変換できなかった行を除外
    potential_nanos = df.dropna(subset=['followers_numeric'])
    potential_nanos = potential_nanos[
        (potential_nanos['followers_numeric'] >= 1000) & 
        (potential_nanos['followers_numeric'] < 10000)
    ]
    
    if not potential_nanos.empty:
        print(f"✅ Nanoカテゴリに該当する可能性のあるユーザーが {len(potential_nanos)} 人見つかりました。")
        print("抽出されたユーザーの例:")
        print(potential_nanos.head())
    else:
        print("❌ Nanoカテゴリに該当するユーザーが見つかりませんでした。このステップで問題がある可能性があります。")
    print("-" * 30)

    # --- Step 4: `pd.cut` を使って実際に分類 ---
    print("ステップ4: `pd.cut` を使って全ユーザーを分類します...")
    bins = [1000, 10000, 100000, 1000000, float('inf')]
    labels = ['Nano', 'Micro', 'Macro', 'Mega']
    
    # 分類を適用
    df['influencer_type_debug'] = pd.cut(df['followers_numeric'], bins=bins, labels=labels, right=False)
    
    # 分類結果を確認
    classified_nanos = df[df['influencer_type_debug'] == 'Nano']
    
    if not classified_nanos.empty:
        print(f"✅ `pd.cut` によって {len(classified_nanos)} 人が 'Nano' に分類されました。")
        print("分類されたユーザーの例:")
        print(classified_nanos.head())
    else:
        print("❌ `pd.cut` で 'Nano' に分類されたユーザーがいませんでした。binやlabelの設定に問題がある可能性があります。")
    print("-" * 30)

    # --- Step 5: 最終的な比較と結論 ---
    print("ステップ5: 最終的な比較と結論")
    
    # ステップ3で見つかったはずのNanoユーザーのセット
    potential_set = set(potential_nanos['Username'])
    # ステップ4で実際に分類されたNanoユーザーのセット
    classified_set = set(classified_nanos['Username'])
    
    missing_users = potential_set - classified_set
    
    if not missing_users:
        print("✅ **結論**: 正常です。Nanoカテゴリに該当する全てのユーザーが正しく分類されています。")
    else:
        print(f"❌ **結論**: 問題が確認されました。{len(missing_users)} 人のユーザーがNanoに分類されるべきなのに、されていません。")
        print("分類から漏れたユーザーの例:", list(missing_users)[:5])

if __name__ == '__main__':
    run_debug()
