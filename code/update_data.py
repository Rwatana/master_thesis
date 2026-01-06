import pandas as pd
import io
import os

def update_and_save_new_csv():
    # ファイル名の定義
    input_csv = 'dataset_A_active_all.csv'
    input_txt = 'influencers.txt'
    output_csv = 'dataset_A_active_all_updated.csv'

    print(f"Checking files...")
    if not os.path.exists(input_csv) or not os.path.exists(input_txt):
        print("Error: Required files not found.")
        return

    try:
        # 1. influencers.txt の読み込み
        with open(input_txt, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # '====' の行を除外
        cleaned_lines = [line for line in lines if "=====" not in line]
        cleaned_content = "".join(cleaned_lines)

        # 読み込みの実行
        # on_bad_lines='skip' を追加して、形式が崩れている行を飛ばすようにします
        try:
            # まずはタブ区切りを試す（txt形式で最も一般的）
            df_info = pd.read_table(io.StringIO(cleaned_content), sep='\t')
            # もし1列しか読み込めなかったらスペース区切りで再試行
            if df_info.shape[1] < 2:
                df_info = pd.read_table(
                    io.StringIO(cleaned_content), 
                    sep=r'\s+', 
                    engine='python', 
                    on_bad_lines='skip'
                )
        except Exception:
            # 失敗した場合はエラー行をスキップしつつ読み込む
            df_info = pd.read_table(
                io.StringIO(cleaned_content), 
                sep=r'\s+', 
                engine='python', 
                on_bad_lines='skip'
            )
        
        # カラム名のクリーニング
        # カラム名の前後の空白を削除
        df_info.columns = [c.strip() for c in df_info.columns]
        
        # 必要なカラムへのリネーム
        df_info = df_info.rename(columns={
            'Username': 'username',
            '#Followers': 'new_followers',
            '#Followees': 'new_following'
        })

        # 2. dataset_A_active_all.csv の読み込み
        df_a = pd.read_csv(input_csv)

        # 3. データの結合 (Merge)
        # usernameをキーにして結合
        df_merged = pd.merge(
            df_a, 
            df_info[['username', 'new_followers', 'new_following']], 
            on='username', 
            how='left'
        )

        # 4. 0だった列を更新
        # text側にデータがあった行のみ更新し、それ以外は元の値を保持
        df_merged['user_followers'] = df_merged['new_followers'].fillna(df_merged['user_followers'])
        df_merged['user_following'] = df_merged['new_following'].fillna(df_merged['user_following'])

        # 一時的なカラムを削除
        df_final = df_merged.drop(columns=['new_followers', 'new_following'])

        # 整数型に変換
        df_final['user_followers'] = df_final['user_followers'].fillna(0).astype(int)
        df_final['user_following'] = df_final['user_following'].fillna(0).astype(int)

        # 5. 新しいCSVとして保存
        df_final.to_csv(output_csv, index=False)
        
        print(f"--- Process Completed ---")
        print(f"New file saved as: {output_csv}")
        print(f"Total rows in TXT (including skipped): {len(cleaned_lines)}")
        print(f"Rows successfully loaded into dataframe: {len(df_info)}")
        
        # 更新後の確認表示
        print("\n[Verification: First 5 rows of updated columns]")
        print(df_final[['username', 'user_followers', 'user_following']].head())

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    update_and_save_new_csv()