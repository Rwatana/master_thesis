import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def plot_edge_importance(csv_path):
    # 1. データの読み込み
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print(f"Skipping empty file: {csv_path}")
        return

    # 2. Importanceの正規化 (0~1スケール)
    # すべて同じ値の場合の除算エラーを避けるためチェック
    min_val = df['Importance'].min()
    max_val = df['Importance'].max()
    if max_val - min_val > 0:
        df['Importance_scaled'] = (df['Importance'] - min_val) / (max_val - min_val)
    else:
        df['Importance_scaled'] = 1.0 # すべて同じ値なら1にする

    # 3. ラベルの作成 (Neighbor名とEdgeIdxのみ)
    # 例: "#CES (e1)"
    df['label'] = df.apply(lambda x: f"{x['Neighbor']} (e{int(x['EdgeIdx'])})", axis=1)

    # 4. プロット用にソート (上位20件を表示,Importanceが高い順に上にくるよう調整)
    df_sorted = df.sort_values('Importance', ascending=True).tail(20)

    # 5. 可視化
    plt.figure(figsize=(10, 8))
    plt.barh(df_sorted['label'], df_sorted['Importance_scaled'], color='#1f77b4')
    
    # タイトルとラベルの設定
    # pos_suffix = csv_path.split('_')[-1].replace('.csv', '') # pos_1 などの識別子を取得
    plt.title(f'Top Edges Importance', fontsize=14)
    plt.xlabel('Importance (Min-Max Scaled)', fontsize=12)
    plt.ylabel('Neighbor Nodes', fontsize=12)
    
    plt.tight_layout()
    
    # 保存と表示
    output_name = csv_path.replace('edges_all_', 'edge_processed_').replace('.csv', '.png')
    plt.savefig(output_name, dpi=300)
    print(f"Saved: {output_name}")
    plt.close()

def main():
    # ディレクトリ内の該当するCSVファイルをすべて取得
    csv_files = glob.glob("node_*_edges_all_pos_*.csv")
    
    if not csv_files:
        print("対象のCSVファイルが見つかりませんでした。パスを確認してください。")
        return

    for file in csv_files:
        plot_edge_importance(file)

if __name__ == "__main__":
    main()

# import pandas as pd
# import matplotlib.pyplot as plt
# import glob
# import os

# def plot_feature_importance(csv_path):
#     # 1. データの読み込み
#     try:
#         df = pd.read_csv(csv_path)
#     except Exception as e:
#         print(f"Error reading {csv_path}: {e}")
#         return
    
#     if df.empty:
#         print(f"Skipping empty file: {csv_path}")
#         return

#     # 2. Importanceの正規化 (0~1スケール)
#     # 最小値を0,最大値を1にするMin-Max Scaling
#     min_val = df['Importance'].min()
#     max_val = df['Importance'].max()
    
#     if max_val - min_val > 0:
#         df['Importance_scaled'] = (df['Importance'] - min_val) / (max_val - min_val)
#     else:
#         # すべての値が同じ場合は便宜上すべて1.0にする
#         df['Importance_scaled'] = 1.0

#     # 3. プロット用にソート (上位20件)
#     # グラフでは上から順に表示したいため,ascending=Trueでソートしてtail(20)をとる
#     df_sorted = df.sort_values('Importance', ascending=True).tail(20)

#     # 4. 可視化
#     plt.figure(figsize=(10, 8))
#     # 特徴量名は 'Name' カラムにあることを想定
#     plt.barh(df_sorted['Name'], df_sorted['Importance_scaled'], color='#2ca02c') # Node用は緑系
    
#     # タイトルとラベルの設定
#     # pos_suffix = csv_path.split('_')[-1].replace('.csv', '')
#     plt.title(f'Top Node Features Importance', fontsize=14)
#     plt.xlabel('Importance (Min-Max Scaled)', fontsize=12)
#     plt.ylabel('Feature Names', fontsize=12)
    
#     plt.grid(axis='x', linestyle='--', alpha=0.7)
#     plt.tight_layout()
    
#     # 5. 保存
#     output_name = csv_path.replace('features_all_', 'feat_processed_').replace('.csv', '.png')
#     plt.savefig(output_name, dpi=300)
#     print(f"Saved: {output_name}")
#     plt.close()

# def main():
#     # ディレクトリ内の node_*_features_all_pos_*.csv をすべて取得
#     csv_files = glob.glob("node_*_features_all_pos_*.csv")
    
#     if not csv_files:
#         print("対象のNode Feature CSVファイルが見つかりませんでした。")
#         return

#     for file in csv_files:
#         plot_feature_importance(file)

# if __name__ == "__main__":
#     main()