import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def process_and_plot(csv_path, is_edge=True):
    # 1. データの読み込み
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return
    
    if df.empty:
        print(f"Skipping empty file: {csv_path}")
        return

    # ターゲットカラムの設定
    # エッジなら Neighbor, ノードなら Name を重複チェックの基準にする
    target_col = 'Neighbor' if is_edge else 'Name'
    
    # 2. 重複排除ロジック
    # 重要度が高い順に並び替え、同じ名前のものは最初（最大値）だけ残す
    df = df.sort_values('Importance', ascending=False)
    df = df.drop_duplicates(subset=[target_col], keep='first')

    # 3. Importanceの正規化 (重複排除後のデータで 0~1スケール)
    min_val = df['Importance'].min()
    max_val = df['Importance'].max()
    
    if max_val - min_val > 0:
        df['Importance_scaled'] = (df['Importance'] - min_val) / (max_val - min_val)
    else:
        df['Importance_scaled'] = 1.0

    # 4. ラベルの作成 (エッジの場合は IDを付与)
    if is_edge:
        df['label'] = df.apply(lambda x: f"{x['Neighbor']} (e{int(x['EdgeIdx'])})", axis=1)
    else:
        df['label'] = df['Name']

    # 5. プロット用に上位20件を抽出（昇順に並び替えて横棒グラフの下から描画）
    df_plot = df.head(20).sort_values('Importance_scaled', ascending=True)

    # 6. 可視化
    plt.figure(figsize=(10, 8))
    color = '#1f77b4' if is_edge else '#2ca02c'
    plt.barh(df_plot['label'], df_plot['Importance_scaled'], color=color)
    
    # タイトルとラベル
    type_str = "Edges" if is_edge else "Node Features"
    # pos_suffix = csv_path.split('_')[-1].replace('.csv', '')
    plt.title(f'Top {type_str} Importance', fontsize=14)
    plt.xlabel('Importance (Min-Max Scaled)', fontsize=12)
    plt.ylabel(target_col, fontsize=12)
    
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 7. 保存
    prefix = 'edge_unique_' if is_edge else 'feat_unique_'
    output_name = csv_path.replace('edges_all_', prefix).replace('features_all_', prefix).replace('.csv', '.png')
    plt.savefig(output_name, dpi=300)
    print(f"Saved: {output_name}")
    plt.close()

def main():
    # エッジファイルの処理
    edge_files = glob.glob("node_*_edges_all_pos_*.csv")
    for f in edge_files:
        process_and_plot(f, is_edge=True)
        
    # ノード特徴量ファイルの処理
    feat_files = glob.glob("node_*_features_all_pos_*.csv")
    for f in feat_files:
        process_and_plot(f, is_edge=False)

if __name__ == "__main__":
    main()