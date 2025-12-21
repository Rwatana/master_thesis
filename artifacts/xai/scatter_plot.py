import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def plot_scatter_importance_impact(csv_path, is_edge=True):
    # 1. データの読み込み
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return
    
    if df.empty:
        return

    # 2. 重複排除（重要度順に並べて最初のみ保持）
    target_col = 'Neighbor' if is_edge else 'Name'
    df = df.sort_values('Importance', ascending=False)
    df = df.drop_duplicates(subset=[target_col], keep='first')

    # 3. Importanceの正規化 (0~1スケール)
    min_val = df['Importance'].min()
    max_val = df['Importance'].max()
    if max_val - min_val > 0:
        df['Importance_scaled'] = (df['Importance'] - min_val) / (max_val - min_val)
    else:
        df['Importance_scaled'] = 1.0

    # 4. スコアインパクトの取得
    # CSVのカラム名 'Score_Impact(unmasked)' を使用
    y_col = 'Score_Impact(unmasked)'

    # 5. 可視化
    plt.figure(figsize=(10, 7))
    color = '#1f77b4' if is_edge else '#2ca02c'
    
    # 散布図のプロット
    plt.scatter(df['Importance_scaled'], df[y_col], alpha=0.6, c=color, edgecolors='w', s=100)

    # 6. 上位5件にラベルを付ける（識別のため）
    top_n = df.head(5)
    for i, row in top_n.iterrows():
        label = f"{row[target_col]}"
        if is_edge:
            label += f" (e{int(row['EdgeIdx'])})"
        
        plt.annotate(label, 
                     (row['Importance_scaled'], row[y_col]),
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center', fontsize=9, alpha=0.8)

    # タイトルとラベル
    type_str = "Edges" if is_edge else "Node Features"
    pos_suffix = csv_path.split('_')[-1].replace('.csv', '')
    plt.title(f'Importance vs Score Impact - {type_str} ({pos_suffix})', fontsize=14)
    plt.xlabel('Importance (Min-Max Scaled 0-1)', fontsize=12)
    plt.ylabel('Score Impact (Ablation)', fontsize=12)
    
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3) # 補助線
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    # 7. 保存
    prefix = 'scatter_edge_' if is_edge else 'scatter_feat_'
    output_name = csv_path.replace('edges_all_', prefix).replace('features_all_', prefix).replace('.csv', '.png')
    plt.savefig(output_name, dpi=300)
    print(f"Saved Scatter Plot: {output_name}")
    plt.close()

def main():
    # エッジとノード特徴量の両方を処理
    edge_files = glob.glob("node_*_edges_all_pos_*.csv")
    feat_files = glob.glob("node_*_features_all_pos_*.csv")

    for f in edge_files:
        plot_scatter_importance_impact(f, is_edge=True)
    
    for f in feat_files:
        plot_scatter_importance_impact(f, is_edge=False)

if __name__ == "__main__":
    main()
