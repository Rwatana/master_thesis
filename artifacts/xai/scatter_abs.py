import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def plot_scatter_abs_fixed_origin(csv_path, is_edge=True):
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

    # 3. Importanceの個別正規化 (0~1スケール)
    # 「スケールは変えなくて良い（＝全体で統一しなくて良い）」という意図に基づきファイル毎に正規化
    min_val = df['Importance'].min()
    max_val = df['Importance'].max()
    if max_val - min_val > 0:
        df['Importance_scaled'] = (df['Importance'] - min_val) / (max_val - min_val)
    else:
        df['Importance_scaled'] = 1.0

    # 4. スコアインパクトの絶対値化
    # 予測を下げる影響（負の値）も寄与度として絶対値で評価
    y_col = 'Impact_abs'
    df[y_col] = df['Score_Impact(unmasked)'].abs()

    # 5. 可視化
    plt.figure(figsize=(10, 7))
    color = '#1f77b4' if is_edge else '#2ca02c'
    
    # 散布図のプロット
    plt.scatter(df['Importance_scaled'], df[y_col], alpha=0.6, c=color, edgecolors='w', s=100)

    # 6. 上位5件にラベルを付ける
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

    # 7. 軸の設定（0の位置を固定）
    plt.xlim(-0.05, 1.05) # X軸(Importance)は0〜1で固定
    
    # 縦軸の0位置を固定。上限は各データの最大値に合わせる
    current_max_impact = df[y_col].max()
    plt.ylim(0, current_max_impact * 1.1 if current_max_impact > 0 else 1.0)

    # タイトルとラベルの設定
    type_str = "Edges" if is_edge else "Node Features"
    pos_suffix = csv_path.split('_')[-1].replace('.csv', '')
    plt.title(f'Importance vs Abs Score Impact - {type_str} ({pos_suffix})', fontsize=14)
    plt.xlabel('Importance (Min-Max Scaled 0-1)', fontsize=12)
    plt.ylabel('Score Impact (Absolute Magnitude)', fontsize=12)
    
    # グリッドの追加（0の位置がわかりやすいよう y=0 に線を強調）
    plt.axhline(0, color='black', linewidth=1.2, alpha=0.5)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    # 8. 保存
    prefix = 'scatter_abs_edge_' if is_edge else 'scatter_abs_feat_'
    output_name = csv_path.replace('edges_all_', prefix).replace('features_all_', prefix).replace('.csv', '.png')
    plt.savefig(output_name, dpi=300)
    print(f"Saved: {output_name}")
    plt.close()

def main():
    # エッジとノード特徴量の両方を処理
    edge_files = glob.glob("node_*_edges_all_pos_*.csv")
    feat_files = glob.glob("node_*_features_all_pos_*.csv")

    for f in edge_files:
        plot_scatter_abs_fixed_origin(f, is_edge=True)
    
    for f in feat_files:
        plot_scatter_abs_fixed_origin(f, is_edge=False)

if __name__ == "__main__":
    main()