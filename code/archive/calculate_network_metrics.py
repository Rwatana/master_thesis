import pandas as pd
import os
import networkx as nx
from tqdm import tqdm
import concurrent.futures
import time
import multiprocessing

# --- 定数定義 ---
MENTIONS_FILE = 'output_mentions_all_parallel.csv'
OUTPUT_CSV_FILE = 'network_centrality_over_time.csv'

# グローバル変数としてメンションデータを定義
df_mentions_global = None

def process_monthly_snapshot(snapshot_date):
    """
    単一の月のスナップショットを処理し,中心性指標とランキングを計算するワーカー関数。
    """
    try:
        # その時点までの全メンションデータでネットワークを構築
        current_mentions = df_mentions_global[df_mentions_global['datetime'] <= snapshot_date]
        if current_mentions.empty:
            return None

        G = nx.from_pandas_edgelist(current_mentions, 'username', 'mention', create_using=nx.DiGraph())
        if G.number_of_nodes() == 0:
            return None
            
        # 中心性指標を計算
        in_degree = nx.in_degree_centrality(G)
        pagerank = nx.pagerank(G, alpha=0.85)
        
        # 結果をDataFrameにまとめる
        df_results = pd.DataFrame({
            'username': list(G.nodes()),
            'in_degree': [in_degree.get(n, 0) for n in G.nodes()],
            'pagerank': [pagerank.get(n, 0) for n in G.nodes()]
        })
        
        # 月ごとのランキングを計算
        df_results['in_degree_rank'] = df_results['in_degree'].rank(method='min', ascending=False).astype(int)
        df_results['pagerank_rank'] = df_results['pagerank'].rank(method='min', ascending=False).astype(int)
        df_results['month'] = snapshot_date
        
        return df_results
        
    except Exception as e:
        print(f"Error processing {snapshot_date}: {e}")
        return None

def main():
    """メインの処理を実行する関数"""
    global df_mentions_global

    print("--- ネットワーク中心性の時系列計算を開始します ---")

    # 1. メンションデータの読み込み
    try:
        print(f"'{MENTIONS_FILE}'を読み込んでいます...")
        df_mentions = pd.read_csv(MENTIONS_FILE, header=0)
        df_mentions.rename(columns={'source': 'username', 'target': 'mention'}, inplace=True)
        df_mentions['datetime'] = pd.to_datetime(df_mentions['timestamp'], unit='s')
        df_mentions_global = df_mentions.drop(columns=['timestamp'])
    except FileNotFoundError:
        print(f"エラー: '{MENTIONS_FILE}' が見つかりません。")
        return
        
    # 2. 月ごとのスナップショット日付を生成
    start_date = df_mentions_global['datetime'].min()
    end_date = df_mentions_global['datetime'].max()
    monthly_snapshots = pd.date_range(start_date, end_date, freq='M')
    
    print(f"{len(monthly_snapshots)}ヶ月分のスナップショットを処理します...")
    num_processes = os.cpu_count()
    print(f"並列処理のために {num_processes} 個のプロセスを開始します...")

    # 3. 並列処理で各月の中心性を計算
    all_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        results_iterator = executor.map(process_monthly_snapshot, monthly_snapshots)
        for result_df in tqdm(results_iterator, total=len(monthly_snapshots), desc="月次スナップショットを処理中"):
            if result_df is not None:
                all_results.append(result_df)
    
    # 4. 全ての結果を結合して保存
    if not all_results:
        print("計算結果がありません。処理を終了します。")
        return
        
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(OUTPUT_CSV_FILE, index=False)
    
    print("\n--- 全ての計算が完了しました ---")
    print(f"✅ {len(final_df)} 件の中心性データを '{OUTPUT_CSV_FILE}' に保存しました。")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"合計処理時間: {end_time - start_time:.2f} 秒")
