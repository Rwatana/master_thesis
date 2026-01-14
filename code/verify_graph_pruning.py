import pandas as pd
from collections import Counter, defaultdict
import os

# --- 1. 定数定義 ---
PREPROCESSED_FILE = 'preprocessed_posts_with_metadata.csv'
HASHTAGS_FILE = 'output_hashtags_all_parallel.csv'
MENTIONS_FILE = 'output_mentions_all_parallel.csv'
INFLUENCERS_FILE = 'influencers.txt'

def verify_pruning_process():
    """
    論文の枝刈り処理を実装し,各段階でのノード数とエッジ数を確認する。
    """
    print("--- 🔬 Starting Graph Pruning Verification Process ---")

    # --- 2. データ読み込み ---
    print("\n[Step 1/5] Loading data files...")
    df_hashtags = pd.read_csv(HASHTAGS_FILE, header=0, low_memory=False)
    df_mentions = pd.read_csv(MENTIONS_FILE, header=0, low_memory=False)
    with open(INFLUENCERS_FILE, 'r', encoding='utf-8') as f: lines = f.readlines()
    lines = [line for line in lines if '===' not in line]
    from io import StringIO
    df_influencers = pd.read_csv(StringIO("".join(lines)), sep='\t', dtype=str)

    # データ整形
    df_hashtags.rename(columns={'source': 'username', 'target': 'hashtag'}, inplace=True)
    df_mentions.rename(columns={'source': 'username', 'target': 'mention'}, inplace=True)
    
    # --- 3. 枝刈り前のノードとエッジを構築 ---
    print("\n[Step 2/5] Building initial graph structure...")
    
    # ノードセットの定義
    influencer_set = set(df_influencers['Username'].astype(str))
    
    # 論文に従い,インフルエンサーがソース元になっているエッジのみを対象とする
    hashtags_from_influencers = df_hashtags[df_hashtags['username'].isin(influencer_set)]
    mentions_from_influencers = df_mentions[df_mentions['username'].isin(influencer_set)]

    hashtag_node_set = set(hashtags_from_influencers['hashtag'].astype(str))
    mention_node_set = set(mentions_from_influencers['mention'].astype(str))
    other_user_node_set = mention_node_set - influencer_set # メンション先がインフルエンサーである場合を除く

    # エッジリストの作成
    initial_edges_ht = list(zip(hashtags_from_influencers['username'], hashtags_from_influencers['hashtag']))
    initial_edges_mt = list(zip(mentions_from_influencers['username'], mentions_from_influencers['mention']))
    initial_all_edges = initial_edges_ht + initial_edges_mt

    print("\n--- 📊 Initial Graph Stats (Before Pruning) ---")
    print(f"Influencer Nodes:    {len(influencer_set):,}")
    print(f"Hashtag Nodes:       {len(hashtag_node_set):,}")
    print(f"Other User Nodes:    {len(other_user_node_set):,}")
    print(f"Total Edges:         {len(initial_all_edges):,}")
    
    # --- 4. 論文の枝刈り処理を実装 ---
    
    # --- 4.1 ノードの枝刈り (接続数が1の補助ノードを削除) ---
    print("\n[Step 3/5] Pruning auxiliary nodes with degree = 1...")
    
    # 補助ノード（ハッシュタグとメンション先）の出現回数（次数）をカウント
    aux_nodes = [edge[1] for edge in initial_all_edges]
    node_degree = Counter(aux_nodes)
    
    # 接続数が2以上の補助ノードだけを保持
    nodes_to_keep = {node for node, degree in node_degree.items() if degree > 1}
    
    # 枝刈り後のエッジリストを作成
    edges_after_node_pruning = [edge for edge in initial_all_edges if edge[1] in nodes_to_keep]
    
    # 枝刈り後のノードセットを再計算
    pruned_influencer_set = {edge[0] for edge in edges_after_node_pruning}
    pruned_hashtag_set = {edge[1] for edge in edges_after_node_pruning if edge[1] in hashtag_node_set}
    pruned_mention_set = {edge[1] for edge in edges_after_node_pruning if edge[1] in mention_node_set}
    pruned_other_user_set = pruned_mention_set - pruned_influencer_set

    print("\n--- 📊 Stats After Node Pruning ---")
    print(f"Remaining Influencer Nodes: {len(pruned_influencer_set):,}")
    print(f"Remaining Hashtag Nodes:    {len(pruned_hashtag_set):,}")
    print(f"Remaining Other User Nodes: {len(pruned_other_user_set):,}")
    print(f"Remaining Edges:            {len(edges_after_node_pruning):,}")

    # --- 4.2 エッジの枝刈り (正規化頻度 < 0.01 のエッジを削除) ---
    print("\n[Step 4/5] Pruning edges with normalized frequency < 0.01...")
    
    # インフルエンサーごとの総エッジ数を計算
    influencer_ht_counts = defaultdict(int)
    influencer_mt_counts = defaultdict(int)
    
    # 現在のエッジリストから,ハッシュタグとメンションを再度分離
    current_edges_ht = [edge for edge in edges_after_node_pruning if edge[1] in hashtag_node_set]
    current_edges_mt = [edge for edge in edges_after_node_pruning if edge[1] in mention_node_set]
    
    for user, _ in current_edges_ht:
        influencer_ht_counts[user] += 1
    for user, _ in current_edges_mt:
        influencer_mt_counts[user] += 1
        
    # 各エッジの出現回数をカウント
    edge_weights_ht = Counter(current_edges_ht)
    edge_weights_mt = Counter(current_edges_mt)

    final_edges = []
    # ハッシュタグエッジのフィルタリング
    for edge, weight in edge_weights_ht.items():
        user = edge[0]
        normalized_freq = weight / influencer_ht_counts[user]
        if normalized_freq >= 0.01:
            final_edges.append(edge)
            
    # メンションエッジのフィルタリング
    for edge, weight in edge_weights_mt.items():
        user = edge[0]
        normalized_freq = weight / influencer_mt_counts[user]
        if normalized_freq >= 0.01:
            final_edges.append(edge)

    # --- 5. 最終結果の表示 ---
    print("\n[Step 5/5] Calculating final graph stats...")

    final_influencer_set = {edge[0] for edge in final_edges}
    final_aux_nodes = {edge[1] for edge in final_edges}
    final_hashtag_set = final_aux_nodes.intersection(hashtag_node_set)
    final_mention_set = final_aux_nodes.intersection(mention_node_set)
    final_other_user_set = final_mention_set - final_influencer_set

    print("\n" + "="*50)
    print("--- 🏆 Final Graph Stats (After All Pruning) ---")
    print("="*50)
    print(f"Influencer Nodes:    {len(final_influencer_set):,}")
    print(f"Hashtag Nodes:       {len(final_hashtag_set):,}")
    print(f"Other User Nodes:    {len(final_other_user_set):,}")
    print(f"Total Nodes:         {len(final_influencer_set | final_hashtag_set | final_other_user_set):,}")
    print(f"Total Edges:         {len(final_edges):,}")
    
    print("\n--- 📜 For Reference: Paper's Final Stats ---")
    print("Influencer Nodes:    18,397")
    print("Hashtag Nodes:       67,695")
    print("Other User Nodes:    20,744")
    print("Total Nodes:         107,832")
    print("Total Edges:         15,090,225 (across all networks)")


if __name__ == '__main__':
    verify_pruning_process()
