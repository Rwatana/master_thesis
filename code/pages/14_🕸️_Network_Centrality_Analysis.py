# import streamlit as st
# import pandas as pd
# import networkx as nx
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from datetime import timedelta

# st.set_page_config(page_title="ネットワーク影響力分析", layout="wide")

# # --- データ読み込み関数 ---
# @st.cache_data
# def load_posts_data(filepath):
#     """投稿データを読み込み、エンゲージメントを計算する"""
#     try:
#         df = pd.read_csv(filepath, parse_dates=['datetime'])
#         df['engagement'] = df['likes'] + df['comments']
#         return df
#     except FileNotFoundError:
#         st.error(f"ファイル '{filepath}' が見つかりません。")
#         return None

# @st.cache_data
# def load_mention_data(filepath):
#     """メンションデータを読み込み、タイムスタンプをdatetimeに変換する"""
#     try:
#         df = pd.read_csv(filepath, header=0, names=['username', 'mention', 'timestamp'])
#         df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
#         return df.drop(columns=['timestamp'])
#     except FileNotFoundError:
#         st.error(f"ファイル '{filepath}' が見つかりません。")
#         return None

# # --- ネットワーク中心性および統計量計算関数（移動窓方式）---
# @st.cache_data
# def calculate_network_metrics_over_time(df_mentions, window_days):
#     """時系列でネットワーク中心性と基本統計量を【移動窓】で計算する"""
#     if df_mentions.empty:
#         return pd.DataFrame()
    
#     df_sorted = df_mentions.sort_values('datetime')
#     start_date, end_date = df_sorted['datetime'].min(), df_sorted['datetime'].max()
    
#     results = []
#     # 2週間ごと（freq='2W'）にスナップショットを作成
#     for snapshot_date in pd.date_range(start_date, end_date, freq='2W'):
#         window_start_date = snapshot_date - timedelta(days=window_days)
        
#         current_mentions = df_sorted[
#             (df_sorted['datetime'] > window_start_date) & 
#             (df_sorted['datetime'] <= snapshot_date)
#         ]
        
#         if current_mentions.empty:
#             continue

#         G = nx.from_pandas_edgelist(current_mentions, 'username', 'mention', create_using=nx.DiGraph())
        
#         if G.number_of_nodes() == 0:
#             continue
        
#         # --- 複数の中心性指標を計算 ---
#         in_degree = nx.in_degree_centrality(G)
#         out_degree = nx.out_degree_centrality(G) # <<< [追加] Out-Degree
#         pagerank = nx.pagerank(G, alpha=0.85)
#         betweenness = nx.betweenness_centrality(G) # <<< [追加] Betweenness
        
#         # ネットワーク全体の統計情報を取得
#         num_nodes = G.number_of_nodes()
#         num_edges = G.number_of_edges()
        
#         for user in G.nodes():
#             results.append({
#                 'datetime': snapshot_date,
#                 'username': user,
#                 'in_degree': in_degree.get(user, 0),
#                 'out_degree': out_degree.get(user, 0), # <<< [追加]
#                 'pagerank': pagerank.get(user, 0),
#                 'betweenness': betweenness.get(user, 0), # <<< [追加]
#                 'num_nodes': num_nodes,
#                 'num_edges': num_edges
#             })
            
#     return pd.DataFrame(results)

# # --- 特定時点のネットワーク図を描画する関数 ---
# @st.cache_data
# def create_network_snapshot_figure(df_mentions, snapshot_date, window_days, top_n=30):
#     """指定された日時のネットワークスナップショットをPlotlyで描画する"""
#     window_start_date = snapshot_date - timedelta(days=window_days)
#     snapshot_mentions = df_mentions[
#         (df_mentions['datetime'] > window_start_date) & 
#         (df_mentions['datetime'] <= snapshot_date)
#     ]

#     if snapshot_mentions.empty:
#         return go.Figure(), pd.DataFrame()

#     G = nx.from_pandas_edgelist(snapshot_mentions, 'username', 'mention', create_using=nx.DiGraph())

#     # 可視化のため、中心性が高い上位Nノードに絞る
#     if G.number_of_nodes() > top_n:
#         top_nodes_dict = dict(sorted(nx.in_degree_centrality(G).items(), key=lambda item: item[1], reverse=True)[:top_n])
#         top_nodes = list(top_nodes_dict.keys())
#         G = G.subgraph(top_nodes)

#     if G.number_of_nodes() == 0:
#         return go.Figure(), pd.DataFrame()
        
#     pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
#     # ノードの作成
#     node_x, node_y, node_text, node_size = [], [], [], []
#     in_degrees = dict(G.in_degree())
#     for node in G.nodes():
#         x, y = pos[node]
#         node_x.append(x)
#         node_y.append(y)
#         degree = in_degrees.get(node, 0)
#         node_text.append(f"{node}<br>In-Degree: {degree}")
#         node_size.append(10 + degree * 5)

#     node_trace = go.Scatter(
#         x=node_x, y=node_y,
#         mode='markers+text',
#         text=[node for node in G.nodes()],
#         textposition="top center",
#         hoverinfo='text',
#         hovertext=node_text,
#         marker=dict(
#             showscale=True,
#             colorscale='YlGnBu',
#             size=node_size,
#             color=[in_degrees.get(node, 0) for node in G.nodes()],
#             colorbar=dict(
#                 thickness=15,
#                 xanchor='left',
#                 title=dict(
#                     text='In-Degree (被メンション数)',
#                     side='right'
#                 )
#             )
#         ))

#     # エッジの作成
#     edge_x, edge_y = [], []
#     for edge in G.edges():
#         x0, y0 = pos[edge[0]]
#         x1, y1 = pos[edge[1]]
#         edge_x.extend([x0, x1, None])
#         edge_y.extend([y0, y1, None])

#     edge_trace = go.Scatter(
#         x=edge_x, y=edge_y,
#         line=dict(width=0.5, color='#888'),
#         hoverinfo='none',
#         mode='lines')

#     fig = go.Figure(data=[edge_trace, node_trace],
#              layout=go.Layout(
#                 showlegend=False,
#                 hovermode='closest',
#                 margin=dict(b=0,l=0,r=0,t=40),
#                 xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#                 yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
#             )
    
#     # トップユーザーのデータフレームを作成
#     top_users_df = pd.DataFrame(
#         sorted(in_degrees.items(), key=lambda x: x[1], reverse=True), 
#         columns=['username', 'in_degree_count']
#     ).head(10)

#     return fig, top_users_df


# # --- UI描画 ---
# st.title("🕸️ ネットワーク影響力とエンゲージメントの比較分析")
# st.info("エンゲージメントの推移と、指定した期間（移動窓）内での短期的なネットワーク影響力の推移を比較します。")

# # --- データの読み込みと準備 ---
# df_posts = load_posts_data('preprocessed_posts_with_metadata.csv')
# df_mentions = load_mention_data('output_mentions_all_parallel.csv')

# if df_posts is None or df_mentions is None:
#     st.stop()
    
# # --- サイドバー ---
# st.sidebar.header("分析対象の選択")
# common_users = sorted(list(set(df_posts['username'].unique()) & set(df_mentions['username'].unique())))
# selected_user = st.sidebar.selectbox("分析したいユーザーを選択:", options=common_users)

# st.sidebar.subheader("グラフ設定")
# rolling_window = st.sidebar.slider("エンゲージメント移動平均の期間（日）:", 1, 60, 30, key="rolling_window")
# centrality_window = st.sidebar.slider("中心性計算の移動窓（日数）:", 7, 180, 60, key="centrality_window")

# # --- 中心性の計算 ---
# with st.spinner(f"ネットワーク中心性を移動窓 {centrality_window} 日で計算中..."):
#     df_metrics = calculate_network_metrics_over_time(df_mentions, centrality_window)

# if df_metrics.empty:
#     st.warning("中心性データを計算できませんでした。")
#     st.stop()
    
# # --- メイン画面 ---
# st.header(f"📈 {selected_user} の分析結果")

# # --- 比較グラフの作成 ---
# user_posts = df_posts[df_posts['username'] == selected_user].set_index('datetime')
# user_centrality = df_metrics[df_metrics['username'] == selected_user].set_index('datetime')
# user_posts_smooth = user_posts[['engagement']].rolling(window=f'{rolling_window}D').mean().dropna()

# fig_comp = make_subplots(specs=[[{"secondary_y": True}]])
# # Y1軸：エンゲージメント
# fig_comp.add_trace(go.Scatter(x=user_posts_smooth.index, y=user_posts_smooth['engagement'], name=f"エンゲージメント({rolling_window}日移動平均)", line=dict(color='royalblue')), secondary_y=False)

# # Y2軸：中心性指標
# fig_comp.add_trace(go.Scatter(x=user_centrality.index, y=user_centrality['in_degree'], name=f"In-Degree (注目度)", line=dict(color='firebrick')), secondary_y=True)
# fig_comp.add_trace(go.Scatter(x=user_centrality.index, y=user_centrality['pagerank'], name=f"PageRank (影響度)", line=dict(color='green', dash='dash')), secondary_y=True)
# fig_comp.add_trace(go.Scatter(x=user_centrality.index, y=user_centrality['betweenness'], name=f"Betweenness (媒介度)", line=dict(color='purple', dash='dot')), secondary_y=True)
# fig_comp.add_trace(go.Scatter(x=user_centrality.index, y=user_centrality['out_degree'], name=f"Out-Degree (発信度)", line=dict(color='orange', dash='dashdot')), secondary_y=True)

# fig_comp.update_layout(title_text=f"エンゲージメント vs ネットワーク中心性（{centrality_window}日移動窓）", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
# fig_comp.update_xaxes(title_text="日付")
# fig_comp.update_yaxes(title_text="エンゲージメント数", secondary_y=False)
# fig_comp.update_yaxes(title_text="中心性スコア", secondary_y=True, range=[0, user_centrality[['in_degree', 'pagerank', 'betweenness', 'out_degree']].max().max() * 1.1]) # Y2軸の範囲を調整

# st.plotly_chart(fig_comp, use_container_width=True)

# st.markdown("---") # 区切り線

# # --- ネットワーク全体の基本指標の推移 ---
# st.header("🌐 ネットワーク全体の基本指標の推移")
# st.info(f"各時点（2週間ごと）で、{centrality_window}日間の移動窓内に存在した総ユーザー数と総メンション数を示します。")

# # 指標プロット用にデータを整形
# df_network_stats = df_metrics[['datetime', 'num_nodes', 'num_edges']].drop_duplicates().set_index('datetime')

# if not df_network_stats.empty:
#     st.line_chart(df_network_stats)
# else:
#     st.warning("ネットワーク統計データを表示できませんでした。")

# st.markdown("---") # 区切り線

# # --- ネットワーク構造の時点分析（スナップショット） ---
# st.header("🔬 ネットワーク構造の時点分析（スナップショット）")
# st.info("上のグラフの日付を選んで、その時点のネットワーク構造を確認できます。")

# # 分析時点を選択
# snapshot_dates = sorted(df_metrics['datetime'].unique())
# if snapshot_dates:
#     selected_date = st.select_slider(
#         "分析したい時点を選択してください:",
#         options=snapshot_dates,
#         format_func=lambda date: pd.to_datetime(date).strftime('%Y-%m-%d')
#     )

#     col1, col2 = st.columns([3, 1])

#     with col1:
#         top_n_nodes = st.slider("表示する上位ユーザー数:", 10, 100, 30, key="top_n")
#         with st.spinner("ネットワーク図を作成中..."):
#             fig_net, df_top_users = create_network_snapshot_figure(df_mentions, selected_date, centrality_window, top_n_nodes)
#             st.plotly_chart(fig_net, use_container_width=True)
            
#     with col2:
#         st.subheader(f"🏆 {pd.to_datetime(selected_date).strftime('%Y-%m-%d')} 時点の上位ユーザー")
#         st.dataframe(df_top_users, use_container_width=True, hide_index=True)
# else:
#     st.warning("分析可能なスナップショットがありません。")

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="ネットワーク影響力分析", layout="wide")

# --- データ読み込み関数 ---
@st.cache_data
def load_centrality_data(filepath):
    """事前に計算された中心性データを読み込む"""
    try:
        return pd.read_csv(filepath, parse_dates=['month'])
    except FileNotFoundError:
        st.error(f"ファイル '{filepath}' が見つかりません。先に `calculate_network_metrics.py` を実行してください。")
        return None

def load_user_post_data(username):
    """ユーザーごとに分割された投稿データを読み込む"""
    filepath = f"user_data/{username}.csv"
    try:
        return pd.read_csv(filepath, parse_dates=['datetime'])
    except FileNotFoundError:
        st.warning(f"ユーザー '{username}' の個別データファイルが見つかりませんでした。")
        return None

# --- UI描画 ---
st.title("🕸️ ネットワーク影響力の時系列分析")
st.info("個人のいいね数推移と、ネットワーク全体での影響力（中心性）の推移を比較し、成長の関連性を探ります。")

# --- データの読み込み ---
df_centrality = load_centrality_data('network_centrality_over_time.csv')
if df_centrality is None:
    st.stop()

# --- サイドバー ---
st.sidebar.header("分析対象の選択")
user_list = sorted(df_centrality['username'].unique())
selected_user = st.sidebar.selectbox("1. 分析したいユーザーを選択:", options=user_list)

if 'run_network_analysis' not in st.session_state:
    st.session_state.run_network_analysis = False
if 'analyzed_user_network' not in st.session_state:
    st.session_state.analyzed_user_network = ""

if st.sidebar.button("分析を開始"):
    st.session_state.run_network_analysis = True
    st.session_state.analyzed_user_network = selected_user
elif selected_user != st.session_state.analyzed_user_network:
    st.session_state.run_network_analysis = False

# --- メイン画面 ---
if st.session_state.run_network_analysis:
    user = st.session_state.analyzed_user_network
    
    with st.spinner(f"'{user}'の投稿データと中心性データを読み込み中..."):
        df_user_posts = load_user_post_data(user)
        df_user_centrality = df_centrality[df_centrality['username'] == user].copy()
    
    st.header(f"📈 {user} の分析結果")

    # --- 比較グラフの作成 ---
    if df_user_posts is not None and not df_user_centrality.empty:
        # いいね数の月次平均を計算
        monthly_likes = df_user_posts.set_index('datetime')['likes'].resample('M').mean()
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                          subplot_titles=("エンゲージメント vs 中心性スコア", "中心性ランキングの推移"))

        # 上段グラフ: いいね数 vs 中心性スコア
        fig.add_trace(go.Scatter(x=monthly_likes.index, y=monthly_likes, name='月間平均いいね数', line=dict(color='royalblue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_user_centrality['month'], y=df_user_centrality['pagerank'], name='PageRank', line=dict(color='green', dash='dash'), yaxis='y2'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_user_centrality['month'], y=df_user_centrality['in_degree'], name='In-Degree', line=dict(color='firebrick', dash='dot'), yaxis='y2'), row=1, col=1)
        
        # 下段グラフ: ランキングの推移
        fig.add_trace(go.Scatter(x=df_user_centrality['month'], y=df_user_centrality['pagerank_rank'], name='PageRank順位', line=dict(color='green', dash='dash')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_user_centrality['month'], y=df_user_centrality['in_degree_rank'], name='In-Degree順位', line=dict(color='firebrick', dash='dot')), row=2, col=1)
        
        # レイアウト設定
        fig.update_layout(height=700, title_text=f"{user} のパフォーマンスとネットワーク影響力の推移")
        fig.update_yaxes(title_text="月間平均いいね数", row=1, col=1, secondary_y=False)
        fig.update_yaxes(title_text="中心性スコア", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="ランキング", row=2, col=1, autorange="reversed") # 順位は逆順に見やすい
        
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("このユーザーの投稿データまたは中心性データが見つかりませんでした。")

    # --- ランキング詳細テーブル ---
    if not df_user_centrality.empty:
        st.subheader("月ごとの中心性スコアとランキング")
        display_df = df_user_centrality[['month', 'pagerank', 'pagerank_rank', 'in_degree', 'in_degree_rank']].sort_values('month', ascending=False)
        st.dataframe(display_df, use_container_width=True)

else:
    st.info("👈 サイドバーで分析したいユーザーを選択し、「分析を開始」ボタンを押してください。")
