import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="成長タイプ分析ダッシュボード", layout="wide")

# --- データ読み込み関数 ---

@st.cache_data
def load_growth_data(filepath):
    """事前に計算されたgrowth_rates.csvを読み込む"""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"ファイル '{filepath}' が見つかりません。先に `1_calculate_growth_rates.py` を実行してください。")
        return None

@st.cache_data
def load_influencer_data(filepath):
    """influencers.txtを読み込む"""
    try:
        return pd.read_csv(filepath, sep='\t', skiprows=[1])
    except FileNotFoundError:
        return None
        
@st.cache_data
def load_average_likes(filepath):
    """preprocessed_posts_with_metadata.csvから平均いいね数を計算する"""
    try:
        df = pd.read_csv(filepath, usecols=['username', 'likes'])
        return df.groupby('username')['likes'].mean().rename('average_likes').reset_index()
    except FileNotFoundError:
        return None

def load_user_post_data(username):
    """ユーザーごとに分割された投稿データを読み込む"""
    filepath = f"user_data/{username}.csv"
    try:
        return pd.read_csv(filepath, parse_dates=['datetime'])
    except FileNotFoundError:
        st.warning(f"ユーザー '{username}' の個別データファイルが見つかりませんでした。")
        return None


# --- 分析関数: 事前計算されたデータで「成長タイプ」を分類 ---
@st.cache_data
def classify_growth_type(df_features, n_clusters=3, method='K-Means'):
    """事前計算された特徴量から成長タイプを分類する"""
    if df_features.empty: return pd.DataFrame()

    features_to_cluster = df_features[['likes_growth_rate', 'average_likes']]
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features_to_cluster)
    
    if method == 'K-Means':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    elif method == '階層的クラスタリング':
        model = AgglomerativeClustering(n_clusters=n_clusters)
    else: # DBSCAN
        model = DBSCAN(eps=0.2, min_samples=5)
    
    df_features['cluster'] = model.fit_predict(scaled_features)
    
    valid_clusters = df_features[df_features['cluster'] != -1]
    cluster_centers = valid_clusters.groupby('cluster')['likes_growth_rate'].mean().sort_values()
    
    growth_map = {idx: f"タイプ {i+1}" for i, idx in enumerate(cluster_centers.index)}
    if method == 'DBSCAN':
        growth_map[-1] = "特異型 (外れ値) 👽"
    elif n_clusters == 3:
        growth_map = {
            cluster_centers.index[0]: "停滞・衰退型 📉",
            cluster_centers.index[1]: "安定成長型 📈",
            cluster_centers.index[2]: "急成長型 🚀"
        }
    elif n_clusters == 5:
        growth_map = {
            cluster_centers.index[0]: "急降下型 📉", cluster_centers.index[1]: "停滞型 ➖",
            cluster_centers.index[2]: "微増型 ↗️", cluster_centers.index[3]: "安定成長型 📈",
            cluster_centers.index[4]: "急成長型 🚀"
        }

    df_features['growth_type'] = df_features['cluster'].map(growth_map)
    return df_features


# --- データの読み込み ---
st.title("👤 成長タイプ分析ダッシュボード")
st.write("サイドバーで分析手法や粒度,ユーザーを選択すると,ダッシュボードが更新されます。")

df_growth = load_growth_data('growth_rates.csv')
df_influencers = load_influencer_data('influencers.txt')
df_avg_likes = load_average_likes('preprocessed_posts_with_metadata.csv')

if any(df is None for df in [df_growth, df_influencers, df_avg_likes]):
    st.stop()

# 分析に必要な特徴量を結合
df_features = pd.merge(df_growth, df_avg_likes, on='username', how='inner')

# --- サイドバー ---
st.sidebar.header("表示設定")

selected_method = st.sidebar.selectbox("分析手法を選択:", options=["K-Means", "階層的クラスタリング", "DBSCAN"])

if selected_method != 'DBSCAN':
    analysis_level = st.sidebar.radio("分析の粒度を選択:", options=["簡易分析 (3タイプ)", "詳細分析 (5タイプ)"])
    num_clusters = 3 if analysis_level == "簡易分析 (3タイプ)" else 5
else:
    st.sidebar.info("DBSCANは自動でグループ数を決定します。")
    num_clusters = 0

# --- 分析の実行とデータ結合 ---
df_classified = classify_growth_type(df_features, num_clusters, selected_method)
df_combined = pd.merge(df_influencers, df_classified, left_on='Username', right_on='username', how='left')

# --- サイドバー（続き） ---
st.sidebar.markdown("---")
selected_user = st.sidebar.selectbox("詳細を見たいユーザーを選択:", options=sorted(df_combined['Username'].unique()))

st.sidebar.markdown("---")
growth_type_options = ["すべてのタイプ"] + df_combined['growth_type'].dropna().unique().tolist()
selected_growth_filter = st.sidebar.radio("一覧表示の絞り込み（オプション）:", options=growth_type_options)

# --- メイン画面 ---
st.markdown("---")
st.header(f"📈 個別分析: {selected_user}")

influencer_info = df_combined[df_combined['Username'] == selected_user]
if not influencer_info.empty:
    info = influencer_info.iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("フォロワー数", f"{int(info['#Followers']):,}")
    col2.metric("フォロー数", f"{int(info['#Followees']):,}")
    col3.metric("総投稿数", f"{int(info['#Posts']):,}")
    col4.metric("成長タイプ", info['growth_type'] if pd.notna(info['growth_type']) else "N/A")
else:
    st.warning(f"{selected_user} の基本情報が見つかりませんでした。")

# --- 個別ユーザーの時系列グラフ（個別ファイルから読み込み） ---
user_posts_df = load_user_post_data(selected_user)
if user_posts_df is not None:
    fig = px.line(
        user_posts_df, x='datetime', y=['likes', 'comments'],
        labels={'datetime': '投稿日時', 'value': '数', 'variable': '指標'},
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

# --- セクション2: ユーザー一覧 ---
st.markdown("---")
st.header("👥 インフルエンサー一覧")

df_to_display = df_combined if selected_growth_filter == "すべてのタイプ" else df_combined[df_combined['growth_type'] == selected_growth_filter]
df_display_final = df_to_display[[
    'Username', '#Followers', 'growth_type', 'average_likes', '#Posts', 'Category'
]].rename(columns={
    'Username': 'ユーザー名', '#Followers': 'フォロワー数', 'growth_type': '成長タイプ',
    'average_likes': '平均いいね数', '#Posts': '投稿数', 'Category': 'カテゴリ'
}).sort_values('フォロワー数', ascending=False).reset_index(drop=True)

st.dataframe(df_display_final, use_container_width=True, column_config={
    "フォロワー数": st.column_config.NumberColumn(format="%d"),
    "平均いいね数": st.column_config.NumberColumn(format="%.1f")
})
