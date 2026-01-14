import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="総合インフルエンサー分析", layout="wide")

# --- データ読み込み関数 ---
@st.cache_data
def load_posts_data(filepath):
    """投稿データを読み込む"""
    try:
        df = pd.read_csv(filepath, parse_dates=['datetime'])
        return df
    except FileNotFoundError:
        st.error(f"ファイル '{filepath}' が見つかりません。")
        return None

@st.cache_data
def load_influencers_data(filepath):
    """インフルエンサーの静的情報を読み込む"""
    try:
        df = pd.read_csv(filepath, sep='\t')
        df.columns = ['username', 'category', 'followers', 'followees', 'posts']
        return df
    except FileNotFoundError:
        st.error(f"ファイル '{filepath}' が見つかりません。")
        return None

# --- 分析関数 1: いいね数の伸び率で「成長タイプ」を分類 ---
@st.cache_data
def analyze_growth_type(_df_posts):
    """ユーザー毎の「いいね数」の成長率を計算し,成長タイプを分類する"""
    user_features = []
    for user in _df_posts['username'].unique():
        user_df = _df_posts[_df_posts['username'] == user]
        if len(user_df) < 2: continue
        
        user_df = user_df.copy()
        user_df['days_since_start'] = (user_df['datetime'] - user_df['datetime'].min()).dt.days
        
        X = user_df[['days_since_start']]
        y = user_df['likes']
        
        model = LinearRegression()
        model.fit(X, y)
        likes_growth_rate = model.coef_[0]
        
        user_features.append({
            'username': user,
            'likes_growth_rate': likes_growth_rate,
            'average_likes': user_df['likes'].mean()
        })
    
    df_features = pd.DataFrame(user_features)
    if df_features.empty: return pd.DataFrame()

    features_to_cluster = df_features[['likes_growth_rate', 'average_likes']]
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features_to_cluster)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    df_features['cluster'] = kmeans.fit_predict(scaled_features)
    
    cluster_centers = df_features.groupby('cluster')['likes_growth_rate'].mean().sort_values()
    
    growth_map = {
        cluster_centers.index[0]: "停滞・衰退型 📉",
        cluster_centers.index[1]: "安定成長型 📈",
        cluster_centers.index[2]: "急成長型 🚀"
    }
    df_features['growth_type'] = df_features['cluster'].map(growth_map)
    # ★修正点1: 必要な列だけを返すように変更
    return df_features.set_index('username')[['growth_type', 'average_likes']]

# --- 分析関数 2: フォロワー数で「階層」を分類 ---
@st.cache_data
def analyze_influencer_tier(_df_influencers):
    """フォロワー数に基づいてインフルエンサーの階層を分類する"""
    df_copy = _df_influencers.copy()
    features_to_cluster = df_copy[['followers']]
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features_to_cluster)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    df_copy['cluster'] = kmeans.fit_predict(scaled_features)
    
    cluster_centers = df_copy.groupby('cluster')['followers'].mean().sort_values()
    
    tier_map = {
        cluster_centers.index[0]: "マイクロインフルエンサー 🌱",
        cluster_centers.index[1]: "ミドルインフルエンサー ✨",
        cluster_centers.index[2]: "トップインフルエンサー 👑"
    }
    df_copy['tier_label'] = df_copy['cluster'].map(tier_map)
    # ★修正点2: 必要な列だけを返すように変更
    return df_copy.set_index('username')[['tier_label', 'followers', 'category', 'followees', 'posts']]

# --- メイン処理 ---
st.title("📊 総合インフルエンサー分析")
st.info("""
インフルエンサーを以下の2つの軸で総合的に評価します。
- **インフルエンサー階層**: 現在のフォロワー数に基づいた影響力の規模
- **成長タイプ**: いいね数の伸び率に基づいた,将来のポテンシャル
""")

df_posts = load_posts_data('output_beauty_category.csv')
df_influencers = load_influencers_data('influencers.txt')

if df_posts is not None and df_influencers is not None:
    
    df_growth_analysis = analyze_growth_type(df_posts)
    df_tier_analysis = analyze_influencer_tier(df_influencers)
    
    # 修正なしでも,これで正常に動作する
    df_combined = df_tier_analysis.join(df_growth_analysis, how='left')
    
    st.markdown("---")
    
    st.header("👤 インフルエンサーの選択とデータ可視化")
    
    user_list = sorted(df_posts['username'].unique())
    selected_user = st.selectbox("分析したいインフルエンサーを選択してください:", user_list)
    
    if selected_user:
        st.subheader(f"分析結果: **{selected_user}**")
        
        col1, col2 = st.columns(2)

        try:
            user_info = df_combined.loc[selected_user]

            with col1:
                st.metric(
                    label="インフルエンサー階層 (フォロワー数ベース)", 
                    value=user_info.get('tier_label', 'N/A')
                )
                st.caption(f"フォロワー数: {int(user_info.get('followers', 0)):,}")

            with col2:
                st.metric(
                    label="成長タイプ (いいね数ベース)",
                    value=user_info.get('growth_type', 'データ不足')
                )
                st.caption(f"平均いいね数: {user_info.get('average_likes', 0):.1f}")

        except KeyError:
            st.warning(f"'{selected_user}' の情報が見つかりませんでした。")

        st.write("---")
        
        st.subheader("❤️ いいね数・コメント数の推移")
        user_posts_df = df_posts[df_posts['username'] == selected_user].sort_values('datetime')
        
        df_melted = user_posts_df.melt(
            id_vars=['datetime'], 
            value_vars=['likes', 'comments'], 
            var_name='指標', 
            value_name='数値'
        )
        
        fig = px.line(
            df_melted, 
            x='datetime', y='数値', color='指標',
            markers=True,
            labels={'datetime': '投稿日時', '数値': '数', '指標': 'エンゲージメント指標'},
            color_discrete_map={'likes': '#636EFA', 'comments': '#FFA15A'}
        )
        st.plotly_chart(fig, use_container_width=True)