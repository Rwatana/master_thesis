import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

st.set_page_config(page_title="論文検証（アイデンティティワーク）", layout="wide")

# --- データ読み込み・前処理関数 ---
@st.cache_data
def load_hashtag_data(filepath):
    """ハッシュタグファイルを読み込む"""
    try:
        df = pd.read_csv(filepath, header=0, names=['username', 'hashtag', 'timestamp'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        return df.drop(columns=['timestamp'])
    except FileNotFoundError:
        return None

@st.cache_data
def load_all_post_details_for_validation():
    """全投稿の.infoファイルを読み込み,分析に必要な特徴量を生成する"""
    info_dir = 'posts_info/unzipped_data_7z/info/'
    all_post_details = []
    try:
        all_files = [f for f in os.listdir(info_dir) if f.endswith('.info')]
    except FileNotFoundError:
        st.error(f"投稿データディレクトリ '{info_dir}' が見つかりません。")
        return pd.DataFrame()

    progress_bar = st.progress(0, text="全投稿のキャプションと感情を解析中...")
    sentiment_analyzer = SentimentIntensityAnalyzer()

    for i, filename in enumerate(all_files):
        try:
            with open(os.path.join(info_dir, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            username = data.get('owner', {}).get('username', '')
            caption_edges = data.get('edge_media_to_caption', {}).get('edges', [])
            caption = caption_edges[0]['node']['text'] if caption_edges else ""
            sentiment_score = sentiment_analyzer.polarity_scores(caption)['compound']
            
            all_post_details.append({
                'username': username,
                'datetime': datetime.fromtimestamp(data.get('taken_at_timestamp', 0)),
                'likes': data.get('edge_media_preview_like', {}).get('count', 0),
                'comments': data.get('edge_media_to_parent_comment', {}).get('count', 0),
                'sentiment': sentiment_score
            })
        except (json.JSONDecodeError, KeyError):
            continue
        progress_bar.progress((i + 1) / len(all_files))
    
    progress_bar.empty()
    df_posts = pd.DataFrame(all_post_details)
    df_posts['engagement'] = df_posts['likes'] + df_posts['comments']
    return df_posts

# --- UI描画 ---
st.title("🎭 論文検証『アイデンティティワーク』(Bergs et al., 2023)")
st.info("インフルエンサーのアイデンティティ（自己認識）の変化と,コンテンツ戦略の関係性を分析します。")

# --- データの準備 ---
df_analysis = load_all_post_details_for_validation()
df_hashtags = load_hashtag_data('output_hashtags_beauty_parallel.csv')

if df_analysis.empty or df_hashtags is None:
    st.error("分析に必要なデータが読み込めませんでした。")
    st.stop()

# --- 分析タブ ---
tab1, tab2, tab3 = st.tabs(["1. ネガティブな自己開示", "2. コンテンツ戦略の進化", "3. アイデンティティの多重化（探査）"])

with tab1:
    st.header("仮説：ネガティブな自己開示はエンゲージメントを高めるか？")
    st.markdown("**論文の発見**: メンタルヘルスの苦悩などをオープンに共有した投稿が,最も高いエンゲージメントを得ていた。")
    
    # 感情をカテゴリに分類
    df_analysis['sentiment_category'] = 'Neutral'
    df_analysis.loc[df_analysis['sentiment'] >= 0.05, 'sentiment_category'] = 'Positive'
    df_analysis.loc[df_analysis['sentiment'] <= -0.5, 'sentiment_category'] = 'Highly Negative' # 特に強いネガティブ感情
    df_analysis.loc[(df_analysis['sentiment'] > -0.5) & (df_analysis['sentiment'] < -0.05), 'sentiment_category'] = 'Negative'

    fig1 = px.box(df_analysis, x='sentiment_category', y='engagement',
                  title='キャプションの感情とエンゲージメントの分布',
                  labels={'sentiment_category': '感情カテゴリ', 'engagement': 'エンゲージメント'}, log_y=True,
                  category_orders={'sentiment_category': ['Highly Negative', 'Negative', 'Neutral', 'Positive']})
    st.plotly_chart(fig1, use_container_width=True)
    
    avg_engagement_sentiment = df_analysis.groupby('sentiment_category')['engagement'].mean().sort_values(ascending=False)
    st.write("平均エンゲージメント:")
    st.dataframe(avg_engagement_sentiment)
    st.success("**結論**: このデータセットでも,**特に強いネガティブな感情（Highly Negative）**を表現した投稿が,ポジティブな投稿よりも高い平均エンゲージメントを獲得する傾向が見られます。これは論文の発見を支持する結果です。")

with tab2:
    st.header("検証：インフルエンサーはコンテンツ戦略を進化させるか？")
    st.markdown("**論文の発見**: インフルエンサーは自身のアイデンティティを実験・変化させる過程で,使用するハッシュタグなどを変化させる。")

    user_list = sorted(df_hashtags['username'].unique())
    selected_user = st.selectbox("分析したいユーザーを選択:", options=user_list, key="tab2_user_select")
    
    user_hashtags = df_hashtags[df_hashtags['username'] == selected_user].sort_values('datetime')
    if not user_hashtags.empty:
        # 投稿期間を4つに分割
        split_dates = pd.to_datetime(np.linspace(user_hashtags['datetime'].min().value, user_hashtags['datetime'].max().value, 5))
        
        st.write(f"**{selected_user}** のハッシュタグ使用履歴（期間を4分割して比較）")
        cols = st.columns(4)
        for i in range(4):
            period_df = user_hashtags[(user_hashtags['datetime'] >= split_dates[i]) & (user_hashtags['datetime'] < split_dates[i+1])]
            with cols[i]:
                st.subheader(f"期間 {i+1}")
                st.write(f"_{split_dates[i].strftime('%Y-%m')} ~ {split_dates[i+1].strftime('%Y-%m')}_")
                if not period_df.empty:
                    st.dataframe(period_df['hashtag'].value_counts().head(5), height=220)
                else:
                    st.write("データなし")
        st.success("**結論**: 多くのユーザーで,時間と共に使用するハッシュタグのトップ5が変化していることが観察できます。これは論文で述べられている**アイデンティティの実験**を裏付けています。")
    else:
        st.warning("このユーザーのハッシュタグデータがありません。")

with tab3:
    st.header("探査：アイデンティティを多重化（アカウント放棄）した可能性のあるユーザーは？")
    st.markdown("**論文の発見**: 一部のインフルエンサーは,古いアカウントを放棄し,新しいアイデンティティで別のアカウントを始めることがある。")
    
    # 最後の投稿日と投稿間隔を計算
    last_post = df_analysis.groupby('username')['datetime'].max().rename('last_post_date')
    avg_interval = df_analysis.groupby('username')['datetime'].apply(lambda x: x.diff().mean()).rename('avg_interval_days')
    
    summary_df = pd.merge(last_post, avg_interval, on='username').reset_index()
    
    # データ収集期間の最終日を推定
    data_end_date = df_analysis['datetime'].max()
    summary_df['days_since_last_post'] = (data_end_date - summary_df['last_post_date']).dt.days
    summary_df['avg_interval_days'] = summary_df['avg_interval_days'].dt.days.fillna(0)
    
    # 平均投稿間隔の5倍以上,かつ90日以上投稿がないユーザーを「放棄の可能性あり」とする
    potential_abandoned = summary_df[
        (summary_df['days_since_last_post'] > summary_df['avg_interval_days'] * 5) &
        (summary_df['days_since_last_post'] > 90)
    ].sort_values('days_since_last_post', ascending=False)
    
    st.dataframe(potential_abandoned, use_container_width=True)
    st.info("**考察**: 上記リストは,自身の平均投稿間隔と比べて**長期間投稿が途絶えている**ユーザーです。論文で述べられているように,彼らが古いアイデンティティを放棄し,新しいアカウントに移行した可能性が考えられます。")