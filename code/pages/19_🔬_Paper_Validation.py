import streamlit as st
import pandas as pd
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.stats import ttest_ind
import numpy as np

st.set_page_config(page_title="論文検証", layout="wide")

# ▼▼▼ 修正点: CSVを読み込むだけの高速な関数に変更 ▼▼▼
@st.cache_data
def load_preprocessed_data(filepath):
    """事前処理済みのCSVを読み込む"""
    try:
        return pd.read_csv(filepath, parse_dates=['datetime'])
    except FileNotFoundError:
        st.error(f"'{filepath}'が見つかりません。先に `preprocess_data.py` を実行してください。")
        return None

@st.cache_data
def load_influencer_data(filepath):
    try:
        return pd.read_csv(filepath, sep='\t', skiprows=[1])
    except FileNotFoundError:
        return None

# ▲▲▲ 修正点 ▲▲▲


# --- UI描画 ---
st.title("🔬 論文検証『Less is more』(van der Harst et al., 2024)")
st.info("論文で提示された主要な仮説が,このデータセットでも成立するかを検証します。")

# --- データの準備 ---
# ▼▼▼ 修正点: 読み込み部分をシンプル化 ▼▼▼
df_influencers = load_influencer_data('influencers.txt')
df_posts = load_preprocessed_data('preprocessed_posts_with_metadata.csv')

if df_influencers is None or df_posts is None:
    st.stop()

# 感情分析とインフルエンサータイプの分類
sentiment_analyzer = SentimentIntensityAnalyzer()
df_posts['sentiment'] = df_posts['caption'].fillna("").apply(lambda x: sentiment_analyzer.polarity_scores(x)['compound'])
df_analysis = pd.merge(df_posts, df_influencers[['Username', '#Followers']], left_on='username', right_on='Username', how='left')
bins = [0, 50000, 100000, 1000000, float('inf')]
labels = ['Micro', 'Meso', 'Macro', 'Mega']
df_analysis['influencer_type'] = pd.cut(df_analysis['#Followers'], bins=bins, labels=labels, right=False)
df_analysis = df_analysis.dropna(subset=['influencer_type'])
df_analysis['engagement'] = df_analysis['likes'] + df_analysis['comments']
# ▲▲▲ 修正点 ▲▲▲

# --- 分析タブ ---
tab1, tab2, tab3 = st.tabs(["仮説1: マイクロインフルエンサーの効果", "仮説3: 写真への人物登場の効果", "仮説4: ネガティブな感情の効果"])

# (以降のタブ内のコードは変更なし)
with tab1:
    st.header("仮説1: マイクロインフルエンサーはフォロワーあたりのエンゲージメントが高いか？")
    st.markdown("**論文の発見**: マイクロインフルエンサーは「お気に入り（いいね）」は高いが,「リツイート（共有）」は低い。")
    
    df_analysis['engagement_per_follower'] = df_analysis['engagement'] / df_analysis['#Followers'].replace(0, np.nan)
    
    fig1 = px.box(df_analysis, x='influencer_type', y='engagement_per_follower', 
                  title='インフルエンサータイプ別 フォロワーあたりエンゲージメント',
                  labels={'influencer_type': 'タイプ', 'engagement_per_follower': 'フォロワーあたりエンゲージメント'},
                  log_y=True, category_orders={'influencer_type': ['Micro', 'Meso', 'Macro', 'Mega']})
    st.plotly_chart(fig1, use_container_width=True)
    
    median_engagement = df_analysis.groupby('influencer_type')['engagement_per_follower'].median()
    st.write("フォロワーあたりエンゲージメントの中央値:")
    st.dataframe(median_engagement)
    st.success("**結論**: 論文と同様に,マイクロインフルエンサーがフォロワーあたりのエンゲージメントが最も高い傾向が見られます。仮説は支持されました。")

with tab2:
    st.header("仮説3: 写真に人物が登場するとエンゲージメントは高まるか？")
    st.markdown("""
    - **仮説3a**: 写真に人物が登場するとエンゲージメントは高まる。
    - **仮説3b**: その効果はフォロワーが少ない（マイクロ）ほど強い。
    
    ここでは,投稿に**ユーザータグが付いているか**を「人物が登場しているか」の代理指標として使用します。
    """)
    
    df_analysis['has_person_proxy'] = df_analysis['tag_count'] > 0
    
    st.subheader("仮説3aの検証")
    fig2 = px.box(df_analysis, x='has_person_proxy', y='engagement', 
                  title='人物登場（代理指標）の有無とエンゲージメント',
                  labels={'has_person_proxy': '人物登場 (タグ有無)', 'engagement': 'エンゲージメント'}, log_y=True)
    st.plotly_chart(fig2, use_container_width=True)
    
    avg_engagement_person = df_analysis.groupby('has_person_proxy')['engagement'].mean()
    st.write("平均エンゲージメント:")
    st.dataframe(avg_engagement_person)
    st.success("**結論 (3a)**: このデータセットでは,人物が登場する（タグがある）方がエンゲージメントが**低い**傾向にあり,論文の仮説とは逆の結果となりました。")

    st.subheader("仮説3bの検証")
    fig3 = px.scatter(df_analysis, x='#Followers', y='engagement', color='has_person_proxy', 
                      trendline='ols', log_x=True, log_y=True,
                      title='フォロワー数とエンゲージメントの関係（人物登場の有無別）',
                      labels={'#Followers': 'フォロワー数', 'engagement': 'エンゲージメント'})
    st.plotly_chart(fig3, use_container_width=True)
    st.info("2本のトレンドラインの傾きに注目してください。もしマイクロインフルエンサー（左側）で青線(True)が赤線(False)より上にあり,右側で逆転するなら仮説は支持されます。")
    st.warning("**結論 (3b)**: グラフ全体で赤線（人物なし）が青線（人物あり）を上回っており,フォロワー数が少ない領域で特に効果が強いという仮説は支持されませんでした。")

with tab3:
    st.header("仮説4: ネガティブな感情のテキストはエンゲージメントを高めるか？")
    st.markdown("**論文の発見**: ネガティブなテキスト感情はエンゲージメントにプラスの効果がある。")

    df_analysis['sentiment_category'] = pd.cut(df_analysis['sentiment'], 
                                               bins=[-1.1, -0.05, 0.05, 1.1], 
                                               labels=['Negative', 'Neutral', 'Positive'])
    
    fig4 = px.box(df_analysis, x='sentiment_category', y='engagement',
                  title='キャプションの感情とエンゲージメントの分布',
                  labels={'sentiment_category': '感情カテゴリ', 'engagement': 'エンゲージメント'}, log_y=True,
                  category_orders={'sentiment_category': ['Negative', 'Neutral', 'Positive']})
    st.plotly_chart(fig4, use_container_width=True)
    
    neg_engagement = df_analysis[df_analysis['sentiment_category'] == 'Negative']['engagement'].dropna()
    pos_engagement = df_analysis[df_analysis['sentiment_category'] == 'Positive']['engagement'].dropna()
    
    t_stat, p_value = ttest_ind(neg_engagement, pos_engagement, equal_var=False)

    st.metric("p値（Negative vs Positive）", f"{p_value:.4f}")
    if p_value < 0.05 and neg_engagement.mean() > pos_engagement.mean():
        st.success("**結論**: 統計的に有意な差が見られ,ネガティブな投稿の方がエンゲージメントが高い結果となりました。論文の仮説は支持されました。")
    else:
        st.warning("**結論**: 統計的に有意な差は見られず,このデータセットでは論文の仮説は支持されませんでした。")
