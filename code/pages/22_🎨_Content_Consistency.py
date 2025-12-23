import streamlit as st
import pandas as pd
import plotly.express as px
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
import os

# --- 初回実行時にストップワードをダウンロード ---
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    st.info("初回実行のため、NLTKのストップワードリストをダウンロードします。")
    nltk.download('stopwords')

st.set_page_config(page_title="コンテンツ一貫性分析", layout="wide")

# --- データ読み込み・分析関数 ---
def load_user_post_data(username):
    """ユーザーごとに分割された投稿データを読み込む"""
    filepath = f"user_data/{username}.csv"
    try:
        return pd.read_csv(filepath, parse_dates=['datetime'])
    except FileNotFoundError:
        st.error(f"ユーザー '{username}' の個別データファイルが見つかりませんでした。先に `aggregate_user_data.py` を実行してください。")
        return None

@st.cache_data
def analyze_user_topics(df, num_topics, num_words):
    """選択されたユーザーの投稿データからトピックモデリングを実行する"""
    if df.empty or 'caption' not in df.columns:
        return pd.DataFrame(), {}

    # テキストの前処理
    stop_words = list(nltk.corpus.stopwords.words('english'))
    stop_words.extend(['beauty', 'makeup', 'like', 'get', 'product', 'use', 'skin', 'look', 'fashion', 'style'])
    
    # min_df=2 に変更し、一人のユーザーの投稿でも単語が拾われやすくする
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, stop_words=stop_words)
    try:
        tfidf = vectorizer.fit_transform(df['caption'].fillna(''))
    except ValueError: # 語彙が少なすぎる場合
        return df, {}


    # LDAモデルの学習
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
    lda.fit(tfidf)

    df['topic'] = lda.transform(tfidf).argmax(axis=1)

    feature_names = vectorizer.get_feature_names_out()
    topic_keywords = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_words = " ".join([feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]])
        topic_keywords[topic_idx] = top_words
        
    return df, topic_keywords

# --- UI描画 ---
st.title("🎨 コンテンツの一貫性（専門性）分析")
st.info("ユーザーを選択し、その人の投稿内容から自動的に「トピック」を抽出します。これにより、特定のテーマに集中している「専門家」か、多様な話題に触れる「ジェネラリスト」かを評価します。")

# --- サイドバー ---
st.sidebar.header("分析対象の選択")

try:
    # user_dataディレクトリからユーザーリストを取得
    user_list = sorted([f.replace('.csv', '') for f in os.listdir('user_data') if f.endswith('.csv')])
except FileNotFoundError:
    st.error("`user_data` ディレクトリが見つかりません。先に `aggregate_user_data.py` を実行してください。")
    user_list = []

if not user_list:
    st.warning("分析対象のユーザーがいません。")
    st.stop()

selected_user = st.sidebar.selectbox("1. 分析したいユーザーを選択:", options=user_list)

st.sidebar.subheader("モデル設定")
num_topics = st.sidebar.slider("検出するトピック数", 2, 8, 3) # デフォルトを3に変更
num_keywords = st.sidebar.slider("各トピックのキーワード数", 3, 10, 5)

# --- 分析開始ボタンと状態管理 ---
if 'run_topic_analysis' not in st.session_state:
    st.session_state.run_topic_analysis = False
if 'analyzed_user_topics' not in st.session_state:
    st.session_state.analyzed_user_topics = ""

if st.sidebar.button("分析を開始"):
    st.session_state.run_topic_analysis = True
    st.session_state.analyzed_user_topics = selected_user
# ユーザーが変更されたら分析状態をリセット
elif selected_user != st.session_state.analyzed_user_topics:
    st.session_state.run_topic_analysis = False

# --- メイン画面 ---
if st.session_state.run_topic_analysis:
    user_to_analyze = st.session_state.analyzed_user_topics
    
    with st.spinner(f"'{user_to_analyze}' の投稿データを読み込み、トピックを分析中..."):
        df_user = load_user_post_data(user_to_analyze)
        
        if df_user is not None:
            df_analyzed, topic_keywords = analyze_user_topics(df_user, num_topics, num_keywords)
            
            st.header(f"分析結果: {user_to_analyze}")
            
            # 1. トピック分布の可視化
            st.subheader("投稿トピックの分布")
            if not df_analyzed.empty and 'topic' in df_analyzed.columns:
                topic_distribution = df_analyzed['topic'].value_counts(normalize=True).mul(100)
                top_topic_percentage = topic_distribution.max()

                if top_topic_percentage > 60:
                    st.success(f"**専門家タイプ**: 投稿の **{top_topic_percentage:.1f}%** が単一のトピック（Topic {topic_distribution.idxmax()}）に集中しています。")
                elif top_topic_percentage > 40:
                    st.info(f"**準専門家タイプ**: 特定のトピックに **{top_topic_percentage:.1f}%** が集まっていますが、他の話題にも触れています。")
                else:
                    st.warning("**ジェネラリストタイプ**: 投稿が複数のトピックに分散しており、多様な内容を発信しています。")

                fig = px.pie(values=topic_distribution.values, names=topic_distribution.index, title=f'{user_to_analyze}の投稿トピック分布')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("このユーザーの投稿からトピックを抽出できませんでした（投稿数が少ない、または語彙が少ない可能性があります）。")

            # 2. 各トピックのキーワード
            st.subheader("検出されたトピックのキーワード")
            if topic_keywords:
                df_topics = pd.DataFrame(list(topic_keywords.items()), columns=['Topic', 'Keywords'])
                st.dataframe(df_topics, use_container_width=True)
            else:
                st.warning("キーワードを抽出できませんでした。")
else:
    st.info("👈 サイドバーで分析したいユーザーを選択し、「分析を開始」ボタンを押してください。")
