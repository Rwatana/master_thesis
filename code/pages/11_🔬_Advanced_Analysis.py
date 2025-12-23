import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime, timedelta
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="高度な戦略分析", layout="wide")

# --- データ読み込み・計算関数 ---
@st.cache_data
def load_influencer_data(filepath):
    try:
        return pd.read_csv(filepath, sep='\t', skiprows=[1])
    except FileNotFoundError:
        st.error(f"ファイル '{filepath}' が見つかりません。")
        return None

@st.cache_data
def load_post_data(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def get_all_posts_with_details(username, info_dir):
    """選択されたユーザーの全投稿の詳細データを取得する"""
    posts_details = []
    try:
        user_post_files = [f for f in os.listdir(info_dir) if f.startswith(f"{username}-")]
        for filename in user_post_files:
            data = load_post_data(os.path.join(info_dir, filename))
            if data:
                caption_edges = data.get('edge_media_to_caption', {}).get('edges', [])
                caption = caption_edges[0]['node']['text'] if caption_edges else ""
                posts_details.append({
                    'datetime': datetime.fromtimestamp(data.get('taken_at_timestamp', 0)),
                    'likes': data.get('edge_media_preview_like', {}).get('count', 0),
                    'comments': data.get('edge_media_to_parent_comment', {}).get('count', 0),
                    'caption': caption,
                    'caption_length': len(caption),
                    'has_question': '?' in caption
                })
        return pd.DataFrame(posts_details).sort_values('datetime').reset_index(drop=True)
    except FileNotFoundError:
        st.error(f"投稿データディレクトリ `{info_dir}` が見つかりません。")
        return pd.DataFrame()

def find_growth_breakpoint(df, metric_col):
    if len(df) < 10: return None
    max_increase, breakpoint_idx = -np.inf, None
    for i in range(int(len(df) * 0.1), int(len(df) * 0.9)):
        increase = df[metric_col][i:].mean() - df[metric_col][:i].mean()
        if increase > max_increase:
            max_increase, breakpoint_idx = increase, i
    return df.index[breakpoint_idx] if breakpoint_idx is not None else None

def generate_wordcloud(text):
    """テキストからワードクラウドを生成する"""
    if not text or text.isspace():
        return None
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

# --- UI描画 ---
st.title("🔬 高度な戦略分析")
st.info("インフルエンサーが成長した要因を**コンテンツ戦略の変化**から分析します。")

# --- データの準備 ---
df_influencers = load_influencer_data('influencers.txt')
info_dir = 'posts_info/unzipped_data_7z/info/'
if df_influencers is None:
    st.stop()

# --- サイドバー ---
st.sidebar.header("分析対象の選択")
user_list = sorted(df_influencers['Username'].unique())
selected_user = st.sidebar.selectbox("分析したいユーザーを選択:", options=user_list)

# --- メイン画面 ---
with st.spinner(f"{selected_user}の全投稿データを解析中..."):
    user_posts_df = get_all_posts_with_details(selected_user, info_dir)

if user_posts_df.empty:
    st.warning("このユーザーの投稿データはありません。")
    st.stop()

breakpoint_idx = find_growth_breakpoint(user_posts_df, 'likes')
if not breakpoint_idx:
    st.warning("明確な成長の転換点を検出できませんでした。")
    st.stop()

breakpoint_date = user_posts_df.loc[breakpoint_idx, 'datetime']
before_df = user_posts_df[user_posts_df.index < breakpoint_idx]
after_df = user_posts_df[user_posts_df.index >= breakpoint_idx]

st.success(f"成長の転換点を **{breakpoint_date.strftime('%Y-%m-%d')}** と推定しました。")
st.markdown("---")

# --- 1. 戦略変化のサマリー ---
st.header("📊 戦略変化のサマリー")
if not before_df.empty and not after_df.empty:
    # 投稿頻度の計算
    days_before = (before_df['datetime'].max() - before_df['datetime'].min()).days + 1
    days_after = (after_df['datetime'].max() - after_df['datetime'].min()).days + 1
    freq_before = (len(before_df) / days_before) * 7 if days_before > 0 else 0
    freq_after = (len(after_df) / days_after) * 7 if days_after > 0 else 0

    # その他の指標
    avg_len_before = before_df['caption_length'].mean()
    avg_len_after = after_df['caption_length'].mean()
    question_rate_before = before_df['has_question'].mean() * 100
    question_rate_after = after_df['has_question'].mean() * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("週あたりの平均投稿数", f"{freq_after:.2f} 回", f"{freq_after - freq_before:.2f} 回")
    col2.metric("平均キャプション長", f"{avg_len_after:.0f} 文字", f"{avg_len_after - avg_len_before:.0f} 文字")
    col3.metric("キャプションで質問する割合", f"{question_rate_after:.1f} %", f"{question_rate_after - question_rate_before:.1f} %")
else:
    st.info("比較のための十分なデータがありません。")


# --- 2. 発信テーマの変化（ワードクラウド） ---
st.markdown("---")
st.header("🎨 発信テーマの変化")
st.write("ワードクラウドで、成長の前後でキャプションで使われる単語がどう変化したかを示します。")

col_wc1, col_wc2 = st.columns(2)
with col_wc1:
    st.subheader("BEFORE (成長前)")
    text_before = " ".join(cap for cap in before_df['caption'])
    fig_before = generate_wordcloud(text_before)
    if fig_before:
        st.pyplot(fig_before)
    else:
        st.write("テキストデータがありません。")

with col_wc2:
    st.subheader("AFTER (成長後)")
    text_after = " ".join(cap for cap in after_df['caption'])
    fig_after = generate_wordcloud(text_after)
    if fig_after:
        st.pyplot(fig_after)
    else:
        st.write("テキストデータがありません。")
