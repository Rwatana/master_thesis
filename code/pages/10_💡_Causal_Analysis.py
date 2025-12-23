import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta

st.set_page_config(page_title="成長要因分析", layout="wide")

# --- データ読み込み関数 ---
@st.cache_data
def load_posts_data(filepath):
    try:
        return pd.read_csv(filepath, parse_dates=['datetime'])
    except FileNotFoundError:
        st.error(f"ファイル '{filepath}' が見つかりません。")
        return None

@st.cache_data
def load_hashtag_mention_data(filepath, target_col_name):
    try:
        df = pd.read_csv(filepath, header=0, names=['username', target_col_name, 'timestamp'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        return df.drop(columns=['timestamp'])
    except FileNotFoundError:
        st.error(f"ファイル '{filepath}' が見つかりません。")
        return None

@st.cache_data
def load_influencer_data(filepath):
    try:
        return pd.read_csv(filepath, sep='\t', skiprows=[1])
    except FileNotFoundError:
        return None

def find_growth_breakpoint(df, metric_col):
    if len(df) < 10: return None
    max_increase, breakpoint_idx = -np.inf, None
    for i in range(int(len(df) * 0.1), int(len(df) * 0.9)):
        increase = df[metric_col][i:].mean() - df[metric_col][:i].mean()
        if increase > max_increase:
            max_increase, breakpoint_idx = increase, i
    return df.index[breakpoint_idx] if breakpoint_idx is not None else None

@st.cache_data
def get_hashtag_past_usage(df_ht, end_date):
    """指定された日付以前の全ハッシュタグ使用回数を計算"""
    past_hashtags = df_ht[df_ht['datetime'] < end_date]
    return past_hashtags['hashtag'].value_counts()

# --- UI描画 ---
st.title("💡 成長要因分析")
st.info("インフルエンサーが成長した要因を「外部要因（有名人からのメンション）」と「内部要因（トレンドの先取り）」の観点から分析します。")

# --- データの読み込み ---
df_posts = load_posts_data('output_beauty_category.csv')
df_hashtags = load_hashtag_mention_data('output_hashtags_beauty_parallel.csv', 'hashtag')
df_mentions = load_hashtag_mention_data('output_mentions_all_parallel.csv', 'mention')
df_influencers = load_influencer_data('influencers.txt')

if any(df is None for df in [df_posts, df_hashtags, df_mentions, df_influencers]):
    st.warning("必要なデータファイルの一部が読み込めませんでした。処理を中断します。")
    st.stop()

famous_users_set = set(df_influencers['Username'].unique())

# --- サイドバー ---
st.sidebar.header("分析対象の選択")
user_list = sorted(df_posts['username'].unique())
selected_user = st.sidebar.selectbox("分析したいユーザーを選択:", options=user_list)
metric_to_analyze = st.sidebar.radio("分析指標:", ('likes', 'comments'))
analysis_window_days = st.sidebar.slider("成長直前の分析期間（日数）", 1, 90, 30)

# --- メイン画面 ---
user_posts_df = df_posts[df_posts['username'] == selected_user].sort_values('datetime').reset_index()
if user_posts_df.empty:
    st.warning("このユーザーの投稿データはありません。")
    st.stop()

breakpoint_idx = find_growth_breakpoint(user_posts_df, metric_to_analyze)
if not breakpoint_idx:
    st.warning("明確な成長の転換点を検出できませんでした。")
    st.stop()

breakpoint_date = user_posts_df.loc[breakpoint_idx, 'datetime']
window_start_date = breakpoint_date - timedelta(days=analysis_window_days)

st.success(f"成長の転換点を **{breakpoint_date.strftime('%Y-%m-%d')}** と推定しました。")
st.write(f"この直前 **{analysis_window_days}日間** ({window_start_date.strftime('%Y-%m-%d')} から) に発生したイベントを分析します。")

# --- グラフ描画 ---
st.header("📈 エンゲージメント推移と分析期間")
fig = px.line(user_posts_df, x='datetime', y=metric_to_analyze, markers=True)

# ▼▼▼ 修正点 ▼▼▼
# 垂直線と注釈を分離
fig.add_vline(x=breakpoint_date.to_pydatetime(), line_width=3, line_dash="dash", line_color="red")
fig.add_annotation(x=breakpoint_date.to_pydatetime(), y=user_posts_df[metric_to_analyze].max(),
                   text="成長点", showarrow=False, yshift=10, font=dict(color="red"))
# ▲▲▲ 修正点 ▲▲▲
                   
fig.add_vrect(x0=window_start_date, x1=breakpoint_date, fillcolor="red", opacity=0.15, line_width=0, annotation_text="分析期間")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.header("📝 分析結果")

# --- 1. 外部要因の分析 ---
st.subheader("外部要因: 有名人からのエンゲージメント")
famous_mentions_in_window = df_mentions[
    (df_mentions['mention'] == selected_user) &
    (df_mentions['username'].isin(famous_users_set)) &
    (df_mentions['datetime'] >= window_start_date) &
    (df_mentions['datetime'] < breakpoint_date)
]

if not famous_mentions_in_window.empty:
    st.success(f"**要因候補**: 分析期間中に、以下の有名インフルエンサーから **{len(famous_mentions_in_window)}** 回のメンションがありました。これが成長のきっかけになった可能性があります。")
    merged_mentions = pd.merge(famous_mentions_in_window, df_influencers[['Username', '#Followers']], left_on='username', right_on='Username', how='left')
    st.dataframe(merged_mentions[['datetime', 'username', '#Followers']].rename(columns={'username': 'メンションした有名人', '#Followers': 'フォロワー数'}))
else:
    st.info("分析期間中に、データセット内の有名人からメンションされた形跡はありませんでした。")


# --- 2. 内部要因の分析 ---
st.subheader("内部要因: トレンドの先取り")
user_hashtags_in_window = df_hashtags[
    (df_hashtags['username'] == selected_user) &
    (df_hashtags['datetime'] >= window_start_date) &
    (df_hashtags['datetime'] < breakpoint_date)
]
user_hashtags_before_window = df_hashtags[
    (df_hashtags['username'] == selected_user) &
    (df_hashtags['datetime'] < window_start_date)
]
new_hashtags_set = set(user_hashtags_in_window['hashtag'].unique()) - set(user_hashtags_before_window['hashtag'].unique())

if new_hashtags_set:
    past_hashtag_counts = get_hashtag_past_usage(df_hashtags, window_start_date)
    
    pioneering_hashtags = []
    for ht in new_hashtags_set:
        past_usage = past_hashtag_counts.get(ht, 0)
        if past_usage < 50:
            pioneering_hashtags.append({'hashtag': ht, 'past_global_usage': past_usage})
    
    if pioneering_hashtags:
        st.success(f"**要因候補**: 分析期間中に、以下の**新しい/ニッチなハッシュタグ**の使用を開始しました。これがトレンドを先取りし、成長に繋がった可能性があります。")
        df_pioneer = pd.DataFrame(pioneering_hashtags).sort_values('past_global_usage')
        st.dataframe(df_pioneer.rename(columns={'hashtag': 'ハッシュタグ', 'past_global_usage': '過去の全体での使用回数'}))
    else:
        st.info("分析期間中に、新しいトレンドを先取りしたと見られるハッシュタグの使用はありませんでした。")
else:
    st.info("分析期間中に、新しいハッシュタグの使用はありませんでした。")