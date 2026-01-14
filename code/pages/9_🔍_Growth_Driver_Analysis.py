import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="成長要因分析", layout="wide")

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
def get_hashtag_analysis_data(df_ht):
    """全ハッシュタグの総使用回数と順位を計算する"""
    total_counts = df_ht['hashtag'].value_counts().reset_index()
    total_counts.columns = ['hashtag', 'total_usage']
    total_counts['rank'] = total_counts['total_usage'].rank(method='min', ascending=False).astype(int)
    return total_counts.set_index('hashtag')

st.title("🔍 成長要因分析 (Before / After)")
st.info("エンゲージメントが急上昇した「転換点」を自動検出し,その前後で何が起きたのかを分析します。")

df_posts = load_posts_data('output_beauty_category.csv')
df_hashtags = load_hashtag_mention_data('output_hashtags_all_parallel.csv', 'hashtag')
df_mentions = load_hashtag_mention_data('output_mentions_all_parallel.csv', 'mention')
df_influencers = load_influencer_data('influencers.txt')

if any(df is None for df in [df_posts, df_hashtags, df_mentions, df_influencers]):
    st.warning("必要なデータファイルの一部が読み込めませんでした。処理を中断します。")
    st.stop()

hashtag_analysis_df = get_hashtag_analysis_data(df_hashtags)
famous_users_set = set(df_influencers['Username'].unique())

st.sidebar.header("分析対象の選択")
user_list = sorted(df_posts['username'].unique())
selected_user = st.sidebar.selectbox("分析したいユーザーを選択:", options=user_list)
metric_to_analyze = st.sidebar.radio("分析指標:", ('likes', 'comments'))

user_posts_df = df_posts[df_posts['username'] == selected_user].sort_values('datetime').reset_index()
if user_posts_df.empty:
    st.warning("このユーザーの投稿データはありません。")
    st.stop()

breakpoint_idx = find_growth_breakpoint(user_posts_df, metric_to_analyze)
if breakpoint_idx:
    breakpoint_date = user_posts_df.loc[breakpoint_idx, 'datetime']
    st.success(f"成長の転換点を **{breakpoint_date.strftime('%Y-%m-%d')}** と推定しました。")
    before_df = user_posts_df[user_posts_df.index < breakpoint_idx]
    after_df = user_posts_df[user_posts_df.index >= breakpoint_idx]
else:
    st.warning("明確な成長の転換点を検出できませんでした。")
    breakpoint_date, before_df, after_df = None, user_posts_df, pd.DataFrame()

st.header("📈 エンゲージメント推移と成長点")
fig = px.line(user_posts_df, x='datetime', y=metric_to_analyze, markers=True)
if breakpoint_date:
    fig.add_vline(x=breakpoint_date.to_pydatetime(), line_width=3, line_dash="dash", line_color="red")
    fig.add_annotation(x=breakpoint_date.to_pydatetime(), y=user_posts_df[metric_to_analyze].max(), text="成長点", showarrow=False, yshift=10, font=dict(color="red"))

famous_mentions_to_user = df_mentions[
    (df_mentions['mention'] == selected_user) &
    (df_mentions['username'].isin(famous_users_set))
].sort_values('datetime')

if not famous_mentions_to_user.empty:
    y_max = user_posts_df[metric_to_analyze].max()
    for _, row in famous_mentions_to_user.iterrows():
        fig.add_annotation(
            x=row['datetime'].to_pydatetime(),
            y=y_max * 0.9,
            text=f"有名人'{row['username']}'からメンション",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, ax=0, ay=-50,
            font=dict(size=10, color="purple"), bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="purple", borderwidth=1, borderpad=4
        )
st.plotly_chart(fig, use_container_width=True)

st.header("9 📊 成長点の前後の比較")
col1, col2 = st.columns(2)

def get_mentions_with_followers(mentions_df, influencers_df):
    if mentions_df.empty:
        return pd.DataFrame(columns=['Username', 'mention_count', '#Followers', 'is_famous'])
    mention_counts = mentions_df['mention'].value_counts().reset_index()
    mention_counts.columns = ['Username', 'mention_count']
    merged_df = pd.merge(mention_counts, influencers_df[['Username', '#Followers']], on='Username', how='left')
    merged_df['is_famous'] = merged_df['#Followers'].notna()
    merged_df['#Followers'] = merged_df['#Followers'].fillna(0).astype(int)
    return merged_df.head(5)

before_hashtags_df = df_hashtags[(df_hashtags['username'] == selected_user) & (df_hashtags['datetime'] < breakpoint_date)] if breakpoint_date else pd.DataFrame()
before_mentions_df = df_mentions[(df_mentions['username'] == selected_user) & (df_mentions['datetime'] < breakpoint_date)] if breakpoint_date else pd.DataFrame()

with col1:
    st.subheader("BEFORE (成長前)")
    avg_metric = before_df[metric_to_analyze].mean()
    st.metric(f"平均{metric_to_analyze.capitalize()}", f"{avg_metric:,.2f}")
    st.write("**よく使われたハッシュタグ TOP5:**")
    st.dataframe(before_hashtags_df['hashtag'].value_counts().head(5))
    st.write("**よくメンションしたユーザー TOP5:**")
    st.dataframe(get_mentions_with_followers(before_mentions_df, df_influencers))

after_hashtags_df = df_hashtags[(df_hashtags['username'] == selected_user) & (df_hashtags['datetime'] >= breakpoint_date)] if breakpoint_date else pd.DataFrame()
after_mentions_df = df_mentions[(df_mentions['username'] == selected_user) & (df_mentions['datetime'] >= breakpoint_date)] if breakpoint_date else pd.DataFrame()

with col2:
    st.subheader("AFTER (成長後)")
    if not after_df.empty:
        avg_metric_after = after_df[metric_to_analyze].mean()
        st.metric(f"平均{metric_to_analyze.capitalize()}", f"{avg_metric_after:,.2f}", delta=f"{avg_metric_after - avg_metric:,.2f}")
        st.write("**よく使われたハッシュタグ TOP5:**")
        st.dataframe(after_hashtags_df['hashtag'].value_counts().head(5))
        st.write("**よくメンションしたユーザー TOP5:**")
        st.dataframe(get_mentions_with_followers(after_mentions_df, df_influencers))

if breakpoint_date and not after_df.empty:
    st.markdown("---")
    st.header("💡 変化の要因分析（新規要素）")

    st.subheader("🚀 新しく使われ始めたハッシュタグのトレンド分析")
    new_hashtags_set = set(after_hashtags_df['hashtag'].unique()) - set(before_hashtags_df['hashtag'].unique())
    
    if new_hashtags_set:
        all_past_hashtags = df_hashtags[df_hashtags['datetime'] < breakpoint_date]
        past_hashtag_counts = all_past_hashtags['hashtag'].value_counts().rename('past_total_usage')
        df_new_ht = pd.DataFrame(list(new_hashtags_set), columns=['hashtag']).set_index('hashtag')
        df_new_ht_analysis = df_new_ht.join(hashtag_analysis_df).join(past_hashtag_counts).fillna(0)
        df_new_ht_analysis = df_new_ht_analysis.sort_values('rank', ascending=True).reset_index()
        
        st.write("`past_total_usage`: 成長点より**前**に全ユーザーが使った総回数。0に近いほど新しいトレンドです。")
        st.write("`total_usage`: **全期間**での総使用回数。 `rank` はその順位です。")
        st.dataframe(df_new_ht_analysis[['hashtag', 'past_total_usage', 'total_usage', 'rank']], use_container_width=True)
    else:
        st.info("成長後に新しく使われ始めたハッシュタグはありませんでした。")

    st.subheader("🤝 新しくメンションし始めたユーザー")
    new_mentions_set = set(after_mentions_df['mention'].unique()) - set(before_mentions_df['mention'].unique())
    
    if new_mentions_set:
        new_mentions_df = pd.DataFrame(list(new_mentions_set), columns=['Username'])
        merged_df = pd.merge(new_mentions_df, df_influencers[['Username', '#Followers', 'Category']], on='Username', how='left')
        merged_df['is_famous'] = merged_df['#Followers'].notna()
        
        famous_new_mentions = merged_df[merged_df['is_famous'] == True]
        unknown_new_mentions = merged_df[merged_df['is_famous'] == False]

        if not famous_new_mentions.empty:
            st.success("▼ 以下の**有名インフルエンサー**（データセット内）との新しい繋がりが確認できました！")
            st.dataframe(famous_new_mentions[['Username', '#Followers', 'Category']], use_container_width=True)
        
        if not unknown_new_mentions.empty:
            st.info("▼ 以下の**新しいユーザー**（データセット外）との繋がりが確認できました。")
            st.dataframe(unknown_new_mentions[['Username']], use_container_width=True)
    else:
        st.info("成長後に新しくメンションし始めたユーザーはいませんでした。")