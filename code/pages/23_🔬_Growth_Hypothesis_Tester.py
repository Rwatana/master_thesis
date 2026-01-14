import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="成長要因 仮説検証", layout="wide")

# --- データ読み込み関数 ---
@st.cache_data
def load_data(filepath, date_col=None):
    """CSVを読み込む汎用関数"""
    try:
        if 'influencers.txt' in filepath:
             df = pd.read_csv(filepath, sep='\t', skiprows=[1])
             df.columns = ['Username', 'Category', '#Followers', '#Followees', '#Posts']
             return df
        return pd.read_csv(filepath, parse_dates=date_col)
    except FileNotFoundError:
        st.error(f"ファイル '{filepath}' が見つかりません。")
        return None

@st.cache_data
def load_hashtag_mention_data(filepath, target_col_name):
    """ハッシュタグ/メンションファイルをヘッダー付きで正しく読み込む"""
    try:
        df = pd.read_csv(filepath, header=0)
        df.rename(columns={'source': 'username', 'target': target_col_name}, inplace=True)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        return df.drop(columns=['timestamp'])
    except FileNotFoundError:
        st.error(f"ファイル '{filepath}' が見つかりません。")
        return None
    except Exception as e:
        st.error(f"'{filepath}' の読み込み中にエラーが発生しました: {e}")
        return None

# --- UI描画 ---
st.title("🔬 成長要因 仮説検証")
st.info("様々な仮説に基づき,インフルエンサーが成長する要因を探ります。")

# --- データの読み込み (軽量なファイルのみ) ---
df_growth = load_data('growth_rates.csv')
df_influencers = load_data('influencers.txt')

if df_growth is None or df_influencers is None:
    st.warning("分析に必要な基本データファイルが読み込めませんでした。")
    st.stop()

if df_growth.empty:
    st.error("成長率データ (`growth_rates.csv`) が空です。バックグラウンドで `1_calculate_growth_rates.py` を実行して,先に集計ファイルを作成してください。")
    st.stop()


# --- サイドバー: ユーザー選択 ---
st.sidebar.header("分析対象の選択")

# ▼▼▼ 修正点: 全ユーザーをリストアップ ▼▼▼
user_list = sorted(df_growth['username'].unique())
# ▲▲▲ 修正点 ▲▲▲

if not user_list:
    st.sidebar.warning("分析対象のユーザーが見つかりません。")
    st.stop()

selected_user = st.sidebar.selectbox("ユーザーを選択:", user_list)


# --- 分析タブ ---
with st.spinner("投稿,メンション,ハッシュタグデータを読み込んでいます..."):
    df_posts = load_data('preprocessed_posts_with_metadata.csv', date_col=['datetime'])
    df_mentions = load_hashtag_mention_data('output_mentions_all_parallel.csv', 'mention')
    df_hashtags = load_hashtag_mention_data('output_hashtags_all_parallel.csv', 'hashtag')

if any(df is None for df in [df_posts, df_mentions, df_hashtags]):
    st.warning("詳細データの読み込みに失敗しました。")
    st.stop()


tab1, tab2, tab3, tab4 = st.tabs(["メンション分析", "ハッシュタグ分析", "投稿頻度分析", "越境影響分析"])

user_posts_df = df_posts[df_posts['username'] == selected_user]

with tab1:
    st.header("仮説：有名になる過程でメンションが増え,特に有名人からのメンションがきっかけになるのではないか？")
    
    mentions_to_user = df_mentions[df_mentions['mention'] == selected_user].copy()
    mentions_to_user.set_index('datetime', inplace=True)
    daily_mentions = mentions_to_user.resample('D').size().rename('mention_count')
    rolling_mentions = daily_mentions.rolling(window='30D').sum()

    likes_ts = user_posts_df.set_index('datetime')['likes'].resample('D').mean().rolling(window='30D').mean()
    
    st.subheader(f"📈 {selected_user}のいいね数 vs 被メンション数（30日移動平均/合計）")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=likes_ts.index, y=likes_ts, name='平均いいね数', line=dict(color='blue')))
    fig1.add_trace(go.Scatter(x=rolling_mentions.index, y=rolling_mentions, name='被メンション数', yaxis='y2', line=dict(color='red', dash='dash')))
    fig1.update_layout(
        yaxis=dict(title='平均いいね数'),
        yaxis2=dict(title='被メンション数', overlaying='y', side='right')
    )
    st.plotly_chart(fig1, use_container_width=True)

    famous_users_set = set(df_influencers['Username'])
    famous_mentions = mentions_to_user[mentions_to_user['username'].isin(famous_users_set)]
    
    if not famous_mentions.empty:
        st.success(f"期間中に **{len(famous_mentions)}** 回,有名人からのメンションがありました。")
        st.dataframe(famous_mentions.reset_index().rename(columns={'username': 'メンションした有名人', 'datetime': '日時'}))
    else:
        st.info("期間中に有名人からのメンションはありませんでした。")

with tab2:
    st.header("仮説：トレンドの先取り（新しいハッシュタグ作成）やトレンドに乗ることが成長要因ではないか？")
    
    user_hashtags = df_hashtags[df_hashtags['username'] == selected_user]
    if not user_hashtags.empty:
        mid_date = user_hashtags['datetime'].min() + (user_hashtags['datetime'].max() - user_hashtags['datetime'].min()) / 2
        
        first_half_tags = set(user_hashtags[user_hashtags['datetime'] < mid_date]['hashtag'].unique())
        second_half_tags = set(user_hashtags[user_hashtags['datetime'] >= mid_date]['hashtag'].unique())
        
        newly_used_tags = second_half_tags - first_half_tags
        
        if newly_used_tags:
            st.subheader("🚀 活動後半に新しく使い始めたハッシュタグ")
            
            hashtag_ranks = df_hashtags['hashtag'].value_counts().reset_index()
            hashtag_ranks.columns = ['hashtag', 'total_usage']
            hashtag_ranks['rank'] = hashtag_ranks.index + 1
            
            new_tags_df = pd.DataFrame(list(newly_used_tags), columns=['hashtag'])
            new_tags_with_ranks = pd.merge(new_tags_df, hashtag_ranks, on='hashtag', how='left').fillna({'total_usage': 1, 'rank': len(hashtag_ranks)})
            
            st.dataframe(new_tags_with_ranks.sort_values('rank'))
            st.info("`rank`の順位が低いほど人気のトレンドに乗ったことを,順位が高いほどニッチ/新しいトレンドを開拓した可能性を示します。")
        else:
            st.info("活動の後半で新しく使い始めたハッシュタグは見つかりませんでした。")
    else:
        st.warning("このユーザーのハッシュタグデータはありません。")

# ▼▼▼ 修正点: 比較ロジックを削除し,選択されたユーザーのみ表示 ▼▼▼
with tab3:
    st.header("仮説：投稿頻度の一貫性が成長に繋がるのではないか？")
    
    user_post_data = df_posts[df_posts['username'] == selected_user]
    
    def calculate_frequency_std(df):
        if df.empty: return 0
        daily_counts = df.set_index('datetime').resample('D').size()
        return daily_counts.std()
        
    freq_std = calculate_frequency_std(user_post_data)
    
    st.metric(f"{selected_user} の投稿頻度のばらつき (標準偏差)", f"{freq_std:.2f}")
    st.info("標準偏差が小さいほど,投稿頻度が**一貫している**ことを示します。")
# ▲▲▲ 修正点 ▲▲▲


with tab4:
    st.header("仮説：他分野のインフルエンサーからのメンションは成長に繋がりやすいのではないか？")

    user_info = df_influencers[df_influencers['Username'] == selected_user]
    if not user_info.empty:
        user_category = user_info['Category'].iloc[0]
        st.write(f"**{selected_user}** のカテゴリ: **{user_category}**")
        
        mentions_to_user_with_category = pd.merge(
            df_mentions[df_mentions['mention'] == selected_user],
            df_influencers[['Username', 'Category']],
            left_on='username', right_on='Username', how='left'
        ).rename(columns={'Category': 'mentioner_category'})
        
        cross_category_mentions = mentions_to_user_with_category[
            mentions_to_user_with_category['mentioner_category'] != user_category
        ].dropna(subset=['mentioner_category'])
        
        if not cross_category_mentions.empty:
            unique_mentioners = cross_category_mentions['username'].nunique()
            st.success(f"このユーザーは,**{cross_category_mentions['mentioner_category'].nunique()}** の異なる分野の **{unique_mentioners}** 人から,合計 **{len(cross_category_mentions)}** 回のメンションを受けています。")
            
            category_counts = cross_category_mentions['mentioner_category'].value_counts()
            fig_pie = px.pie(values=category_counts.values, names=category_counts.index, title="どの分野からのメンションが多いか")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("このユーザーは,他分野からのメンションを受けていませんでした。")
    else:
        st.warning("このユーザーのカテゴリ情報が見つかりません。")

