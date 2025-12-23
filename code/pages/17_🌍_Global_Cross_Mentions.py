import streamlit as st
import pandas as pd

st.set_page_config(page_title="全体 越境メンション分析", layout="wide")

# --- データ読み込み関数 ---
@st.cache_data
def load_mention_data(filepath):
    """メンションファイルを読み込む"""
    try:
        df = pd.read_csv(filepath, header=0, names=['username', 'mention', 'timestamp'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        return df.drop(columns=['timestamp'])
    except FileNotFoundError:
        st.error(f"ファイル '{filepath}' が見つかりません。")
        return None

@st.cache_data
def load_influencer_data(filepath):
    """influencers.txtを読み込む"""
    try:
        return pd.read_csv(filepath, sep='\t', skiprows=[1])
    except FileNotFoundError:
        return None

# --- 全ペア計算関数 ---
@st.cache_data
def calculate_all_cross_mentions(df_mentions, df_influencers):
    """
    全ての異カテゴリ間メンションを抽出し、詳細ペアと個人サマリーの2つのテーブルを作成する。
    """
    if df_mentions is None or df_influencers is None:
        return pd.DataFrame(), pd.DataFrame()

    df_merged = pd.merge(df_mentions, df_influencers[['Username', 'Category']], left_on='username', right_on='Username', how='inner').rename(columns={'Category': 'mentioner_category'})
    df_merged = pd.merge(df_merged, df_influencers[['Username', 'Category']], left_on='mention', right_on='Username', how='inner').rename(columns={'Category': 'mentioned_category'})
    cross_category_df = df_merged[df_merged['mentioner_category'] != df_merged['mentioned_category']]

    # 1. 詳細ペアテーブルの作成
    pair_summary = cross_category_df.groupby(['mention', 'mentioned_category', 'mentioner_category']).agg(
        total_mentions=('username', 'count'),
        unique_mentioners=('username', 'nunique')
    ).reset_index()
    pair_summary.rename(columns={
        'mention': 'メンションされたユーザー', 'mentioned_category': '自分のカテゴリ',
        'mentioner_category': 'メンション元のカテゴリ', 'total_mentions': '合計メンション回数',
        'unique_mentioners': 'ユニークなメンション元の人数'
    }, inplace=True)
    
    # 2. 個人サマリーテーブルの作成
    personal_summary = cross_category_df.groupby('mention').agg(
        total_cross_mentions=('username', 'count'),
        unique_cross_mentioners=('username', 'nunique'),
        unique_cross_categories=('mentioner_category', 'nunique')
    ).reset_index()
    personal_summary.rename(columns={
        'mention': 'メンションされたユーザー', 'total_cross_mentions': '異分野からの総メンション数',
        'unique_cross_mentioners': '異分野のユニークユーザー数', 'unique_cross_categories': '異分野のカテゴリ数'
    }, inplace=True)
    
    # 個人サマリーに本人のカテゴリ情報を追加
    personal_summary = pd.merge(personal_summary, df_influencers[['Username', 'Category']], left_on='メンションされたユーザー', right_on='Username', how='left')
    
    return pair_summary.sort_values(['メンションされたユーザー', '合計メンション回数'], ascending=[True, False]), personal_summary.sort_values('異分野からの総メンション数', ascending=False)


# --- UI描画 ---
st.title("🌍 全体 越境メンション分析")
st.info("各インフルエンサーが、どのカテゴリのユーザーから、どれだけ多くの注目（メンション）を集めているかを分析します。")

# --- データの読み込み ---
df_mentions = load_mention_data('output_mentions_all_parallel.csv')
df_influencers = load_influencer_data('influencers.txt')

with st.spinner("全体の越境メンション関係を計算中..."):
    pair_summary_df, personal_summary_df = calculate_all_cross_mentions(df_mentions, df_influencers)

if pair_summary_df.empty:
    st.warning("異カテゴリ間のメンションが見つかりませんでした。")
    st.stop()
    
# --- サイドバー フィルター ---
st.sidebar.header("フィルター")

# ▼▼▼ 修正点: ユーザー絞り込みを追加 ▼▼▼
all_users = sorted(pair_summary_df['メンションされたユーザー'].unique())
selected_user = st.sidebar.selectbox(
    "ユーザーで絞り込む (任意):",
    options=['（全てのユーザー）'] + all_users,
    index=0 # デフォルトは「全てのユーザー」
)

# ユーザーが選択された場合、データフレームをフィルタリング
if selected_user != '（全てのユーザー）':
    pair_summary_df = pair_summary_df[pair_summary_df['メンションされたユーザー'] == selected_user]
    personal_summary_df = personal_summary_df[personal_summary_df['メンションされたユーザー'] == selected_user]
# ▲▲▲ 修正点 ▲▲▲

all_categories = sorted(pair_summary_df['自分のカテゴリ'].unique())
selected_my_category = st.sidebar.multiselect(
    "自分のカテゴリで絞り込み:",
    options=all_categories,
    default=all_categories
)

if not selected_my_category:
    st.warning("少なくとも1つのカテゴリを選択してください。")
    st.stop()
    
# カテゴリでフィルタリング
pair_summary_df = pair_summary_df[pair_summary_df['自分のカテゴリ'].isin(selected_my_category)]
personal_summary_df = personal_summary_df[personal_summary_df['Category'].isin(selected_my_category)]


# --- メイン画面 ---

# ▼▼▼ 新しいセクションを追加 ▼▼▼
st.header("サマリー：誰が分野を超えて注目されているか")
st.write("各ユーザーが、異分野から受けた総メンション数、ユニークユーザー数、カテゴリ数の集計です。")
st.dataframe(personal_summary_df[['メンションされたユーザー', 'Category', '異分野からの総メンション数', '異分野のユニークユーザー数', '異分野のカテゴリ数']], use_container_width=True)
# ▲▲▲ 新しいセクション ▲▲▲


st.markdown("---")
st.header("詳細：どの分野から注目されているか")
st.write(f"合計 **{len(pair_summary_df)}** パターンの越境メンション関係が見つかりました。")
st.dataframe(pair_summary_df, use_container_width=True)