import streamlit as st
import pandas as pd

st.set_page_config(page_title="全ペア分析", layout="wide")

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
        # #Followees列の型を数値として読み込むために前処理
        df = pd.read_csv(filepath, sep='\t', skiprows=[1])
        df['#Followees'] = pd.to_numeric(df['#Followees'], errors='coerce')
        return df
    except FileNotFoundError:
        return None

# --- 全ペア計算関数 ---
@st.cache_data
def calculate_all_cross_category_pairs(df_mentions, df_influencers):
    """全ての異カテゴリ間メンションペアを抽出し、集計する"""
    if df_mentions is None or df_influencers is None:
        return pd.DataFrame()

    # 1. メンションのペアと回数を集計
    mention_counts = df_mentions.groupby(['username', 'mention']).size().reset_index(name='mention_count')

    # 2. メンションした側(source)のカテゴリ情報を結合
    df_merged = pd.merge(
        mention_counts,
        df_influencers[['Username', 'Category']],
        left_on='username',
        right_on='Username',
        how='inner' # influencers.txtに存在するユーザーのみを対象
    ).rename(columns={'Category': 'mentioner_category'})

    # 3. メンションされた側(target)のカテゴリ情報を結合
    df_merged = pd.merge(
        df_merged,
        df_influencers[['Username', 'Category']],
        left_on='mention',
        right_on='Username',
        how='inner' # influencers.txtに存在するユーザーのみを対象
    ).rename(columns={'Category': 'mentioned_category'})

    # 4. 異なるカテゴリ間のペアのみをフィルタリング
    cross_category_pairs = df_merged[df_merged['mentioner_category'] != df_merged['mentioned_category']]
    
    # 不要な列を削除し、列名を整形
    final_df = cross_category_pairs[['username', 'mentioner_category', 'mention', 'mentioned_category', 'mention_count']]
    final_df.columns = ['メンションしたユーザー', 'メンション元カテゴリ', 'メンションされたユーザー', 'メンション先カテゴリ', 'メンション回数']
    
    return final_df.sort_values('メンション回数', ascending=False)


# --- UI描画 ---
st.title("🌐 異カテゴリ間メンション 全ペア分析")
st.info("データセット内の全インフルエンサーのうち、異なるカテゴリに属するユーザー同士のメンション関係を分析します。")

# --- データの読み込み ---
df_mentions = load_mention_data('output_mentions_all_parallel.csv')
df_influencers = load_influencer_data('influencers.txt')

with st.spinner("全メンションペアを計算中..."):
    all_pairs_df = calculate_all_cross_category_pairs(df_mentions, df_influencers)

if all_pairs_df.empty:
    st.warning("異カテゴリ間のメンションペアが見つかりませんでした。")
    st.stop()

# --- メイン画面 ---
st.header("全ペアリスト")
st.write(f"合計 **{len(all_pairs_df)}** 組の異カテゴリ間メンションのペアが見つかりました。")

st.dataframe(all_pairs_df, use_container_width=True)
