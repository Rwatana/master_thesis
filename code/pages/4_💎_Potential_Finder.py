# -*- coding: utf-8 -*-
"""
Streamlit application to verify influencer engagement statistics from a research paper
against a user's dataset.
This app calculates engagement rates based on user-defined metrics and compares
the resulting distribution of influencers with the paper's Table 2.
"""

import streamlit as st
import pandas as pd
import altair as alt
from io import StringIO

# --- 定数定義 (Constants) ---
PREPROCESSED_FILE = 'preprocessed_posts_with_metadata.csv'
INFLUENCERS_FILE = 'influencers.txt'

# --- ページの基本設定 (Page Configuration) ---
st.set_page_config(
    page_title="Verify Paper Statistics",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Verify Paper's Statistics (Table 2)")
st.markdown("""
論文に記載されている**Table 2**のインフルエンサー分布を、あなたのデータセットで検証します。  
サイドバーで**エンゲージメント率 `E(·)` の定義**と**分析対象月**を選択してください。
""")

# --- データ読み込みとキャッシュ (Data Loading and Caching) ---

@st.cache_data
def get_paper_table():
    """論文のTable 2のデータをDataFrameとして作成します。"""
    paper_data = {
        'Relevance Level': [5, 4, 3, 2, 1, 0],
        'Engagement Rate E(·)': [
            "E(·) >= 0.10",
            "0.10 > E(·) >= 0.07",
            "0.07 > E(·) >= 0.05",
            "0.05 > E(·) >= 0.03",
            "0.03 > E(·) >= 0.01",
            "0.01 > E(·)"
        ],
        'Number of Influencers': [1274, 1678, 2321, 4509, 6882, 1734],
        'Percentage': ["6.92%", "9.12%", "12.62%", "24.51%", "37.41%", "9.42%"]
    }
    return pd.DataFrame(paper_data).set_index('Relevance Level')

@st.cache_data
def load_data():
    """投稿データとインフルエンサーデータを読み込み、前処理してマージします。"""
    try:
        df_posts = pd.read_csv(PREPROCESSED_FILE, parse_dates=['datetime'], low_memory=False)
        if 'comments' not in df_posts.columns:
            df_posts['comments'] = 0

        with open(INFLUENCERS_FILE, 'r', encoding='utf-8') as f:
            lines = [line for line in f.readlines() if '===' not in line]
        
        df_influencers = pd.read_csv(StringIO("".join(lines)), sep='\t', dtype=str)
        df_influencers = df_influencers.rename(columns={'#Followers': 'followers', 'Username': 'username'})
        
        df_influencers['followers'] = pd.to_numeric(df_influencers['followers'], errors='coerce')
        df_posts['month'] = df_posts['datetime'].dt.to_period('M').dt.start_time
        
        df_merged = pd.merge(df_posts, df_influencers[['username', 'followers']], on='username', how='left')
        
        month_list = sorted(df_merged['month'].unique(), reverse=True)
        return df_merged, month_list

    except FileNotFoundError as e:
        st.error(f"エラー: {e.filename} が見つかりません。必要なファイルをアップロードしてください。")
        return None, None
    except Exception as e:
        st.error(f"データの読み込み中にエラーが発生しました: {e}")
        return None, None

# --- ヘルパー関数 (Helper Functions) ---

def assign_relevance_levels(engagement_rate):
    """エンゲージメント率に基づいて関連性レベル（0〜5）を割り当てます。"""
    if engagement_rate >= 0.10: return 5
    if engagement_rate >= 0.07: return 4
    if engagement_rate >= 0.05: return 3
    if engagement_rate >= 0.03: return 2
    if engagement_rate >= 0.01: return 1
    return 0

# --- サイドバー (Sidebar for User Input) ---
st.sidebar.header("⚙️ Analysis Configuration")
df, month_list = load_data()

if df is not None and month_list is not None:
    selected_month = st.sidebar.selectbox(
        "🗓️ 分析対象の月を選択",
        month_list,
        format_func=lambda date: date.strftime('%Y-%m')
    )

    numerator_option = st.sidebar.selectbox(
        "分子 (Numerator)",
        ("いいね数", "いいね数 + コメント数"),
        key="numerator",
        help="エンゲージメント率の分子を選択します。"
    )

    denominator_option = st.sidebar.radio(
        "分母 (Denominator)",
        ("投稿数", "フォロワー数"),
        key="denominator",
        help="**投稿数**: 1投稿あたりのエンゲージメント数を計算します。\n\n**フォロワー数**: 論文の定義に基づき、フォロワー数あたりの平均エンゲージメント数を計算します。"
    )

    # --- メイン処理 (Main Processing) ---
    st.markdown("---")
    
    # 1. 選択された月のデータをフィルタリング
    df_month = df[df['month'] == selected_month]

    if not df_month.empty:
        # 2. 月内のインフルエンサーごとにエンゲージメントを集計
        monthly_agg = df_month.groupby(['username', 'followers']).agg(
            total_likes=('likes', 'sum'),
            total_comments=('comments', 'sum'),
            post_count=('datetime', 'size')
        ).reset_index()

        # 3. エンゲージメント率 E(·) を計算
        # 3a. 分子を決定
        if numerator_option == "いいね数":
            monthly_agg['numerator_total'] = monthly_agg['total_likes']
        else:
            monthly_agg['numerator_total'] = monthly_agg['total_likes'] + monthly_agg['total_comments']

        # 3b. 分母に応じて計算方法を変更
        if denominator_option == "投稿数":
            # [計算A] 1投稿あたりのエンゲージメント数 (Engagement Per Post)
            monthly_agg['engagement_rate'] = monthly_agg.apply(
                lambda row: row['numerator_total'] / row['post_count'] if row['post_count'] > 0 else 0, axis=1
            )
        else: # フォロワー数
            # [計算B] 論文の定義に準拠した計算
            # B-1. まず、1投稿あたりの「平均」エンゲージメント数を計算
            monthly_agg['avg_engagement_per_post'] = monthly_agg.apply(
                lambda row: row['numerator_total'] / row['post_count'] if row['post_count'] > 0 else 0, axis=1
            )
            # B-2. 次に、その平均値をフォロワー数で割る
            monthly_agg['engagement_rate'] = monthly_agg.apply(
                lambda row: row['avg_engagement_per_post'] / row['followers'] if pd.notna(row['followers']) and row['followers'] > 0 else 0, axis=1
            )
        
        # 4. 各インフルエンサーに関連性レベルを割り当て
        monthly_agg['Relevance Level'] = monthly_agg['engagement_rate'].apply(assign_relevance_levels)

        # 5. 分布を計算
        counts = monthly_agg['Relevance Level'].value_counts()
        percentages = monthly_agg['Relevance Level'].value_counts(normalize=True) * 100
        
        result_df = pd.DataFrame({
            'Number of Influencers': counts,
            'Percentage': percentages.map('{:.2f}%'.format)
        }).sort_index()
        
        result_df = result_df.reindex(range(6), fill_value=0)
        result_df['Percentage'] = result_df.apply(
            lambda row: '0.00%' if row['Number of Influencers'] == 0 else row['Percentage'], axis=1
        )

        # --- 結果表示 (Displaying Results) ---
        st.subheader(f"📊 {selected_month.strftime('%Y-%m')} の分析結果")
        st.metric(
            label="分析対象の総インフルエンサー数",
            value=f"{len(monthly_agg):,} 人"
        )

        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.subheader("📄 論文の分布 (Table 2)")
            st.dataframe(get_paper_table(), use_container_width=True)
            
        with col2:
            st.subheader("📈 あなたのデータの分布")
            st.dataframe(result_df, use_container_width=True)

        st.markdown("---")
        st.subheader(" visually Comparing the Distributions")
        
        # グラフ用にデータを整形
        chart_df_paper = get_paper_table().reset_index()
        chart_df_paper['Source'] = 'Paper'
        
        chart_df_your = result_df.reset_index()
        chart_df_your.rename(columns={'Number of Influencers': 'Number of Influencers Your Data'}, inplace=True)
        chart_df_your['Source'] = 'Your Data'
        
        combined_chart_df = pd.merge(chart_df_paper, chart_df_your[['Relevance Level', 'Number of Influencers Your Data', 'Source']], on='Relevance Level')
        combined_chart_df = combined_chart_df.melt(
            id_vars=['Relevance Level', 'Engagement Rate E(·)'], 
            value_vars=['Number of Influencers', 'Number of Influencers Your Data'],
            var_name='Source',
            value_name='Count'
        )
        combined_chart_df['Source'] = combined_chart_df['Source'].map({
            'Number of Influencers': 'Paper',
            'Number of Influencers Your Data': 'Your Data'
        })
        
        bar_chart = alt.Chart(combined_chart_df).mark_bar().encode(
            x=alt.X('Relevance Level:O', title='Relevance Level', sort=alt.SortField('Relevance Level', order='ascending')),
            y=alt.Y('Count:Q', title='Number of Influencers'),
            color=alt.Color('Source:N', title='データソース'),
            xOffset='Source:N',
            tooltip=['Relevance Level', 'Count', 'Source', 'Engagement Rate E(·)']
        ).properties(
            height=450
        ).interactive()

        st.altair_chart(bar_chart, use_container_width=True)

    else:
        st.warning(f"**{selected_month.strftime('%Y-%m')}** には投稿データがありません。他の月を選択してください。")

elif df is None:
    st.warning("データを読み込めませんでした。`preprocessed_posts_with_metadata.csv` と `influencers.txt` が正しい場所にあるか確認してください。")

else:
    st.info("データをロード中です...")
