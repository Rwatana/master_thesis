import streamlit as st
import pandas as pd

st.set_page_config(page_title="分析結果サマリー", layout="wide")

# 提供されたJSONデータをPythonの辞書として定義
analysis_data = {
    "caption_vs_likes": {
        "correlation": -0.050074022394243585
    },
    "user_tags_vs_likes": {
        "avg_likes_with_tags": 5441.736228807884,
        "avg_likes_without_tags": 7446.779368108446
    },
    "post_timing_vs_likes": {
        "avg_likes_by_weekday": {
            "Monday": 5934.782413198788, "Tuesday": 5821.494517164844,
            "Wednesday": 5969.346745447232, "Thursday": 5822.5932676999055,
            "Friday": 5772.397010001317, "Saturday": 6446.986654674323,
            "Sunday": 6661.156781463716
        },
        "avg_likes_by_hour": {
            "0": 5702.099074377457, "1": 5777.851555132034, "2": 6136.332077098004,
            "3": 6241.13620205045, "4": 6543.954955857834, "5": 7147.143716411398,
            "6": 7986.126092074658, "7": 7942.611070678333, "8": 8326.850804557711,
            "9": 7199.347385940529, "10": 6345.5030149690565, "11": 6648.739351756784,
            "12": 6441.483244523387, "13": 6942.844894337796, "14": 6043.841918815858,
            "15": 4180.100955395793, "16": 3353.141281270878, "17": 3444.1781392235607,
            "18": 3791.408659783629, "19": 4081.575317070965, "20": 4410.014119010348,
            "21": 5063.685499351492, "22": 4917.2127669206975, "23": 5556.761169464439
        }
    }
}

# --- UI描画 ---
st.title("📊 分析結果サマリー")
st.write("投稿メタデータとエンゲージメント（いいね数）の関係性についての分析結果です。")

tab1, tab2, tab3 = st.tabs(["キャプション", "ユーザータグ", "投稿時間"])

# --- タブ1: キャプションの長さ ---
with tab1:
    st.header("✍️ キャプションの長さ")
    corr = analysis_data["caption_vs_likes"]["correlation"]
    
    st.metric("キャプション長と「いいね数」の相関係数", f"{corr:.3f}")
    
    if -0.2 < corr < 0.2:
        st.info("**考察**: キャプションの長さと「いいね数」には、ほとんど相関関係は見られませんでした。")
    elif corr >= 0.2:
        st.success("**考察**: キャプションが長いほど「いいね」が多くなる、弱い正の相関が見られます。")
    else:
        st.success("**考察**: キャプションが短いほど「いいね」が多くなる、弱い負の相関が見られます。")

# --- タブ2: ユーザータグ ---
with tab2:
    st.header("👥 ユーザータグ")
    with_tags = analysis_data["user_tags_vs_likes"]["avg_likes_with_tags"]
    without_tags = analysis_data["user_tags_vs_likes"]["avg_likes_without_tags"]

    col1, col2 = st.columns(2)
    col1.metric("タグあり投稿の平均いいね", f"{with_tags:,.0f}")
    col2.metric("タグなし投稿の平均いいね", f"{without_tags:,.0f}", delta=f"{without_tags - with_tags:,.0f}")
    
    if without_tags > with_tags:
        st.warning("**考察**: このデータセットでは、意外にもユーザータグを**付けない方が**平均いいね数が多いという結果になりました。")
    else:
        st.success("**考察**: ユーザータグを付けることで、平均いいね数が多くなる傾向が見られます。")
        
    # 比較のための棒グラフ
    df_tags = pd.DataFrame({
        '種類': ['タグあり', 'タグなし'],
        '平均いいね数': [with_tags, without_tags]
    }).set_index('種類')
    st.bar_chart(df_tags)


# --- タブ3: 投稿時間 ---
with tab3:
    st.header("🕒 投稿時間")
    
    st.subheader("曜日ごとの傾向")
    weekday_data = analysis_data["post_timing_vs_likes"]["avg_likes_by_weekday"]
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    df_weekday = pd.DataFrame(list(weekday_data.items()), columns=['曜日', '平均いいね数'])
    df_weekday['曜日'] = pd.Categorical(df_weekday['曜日'], categories=weekday_order, ordered=True)
    df_weekday = df_weekday.sort_values('曜日').set_index('曜日')
    
    st.bar_chart(df_weekday)
    st.success("**考察**: 週末（特に日曜日）のエンゲージメントが最も高い傾向があります。")

    st.subheader("時間帯ごとの傾向")
    hour_data = analysis_data["post_timing_vs_likes"]["avg_likes_by_hour"]
    
    df_hour = pd.DataFrame(list(hour_data.items()), columns=['時間', '平均いいね数'])
    df_hour['時間'] = df_hour['時間'].astype(int)
    df_hour = df_hour.sort_values('時間').set_index('時間')

    st.bar_chart(df_hour)
    st.success("**考察**: 早朝から午前中（特に8時台）にかけてエンゲージメントが高く、夕方にかけて一度落ち込む傾向が見られます。")
