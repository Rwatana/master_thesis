import pandas as pd
import os
import json
from datetime import datetime
import plotly.express as px
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed # 並列処理ライブラリをインポート

# --- データ読み込み・前処理関数 (Streamlit非依存) ---

def load_influencer_data(filepath):
    """influencers.txtを読み込む"""
    try:
        df = pd.read_csv(filepath, sep='\t', skiprows=[1])
        print(f"✅ Successfully loaded '{filepath}'.")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File not found at '{filepath}'.")
        return None

def process_single_file(filepath):
    """
    単一の.infoファイルを処理し,辞書として結果を返すワーカー関数
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        caption_edges = data.get('edge_media_to_caption', {}).get('edges', [])
        caption = caption_edges[0]['node']['text'] if caption_edges else ""
        timestamp = data.get('taken_at_timestamp', 0)
        dt_object = datetime.fromtimestamp(timestamp)
        
        return {
            'username': data.get('owner', {}).get('username', ''),
            'likes': data.get('edge_media_preview_like', {}).get('count', 0),
            'comments': data.get('edge_media_to_parent_comment', {}).get('count', 0),
            'caption_length': len(caption),
            'tag_count': len(data.get('edge_media_to_tagged_user', {}).get('edges', [])),
            'weekday': dt_object.strftime('%A'),
            'hour': dt_object.hour
        }
    except (json.JSONDecodeError, KeyError, FileNotFoundError):
        return None # エラーが発生した場合はNoneを返す

def load_all_post_metadata(info_dir, influencers_df):
    """
    指定されたカテゴリの全投稿の.infoファイルを並列処理で読み込む
    """
    if influencers_df is None or influencers_df.empty:
        print("⚠️ Warning: Influencer dataframe is empty. Cannot process post metadata.")
        return pd.DataFrame()

    usernames_to_process = set(influencers_df['Username'])
    
    try:
        all_files = os.listdir(info_dir)
        print(f"Found {len(all_files)} total files in '{info_dir}'.")
    except FileNotFoundError:
        print(f"❌ Error: Post data directory '{info_dir}' not found.")
        return pd.DataFrame()
    
    # 処理対象のファイルパスリストを作成
    filepaths_to_process = [
        os.path.join(info_dir, f) 
        for f in all_files if f.split('-')[0] in usernames_to_process and f.endswith('.info')
    ]
    print(f"Found {len(filepaths_to_process)} relevant post files to process for the selected categories.")

    all_post_details = []
    # ProcessPoolExecutorを使用してファイルを並列で処理
    with ProcessPoolExecutor() as executor:
        # futureオブジェクトの辞書を作成
        future_to_filepath = {executor.submit(process_single_file, fp): fp for fp in filepaths_to_process}
        
        # tqdmで進捗を表示しながら完了したタスクを処理
        for future in tqdm(as_completed(future_to_filepath), total=len(filepaths_to_process), desc="Processing post metadata"):
            result = future.result()
            if result is not None: # 正常に処理できた結果のみを追加
                all_post_details.append(result)
        
    return pd.DataFrame(all_post_details)

def main(categories):
    """メインの分析処理を実行し,結果をファイルに保存する"""
    
    # --- 1. 出力ディレクトリの作成 ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"analysis_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n🚀 Starting analysis for categories: {', '.join(categories)}")
    print(f"💾 Results will be saved to: '{output_dir}/'")
    
    summary_results = {}

    # --- 2. データの準備 ---
    df_influencers_full = load_influencer_data('influencers.txt')
    info_dir = 'posts_info/unzipped_data_7z/info/'

    if df_influencers_full is None:
        print("❌ Critical Error: Could not load 'influencers.txt'. Aborting.")
        return

    df_influencers_filtered = df_influencers_full[df_influencers_full['Category'].isin(categories)]
    df_metadata = load_all_post_metadata(info_dir, df_influencers_filtered)

    if df_metadata.empty:
        print("⚠️ Warning: No post data found for the selected categories. Aborting.")
        return

    # --- 3. 分析の実行と結果の保存 ---
    
    print("\n[Analysis 1/3] Analyzing caption length vs. engagement...")
    correlation = df_metadata['caption_length'].corr(df_metadata['likes'])
    summary_results['caption_vs_likes'] = {'correlation': correlation}
    fig_caption = px.scatter(
        df_metadata, x='caption_length', y='likes', trendline='ols',
        trendline_color_override='red', title='Caption Length vs. Likes',
        labels={'caption_length': 'Caption Length', 'likes': 'Like Count'}
    )
    fig_caption.write_html(os.path.join(output_dir, "1_caption_vs_likes.html"))
    print(f"  - Correlation: {correlation:.3f}")
    print(f"  - Saved graph to '1_caption_vs_likes.html'")

    print("\n[Analysis 2/3] Analyzing user tags vs. engagement...")
    df_metadata['has_tags'] = df_metadata['tag_count'] > 0
    avg_likes_with_tags = df_metadata[df_metadata['has_tags'] == True]['likes'].mean()
    avg_likes_without_tags = df_metadata[df_metadata['has_tags'] == False]['likes'].mean()
    summary_results['user_tags_vs_likes'] = {
        'avg_likes_with_tags': avg_likes_with_tags,
        'avg_likes_without_tags': avg_likes_without_tags
    }
    fig_tags = px.box(
        df_metadata, x='has_tags', y='likes', title='User Tags vs. Likes Distribution',
        labels={'has_tags': 'Has User Tags', 'likes': 'Like Count'}
    )
    fig_tags.write_html(os.path.join(output_dir, "2_tags_vs_likes.html"))
    print(f"  - Avg Likes (With Tags): {avg_likes_with_tags:,.0f}")
    print(f"  - Avg Likes (Without Tags): {avg_likes_without_tags:,.0f}")
    print(f"  - Saved graph to '2_tags_vs_likes.html'")

    print("\n[Analysis 3/3] Analyzing post timing vs. engagement...")
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    avg_likes_by_weekday = df_metadata.groupby('weekday')['likes'].mean().reindex(weekday_order)
    avg_likes_by_hour = df_metadata.groupby('hour')['likes'].mean()
    summary_results['post_timing_vs_likes'] = {
        'avg_likes_by_weekday': avg_likes_by_weekday.to_dict(),
        'avg_likes_by_hour': avg_likes_by_hour.to_dict()
    }
    fig_weekday = px.bar(avg_likes_by_weekday, y='likes', title='Average Likes by Day of Week')
    fig_weekday.write_html(os.path.join(output_dir, "3a_avg_likes_by_weekday.html"))
    fig_hour = px.bar(avg_likes_by_hour, y='likes', title='Average Likes by Hour of Day')
    fig_hour.write_html(os.path.join(output_dir, "3b_avg_likes_by_hour.html"))
    heatmap_data = df_metadata.pivot_table(index='weekday', columns='hour', values='likes', aggfunc='mean').reindex(weekday_order)
    fig_heatmap = px.imshow(heatmap_data, labels=dict(x="Hour of Day", y="Day of Week", color="Avg Likes"))
    fig_heatmap.write_html(os.path.join(output_dir, "3c_heatmap_weekday_hour.html"))
    print(f"  - Saved weekday, hour, and heatmap graphs.")

    with open(os.path.join(output_dir, "summary_metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=4, ensure_ascii=False)
    print(f"\n✅ Analysis complete. All results saved in '{output_dir}/'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run metadata engagement analysis and save results to files."
    )
    parser.add_argument(
        '--categories',
        nargs='+',
        default=['beauty', 'fashion'],
        help="A list of categories to analyze (e.g., --categories beauty fashion travel)"
    )
    args = parser.parse_args()
    
    main(args.categories)

