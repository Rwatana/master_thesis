import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def calculate_growth_for_user(user_data):
    """
    単一ユーザーのデータフレームを受け取り、成長率を計算するワーカー関数。
    並列処理のために独立したファイルに定義されています。
    """
    username, user_df = user_data
    if len(user_df) < 2:
        return None

    # データ型が失われる可能性があるため、datetime型に再変換
    user_df = user_df.copy()
    user_df['datetime'] = pd.to_datetime(user_df['datetime'])

    user_df = user_df.sort_values('datetime')
    start_date = user_df['datetime'].min()
    user_df['days_since_start'] = (user_df['datetime'] - start_date).dt.days
    
    model = LinearRegression()
    X = user_df[['days_since_start']]
    
    # likesの傾き
    model.fit(X, user_df['likes'])
    likes_slope = model.coef_[0]
    
    # commentsの傾き
    model.fit(X, user_df['comments'])
    comments_slope = model.coef_[0]

    return {
        'username': username,
        'likes_growth_rate': likes_slope,
        'comments_growth_rate': comments_slope
    }

