import pandas as pd
import numpy as np

def collect_user_interactions():
    """
    Simulate collection of user interaction data.
    In practice, this would pull data from logs or a database.
    """
    # Example data
    data = pd.DataFrame({
        'user_id': [1, 2, 1, 3],
        'video_id': [101, 102, 103, 104],
        'click': [1, 1, 0, 1],
        'watch_time': [300, 200, 0, 400],
        'like': [1, 0, 0, 1],
        'share': [0, 0, 0, 1],
        'timestamp': pd.to_datetime(['2023-09-20 10:00', '2023-09-20 11:00', '2023-09-20 12:00', '2023-09-20 13:00'])
    })
    return data

def collect_content_metadata():
    """
    Simulate collection of content metadata.
    """
    metadata = pd.DataFrame({
        'video_id': [101, 102, 103, 104],
        'title': ['Funny Cats', 'Cooking Pasta', 'News Update', 'Gaming Highlights'],
        'description': ['Cats doing funny things', 'How to cook pasta', 'Latest news update', 'Top gaming moments'],
        'category': ['Pets', 'Food', 'News', 'Gaming'],
        'upload_date': pd.to_datetime(['2023-09-18', '2023-09-19', '2023-09-19', '2023-09-20']),
        'duration': [600, 900, 300, 1200]
    })
    return metadata

def collect_contextual_info():
    """
    Simulate collection of contextual information.
    """
    context = pd.DataFrame({
        'user_id': [1, 2, 1, 3],
        'device_type': ['Mobile', 'Desktop', 'Mobile', 'Tablet'],
        'location': ['USA', 'Canada', 'USA', 'UK']
    })
    return context

def feature_engineering(interactions, metadata, context):
    """
    Merge and process data to create feature set.
    """
    # Merge dataframes
    df = interactions.merge(metadata, on='video_id')
    df = df.merge(context, on='user_id')
    
    # Feature extraction
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Encode categorical variables
    df = pd.get_dummies(df, columns=['category', 'device_type', 'location'])
    
    # Normalize numerical features
    df['watch_time'] = df['watch_time'] / df['duration']
    df['watch_time'].fillna(0, inplace=True)
    
    # Drop unnecessary columns
    df.drop(['title', 'description', 'timestamp', 'upload_date'], axis=1, inplace=True)
    
    return df

if __name__ == "__main__":
    interactions = collect_user_interactions()
    metadata = collect_content_metadata()
    context = collect_contextual_info()
    features = feature_engineering(interactions, metadata, context)
    print(features.head())