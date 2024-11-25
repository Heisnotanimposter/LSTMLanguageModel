# serving.py

import numpy as np
from tensorflow.keras.models import load_model
from shallow_tower import build_shallow_tower
from data_collection import feature_engineering

def load_models():
    """
    Load the trained models.
    """
    shallow_tower = build_shallow_tower(input_shape=128)
    multi_task_model = load_model('best_model.h5')
    return shallow_tower, multi_task_model

def generate_scores(shallow_tower, multi_task_model, user_features):
    """
    Generate scores for videos.
    """
    shallow_output = shallow_tower.predict(user_features)
    predictions = multi_task_model.predict(shallow_output)
    
    # Combine scores
    click_score = predictions[0]
    watch_time_score = predictions[1]
    like_score = predictions[2]
    
    combined_score = 0.5 * click_score + 0.3 * watch_time_score + 0.2 * like_score
    return combined_score

def recommend_videos(user_id, candidate_videos):
    """
    Recommend top-N videos to the user.
    """
    # Load models
    shallow_tower, multi_task_model = load_models()
    
    # Collect user features
    interactions = collect_user_interactions()  # From data_collection.py
    metadata = collect_content_metadata()
    context = collect_contextual_info()
    features = feature_engineering(interactions, metadata, context)
    
    # Filter features for the user and candidate videos
    user_features = features[(features['user_id'] == user_id) & (features['video_id'].isin(candidate_videos))]
    
    # Generate scores
    scores = generate_scores(shallow_tower, multi_task_model, user_features)
    
    # Rank videos
    user_features['score'] = scores
    top_videos = user_features.sort_values('score', ascending=False)['video_id'].tolist()
    return top_videos

if __name__ == "__main__":
    user_id = 1
    candidate_videos = [101, 102, 103, 104]
    recommendations = recommend_videos(user_id, candidate_videos)
    print("Recommended Videos:", recommendations)