# training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from shallow_tower import build_shallow_tower
from multi_task_learning import build_multi_task_model, compile_multi_task_model

def train_model(features, labels):
    """
    Train the recommendation model.
    """
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2)
    
    # Build models
    shallow_tower = build_shallow_tower(X_train.shape[1])
    shallow_output_train = shallow_tower.predict(X_train)
    shallow_output_val = shallow_tower.predict(X_val)
    
    multi_task_model = build_multi_task_model(shallow_output_train.shape[1])
    multi_task_model = compile_multi_task_model(multi_task_model)
    
    # Callbacks
    early_stopping = EarlyStopping(patience=3)
    model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
    
    # Train
    history = multi_task_model.fit(
        shallow_output_train, 
        y_train, 
        validation_data=(shallow_output_val, y_val),
        epochs=20,
        callbacks=[early_stopping, model_checkpoint]
    )
    return multi_task_model

if __name__ == "__main__":
    # Simulate features and labels
    features = pd.DataFrame(np.random.rand(100, 128))
    labels = {
        'click_output': np.random.randint(2, size=(100, 1)),
        'watch_time_output': np.random.rand(100, 1),
        'like_output': np.random.randint(2, size=(100, 1))
    }
    model = train_model(features, labels)