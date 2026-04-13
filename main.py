import pandas as pd
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST
from neuralforecast.utils import AirPassengersDF

def main():
    print("🚀 Initializing Modern Time Series Forecasting Pipeline...")
    
    # 1. Load sample dataset (AirPassengers)
    # NeuralForecast expects columns: unique_id, ds (datetime), y (value)
    df = AirPassengersDF
    df['unique_id'] = 1.0 # Single time series
    
    # 2. Split data into train and test
    # AirPassengers has 144 observations. Use 120 for train, 24 for test.
    Y_train_df = df.iloc[:120,:].copy()
    Y_test_df = df.iloc[120:,:].copy()
    
    # 3. Initialize PatchTST Model
    # 2025 Standard Config: 
    # - h (horizon): 24 steps
    # - input_size: look-back window of 48
    # - patch_len: 16 (from paper)
    # - stride: 8 (from paper)
    model = PatchTST(h=24, 
                    input_size=48, 
                    patch_len=16, 
                    stride=8,
                    max_steps=100) # Fast training for demo
    
    # 4. Fit Model
    nf = NeuralForecast(models=[model], freq='ME')
    print("📈 Training PatchTST...")
    nf.fit(df=Y_train_df)
    
    # 5. Predict
    print("🔮 Generating Forecast...")
    Y_hat_df = nf.predict()
    
    # 6. Visualize results
    print("🖼️ Visualizing Results...")
    plt.figure(figsize=(12, 6))
    plt.plot(Y_train_df['ds'], Y_train_df['y'], label='Train', color='#2E86C1')
    plt.plot(Y_test_df['ds'], Y_test_df['y'], label='Actual', color='#AED6F1')
    plt.plot(Y_hat_df['ds'], Y_hat_df['PatchTST'], label='PatchTST Forecast', color='#E74C3C', linestyle='--')
    
    plt.title('PatchTST: Transformer-based Time Series Forecasting', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Passengers', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save the plot
    plt.savefig('forecast_results.png')
    print("✅ Forecast complete. Results saved to 'forecast_results.png'.")

if __name__ == "__main__":
    main()
