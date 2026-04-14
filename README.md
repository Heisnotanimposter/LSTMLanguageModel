# Time Series Forecasting Excellence 2025

This repository has been modernized from a legacy LSTM placeholder into a state-of-the-art Time Series Forecasting project based on 2025 research. It leverages **PatchTST** (Transformer-based patching) and **Foundation Models** for superior accuracy in long-term forecasting.

## 🚀 Key Architectures

### 1. PatchTST (Patch Time Series Transformer)
PatchTST is the primary model in this repository. Unlike traditional LSTMs or point-wise Transformers, it:
- **Segments** time series into subseries-level patches.
- Uses **Channel-Independence** to process multivariate data more effectively.
- Reduces complexity from $O(L^2)$ to $O(N^2)$ where $N \approx L/S$, allowing for much longer look-back windows.

### 2. Foundation Models (Chronos)
Integrated support for Amazon's **Chronos**, which treats time series forecasting as a language modeling task. It offers high-quality zero-shot performance on unseen datasets.

## 📁 Research & Documentation
The `papers/` directory contains the foundational research for this implementation:
- `patchtst.pdf`: A Time Series is Worth 64 Words.
- `chronos.pdf`: Learning the Language of Time Series.
- `itransformer.pdf`: Inverted Transformers for Time Series.

## 🛠️ Getting Started

### Prerequisites
Install the required Python packages:
```bash
pip install -r requirements.txt
```

### Usage
Run the main pipeline to train PatchTST on a sample dataset and visualize the forecast:
```bash
python main.py
```

## 📈 Benchmarks
Based on the provided research, this architecture is expected to outperform LSTMs by **20-40%** on standard benchmarks like ETT, Traffic, and Electricity.


