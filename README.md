# Stock Price Prediction with LSTM

A deep learning project that uses an LSTM neural network to predict stock prices based on historical data from the Alpha Vantage API.

## Features

- Downloads historical adjusted close prices for any stock symbol
- Normalizes data and creates time series windows for LSTM input
- Trains a 2-layer LSTM model with PyTorch
- Visualizes training/validation splits and predictions
- Predicts the next trading day's closing price

## Requirements

See `requirements.txt` for dependencies (PyTorch, Alpha Vantage, NumPy, Matplotlib)

## Usage

```bash
python project.py
```

Modify the `config` dictionary in `project.py` to change:

- Stock symbol (default: IBM)
- Window size for time series (default: 20 days)
- Model hyperparameters (LSTM layers, size, dropout)
- Training parameters (batch size, epochs, learning rate)

## Notes

- Uses the Alpha Vantage demo API key by default (get your own at https://www.alphavantage.co/support/#api-key)
- Configure `device` in training config for CPU/GPU/Apple Silicon acceleration
- Training on CPU is recommended for this small dataset
