import numpy as np
from utils.data_loader import get_latest_data
from utils.indicators import add_indicators

def predict_trend(model, scaler, interval="1m", shift_period=3):
    df = get_latest_data(interval=interval)
    df = add_indicators(df)
    feature_cols = ["close", "ma5", "ma10", "rsi", "return_1min", "return_3min"]
    df[feature_cols] = scaler.transform(df[feature_cols])
    sequence = df[feature_cols].iloc[-10:].values
    X = np.expand_dims(sequence, axis=0)
    prob = model.predict(X)[0][0]
    return ("上涨" if prob > 0.5 else "下跌", prob if prob > 0.5 else 1 - prob)
