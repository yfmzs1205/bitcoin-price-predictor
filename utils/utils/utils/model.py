import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_and_train_model(df):
    feature_cols = ["close", "ma5", "ma10", "rsi", "return_1min", "return_3min"]
    sequence_length = 10
    df = df.copy()
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    X, y = [], []
    for i in range(len(df) - sequence_length):
        X.append(df[feature_cols].iloc[i:i+sequence_length].values)
        y.append(df["target"].iloc[i+sequence_length])

    X, y = np.array(X), np.array(y)
    model = Sequential([
        LSTM(64, input_shape=(X.shape[1], X.shape[2])),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    return model, scaler
