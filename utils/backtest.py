import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.data_loader import get_latest_data
from utils.indicators import add_indicators, create_labels
from utils.model import build_and_train_model

def backtest_model(shift_period=3):
    df = get_latest_data()
    df = add_indicators(df)
    df = create_labels(df, shift_period=shift_period)

    feature_cols = ["close", "ma5", "ma10", "rsi", "return_1min", "return_3min"]
    sequence_length = 10

    X, y = [], []
    for i in range(len(df) - sequence_length):
        X.append(df[feature_cols].iloc[i:i+sequence_length].values)
        y.append(df["target"].iloc[i+sequence_length])

    X, y = np.array(X), np.array(y)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model, _ = build_and_train_model(df.iloc[:split + sequence_length])
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    return {
        "准确率": accuracy_score(y_test, y_pred),
        "精确率": precision_score(y_test, y_pred),
        "召回率": recall_score(y_test, y_pred),
        "F1分数": f1_score(y_test, y_pred)
    }
