def add_indicators(df):
    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma10"] = df["close"].rolling(window=10).mean()
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))
    df["return_1min"] = df["close"].pct_change(1)
    df["return_3min"] = df["close"].pct_change(3)
    df = df.dropna()
    return df

def create_labels(df, shift_period=3):
    df["target"] = (df["close"].shift(-shift_period) > df["close"]).astype(int)
    return df.dropna()
