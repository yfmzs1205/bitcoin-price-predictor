import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.data import fetch_binance_klines
from utils.model import train_xgb_model, predict_trend

st.set_page_config(page_title="比特币价格趋势预测", layout="wide")
st.title("📈 比特币价格趋势预测系统")

interval = st.selectbox("选择预测周期", ["1m", "3m", "5m"])
df = fetch_binance_klines(symbol="BTCUSDT", interval=interval, limit=200)

st.subheader("历史 K 线图")
fig = go.Figure(data=[
    go.Candlestick(
        x=df["timestamp"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"]
    )
])
st.plotly_chart(fig, use_container_width=True)

st.subheader("模型训练与趋势预测")
feature_df = df.drop(columns=["timestamp", "target"])
model, acc, report = train_xgb_model(df.drop(columns=["timestamp"]), target_column='target')
st.success(f"模型训练完成，准确率约为：{acc*100:.2f}%")

st.subheader("📊 预测结果")
latest = feature_df.iloc[[-1]]
prediction, proba = predict_trend(model, latest)
trend = "📈 上涨" if prediction == 1 else "📉 下跌"
confidence = proba[1]*100 if prediction == 1 else proba[0]*100
st.metric(label="最新预测趋势", value=trend, delta=f"{confidence:.2f}%")

st.subheader("📈 历史趋势打分报告")
st.dataframe(pd.DataFrame(report).transpose().round(2))
