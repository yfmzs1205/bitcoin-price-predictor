import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.data_loader import get_latest_data
from utils.indicators import add_indicators, create_labels
from utils.model import build_and_train_model
from utils.predictor import predict_trend
from utils.backtest import backtest_model

st.set_page_config(page_title="比特币价格趋势预测系统", layout="wide")

st.title("📈 比特币价格趋势预测系统")
st.markdown("支持预测未来 **1分钟、3分钟、5分钟** 的趋势方向，提供可视化图表与交易建议。")

@st.cache_resource(show_spinner=True)
def prepare_models():
    df = get_latest_data()
    df = add_indicators(df)
    df1 = create_labels(df.copy(), shift_period=1)
    df3 = create_labels(df.copy(), shift_period=3)
    df5 = create_labels(df.copy(), shift_period=5)
    m1, s1 = build_and_train_model(df1)
    m3, s3 = build_and_train_model(df3)
    m5, s5 = build_and_train_model(df5)
    return m1, s1, m3, s3, m5, s5, df

with st.spinner("正在训练模型并准备预测..."):
    model1, scaler1, model3, scaler3, model5, scaler5, df = prepare_models()

trend1, conf1 = predict_trend(model1, scaler1, interval="1m")
trend3, conf3 = predict_trend(model3, scaler3, interval="1m")
trend5, conf5 = predict_trend(model5, scaler5, interval="1m")

# 展示指标
st.subheader("📊 趋势预测结果")
cols = st.columns(3)
cols[0].metric("未来 1 分钟", trend1, f"{conf1:.2%}")
cols[1].metric("未来 3 分钟", trend3, f"{conf3:.2%}")
cols[2].metric("未来 5 分钟", trend5, f"{conf5:.2%}")

# 智能建议
def get_suggestion(*args):
    up_count = sum([t == "上涨" for t in args[::2]])
    avg_conf = sum(args[1::2]) / 3
    if up_count >= 2 and avg_conf > 0.55:
        return "📈 建议：做多（多数预测上涨，置信度较高）"
    elif up_count <= 1 and avg_conf > 0.55:
        return "📉 建议：做空（多数预测下跌，置信度较高）"
    else:
        return "⏸ 建议：观望（预测结果不一致或信心不足）"

st.subheader("🧠 交易建议")
st.warning(get_suggestion(trend1, conf1, trend3, conf3, trend5, conf5))

# 可视化：K线图
st.subheader("🕯️ 最近价格 K 线图（1分钟）")
df_plot = df.tail(100).copy()
fig = go.Figure(data=[go.Candlestick(
    x=df_plot["timestamp"],
    open=df_plot["open"],
    high=df_plot["high"],
    low=df_plot["low"],
    close=df_plot["close"],
    increasing_line_color='green',
    decreasing_line_color='red'
)])
fig.update_layout(xaxis_rangeslider_visible=False, height=400, margin=dict(t=10, b=10))
st.plotly_chart(fig, use_container_width=True)

# 回测结果展示
st.subheader("📈 模型回测结果")
backtest_results = backtest_model(shift_period=3)
for metric, value in backtest_results.items():
    st.write(f"{metric}: {value:.2f}")

st.caption("数据来源：Binance | 模型类型：LSTM | 训练周期：最近500分钟 | 更新频率：每次刷新页面")
