import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go

# جلب بيانات السهم (وهمية - استبدلها لاحقاً ب API حقيقي)
def get_stock_data(ticker):
    dates = pd.date_range(end=pd.Timestamp.today(), periods=30)
    data = {
        'Date': dates,
        'Open': np.random.uniform(100, 110, 30),
        'High': np.random.uniform(110, 115, 30),
        'Low': np.random.uniform(95, 105, 30),
        'Close': np.random.uniform(100, 110, 30),
        'RSI': np.random.uniform(30, 70, 30),
        'SMA_20': np.random.uniform(100, 108, 30),
        'SMA_50': np.random.uniform(99, 107, 30),
        'MACD': np.random.uniform(-1, 1, 30),
        'Target': np.random.choice([0,1], 30)  # عشوائي صعود/هبوط
    }
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    return df

# تدريب النموذج
def train_predictor(df):
    features = ['RSI', 'SMA_20', 'SMA_50', 'MACD']
    X = df[features]
    y = df['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc

# كشف أنماط الشموع
def detect_candlestick_patterns(df):
    patterns = []
    for i in range(len(df)):
        o = df['Open'].iloc[i]
        h = df['High'].iloc[i]
        l = df['Low'].iloc[i]
        c = df['Close'].iloc[i]
        body = abs(c - o)
        candle_range = h - l
        upper_shadow = h - max(c, o)
        lower_shadow = min(c, o) - l

        pattern = ""

        if body < candle_range * 0.3 and (upper_shadow > body * 2 or lower_shadow > body * 2):
            pattern = "Pin Bar"
        elif body < candle_range * 0.1:
            pattern = "Doji"
        elif lower_shadow > body * 2 and upper_shadow < body:
            pattern = "Hammer"

        if i > 0:
            prev_o = df['Open'].iloc[i-1]
            prev_c = df['Close'].iloc[i-1]
            if prev_c < prev_o and c > o and o < prev_c and c > prev_o:
                pattern = "Bullish Engulfing"
            elif prev_c > prev_o and c < o and o > prev_c and c < prev_o:
                pattern = "Bearish Engulfing"

        patterns.append(pattern)
    return patterns

# رسم الشموع باستخدام Plotly
def plot_candlestick(df):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='الشموع اليابانية'
    )])
    fig.update_layout(xaxis_rangeslider_visible=False, height=400)
    return fig

# رسم RSI
def plot_rsi(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    fig.update_layout(height=250, title="مؤشر RSI")
    return fig

# الصفحة الرئيسية

st.set_page_config(page_title="تحليل الأسهم المتقدم مع AI والشموع", layout="wide")
st.title("📊 تحليل الأسهم المتقدم مع التنبؤ والشموع اليابانية")

ticker = st.text_input("ادخل رمز السهم (مثال: AAPL)", "AAPL")

if st.button("ابدأ التحليل"):
    with st.spinner("جلب البيانات وتدريب النموذج..."):
        df = get_stock_data(ticker)
        model, acc = train_predictor(df)
        df['Pattern'] = detect_candlestick_patterns(df)

    st.success(f"✅ دقة النموذج: {acc:.2%}")

    latest = df[["RSI", "SMA_20", "SMA_50", "MACD"]].iloc[-1:]
    prediction = model.predict(latest)[0]

    if prediction == 1:
        st.markdown(f"📈 التوقع: السهم <b>{ticker}</b> متجه للصعود", unsafe_allow_html=True)
    else:
        st.markdown(f"📉 التوقع: السهم <b>{ticker}</b> متجه للهبوط", unsafe_allow_html=True)

    # تخطيط الشموع اليابانية و RSI بشكل جانبي
    col1, col2 = st.columns([3,1])

    with col1:
        st.subheader("مخطط الشموع اليابانية")
        st.plotly_chart(plot_candlestick(df), use_container_width=True)

        st.subheader("الأنماط المكتشفة للشموع")
        st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Pattern']])

    with col2:
        st.subheader("مؤشرات فنية")
        st.plotly_chart(plot_rsi(df), use_container_width=True)

        st.markdown("### ملخص المؤشرات")
        st.write(f"- متوسط RSI الحالي: {df['RSI'].iloc[-1]:.2f}")
        st.write(f"- متوسط SMA_20 الحالي: {df['SMA_20'].iloc[-1]:.2f}")
        st.write(f"- متوسط SMA_50 الحالي: {df['SMA_50'].iloc[-1]:.2f}")
        st.write(f"- مؤشر MACD الحالي: {df['MACD'].iloc[-1]:.2f}")
