import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- دالة جلب بيانات السهم (وهمية هنا، يجب استبدالها بجلب البيانات الحقيقية) ---
def get_stock_data(ticker):
    # هنا مكان استدعاء API لجلب بيانات يومية: Open, High, Low, Close, ... الخ
    # للعرض نستخدم بيانات وهمية:
    data = {
        'Open': [100, 102, 101, 98, 99, 100, 102, 103, 104, 105],
        'High': [103, 105, 104, 100, 102, 104, 106, 107, 108, 109],
        'Low': [99, 101, 97, 95, 98, 99, 101, 102, 103, 104],
        'Close': [102, 101, 98, 99, 101, 103, 105, 106, 107, 108],
        'RSI': np.random.uniform(30, 70, 10),
        'SMA_20': np.random.uniform(100, 105, 10),
        'SMA_50': np.random.uniform(99, 104, 10),
        'MACD': np.random.uniform(-1, 1, 10),
        'Target': [1, 0, 0, 1, 1, 1, 0, 1, 1, 0]  # صعود=1، هبوط=0
    }
    df = pd.DataFrame(data)
    return df

# --- دالة تدريب النموذج ---
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

# --- دالة كشف أنماط الشموع ---
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

        # Pin Bar: جسم صغير وظل علوي أو سفلي طويل
        if body < candle_range * 0.3 and (upper_shadow > body * 2 or lower_shadow > body * 2):
            pattern = "Pin Bar"
        
        # Doji: جسم صغير جداً
        elif body < candle_range * 0.1:
            pattern = "Doji"
        
        # Hammer: جسم صغير وظل سفلي طويل وظل علوي صغير
        elif lower_shadow > body * 2 and upper_shadow < body:
            pattern = "Hammer"
        
        # Engulfing يحتاج يومين
        if i > 0:
            prev_o = df['Open'].iloc[i-1]
            prev_c = df['Close'].iloc[i-1]
            # صعودي Engulfing
            if prev_c < prev_o and c > o and o < prev_c and c > prev_o:
                pattern = "Bullish Engulfing"
            # هبوطي Engulfing
            elif prev_c > prev_o and c < o and o > prev_c and c < prev_o:
                pattern = "Bearish Engulfing"

        patterns.append(pattern)
    return patterns

# --- واجهة Streamlit ---

st.set_page_config(page_title="AI Stock Predictor with Candlestick Patterns", layout="wide")
st.title("🔮 التنبؤ باتجاه السهم مع تحليل الشموع اليابانية")

ticker = st.text_input("ادخل رمز السهم", "AAPL")

if st.button("ابدأ التحليل"):
    df = get_stock_data(ticker)

    # تدريب النموذج والتنبؤ
    model, acc = train_predictor(df)
    st.success(f"✅ دقة النموذج: {acc:.2%}")

    latest = df[["RSI", "SMA_20", "SMA_50", "MACD"]].iloc[-1:]
    prediction = model.predict(latest)[0]

    if prediction == 1:
        st.markdown(f"📈 السهم <b>{ticker}</b> متوقع له <b>الصعود</b>", unsafe_allow_html=True)
    else:
        st.markdown(f"📉 السهم <b>{ticker}</b> متوقع له <b>الهبوط</b>", unsafe_allow_html=True)

    # تحليل أنماط الشموع اليابانية
    df['Pattern'] = detect_candlestick_patterns(df)
    st.subheader("أنماط الشموع اليابانية المكتشفة")
    st.write(df[['Open', 'High', 'Low', 'Close', 'Pattern']])
