import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import requests

# إعداد الصفحة
st.set_page_config(page_title="AI Stock Predictor", layout="wide")
st.title("🔮 التنبؤ باتجاه السهم")

# إدخال المستخدم
ticker = st.text_input("ادخل رمز السهم", "AAPL")
bot_token = st.text_input("Telegram Bot Token", type="password")
chat_id = st.text_input("Telegram Chat ID")

# تحميل البيانات
@st.cache
def get_stock_data(ticker):
    df = yf.download(ticker, period="1y", interval="1d")
    df['RSI'] = compute_rsi(df['Close'])
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df = df.dropna()
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# تدريب النموذج
def train_predictor(df):
    X = df[['RSI', 'SMA_20', 'SMA_50', 'MACD']]
    y = (df['Close'].shift(-1) > df['Close']).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc

# رسم البيانات
def plot_stock_data(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'],
                name="Candlesticks"))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='SMA 20'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='SMA 50'))
    st.plotly_chart(fig)

# إرسال تنبيه عبر Telegram
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    params = {'chat_id': chat_id, 'text': message, 'parse_mode': 'HTML'}
    response = requests.get(url, params=params)
    return response

# تنفيذ التحليل
if st.button("ابدأ التحليل"):
    df = get_stock_data(ticker)
    model, acc = train_predictor(df)
    st.success(f"✅ دقة النموذج: {acc:.2%}")
    plot_stock_data(df)

    latest = df[["RSI", "SMA_20", "SMA_50", "MACD"]].iloc[-1:]
    prediction = model.predict(latest)[0]

    if prediction == 1:
        msg = f"📈 السهم <b>{ticker}</b> متوقع له <b>الصعود</b>"
        st.markdown(msg, unsafe_allow_html=True)
        if bot_token and chat_id:
            send_telegram_message(msg)
    else:
        msg = f"📉 السهم <b>{ticker}</b> متوقع له <b>الهبوط</b>"
        st.markdown(msg, unsafe_allow_html=True)
        if bot_token and chat_id:
            send_telegram_message(msg)
