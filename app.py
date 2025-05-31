import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import yfinance as yf
import ta  # مكتبة التحليل الفني

# إعداد واجهة الصفحة
st.set_page_config(page_title="AI Stock Predictor", layout="wide")
st.title("🔮 التنبؤ باتجاه السهم")

# ----------------- الدوال -----------------

# تحميل بيانات السهم وتحليلها
def get_stock_data(ticker):
    df = yf.download(ticker, period="6mo", interval="1d")
    df.dropna(inplace=True)

    df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    df["SMA_20"] = ta.trend.SMAIndicator(df["Close"], window=20).sma_indicator()
    df["SMA_50"] = ta.trend.SMAIndicator(df["Close"], window=50).sma_indicator()
    df["MACD"] = ta.trend.MACD(df["Close"]).macd_diff()

    # إنشاء العمود الهدف: 1 إذا أغلق السهم أعلى من اليوم السابق، 0 إذا أقل
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)

    return df

# تدريب النموذج
def train_predictor(df):
    features = ["RSI", "SMA_20", "SMA_50", "MACD"]
    X = df[features]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, acc

# إرسال تنبيه إلى Telegram
def send_telegram_alert(bot_token, chat_id, message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    response = requests.post(url, data=payload)
    return response

# ----------------- واجهة المستخدم -----------------

ticker = st.text_input("📌 أدخل رمز السهم (مثال: AAPL)", "AAPL")
bot_token = st.text_input("🔐 Telegram Bot Token (اختياري)", type="password")
chat_id = st.text_input("💬 Telegram Chat ID (اختياري)")

if st.button("ابدأ التحليل"):
    try:
        df = get_stock_data(ticker)
        model, acc = train_predictor(df)

        st.success(f"✅ دقة النموذج: {acc:.2%}")

        latest = df[["RSI", "SMA_20", "SMA_50", "MACD"]].iloc[-1:]
        prediction = model.predict(latest)[0]

        if prediction == 1:
            msg = f"📈 السهم <b>{ticker}</b> متوقع له <b>الصعود</b>"
        else:
            msg = f"📉 السهم <b>{ticker}</b> متوقع له <b>الهبوط</b>"

        st.markdown(msg, unsafe_allow_html=True)

        # إرسال تنبيه عبر Telegram إذا كانت البيانات موجودة
        if bot_token and chat_id:
            response = send_telegram_alert(bot_token, chat_id, msg)
            if response.status_code == 200:
                st.success("📤 تم إرسال التنبيه إلى Telegram.")
            else:
                st.error("❌ فشل في إرسال التنبيه عبر Telegram.")
    except Exception as e:
        st.error(f"حدث خطأ أثناء التحليل: {e}")
