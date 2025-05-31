import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# إعدادات Finnhub (ضع التوكن الخاص بك)
FINNHUB_API_KEY = "d0s84hpr01qkkpltj8j0d0s84hpr01qkkpltj8jg"

# --- دالة جلب بيانات السهم من Finnhub ---
def fetch_stock_data(ticker):
    url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={FINNHUB_API_KEY}"
    res = requests.get(url)
    if res.status_code == 200:
        data = res.json()
        # current price, high price, low price, open price, previous close
        return data
    return None

# --- دالة جلب الأخبار ---
def fetch_news(ticker):
    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from=2025-05-01&to=2025-05-31&token={FINNHUB_API_KEY}"
    res = requests.get(url)
    if res.status_code == 200:
        news = res.json()
        return news
    return []

# --- تحليل المشاعر (مثال مبسط جداً، يمكنك استبداله بنموذج AI) ---
def sentiment_analysis(text):
    text = text.lower()
    if any(word in text for word in ["good", "positive", "up", "gain", "profit"]):
        return "إيجابي"
    elif any(word in text for word in ["bad", "negative", "down", "loss", "fall"]):
        return "سلبي"
    else:
        return "محايد"

# --- واجهة المستخدم ---
st.title("📈 نظام متكامل لتحليل الأسهم مع تنبيهات متقدمة")

# المحفظة (مثال)
if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = {}

ticker = st.text_input("رمز السهم (مثال: AAPL)").upper()

if ticker:
    data = fetch_stock_data(ticker)
    if data:
        st.subheader(f"بيانات السهم: {ticker}")
        st.write(f"السعر الحالي: {data['c']}")
        st.write(f"أعلى سعر خلال اليوم: {data['h']}")
        st.write(f"أدنى سعر خلال اليوم: {data['l']}")
        st.write(f"سعر الافتتاح: {data['o']}")
        st.write(f"سعر الإغلاق السابق: {data['pc']}")

        # تنبيه بسيط لحركات كبيرة
        change_percent = ((data['c'] - data['pc']) / data['pc']) * 100
        if abs(change_percent) > 5:
            st.warning(f"⚠️ تغير كبير بالسعر: {change_percent:.2f}%")

        # جلب الأخبار وتحليل المشاعر
        news = fetch_news(ticker)
        if news:
            st.subheader("أخبار وتحليل المشاعر")
            for article in news[:5]:
                sentiment = sentiment_analysis(article['headline'])
                st.markdown(f"**{article['datetime']}** - {article['headline']} - *المشاعر: {sentiment}*")
        else:
            st.info("لا توجد أخبار متاحة حالياً.")

        # إدارة المحفظة - إضافة سهم
        with st.expander("إدارة المحفظة"):
            qty = st.number_input("كمية الأسهم التي تملكها", min_value=0, step=1)
            buy_price = st.number_input("سعر الشراء لكل سهم", min_value=0.0, step=0.01)
            if st.button("إضافة للسجل"):
                st.session_state.portfolio[ticker] = {"quantity": qty, "buy_price": buy_price, "current_price": data['c']}
                st.success(f"تمت إضافة {ticker} إلى المحفظة")

        # عرض المحفظة
        if st.session_state.portfolio:
            st.subheader("📊 محفظتك")
            df_portfolio = pd.DataFrame.from_dict(st.session_state.portfolio, orient='index')
            df_portfolio['قيمة السوق'] = df_portfolio['quantity'] * df_portfolio['current_price']
            df_portfolio['الربح/الخسارة'] = (df_portfolio['current_price'] - df_portfolio['buy_price']) * df_portfolio['quantity']
            st.dataframe(df_portfolio)

else:
    st.info("ادخل رمز سهم لبدء التحليل.")

