import streamlit as st
import requests
import pandas as pd
import openai
import plotly.express as px
import yfinance as yf
# مفاتيح API
openai.api_key = "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"

def fetch_stock_data(ticker):
    df = yf.download(ticker, period="1mo", interval="1d")
    return df
# دوال جلب البيانات وإرسال تنبيهات تليجرام ...


def fetch_news(ticker):
    FINNHUB_API_KEY = "d0s63s1r01qkkplt7130d0s63s1r01qkkplt713g"
    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from=2023-01-01&to=2023-12-31&token={FINNHUB_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return []

# تحليل مشاعر باستخدام OpenAI
def sentiment_analysis_openai(text):
    prompt = f"صنف المشاعر لهذا النص المالي إلى: إيجابي، سلبي، أو محايد.\nالنص: {text}\nالنتيجة:"
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=5,
        temperature=0
    )
    sentiment = response.choices[0].text.strip()
    return sentiment

# إرسال رسالة تليجرام
def send_telegram_message(bot_token, chat_id, message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    response = requests.post(url, data=payload)
    return response.status_code == 200

# عرض الرسم البياني للمحفظة
def plot_portfolio_profit_loss(df_portfolio):
    fig = px.bar(df_portfolio, x=df_portfolio.index, y='الربح/الخسارة',
                 title='الربح / الخسارة لكل سهم في المحفظة',
                 labels={'x':'السهم', 'الربح/الخسارة':'الدولار'})
    st.plotly_chart(fig)

# الواجهة الرئيسية
st.title("📊 نظام تحليل الأسهم وإدارة المحفظة")

ticker = st.text_input("رمز السهم").upper()
bot_token = st.text_input("Telegram Bot Token", type="password")
chat_id = st.text_input("Telegram Chat ID")

if ticker:
    # جلب الأخبار من Finnhub (مثال)
    news = fetch_news(ticker)
    if news:
        st.subheader("أخبار وتحليل المشاعر (OpenAI)")
        for article in news[:5]:
            sentiment = sentiment_analysis_openai(article['headline'])
            st.markdown(f"{article['datetime']} - **{article['headline']}** - *المشاعر: {sentiment}*")

# إضافة سهم للمحفظة
if 'portfolio' not in st.session_state:
    st.session_state['portfolio'] = {}

with st.expander("إدارة المحفظة"):
    new_ticker = st.text_input("أضف سهم للمحفظة")
    qty = st.number_input("كمية الأسهم", min_value=0, step=1)
    buy_price = st.number_input("سعر الشراء", min_value=0.0, step=0.01)
    if st.button("أضف للسجل"):
        if new_ticker and qty > 0 and buy_price > 0:
            st.session_state.portfolio[new_ticker] = {"quantity": qty, "buy_price": buy_price}
            st.success(f"تمت إضافة {new_ticker} إلى المحفظة")

# عرض المحفظة
if st.session_state.portfolio:
    df = pd.DataFrame.from_dict(st.session_state.portfolio, orient='index')
    # تحديث السعر الحالي لكل سهم (تحتاج تنفيذ جلب السعر)
    df['current_price'] = df.index.to_series().apply(lambda x: fetch_stock_data(x)['c'] if fetch_stock_data(x) else 0)
    df['قيمة السوق'] = df['quantity'] * df['current_price']
    df['الربح/الخسارة'] = (df['current_price'] - df['buy_price']) * df['quantity']
    st.dataframe(df)
    plot_portfolio_profit_loss(df)



