import yfinance as yf
import requests
import streamlit as st

session = requests.Session()

session.headers.update(
    {
        "User-Agent":
        "Mozilla/5.0"
    }
)

@st.cache_data(ttl=3600)
def get_price_data(
        ticker,
        start,
        end):

    try:

        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            session=session
        )

        if df.empty:
            return df

        if hasattr(df.columns, "levels"):
            df.columns=df.columns.droplevel(1)

        return df

    except Exception as e:

        print(e)
        return None


@st.cache_data(ttl=3600)
def get_company_info(
        ticker):

    try:

        return yf.Ticker(
            ticker,
            session=session
        ).info

    except:
        return {}