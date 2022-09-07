import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.title('Stock performance comparison')

tickers=('TSLA','AAPL','MSFT','BTC-USD','ETH-USD','AMD','AMZN')

dropdown=st.multiselect('Pick your assets',tickers)

start=st.date_input('Start',value =pd.to_datetime('2022-07-03'))
end=st.date_input('End',value=pd.to_datetime('today'))

def relativeret(df):
    rel=df.pct_change()
    cumret=(1+rel).cumprod()-1
    cumret=cumret.fillna(0)
    return cumret


if len(dropdown)>0:
   df=relativeret(yf.download(dropdown,start,end)['Adj Close'])
   st.header('Returns of {}'.format(dropdown))
   st.dataframe(df)
   st.line_chart(df)
   st.area_chart(df)
   st.bar_chart(df)
    
st.title('Stock time series analysis')
tickers=('TSLA','AAPL','MSFT','BTC-USD','ETH-USD','AMD','AMZN')
dropdown=st.multiselect('Pick your assets',tickers)
df=yf.download(dropdown,start,end)['Adj Close'])
business_monthly = df.resample('BM')
fig, ax = plt.subplots(figsize=(16, 4))
business_monthly['Adj Close'].plot(title='Stock Close Price monthly', ax=ax)
