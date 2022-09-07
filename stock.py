import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.title('Stock performance comparison')

tickers=('TSLA','AAPL','MSFT','BTC-USD','ETH-USD','AMD','AMZN')

dropdown=st.multiselect('Pick your assets',tickers,key=1)

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
dropdown1=st.multiselect('Pick your assets',tickers,key=2)
start=st.date_input('Start',value =pd.to_datetime('2022-07-12'))
end=st.date_input('End',value=pd.to_datetime('2022-09-08'))
dataset = yf.download(dropdown1,start,end)
weekly_Monday = dataset.asfreq('W-Mon')
fig, ax = plt.subplots(figsize=(16, 4))
weekly_Monday['Adj Close'].plot(title='Weekly Stock Adj Close for Monday', ax=ax)
st.line_chart(dataset)
