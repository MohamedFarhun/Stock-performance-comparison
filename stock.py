import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt


st.title('Stock performance comparison')

tickers=('TSLA','AAPL','MSFT','BTC-USD','ETH-USD','AMD','AMZN')

dropdown=st.multiselect('Pick your assets',tickers,key=1,default='TSLA')

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
dropdown=st.multiselect('Pick your assets',tickers,key=2,default='TSLA')
start=st.date_input('Start',value =pd.to_datetime('2022-07-12'))
end = st.date_input('end',dt.date(2022,8, 12))
dataset = yf.download(dropdown,start,end)['Adj Close']
st.title('Weekly Stock Adj Close for Monday')
weekly_Monday = dataset.asfreq('W-Mon')
fig, ax = plt.subplots(figsize=(16, 4))
st.line_chart(dataset)

st.title('Weekly Stock Average for Monday')
tickers=('TSLA','AAPL','MSFT','BTC-USD','ETH-USD','AMD','AMZN')
dropdown=st.multiselect('Pick your assets',tickers,key=3,default='TSLA')
start = st.date_input('Start',dt.date(2021,8, 12))
end = st.date_input('end',dt.date(2022,8, 13))
dataset = yf.download(dropdown,start,end)['Adj Close']
weekly_avg = dataset.resample('W').mean()
fig, ax = plt.subplots(figsize=(16, 4))
st.line_chart(dataset)

st.title('Stock Time Returns Analysis')
tickers=('TSLA','AAPL','MSFT','BTC-USD','ETH-USD','AMD','AMZN')
dropdown=st.multiselect('Pick your assets',tickers,key=4,default='TSLA')
start = dt.date.today() - dt.timedelta(days = 365*5)
end = dt.date.today()
data = yf.download(dropdown,start,end)['Adj Close']
st.line_chart(data)

st.title('Profit and Loss in Trading')
tickers=['TSLA','AAPL','MSFT','BTC-USD','ETH-USD','AMD','AMZN']
dropdown=st.multiselect('Pick your assets',tickers,key=5,default='TSLA')
start = dt.date.today() - dt.timedelta(days = 365*5)
end = dt.date.today()
dataset= yf.download(dropdown,start,end)['Adj Close']
Start = 5000
dataset['Shares'] = 0
dataset['PnL'] = 0
dataset['End'] = Start
dataset['Adj Close']=dataset
AdjClose=int['Adj Close']
dataset['Shares'] = dataset['End'] / AdjClose
dataset['PnL'] = dataset['Shares'] * (dataset['Adj Close'] - AdjClose)
dataset['End'] = dataset['End'] + dataset['PnL']
st.line_chart(dataset)
