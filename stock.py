import streamlit as st
import yfinance as yf
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


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
st.line_chart(dataset)

st.title('Weekly Stock Average for Monday')
tickers=('TSLA','AAPL','MSFT','BTC-USD','ETH-USD','AMD','AMZN')
dropdown=st.multiselect('Pick your assets',tickers,key=3,default='TSLA')
start = st.date_input('Start',dt.date(2021,8, 12))
end = st.date_input('end',dt.date(2022,8, 13))
dataset = yf.download(dropdown,start,end)['Adj Close']
weekly_avg = dataset.resample('W').mean()
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
start = st.date_input('Start',dt.date(2021,8, 13))
end = st.date_input('end',dt.date(2022,8, 14))
dataset= yf.download(dropdown,start,end)['Adj Close']
Start = 5000
dataset['Shares'] = 3
dataset['PnL'] = 42
dataset['End'] = Start
dataset['Adj Close']=300
dataset['Shares'] = dataset['End'] / dataset['Adj Close']
dataset['PnL'] = dataset['Shares'] * (dataset['Adj Close'] - dataset['Adj Close'])
dataset['End'] = dataset['End'] + dataset['PnL']
st.line_chart(dataset)

st.title('Stock Price Predictions')
tickers=['TSLA','AAPL','MSFT','BTC-USD','ETH-USD','AMD','AMZN']
dropdown=st.radio('Pick your assets',tickers)
start = st.date_input('Start',dt.date(2021,8, 14))
end = st.date_input('end',dt.date(2022,8, 15))
df= yf.download(dropdown,start,end)
x_train = df[1:5]
y_train = df['Adj Close']
x_train = x_train.values[:-1]
y_train = y_train.values[1:]
x_train,x_test,y_train,y_test=train_test_split(x_train,x_train,test_size=0.1,random_state=3)
x_train=x_train.reshape(-1,1)
y_train=y_train.reshape(-1,1)
lr = LinearRegression()
fit=lr.fit(x_train, y_train)
st.text('Regression type:-')
st.text(fit)
x_test=df.iloc[:,:1]
y_test=df['Adj Close']
score=lr.score(x_test, y_test)
st.text('The accuracy score is:-')
st.text(score)

st.title('ValueAtRisk')
tickers=['TSLA','AAPL','MSFT','BTC-USD','ETH-USD','AMD','AMZN']
dropdown=st.multiselect('Pick your assets',tickers,key=6,default='TSLA')
start = st.date_input('Start',dt.date(2021,8, 15))
end = st.date_input('end',dt.date(2022,8, 16))
df= yf.download(dropdown,start,end)
barchart=df["Adj Close"].pct_change()
st.bar_chart(barchart)
Adjclose=df["Adj Close"].pct_change().std()
st.text('The standard deviation is:-')
st.text(Adjclose)
st.header('Value at risk-return')
returns = df["Adj Close"].pct_change().dropna()
mean = returns.mean()
sigma = returns.std()
quantile=returns.quantile(0.05)
st.text('Return of quantile is:-')
st.text(quantile)

st.title('Time Series Stock Forecast')
tickers=['TSLA','AAPL','MSFT','BTC-USD','ETH-USD','AMD','AMZN']
dropdown=st.multiselect('Pick your assets',tickers,key=7,default='TSLA')
start = st.date_input('Start',dt.date(2021,8, 16))
end = st.date_input('end',dt.date(2022,8, 17))
df= yf.download(dropdown,start,end)
df_train = df[:740]
df_test = df[740:]
mdl = Prophet(interval_width=0.95,daily_seasonality=True,yearly_seasonality=True)
df['Adj Close'].plot(figsize=(12,8))
df = pd.rename(columns={'Date':'ds', 'Adj Close':'y'})
mdl.fit(df_train)
future = mdl.make_future_dataframe(periods=24, freq='MS')
forecast = mdl.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
mdl=mdl.plot(forecast);
st.line_chart(mdl)

