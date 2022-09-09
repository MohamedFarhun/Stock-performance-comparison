import streamlit as st
import yfinance as yf
import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
import math
import statistics as sp
import scipy.stats as stats


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
Price= pd.DataFrame(np.log(df['Adj Close']))
st.line_chart(Price)

st.title('Stock Investment Portfolio-Risk and Return')
tickers=['TSLA','AAPL','MSFT','BTC-USD','ETH-USD','AMD','AMZN']
dropdown=st.multiselect('Pick your assets',tickers,key=8,default='TSLA')
start = st.date_input('Start',dt.date(2021,8, 17))
end = st.date_input('end',dt.date(2022,8, 18))
df= yf.download(dropdown,start,end)
df = pd.DataFrame()
data = []
for ticker in tickers:
    df = pd.merge(df, pd.DataFrame(yf.download(tickers, fields='price', start=start, end=end)['Adj Close']), right_index=True, left_index=True, how='outer')
    data.append(ticker)
    break;
rets = df.pct_change()
rets=rets.columns
rets=df.std()
rets1=df.mean()
st.text('Standard deviation')
st.bar_chart(rets)
st.text('Mean')
st.bar_chart(rets1)
st.header('Risk vs Return')
chart_data = pd.DataFrame(np.random.randn(50, 2),columns=["Risk","Returns"])
st.bar_chart(chart_data)

st.title('Stock Covariance & Correlations')
dropdown = st.selectbox('Choose any one to have analysis',('TSLA','AAPL','MSFT','BTC-USD','ETH-USD','AMD','AMZN'),key=9)
st.write('You selected:', dropdown)
start = st.date_input('Start',dt.date(2021,8, 18))
end = st.date_input('end',dt.date(2022,8, 19))
dataset= yf.download(dropdown,start,end)
stocks_returns = np.log(dataset / dataset.shift(1))
variance= stocks_returns.var()
st.text('Variance:-')
st.text(variance)
standard_deviation= stocks_returns.var()*250
st.text('Standard Deviation:-')
st.text(standard_deviation)
cov_matrix = stocks_returns.cov()*250
st.text('Covariance:-')
st.text(cov_matrix)
st.text('Correlation:-')
corr_matrix = stocks_returns.corr()*250
st.text(corr_matrix)

st.title('Stock Linear Regression(Graphical representation)')
tickers=['TSLA','AAPL','MSFT','BTC-USD','ETH-USD','AMD','AMZN']
dropdown = st.selectbox('Choose any one to have analysis',('TSLA','AAPL','MSFT','BTC-USD','ETH-USD','AMD','AMZN'),key=10)
start = st.date_input('Start',dt.date(2021,8, 19))
end = st.date_input('end',dt.date(2022,8, 20))
dataset= yf.download(dropdown,start,end)
dataset['Returns'] = np.log(dataset['Adj Close'] / dataset['Adj Close'].shift(1))
dataset = dataset.dropna()
X = dataset['Open']
Y = dataset['Adj Close']
st.bar_chart(X)
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
X_train = np.array(X_train).reshape(-1,1)
y_train = np.array(y_train).reshape(-1,1)
X_test = np.array(X_test).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1)
linregression=LinearRegression()
st.text('Regression is:-')
st.text(linregression)
linregression.fit(X_train,y_train)
y_pred = linregression.predict(X_test)
st.text('Intercept is:-')
intercept=linregression.intercept_
st.text(intercept)
st.text('Slope is:-')
Slope=linregression.coef_
st.text(Slope)
predicted_value=linregression.predict(X_train)
st.subheader('Predicted graph')
st.line_chart(predicted_value)

st.title('Stock Statistics')
tickers=['TSLA','AAPL','MSFT','BTC-USD','ETH-USD','AMD','AMZN']
dropdown=st.multiselect('Pick your assets',tickers,key=11,default='TSLA')
start = st.date_input('Start',dt.date(2021,8, 20))
end = st.date_input('end',dt.date(2022,8, 21))
df= yf.download(dropdown,start,end)
returns = df['Adj Close'].pct_change()[1:].dropna()
mean=sp.mean(returns)
median=sp.median(returns)
median_low=sp.median_low(returns)
median_high=sp.median_high(returns)
median_grouped=sp.median_grouped(returns)
st.write('Mean of {} is:-'.format(dropdown),mean)
