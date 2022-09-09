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
from scipy.stats.mstats import gmean
from statsmodels.stats.stattools import jarque_bera
from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import binom
from scipy.stats import poisson
from scipy.stats import expon
import seaborn as sns


st.title('Stock performance comparison')

def add_bg_from_url():
    st.markdown(f"""
         <style>
         .stApp {{
             background-image: url("https://img.freepik.com/free-photo/gray-abstract-wireframe-technology-background_53876-101941.jpg?w=996&t=st=1662725888~exp=1662726488~hmac=3b148d7688fd138851b5a25f611ca9cf08bf2d28382e218d5896f95f6baa2bd4");
             background-attachment: fixed;
             background-size: cover}}
             </style>""",unsafe_allow_html=True)
add_bg_from_url() 

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
df= yf.download(dropdown,start,end)['Adj Close']
Start = 5000
df['Shares'] = 0
df['PnL'] = 0
df['End'] = Start
df['Shares'] = pd.DataFrame(df['End'].shift(1) / df['Adj Close'].shift(1))
df['PnL'] = pd.DataFrame(df['Shares'] * (df['Adj Close'] - df['Adj Close'].shift(1)))
df['End'] = pd.DataFrame(df['End'].shift(1) + df['PnL'])
plt.figure(figsize=(16,8))
plt.plot(df['PnL'])
plt.title('Profit and Loss for Daily')
plt.xlabel('Date')
plt.ylabel('Price')
st.pyplot(plt)
plt.close()

st.title('Stock Price Predictions-Accuracy Score')
tickers=['TSLA','AAPL','MSFT','BTC-USD','ETH-USD','AMD','AMZN']
dropdown = st.selectbox('Choose any one to have analysis',('TSLA','AAPL','MSFT','BTC-USD','ETH-USD','AMD','AMZN'),key=13)
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
st.write('Regression type of {} is:-'.format(dropdown))
st.text(fit)
x_test=df.iloc[:,:1]
y_test=df['Adj Close']
score=lr.score(x_test, y_test)
st.write('Accuracy score of {} is:-'.format(dropdown),score)

st.title('ValueAtRisk')
tickers=['TSLA','AAPL','MSFT','BTC-USD','ETH-USD','AMD','AMZN']
dropdown=st.multiselect('Pick your assets',tickers,key=6,default='TSLA')
start = st.date_input('Start',dt.date(2021,8, 15))
end = st.date_input('end',dt.date(2022,8, 16))
df= yf.download(dropdown,start,end)
barchart=df["Adj Close"].pct_change()
st.bar_chart(barchart)
Adjclose=df["Adj Close"].pct_change().std()
st.header('Standard deviation')
st.write('Standard deviation of {} is:-'.format(dropdown),Adjclose)
st.header('Value at risk-return')
returns = df["Adj Close"].pct_change().dropna()
mean = returns.mean()
sigma = returns.std()
quantile=returns.quantile(0.05)
st.write('Return of {} quantile is:-'.format(dropdown),quantile)

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
rets = df.pct_change(periods=3)
scatter_matrix(rets, diagonal='kde', figsize=(10, 10))
corr = rets.corr()
plt.imshow(corr, cmap='Blues', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns)
plt.yticks(range(len(corr)), corr.columns)
st.pyplot(plt)
plt.close()
plt.bar(rets.columns, rets.std(), color=['red', 'blue', 'green', 'orange', 'cyan'])
plt.title("Stock Risk")
plt.xlabel("Stock Symbols")
plt.ylabel("Standard Deviations")
st.pyplot(plt)
plt.close()
plt.bar(rets.columns, rets.mean(), color=['red', 'blue', 'green', 'orange', 'cyan'])
plt.title("Average Returns")
plt.xlabel("Stock Symbols")
plt.ylabel("Returns")
st.pyplot(plt)
plt.close()

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
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
X_train = np.array(X_train).reshape(-1,1)
y_train = np.array(y_train).reshape(-1,1)
X_test = np.array(X_test).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1)
linregression=LinearRegression()
st.write('Regression type of {} is:-'.format(dropdown))
st.text(linregression)
linregression.fit(X_train,y_train)
y_pred = linregression.predict(X_test)
intercept=linregression.intercept_
st.write('Intercept of {} is:-'.format(dropdown),intercept)
Slope=linregression.coef_
st.write('Slope of {} is:-'.format(dropdown),Slope)
plt.scatter(X_train,y_train)
ax=plt.plot(X_train,linregression.predict(X_train),'r')
st.subheader('Predicted graph')
st.pyplot(plt)
plt.close()

st.title('Stock Statistics')
tickers=['TSLA','AAPL','MSFT','BTC-USD','ETH-USD','AMD','AMZN']
dropdown = st.selectbox('Choose any one to have analysis',('TSLA','AAPL','MSFT','BTC-USD','ETH-USD','AMD','AMZN'),key=11)
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
st.write('Median of {} is:-'.format(dropdown),median)
st.write('Median_low of {} is:-'.format(dropdown),median_low)
st.write('Median_high of {} is:-'.format(dropdown),median_high)
st.write('Median_grouped of {} is:-'.format(dropdown),median_grouped)
Standard_deviation=returns.std()
st.write('Standard_deviation of {} is:-'.format(dropdown),Standard_deviation)
T = len(returns)
init_price = df['Adj Close'][0]
final_price = df['Adj Close'][T]
st.write('init_price of {} is:-'.format(dropdown),init_price)
st.write('final_price of {} is:-'.format(dropdown),final_price)
ratios = returns + np.ones(len(returns))
R_G = gmean(ratios) - 1
final_price_as_computed_with_RG=init_price*(1 + R_G)**T
st.write('Final_price_as_computed_with_RG of {} is:-'.format(dropdown),final_price_as_computed_with_RG)
Harmonic_mean=len(returns)/np.sum(1.0/returns)
st.write('Harmonic_mean of {} is:-'.format(dropdown),Harmonic_mean)
skew=stats.skew(returns)
st.write('Skew of {} is:-'.format(dropdown),skew)
kurtosis=stats.kurtosis(returns)
st.write('Excess Kurtosis of {} is:-'.format(dropdown),kurtosis)
chart_data = pd.DataFrame(np.random.randn(20, 3),columns=['skew', 'mean', 'median'])
st.line_chart(chart_data)
st.bar_chart(chart_data)
st.area_chart(chart_data)
st.write('Excess Kurtosis of {} is:-'.format(dropdown),kurtosis)
_, pvalue, _, _ = jarque_bera(returns)
st.write('The returns of {} :-'.format(dropdown))
if pvalue > 0.05:
    st.write('The returns are likely normal')
else:
    st.write('The returns are likely not normal.')

  
st.title('Stock datascience-analysis')
tickers=['TSLA','AAPL','MSFT','BTC-USD','ETH-USD','AMD','AMZN']
dropdown = st.selectbox('Choose any one to have analysis',('TSLA','AAPL','MSFT','BTC-USD','ETH-USD','AMD','AMZN'),key=12)
start = st.date_input('Start',dt.date(2021,8, 21))
end = st.date_input('end',dt.date(2022,8, 22))
df= yf.download(dropdown,start,end)
df['Returns'] = df['Adj Close'].pct_change()[1:].dropna()
mean=df.mean()
median=df.median()
mode=df.mode()
min=df['Returns'].min()
max = df['Returns'].max()
standard_deviation=df.std()
mu = df['Returns'].mean()
sigma = df['Returns'].std()
[n,bins,patches] = plt.hist(df['Returns'], 100)
s =norm.pdf(bins, mu, sigma)
plt.plot(bins, s, color='y', lw=2)
st.subheader('Stock Returns on Normal Distribution')
st.pyplot(plt)
plt.close()
mu, std = norm.fit(dataset['Returns'])
plt.hist(dataset['Returns'], bins=25, alpha=0.6, color='g')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)
st.pyplot(plt)
plt.close()

df2= yf.download(dropdown,start,end)
stock_ret = df['Adj Close'].pct_change().dropna()
mkt_ret = df2['Adj Close'].pct_change().dropna()
beta, alpha, r_value, p_value, std_err = stats.linregress(mkt_ret, stock_ret)
print(beta, alpha)
mu, std = gamma.stats(dataset['Returns'])
plt.hist(dataset['Returns'], bins=25, color='g')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 1171)
p = gamma.pdf(x, alpha, scale=1/beta)
plt.plot(x, p, 'k', linewidth=2)
plt.title("Gamma Distribution for Stock")
st.pyplot(plt)
plt.close()

mu = dataset['Returns'].mean()
dist = poisson.rvs(mu=mu, loc=0, size=1000)
print("Mean: %g" % np.mean(dataset['Returns']))
print("SD: %g" % np.std(dataset['Returns'], ddof=1))
plt.hist(dist, bins=10)
plt.title('Poisson Distribution Curve')
st.pyplot(plt)
plt.close()

mu = dataset['Returns'].mean()
sigma = dataset['Returns'].std()
x_m = dataset['Returns'].max()
def plot_exponential(x_range, mu=0, sigma=1, cdf=False, **kwargs):
    if cdf:
        y = expon.cdf(x, mu, sigma)
    else:
        y = expon.pdf(x, mu, sigma)
    plt.plot(x, y, **kwargs)
x = np.linspace(0, x_m, 5000)
plot_exponential(x, 0, 1, color='red', lw=2, ls='-', alpha=0.5, label='pdf')
plot_exponential(x, 0, 1, cdf=True, color='blue', lw=2, ls='-', alpha=0.5, label='cdf')
plt.title('Probability and Cumulative distribution function')
plt.xlabel('Adj Close')
plt.ylabel('Probability')
plt.legend(loc='best')
st.pyplot(plt)
plt.close()

n = 10 
p = 0.5 
k = np.arange(0,21) 
binomial = binom.pmf(k, n, p)
data_binom = binom.rvs(n=len(dataset['Adj Close']),p=0.5,size=1000)
plt.figure(figsize=(16,10))
plt.title("Binomial Distribution")
ax = sns.distplot(data_binom,kde=False,color='skyblue',hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Binomial Distribution', ylabel='Frequency')
st.pyplot(plt)
plt.close()
st.header('SNS Pairplot')
sac=sns.pairplot(df, kind="scatter")
st.pyplot(sac)
plt.close()
