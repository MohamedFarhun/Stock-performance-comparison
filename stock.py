import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.title('Stock Market Prediction')

tickers=('TSLA','AAPL','MSFT','BTC-USD','ETH-USD','AMD')

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
   st.line_chart(df)


st.title('Stock measure of centres')
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
data=[108.76000,105.506667,112.380952,146.958095,143.490000,126.839999,119.630000,112.683913,95.797500,94.271428,90.852858,82.904499]
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.set_title("Measures of Center")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.scatter(months,data)
import statistics as st
ax.plot([st.mean(data)], [st.mean(data)], color='r', marker="o", markersize=15)
ax.plot([st.median(data)], [st.median(data)], color='g', marker="o", markersize=15)
plt.annotate("Mean", (st.mean(data), st.mean(data)+0.3), color='r')
plt.annotate("Median", (st.median(data), st.median(data)+0.3), color='g')
#plt.annotate("Mode", (st.mode(data), st.mode(data)+0.3), color='k')
plt.show()
