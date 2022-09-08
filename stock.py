import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

with st.form("my_form"):
    st.write("Inside the form")
    slider_val = st.slider("Form slider")
    checkbox_val = st.checkbox("Form checkbox")

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("slider", slider_val, "checkbox", checkbox_val)

st.write("Outside the form")

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
start=st.date_input('Start',value =pd.to_datetime('2022-07-12'))
end=st.date_input('End',value=pd.to_datetime('2022-08-05'))
st.subheader('Please add a stock to rectify the error')
dataset = yf.download(tickers,start,end)['Adj Close']
st.title('Weekly Stock Adj Close for Monday')
weekly_Monday = dataset.asfreq('W-Mon')
fig, ax = plt.subplots(figsize=(16, 4))
st.line_chart(dataset)
