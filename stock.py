import streamlit as st
from datetime import date 
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

start_date = "2000-01-01"
end_date = "2021-12-01"


st.title("Stock Prediciton App")
stocks = ("HDFCBANK.NS","KOTAKBANK.NS","ICICIBANK.NS","AXISBANK.NS","YESBANK.NS","BTC-USD","^NSEI")

selected_stocks = st.selectbox("Select dataset for predict",stocks)


n_years = st.slider("Years of prediction:",1,10)
period = n_years * 365

@st.cache
def load_data(ticker):
    data = yf.download(ticker,start_date, end_date)
    data.reset_index(inplace = True)
    return data
data_load = st.text("Load data...")
data = load_data(selected_stocks)
data_load.text("Loading Data done!.....")

st.subheader('Raw Data')
st.write(data.tail())

def plot_raw():
   fig = go.Figure()
   fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name = 'Stock Open'))
   fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name = 'Stock Close'))
   fig.layout.update(title_text= "Time Series Data",xaxis_rangeslider_visible=True)
   st.plotly_chart(fig)
plot_raw()  


df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date":"ds","Close":"y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecasting = m.predict(future)

st.subheader('Forecasting Data')
st.write(forecasting.tail())

st.write('Forecast Data')
fig1 = plot_plotly(m,forecasting)
st.plotly_chart(fig1)

st.write('Forecast components')
fig2 = m.plot_components(forecasting)
st.write(fig2)
