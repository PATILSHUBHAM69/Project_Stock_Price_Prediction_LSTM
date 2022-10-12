import streamlit as st
st.title('Stock Price Prediction!')
stocks = ("WIPRO.NS","SUNPHARMA.NS","CIPLA.NS","TATASTEEL.NS","ITC.NS","INFY.NS","ONGC.NS","TCS.NS","MARUTI.NS","RELIANCE.NS","HDFC.NS","ICICIBANK.NS","SBIN.NS")
selected_stock = st.selectbox(" Select Stock for Prediction", stocks)
n = st.slider('Number of  days want to predict from today ', 1, 90)
if st.button('Proceed to Predict'):
    
    import pandas as pd
    import numpy as np
    import datetime as dt
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.layers import LSTM
    import matplotlib.pyplot as plt

       
    from plotly import graph_objs as go
    import matplotlib.pyplot as plt
    import plotly.express as px
    from plotly.subplots import make_subplots    
    from datetime import datetime,timedelta
    u=datetime.today()
    i=(str(u.date()))
    import yfinance as yf
    df=yf.download(selected_stock,start='2010-1-1',end=i)
    df=df.reset_index()
    all_time_high = df['Close'].max().round()
    st.subheader("All Time High")
    st.write(all_time_high)
    all_time_low = df['Close'].min().round()
    st.subheader("All Time Low")
    st.write(all_time_low)
    last_one_year_data = df['Close'].tail(250)
    last_one_year_high = last_one_year_data.max().round()
    last_one_year_low = last_one_year_data.min().round()
    st.subheader("52 Weeks High")
    st.write(last_one_year_high)
    st.subheader("52 Weeks Low")
    st.write(last_one_year_low)
    
    st.subheader("Time Series Data")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = df['Date'], y=df['Open'], name='Stock_Open'))
    fig.add_trace(go.Scatter(x = df['Date'], y=df['Close'], name='Stock_Close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
    st.subheader("ACTUAL PRICE AND AVERAGE OF LAST 100 DAYS AND AVERAGE OF LAST 200 DAYS")
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig_ma = plt.figure(figsize = (30,10))
    plt.plot(df.Close,label='CLOSE PRICE OF STOCK')
    plt.plot(ma100, 'g',label='AVERAGE OF LAST 100 DAYS CLOSE PRICE OF STOCK')
    plt.plot(ma200, 'r',label='AVERAGE OF LAST 200 DAYS CLOSE PRICE OF STOCK')
    plt.title("ORIGENAL AND AVERAGE OF LAST 100 DAYS AND AVERAGE OF LAST 200 DAYS")
    plt.xlabel('RECORDS')
    plt.ylabel('PRICE')
    plt.legend()
    st.pyplot(fig_ma)
    df=df[['Close']]

    
     from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler(feature_range=(0,1))
    df=scaler.fit_transform(np.array(df).reshape(-1,1))
    training_size=int(len(df))
    train_data=df[0:training_size,:]
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
            return np.array(dataX), np.array(dataY)
    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)    
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)

    model=Sequential()
    model.add(LSTM(30,return_sequences=True,input_shape=(100,1)))
    model.add(LSTM(10,return_sequences=True))
    model.add(Dropout(0.2)) 
    model.add(LSTM(10))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')

    history=model.fit(X_train,y_train,epochs=2,batch_size=100,verbose=1)
    x=(len(train_data)-100)
    x_input=train_data[x:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    
    from numpy import array
    lst_output=[]
    n_steps=100
    i=0
    while(i<n):
        if(len(temp_input)>100):
            x_input=np.array(temp_input[1:])
            #print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1
    final_output = scaler.inverse_transform(lst_output[:])
    prediction = final_output.round().tolist()
    
    st.subheader('Prediction for next enter days')
    import datetime
    today = datetime.date.today()
    m =0
    days = []
    while m < n:
        tomorrow=today + datetime.timedelta(days = 1+m)
        m = m+1
        t=tomorrow.strftime("%d/%m/%Y")
        days.append(t)
    df3 = pd.DataFrame({'Date':days, 'Price':prediction})
    st.write(df3)
