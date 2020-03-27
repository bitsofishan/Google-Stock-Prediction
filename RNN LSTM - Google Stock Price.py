import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset_train=pd.read_csv(r"C:\Users\Ishan\Documents\Python Scripts\Datasets\Google Stock Price\Google_Stock_Price_Train.csv")
training_set=dataset_train.iloc[:,1:2].values

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
training_set_scaled=scaler.fit_transform(training_set)

X_train=[]
y_train=[]
for i in range(60,1250):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
X_train,y_train=np.array(X_train),np.array(y_train)

X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout,LSTM
regressor=Sequential()
regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))
regressor.compile(optimizer="adam",loss="mean_squared_error")
regressor.fit(X_train,y_train,epochs=100,batch_size=32)

dataset_test=pd.read_csv(r"C:\Users\Ishan\Documents\Python Scripts\Datasets\Google Stock Price\Google_Stock_Price_Test.csv")
real_stock_price=dataset_test.iloc[:,1:2].values

dataset_total=pd.concat((dataset_train.iloc[:,1:2],dataset_test.iloc[:,1:2]),axis=0)
inputs=dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs=inputs.reshape(-1,1)
inputs=scaler.transform(inputs)
X_test=[]
for i in range(60,180):
    X_test.append(inputs[i-60:i,0])
X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_stock_price=regressor.predict(X_test)
predicted_stock_price=scaler.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price,color="red",label="real")
plt.plot(predicted_stock_price,color="blue",label="predicted")
plt.title("Google Stock")
plt.xlabel("Time")
plt.ylabel("Stock price")
plt.legend()
plt.show()
















