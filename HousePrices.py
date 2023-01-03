#---------------------------------------- Importing Libraries
from pandas import read_excel
# from keras.models import Sequential
# from keras.layers import Dense, activation
import numpy
# import os
# os.system("clear")
#-----------------------------------------Reading Excel file
Dataset_directory = "/home/hosein/Desktop/HousePricing-MachinLearning/Assignment 2_BUSI 651_House Prices.xlsx"

dataset = read_excel(Dataset_directory)
datalength = len(dataset)
print(datalength)

# #-----------------------------------------deviding dataset into Y and X datasets

# x_data = dataset[["GarageCars","GarageArea","OverallQual","GrLivArea"]]
# y_data = dataset["SalePrice"]


# #-----------------------------------------deviding X and Y datasets into train and test datasets

# train_len=int(datalength*0.8)
# x_train=x_data[:train_len+1]
# y_train=y_data[:train_len+1]
# x_test=x_data[train_len+1:]
# y_test=y_data[train_len+1:]

# #------------------------------------------Design Model

# price_model = Sequential()
# price_model.add(Dense(64,activation = "relu",input_dim = 4)) #first hidden layer with 4 input
# price_model.add(Dense(64,activation="relu"))#Second hidden layer
# price_model.add(Dense(64,activation="relu"))#Third hidden layer
# price_model.add(Dense(1))

# #-----------------------------------------Compile and fit model

# price_model.compile(loss="mean_squared_error",optimizer="adam")
# price_model.fit(x_train,y_train,epochs=1000)

# #-----------------------------------------Testing the model

# y_predict = price_model.predict(x_test)

# #-----------------------------------------reshaping

# y_predicted_1 = y_predict.reshape(1,datalength-train_len-1)
# y_test_1 = y_test.to_numpy()
# APE = abs( numpy.subtract(y_predicted_1, y_test_1) )/y_test_1*100
# MAPE = numpy.mean(APE)
# print(MAPE)
# #-----------------------------------------Prediction
# # print(price_model.predict([[2,600,7,2200]]))








