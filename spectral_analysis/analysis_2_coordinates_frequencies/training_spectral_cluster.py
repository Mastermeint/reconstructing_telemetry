from os import truncate
from functions import total_consumption_meetdata_and_maximum_peak
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import fsolve, minimize
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import pandas as pd


unique_meetdata = pd.read_csv('../Alliander_data/unique_meetdata.csv')
spectral_df = pd.read_csv('../Alliander_data/spectral_multiple_frequencies_cluster_profiles.csv')

first_column = unique_meetdata.columns[0]

numeration_customers_unique = unique_meetdata.iloc[:,[0]]
numeration_customers_unique.rename(columns={ numeration_customers_unique.columns[0]: "Customer" }, inplace = True)
print(numeration_customers_unique.head())
unique_meetdata = unique_meetdata.drop([first_column], axis=1)
print(unique_meetdata.head())

unique_meetdata = total_consumption_meetdata_and_maximum_peak(unique_meetdata)

print(unique_meetdata["SUM"].head())
print(unique_meetdata["MAX"].head())

column_list = list(spectral_df)

spectral_df["SUM"] = unique_meetdata["SUM"]
spectral_df["MAX"] = unique_meetdata["MAX"]

spectral_df = spectral_df.drop(["Spectral Frequency 1 X","Spectral Frequency 1 Y","Spectral Frequency 2 X","Spectral Frequency 2 Y"], axis = 1)
profiles = list(spectral_df["Profiles"])
profiles_np_zeros = np.zeros(len(profiles))
profiles_np = profiles_np_zeros.copy()

for i in range(len(profiles)):
    
    #print(type(profiles[i]))
    if profiles[i] == "['KVKSEGMENT_1']": profiles_np[i] = 1
    if profiles[i] == "['KVKSEGMENT_2']": profiles_np[i] = 2
    if profiles[i] == "['KVKSEGMENT_3']": profiles_np[i] = 3
    if profiles[i] == "['KVKSEGMENT_4']": profiles_np[i] = 4
    if profiles[i] == "['KVKSEGMENT_5']": profiles_np[i] = 5
    if profiles[i] == "['KVKSEGMENT_6']": profiles_np[i] = 6
    if profiles[i] == "['KVKSEGMENT_7']": profiles_np[i] = 7
    if profiles[i] == "['KVKSEGMENT_8']": profiles_np[i] = 8
    if profiles[i] == "['KVKSEGMENT_9']": profiles_np[i] = 9
    if profiles[i] == "['KVKSEGMENT_10']": profiles_np[i] = 10
    if profiles[i] == "['KVKSEGMENT_11']": profiles_np[i] = 11
    if profiles[i] == "['KVKSEGMENT_12']": profiles_np[i] = 12
    if profiles[i] == "['KVKSEGMENT_13']": profiles_np[i] = 13
    if profiles[i] == "['KVKSEGMENT_14']": profiles_np[i] = 14
    if profiles[i] == "['KVKSEGMENT_15']": profiles_np[i] = 15
    if profiles[i] == "['KVKSEGMENT_16']": profiles_np[i] = 16
    if profiles[i] == "['KVKSEGMENT_17']": profiles_np[i] = 17
    if profiles[i] == "['KVKSEGMENT_18']": profiles_np[i] = 18
    if profiles[i] == "['KVKSEGMENT_19']": profiles_np[i] = 19
    if profiles[i] == "['KVKSEGMENT_20']": profiles_np[i] = 20
    if profiles[i] == "['PV']": profiles_np[i] = 21
    if profiles[i] == "['WIND']": profiles_np[i] = 22
    if profiles[i] == "['E3A']": profiles_np[i] = 23
    if profiles[i] == "['E3B']": profiles_np[i] = 24
    if profiles[i] == "['E3C']": profiles_np[i] = 25
    if profiles[i] == "['E3D']": profiles_np[i] = 26


spectral_df["Numeration Profiles"] = profiles_np

spectral_df.to_csv('../Alliander_data/spectral_preparation_and_check.csv',chunksize=100, index = False)

spectral_df = spectral_df.drop(["Customer","Profiles"], axis = 1)

spectral_df.to_csv('../Alliander_data/spectral_training.csv',chunksize=100, index = False)

training_set_dimension = 1000

spectral_df = pd.read_csv('../Alliander_data/spectral_training.csv', nrows = training_set_dimension)
spectral_df_to_cluster = pd.read_csv('../Alliander_data/spectral_training.csv', skiprows = training_set_dimension+1, nrows = 1)
y = spectral_df["K-means cluster"].to_numpy() 
print("Here is y: ")
print(y)

spectral_df = spectral_df.drop(["K-means cluster"], axis = 1)
first_column = spectral_df_to_cluster.columns[0]
spectral_df_to_cluster = spectral_df_to_cluster.drop([first_column], axis = 1)
X = spectral_df.to_numpy()
print(np.shape(X))
row = spectral_df_to_cluster.to_numpy()

print(row)
print(np.shape(row))
print(np.shape(y))
# define model
model = GaussianProcessClassifier(1*RBF(1.0))
# fit model
model.fit(X, y)
# make a prediction
yhat = model.predict(row)
# summarize prediction
print('Predicted Class: %d' % yhat)
print(y)
spectral_df["Training"] = pd.DataFrame(data = y)
#spectral_df = spectral_df.append(pd.DataFrame(['SUM','MAX', 'Numeration profiles', 'Predictions'], columns=list(spectral_df)), ignore_index=True)
listOfSeries = [pd.Series(['MAX_pred', 'MAX_pred', 'Numeration profiles', 'Predictions'], index=spectral_df.columns ) ]
# Pass a list of series to the append() to add 
# multiple rows to dataframe
spectral_df = spectral_df.append(  listOfSeries,
                        ignore_index=True)
#spectral_df = spectral_df.append({'SUM': 0, 'MAX': 0, 'Numeration profiles': 0, 'Training': 0}, ignore_index=True)
row = np.append(row,yhat)
print(row)
spectral_df = spectral_df.append(pd.DataFrame(row.reshape(1,-1), columns=list(spectral_df)), ignore_index=True)
print(spectral_df)
spectral_df.to_csv('../Alliander_data/spectral_after_training.csv', index = False)



