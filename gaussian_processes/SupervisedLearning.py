import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
import pandas as pd

meetdata_df = pd.read_csv('../Alliander_data/unique_meetdata.csv', nrows = 11)
first_column = meetdata_df.columns[0]
#customer = meetdata_df
meetdata_df = meetdata_df.drop([first_column], axis=1)

X = meetdata_df.to_numpy()
row = X[10,:]
print(row)
X = X[:9,:]
print(X)
y = [8,21,6,21,20,17,4,23,20]
# define model
model = GaussianProcessClassifier()
# fit model
model.fit(X, y)

# make a prediction
yhat = model.predict([row])
# summarize prediction
print('Predicted Class: %d' % yhat)





training_set_dimension = 10
meetdata_df = pd.read_csv('../Alliander_data/unique_meetdata.csv', nrows = training_set_dimension)
meetdata_df_to_cluster = pd.read_csv('../Alliander_data/unique_meetdata.csv', skiprows = training_set_dimension+2, nrows = 1)
first_column = meetdata_df.columns[0]
values_first_column = meetdata_df.iloc[:,0]
#customer = meetdata_df
meetdata_df = meetdata_df.drop([first_column], axis=1)

first_column_1 = meetdata_df_to_cluster.columns[0]
meetdata_df_to_cluster = meetdata_df_to_cluster.drop([first_column_1], axis=1)

numbers_customers = values_first_column

print("Dataframe numbers customers: ")
print(numbers_customers)

connect_df = pd.read_csv('../Alliander_data/sorted_connect.csv')

y = np.zeros(training_set_dimension)

numbers_customers = numbers_customers.to_numpy()
print(numbers_customers)
profiles = []

def select_profile_costumer_cluster(connect_df,number_costumer):
    connect_df.sort_values(by = 'RND_ID')
    random_ids = connect_df['RND_ID']
    random_ids_np = random_ids.to_numpy()
    profile = []
    for i in range(len(random_ids_np)):
        if random_ids_np[i] == number_costumer:
            baseload_profiles = connect_df['BASELOAD_PROFILE']

            if baseload_profiles[i].isdigit() == True:
                baseload_profiles[i] = "KVKSEGMENT_{}".format(baseload_profiles[i])

            profile.append(baseload_profiles[i]) 

    return profile

for i in range(len(numbers_customers)):
    #print(number_costumer[i])
    profile_costumer = select_profile_costumer_cluster(connect_df,numbers_customers[i])
    #print(profile_costumer)
    profiles.append(profile_costumer)

print("numbers customers: ")
print(numbers_customers)
print("profiles: ")
print(profiles)

for i in range(len(numbers_customers)):
    profile_costumer = select_profile_costumer_cluster(connect_df,numbers_customers[i])
    
    print(profiles[i])
    if profiles[i] == ['KVKSEGMENT_1']: y[i] = 1
    if profiles[i] == ['KVKSEGMENT_2']: y[i] = 2
    if profiles[i] == ['KVKSEGMENT_3']: y[i] = 3
    if profiles[i] == ['KVKSEGMENT_4']: y[i] = 4
    if profiles[i] == ['KVKSEGMENT_5']: y[i] = 5
    if profiles[i] == ['KVKSEGMENT_6']: y[i] = 6
    if profiles[i] == ['KVKSEGMENT_7']: y[i] = 7
    if profiles[i] == ['KVKSEGMENT_8']: y[i] = 8
    if profiles[i] == ['KVKSEGMENT_9']: y[i] = 9
    if profiles[i] == ['KVKSEGMENT_10']: y[i] = 10
    if profiles[i] == ['KVKSEGMENT_11']: y[i] = 11
    if profiles[i] == ['KVKSEGMENT_12']: y[i] = 12
    if profiles[i] == ['KVKSEGMENT_13']: y[i] = 13
    if profiles[i] == ['KVKSEGMENT_14']: y[i] = 14
    if profiles[i] == ['KVKSEGMENT_15']: y[i] = 15
    if profiles[i] == ['KVKSEGMENT_16']: y[i] = 16
    if profiles[i] == ['KVKSEGMENT_17']: y[i] = 17
    if profiles[i] == ['KVKSEGMENT_18']: y[i] = 18
    if profiles[i] == ['KVKSEGMENT_19']: y[i] = 19
    if profiles[i] == ['KVKSEGMENT_20']: y[i] = 20
    if profiles[i] == ['PV']: y[i] = 21
    if profiles[i] == ['WIND']: y[i] = 22
    if profiles[i] == ['E3A']: y[i] = 23
    if profiles[i] == ['E3B']: y[i] = 24
    if profiles[i] == ['E3C']: y[i] = 25
    if profiles[i] == ['E3D']: y[i] = 26

print("Here is y: ")
print(y)
y = np.random.randint(1,26,training_set_dimension)
X = meetdata_df.to_numpy()
print(np.shape(X))
row = meetdata_df_to_cluster.to_numpy()
#row = np.transpose(row)
print(row)
print(np.shape(row))
print(np.shape(y))
# define model
model = GaussianProcessClassifier()
# fit model
model.fit(X, y)
# make a prediction
yhat = model.predict(row)
# summarize prediction
print('Predicted Class: %d' % yhat)


