import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

connect_df = pd.read_csv('../Alliander_data/sorted_connect.csv')
meetdata_df = pd.read_csv('../Alliander_data/unique_meetdata.csv', nrows=1500)
first_column = meetdata_df.columns[0]
#customer = meetdata_df
customer = meetdata_df.drop([first_column], axis=1)
profile = pd.read_csv('../Alliander_data/neat_profiles.csv')

def set_correct_index_profiles(profile_df, meetdata_df):
    
    index = pd.to_datetime(meetdata_df.iloc[3].index)

    print(index)
    profile_df = profile_df.drop(index = np.arange(5664,5760,1))

    print('profile shape: ')
    print(profile_df.shape)
    profile_df.index = index
    first_column = profile_df.columns[0]
# Delete first
    profile_df = profile_df.drop([first_column], axis=1)
    
    profile_df.to_csv('../Alliander_data/neat_profiles_fitted.csv')
    return profile_df

print(set_correct_index_profiles(profile,customer).head())    
# get customer number 4 with profile 'KVKSEGMENT_6'
profile = profile['KVKSEGMENT_8']
y = customer.iloc[7]
customer = customer.iloc[0:10]
print('Customers:')
print(customer)

# customer number 4 has profile: 'KVKSEGMENT_6'

print('profile: ')
print(profile.head())
print(profile.tail())

# transpose customer and convert index to datetime!!

print('customer: ')
print(customer.head())
print(customer.tail())

customer.index = pd.to_datetime(customer.index)

n = 10000
customer_np = customer.to_numpy()

customer_np = customer_np[:,1000:(1000+n)]

print(customer_np)


from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, ConstantKernel, RBF
'''
k0 = WhiteKernel(noise_level=0.3**2, noise_level_bounds=(0.1**2, 0.5**2))

k1 = ConstantKernel(constant_value=2) * \
  ExpSineSquared(length_scale=1.0, periodicity=40, periodicity_bounds=(35, 45))

kernel_1  = k0 + k1 
'''
k0 = WhiteKernel(noise_level=0.3**2, noise_level_bounds=(0.1**2, 0.5**2))

k1 = ConstantKernel(constant_value=100) * \
  ExpSineSquared(length_scale=100.0, periodicity=40, periodicity_bounds=(10, 155))

k2 = ConstantKernel(constant_value=100, constant_value_bounds=(10, 1000)) * \
  RBF(length_scale=100.0, length_scale_bounds=(1, 1e4)) 

k3 = ConstantKernel(constant_value=100) * \
  ExpSineSquared(length_scale=100.0, periodicity=20, periodicity_bounds=(10, 150))

k4 = ConstantKernel(constant_value=100) * \
  ExpSineSquared(length_scale=1, periodicity=4, periodicity_bounds=(1, 15))

kernel_1  = k0 + k1 + k2 + k3 + k4

from sklearn.gaussian_process import GaussianProcessRegressor

gp1 = GaussianProcessRegressor(
    kernel=kernel_1, 
    n_restarts_optimizer=10, 
    normalize_y=True,
    alpha=0.0
)

X = customer_np
print(X.shape)
X = np.transpose(X)
print(customer_np)
y = np.transpose(y)
y = y.to_numpy()
y = y[1000:1000+(1000+n)]

print(X)
print(y)
#X = data_df['t'].values.reshape(n, 1)
#y = data_df['y1'].values.reshape(n, 1)



#gp1.fit(X, y)
GaussianProcessRegressor(alpha=0.0,
                         kernel=WhiteKernel(noise_level=0.09) + 1.41**2 * ExpSineSquared(length_scale=1, periodicity=40),
                         n_restarts_optimizer=10, normalize_y=True)

# Generate predictions.
y_pred, y_std = gp1.predict(X, return_std=True)
print('Prediction: ')
print(y_pred)




