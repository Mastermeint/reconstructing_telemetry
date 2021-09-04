import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

connect_df = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/sorted_connect.csv')
meetdata_df = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/gv_meetdata_select.csv', nrows=50)
first_column = meetdata_df.columns[0]
#customer = meetdata_df
customer = meetdata_df.drop([first_column], axis=1)
profile = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/neat_profiles.csv')

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
    
    profile_df.to_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/neat_profiles_fitted.csv')
    return profile_df

print(set_correct_index_profiles(profile,customer).head())    
# get customer number 4 with profile 'KVKSEGMENT_6'
profile = profile['KVKSEGMENT_8']

customer = customer.iloc[42]
customer_np = customer.to_numpy()
plt.plot(customer_np)
plt.show()

print('Customer 43:')
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

#print('Customer index: ')
#print(customer.index[5663]) #this day corresponds to the last day of February, last 15 min: 2020-02-28 23:45:00
profile = profile.drop(index = np.arange(5664,5760,1)) #this is dropping the last day of Febuary because of the leaf year
print('Profile shape: ')
print(profile.shape)
profile.index = customer.index
print('Profile index: ')
print(profile.index)
print("Profile dataframe: ")    
print(profile)
n = 2500
customer_np = customer.to_numpy()
customer_np = customer_np[25000:(25000+n)]

sns.set_style(
    style='darkgrid', 
    rc={'axes.facecolor': '.9', 'grid.color': '.8'}
)
sns.set_palette(palette='deep')
sns_c = sns.color_palette(palette='deep')
#matplotlib inline
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['figure.dpi'] = 100
plt.show()
# Number of samples. 

# Generate "time" variable. 

t = np.arange(n)

data_df = pd.DataFrame({'t' : t})

# Generate seasonal variables. 
def seasonal(t, amplitude, period):
    """Generate a sinusoidal curve."""
    y1 = amplitude * np.sin((2*np.pi)*t/period) 
    return y1

# Add two seasonal components. 
data_df['s1'] = data_df['t'].apply(lambda t : seasonal(t, amplitude=2, period=40))

# Define target variable. 
data_df['y1'] = data_df['s1']

#fig, ax = plt.subplots()
#sns.lineplot(x='t', y='s1', data=data_df, color=sns_c[0], label='s1', ax=ax) 
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#ax.set(title='Seasonal Component', xlabel='t', ylabel='');
#plt.show()
# Set noise standard deviation. 
sigma_n = 0.3

data_df['epsilon'] = np.random.normal(loc=0, scale=sigma_n, size=n)
# Add noise to target variable. 
data_df ['y1'] = data_df ['y1'] + data_df ['epsilon']

#fig, ax = plt.subplots()
#sns.lineplot(x='t', y='y1', data=data_df, color=sns_c[0], label='y1', ax=ax) 
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#ax.set(title='Sample Data 1', xlabel='t', ylabel='');
#plt.show()

from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, ConstantKernel, RBF
#'''
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
'''
from sklearn.gaussian_process import GaussianProcessRegressor

gp1 = GaussianProcessRegressor(
    kernel=kernel_1, 
    n_restarts_optimizer=10, 
    normalize_y=True,
    alpha=0.0
)

X = np.arange(0,n,1).reshape(n,1)
y = customer_np.reshape(n, 1)

#X = data_df['t'].values.reshape(n, 1)
#y = data_df['y1'].values.reshape(n, 1)

prop_train = 0.7
n_train = round(prop_train * n)

X_train = X[:n_train]
y_train = y[:n_train]

X_test = X[n_train:]
y_test = y[n_train:]

gp1_prior_samples = gp1.sample_y(X=X_train, n_samples=100)

fig, ax = plt.subplots()
for i in range(100):
    sns.lineplot(x=X_train[...,0], y = gp1_prior_samples[:, i], color=sns_c[1], alpha=0.2, ax=ax)
sns.lineplot(x=X_train[...,0], y=y_train[..., 0], color=sns_c[0], label='y1', ax=ax) 
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set(title='GP1 Prior Samples', xlabel='t');
plt.show()

gp1.fit(X_train, y_train)
GaussianProcessRegressor(alpha=0.0,
                         kernel=WhiteKernel(noise_level=0.09) + 1.41**2 * ExpSineSquared(length_scale=1, periodicity=40),
                         n_restarts_optimizer=10, normalize_y=True)

# Generate predictions.
y_pred, y_std = gp1.predict(X, return_std=True)

data_df['y_pred'] = y_pred
data_df['y_std'] = y_std
data_df['y_pred_lwr'] = data_df['y_pred'] - 2*data_df['y_std']
data_df['y_pred_upr'] = data_df['y_pred'] + 2*data_df['y_std']

fig, ax = plt.subplots()

ax.fill_between(
    x=data_df['t'], 
    y1=data_df['y_pred_lwr'], 
    y2=data_df['y_pred_upr'], 
    color=sns_c[2], 
    alpha=0.15, 
    label='credible_interval'
)
data_df ['y1'] = pd.DataFrame(data = customer_np)

sns.lineplot(x='t', y='y1', data=data_df, color=sns_c[0], label = 'y1', ax=ax)
sns.lineplot(x='t', y='y_pred', data=data_df, color=sns_c[2], label='y_pred', ax=ax)

ax.axvline(n_train, color=sns_c[3], linestyle='--', label='train-test split')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set(title='Prediction Sample 1', xlabel='t', ylabel='');
plt.show()