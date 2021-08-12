import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
from datetime import datetime

profile = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/neat_profiles.csv')
customer = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/unique_meetdata.csv', index_col = 0, nrows = 100)

'''

This function is giving to the neat profiles the correct indexing and is removing the exciding rows.

profile_df is the one extracted from neat_profiles, which has been created through profielen.csv and edsn_profielen.csv
See load_save_profiles.py to learn how neat_profiles.csv is created.

meetdata_df can be extracted directly from gv_meetdata.csv or it is more convenient to use unique_meetdata.csv
See save_unique_profile_customer.py to learn how unique_meetdata.csv is created.

'''
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
profile = profile['KVKSEGMENT_6']
customer = customer.iloc[17]
# customer number 4 has profile: 'KVKSEGMENT_6'

print('profile: ')
print(profile.head())
print(profile.tail())
print('profile index type:')
print(type(profile.index[1]))
print('profile shape: ')
print(profile.shape)
print()

# transpose customer and convert index to datetime!!

print('customer: ')
print(customer.head())
print(customer.tail())

print("Customer index before conversion: ")

print(customer.index)

customer.index = pd.to_datetime(customer.index)

print("Customer index after conversion: ")
print(customer.index)

print('customer index type: ')
print(type(customer.index[1]))
idx1 = pd.Index(profile.index)
idx2 = pd.Index(customer.index)
print('customer shape: ')
print(customer.shape)
print('profile shape: ')
print(profile.shape)

print('Profile index: ')
print(profile.index)
print('Customer index: ')
print(customer.index[5663]) #this day corresponds to the last day of February, last 15 min: 2020-02-28 23:45:00
profile = profile.drop(index = np.arange(5664,5760,1))
print('Profile first element: ')
print(profile.iloc[0])
print('profile shape: ')
print(profile.shape)
profile.index = customer.index

print(profile.index)
print("Profile dataframe: ")    
print(profile)


# Here is the second method tried

profile_np = profile.to_numpy()
customer_np = customer.to_numpy()


def fun(x,a,b):
            
            y = a*(np.sum(customer_np)/np.amax(customer_np))*x**b 

            return y

fun(profile_np[1],1,1)

parameters, covariance = curve_fit(fun, profile_np, customer_np)

fit_A = parameters[0]
fit_B = parameters[1]
print(fit_A)
print(fit_B)

plt.plot(customer_np[2000:4000], label = r"real load")
plt.plot(fit_A*(np.sum(customer_np)/np.amax(customer_np))*profile_np[2000:4000]**fit_B, label = r"estimated load exponential")
plt.plot(np.sum(customer_np)*profile_np[2000:4000], label = r"estimated load linear")
plt.legend()
plt.show()

def optimal_parameters(profile_np, customer_np):
    
    def fun(x,a,b):
        y = a*(np.sum(customer_np)/np.amax(customer_np))*x**b 
        return y

    parameters, _ = curve_fit(fun, profile_np, customer_np)

    optimal_a = parameters[0]
    optimal_b = parameters[1]
        
    return optimal_a, optimal_b

fit_A, fit_B = optimal_parameters(profile_np, customer_np)

plt.plot(customer_np[2000:4000], label = r"real load")
plt.plot(fit_A*(np.sum(customer_np)/np.amax(customer_np))*profile_np[2000:4000]**fit_B, label = r"estimated load exponential")
plt.plot(np.sum(customer_np)*profile_np[2000:4000], label = r"estimated load linear")
plt.legend()
plt.show()
#Compute absolute error

error_exp = np.abs(fit_A*(np.sum(customer_np)/np.amax(customer_np))*profile_np**fit_B - customer_np)
error_lin = np.abs(np.sum(customer_np)*profile_np - customer_np)

absolute_error_exp = np.sum(error_exp)
absolute_error_lin = np.sum(error_lin)

print("Absolute error exp fit: ")
print(absolute_error_exp)
print("Absolute error lin fit: ")
print(absolute_error_lin)