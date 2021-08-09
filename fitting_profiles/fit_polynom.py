# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
import pandas as pd


# imports for data management
from load_data_scripts import load_save_profiles
from load_data_scripts import save_unique_profile_customer


def emperical_fit(data, a, b):
    return a*(data)**b


_, profile = load_save_profiles.load_profiles()
customer = save_unique_profile_customer.unique_profile(nrows=4, reload=1)

# get customer number 4 with profile 'KVKSEGMENT_6'
profile = profile['KVKSEGMENT_6']
customer = customer.iloc[4]
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
customer.index = pd.to_datetime(customer.index)
print('customer index type: ')
print(type(customer.index[1]))
idx1 = pd.Index(profile.index)
idx2 = pd.Index(customer.index)
print('customer shape: ')
print(customer.shape)
print('difference: ')
print(idx1.difference(idx2))
# popt, pcov = curve_fit(emperical_fit, profile, customer)
#
# profile.plot()
# customer.plot(alpha=0.3)
# plt.legend()
# plt.show()
# plt.plot(customer.columns, emperical_fit(profile, *popt), 'g--',
#          label='fit')
#
# plt.legend()
# plt.show()
