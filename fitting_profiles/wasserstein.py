import time
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

start = time.time()
# imports for data management
# from load_data_scripts import load_save_profiles
from load_data_scripts import save_unique_profile_customer


# _, profile = load_save_profiles.load_profiles()
customers = save_unique_profile_customer.unique_customer(nrows=300, reload=1)

date_num = customers.shape[1]
uniform_np = (1/date_num)*np.ones((date_num,))

comp_start = time.time()

# uniform_dist = customers.apply(lambda x: print(x.shape),
#                                axis=1)
print('rows: ', customers.index)
uniform_dist = customers.apply(lambda x:
                               wasserstein_distance(x.values, uniform_np),
                               axis=1)

print('uniform distances: ')
print(uniform_dist)
end = time.time()
print('total running time: ', end - start)
print('computational time: ', end - comp_start)
# start1 = time.time()
# wass = wasserstein_distance(profile_6, profile_7)
# end1 = time.time()
# print('wass 6 and 7: ')
# print(wass)
# print('done in: ', end1-start1)
#
# start2 = time.time()
# wass = wasserstein_distance(profile_7, profile_8)
# end2 = time.time()
# print('wass 7 and 8: ')
# print(wass)
# print('dont in ', end2 - start2)
#
#
# # profile_6.plot()
# # profile_7.plot()
# plt.legend()
# plt.show()
#
# profile_7.plot()
# profile_8.plot()
# plt.legend()
# plt.show()
