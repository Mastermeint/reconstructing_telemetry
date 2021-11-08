import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

# imports for data management
# from load_data_scripts import load_save_profiles
from load_data_scripts import save_unique_profile_customer

start = time.time()

print('starting to read stuff')
# _, profile = load_save_profiles.load_profiles()
customers = save_unique_profile_customer.unique_customer(nrows=5000)

print('read all the stuff')
customer_num = len(customers.index)
print('numbers of customers left: ', customer_num)

date_num = customers.shape[1]
print('datenum: ', date_num)
# uniform_np = (1/date_num)*np.ones((date_num,))


# uniform_dist = customers.apply(lambda x: print(x.shape),
#                                axis=1)
# there are 672 points of 15 min intervals in a week
weekly_oscillation = ((-1*np.cos(np.arange(date_num)/672*np.pi*2)+1))
weekly_oscillation = weekly_oscillation/np.sum(weekly_oscillation)
weekly_df = pd.DataFrame(weekly_oscillation,
                         index=customers.columns)

monthly_oscillation = (-1*np.cos(np.arange(date_num)/2880*np.pi*2)+1)
montly_oscillation = monthly_oscillation/np.sum(monthly_oscillation)
monthly_df = pd.DataFrame(monthly_oscillation,
                          index=customers.columns)

wass_weekly_dist = customers.apply(lambda x:
                                   wasserstein_distance((x.values)/x.sum(),
                                                        weekly_oscillation),
                                   axis=1)

wass_monthly_dist = customers.apply(lambda x:
                                    wasserstein_distance((x.values)/x.sum(),
                                                         monthly_oscillation),
                                    axis=1)

end = time.time()
print('this took {} seconds '.format(end-start))

print()
print()
print('WEEKLY STUFF:')
print('weekly fun:')
print(weekly_df.head())

wass_weekly_dist = wass_weekly_dist*date_num
print('min, max values are: ', wass_weekly_dist.min(), ' ',
      wass_weekly_dist.max())
print('now removing the upper quantile:')
q = wass_weekly_dist.quantile(0.95)
wass_weekly_dist = wass_weekly_dist[wass_weekly_dist < q]

print(' new min, max values are: ', wass_weekly_dist.min(), ' ',
      wass_weekly_dist.max())
print('wass_dist head: ')
print(wass_weekly_dist.head())

print('MONTHLY STUFF: ')
print('monthly fun:')
print(monthly_df.head())
wass_monthly_dist = wass_monthly_dist*date_num
print('min, max values are: ', wass_monthly_dist.min(), ' ',
      wass_monthly_dist.max())
print('now removing the upper quantile:')
q = wass_monthly_dist.quantile(0.95)
wass_monthly_dist = wass_monthly_dist[wass_monthly_dist < q]

print(' new min, max values are: ', wass_monthly_dist.min(), ' ',
      wass_monthly_dist.max())
print('wass_dist head: ')
print(wass_monthly_dist.head())


print('PLOTTING STUFF')
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
wass_weekly_dist.plot.hist(ax=axes[0], bins=50, title='weekly distances')
wass_monthly_dist.plot.hist(ax=axes[1], bins=50, title='monthly distances')
# plt.xlim((0, 0.10))
# wass_weekly_dist.plot.hist(bins=50, logx=True)
plt.show()

# TODO: create own pairwise_distances with hookers and blackjack

# comp_start = time.time()
# print('comp_start: ', comp_start)
# wass_time_lis = []
#
# pairwise_np = np.zeros((customer_num, customer_num))
# for row in range(customer_num):
#     for column in range(row, customer_num):
#         start_wass = time.time()
#         dist = wasserstein_distance(customers.iloc[row],
#                                     customers.iloc[column])
#         pairwise_np[row, column] = dist
#         end_wass = time.time()
#         wass_time_lis.append(end_wass - start_wass)
#
# pairwise_matrix = pd.DataFrame(pairwise_np,
#                                index=customers.index,
#                                columns=customers.index)
# # uniform_dist = pairwise_distances(customers.iloc[:300],
#                                   # metric=wasserstein_distance)
#
# print('uniform distances: ')
# print(pairwise_matrix)
# end = time.time()
# print('total running time: ', end - start)
# print('computational time: ', end - comp_start)
# plt.hist(wass_time_lis)
# plt.show()
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
