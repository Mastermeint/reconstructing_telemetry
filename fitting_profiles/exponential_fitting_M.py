import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
from scipy.optimize import fsolve, minimize

from load_data_scripts import load_save_profiles
from load_data_scripts import save_unique_profile_customer


connection, profiles = load_save_profiles.load_profiles()
customers = save_unique_profile_customer.unique_customer(nrows=200)


# drop leapday '2020-02-29'
def drop_leap_day(df):
    df = df[~((df.index.month == 2) & (df.index.day == 29))]
    return df


profiles = drop_leap_day(profiles)


def optimal_parameters(profile_np, customer_np):

    def exp_eq(x):
        a, b = x
        SJV = customer_np.sum()
        SJV_error = SJV*(a*profile_np.pow(b).sum() - 1)
        max_error = SJV*(np.amax(a*profile_np.pow(b)) -
                         np.amax(customer_np)/SJV)
        return (SJV_error, max_error)

    a, b = fsolve(exp_eq, (0.9, 0.9), factor=0.3, xtol=1.3e-10,
                  maxfev=800)

    return a, b


improvement_lis = [0]*customers.index.size
count = 0
better = []
worse = []

# print('connection: ', connection)
for customer_id in customers.index:
    # find corresponding profile
    profile_type = connection.loc[customer_id, 'BASELOAD_PROFILE']
    print('customer {} with profile type {}'.format(customer_id, profile_type))
    profile_customer = profiles[profile_type]
    customer = customers.loc[customer_id]

    res = optimal_parameters(profile_customer, customer)

    print('fitting res=(a,b): ', res)
    est_exp_load = res[0]*customer.sum()*profile_customer.pow(res[1])
    est_lin_load = customer.sum()*profile_customer

    error_exp = np.abs(est_exp_load - customer)
    error_lin = np.abs(est_lin_load - customer)

    absolute_error_exp = np.sqrt(np.square(error_exp).sum())
    absolute_error_lin = np.sqrt(np.square(error_lin).sum())

    print("Absolute error exp fit: ")
    print(absolute_error_exp)
    print("Absolute error lin fit: ")
    print(absolute_error_lin)
    print('relative improvement: ')
    improvement = (1 - absolute_error_exp/absolute_error_lin)*100
    improvement_lis[count] = improvement
    if improvement > 0:
        better.append((improvement, res))

    else:
        worse.append((improvement, res))
    count += 1
    print("{:.4f}%".format((1 - absolute_error_exp/absolute_error_lin)*100))
    print()
    print()

print('improvements: ')
print(improvement_lis)
print('better lis: ')
print(better)
print('better sum: ')
print(sum([x for (x,_) in better]))
# print('length {} and sum {}'.format(len(better), sum(better)))
print()

print('worse lis: ')
print(worse)
print('worse sum:')
print(sum([x for (x,_) in worse]))
#print('length {} and sum {}'.format(len(worse), sum(worse)))
plt.hist(improvement_lis)
plt.show()
# plt.show(block=True)

# too lazy to check screen: notify me when done
try:
    import os
    duration = 0.5  # seconds
    freq = 600  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
except _:
    import winsound
    duration = 1000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)


# # plot the estimates with the actual telemetry
# plt.figure(figsize=(12, 5))
# plt.xlabel('timestamp every 15 min')
# plt.ylabel('KWH usage')
#
# ax1 = est_exp_load.plot(color='red', grid=True,
#                         label="estimated load exponential",
#                         alpha=0.5)
# ax2 = est_lin_load.plot(color='green', grid=True,
#                         label="estimated load linear",
#                         alpha=0.5)
# ax3 = customer.plot(color='blue', grid=True, label="real load",
#                     alpha=0.3)
#
# # h1, l1 = ax1.get_legend_handles_labels()
# # h2, l2 = ax2.get_legend_handles_labels()
#
# plt.legend()
# plt.show()
