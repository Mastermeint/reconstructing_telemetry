import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
# import pandas as pd
import numpy as np
from scipy.optimize import fsolve  # , minimize

from load_data_scripts import load_save_profiles
from load_data_scripts import save_unique_profile_customer


connection, profiles = load_save_profiles.load_profiles()
customers = save_unique_profile_customer.unique_customer(nrows=200)


# drop leapday '2020-02-29'
def drop_leap_day(df):
    df = df[~((df.index.month == 2) & (df.index.day == 29))]
    return df


profiles = drop_leap_day(profiles)


def optimal_parameters(profile, customer):

    def exp_eq(x):
        a, b = x
        SJV = customer.sum()
        SJV_error = SJV*(a*profile.pow(b).sum() - 1)
        max_error = SJV*(a*profile.max()**b -
                         customer.max()/SJV)
        return (SJV_error, max_error)

    # max_ratio = (np.amax(customer_np)/customer_np.sum())/np.amax(profile_np)
    print('max profile: ', profile.max())
    print('max customer: ', customer.max())
    SJV = customer.sum()
    guess_b = np.log(customer.max()/SJV)/np.log(profile.max())
    guess_a = customer.sum()/profile.pow(guess_b).sum()
    guess = (guess_a, guess_b)

    print('initial guesses (a,b); ', guess)

    a, b = fsolve(exp_eq, guess, factor=0.3)
    # guess = (1.0, 1.0)
    # a, b = diagbroyden(exp_eq, guess)
    # x = Symbol('x')
    # a, b = solve(exp_eq, x)

    print('result of error function(should be around (0,0)): ')
    print(exp_eq((a, b)))
    return a, b


improvement_lis = [0]*customers.index.size
count = 0
better = []
worse = []


def get_customer_with_profile(profile):
    # select all customers having this profile
    sub_customers_id = connection[connection['BASELOAD_PROFILE'] == profile]
    sub_customers = customers[sub_customers_id]
    return sub_customers


for customer_id in customers.index:
    # find corresponding profile
    profile_type = connection.loc[customer_id, 'BASELOAD_PROFILE']
    print('customer {} with profile type {}'.format(customer_id, profile_type))
    profile_customer = profiles[profile_type]
    customer = customers.loc[customer_id]
    print('this is profile: ', profile_customer.name)
    print('with sum: ', profile_customer.sum())
    print('and customer has SJV of: ', customer.sum())

    res = optimal_parameters(profile_customer, customer)
    # res = optimal_parameters_5(profile_customer, customer)
    print('fitting res=(a,b): ', res)
    a, b = res
    SJV = customer.sum()
    max_error = SJV/100*(a*(profile_customer.max()**b) -
                         customer.max()/SJV)
    print('max diff: ', max_error)
    est_exp_load = a*customer.sum()*(profile_customer.pow(b))
    est_lin_load = customer.sum()*profile_customer

    error_exp = np.abs(est_exp_load - customer)
    error_lin = np.abs(est_lin_load - customer)

    absolute_error_exp = np.sqrt(np.square(error_exp).sum())
    absolute_error_lin = np.sqrt(np.square(error_lin).sum())

    print('relative improvement: ')
    improvement = (1 - absolute_error_exp/absolute_error_lin)*100
    improvement_lis[count] = improvement
    if improvement > 0:
        better.append((customer_id, improvement, res))
        print('better: ')
    else:
        worse.append((customer_id, improvement, res))
        print('worse: ')
    count += 1
    print("{:.4f}%".format((1 - absolute_error_exp/absolute_error_lin)*100))


    plt.figure(figsize=(12, 5))
    plt.xlabel('timestamp every 15 min')
    plt.ylabel('KWH usage')

    plt.title("customer {} from profile {}".format(
        customer_id, profile_type))
    ax1 = est_exp_load.plot.line(color='red', grid=True,
                                 label="estimated load exponential",
                                 alpha=0.8)
    ax2 = est_lin_load.plot.line(color='green', grid=True,
                                 label="estimated load linear",
                                 alpha=0.5)
    ax3 = customer.plot.line(color='blue', grid=True,
                             label="real load",
                             alpha=0.3)

    plt.legend()
    plt.show()
    print("{:.4f}%".format((1 - absolute_error_exp/absolute_error_lin)*100))
    print()
    print()

print('improvements: ')
print(improvement_lis[:10])
print('better lis: ')
print(better[:10])
print('better sum: ')
print(sum([x for (_, x, _) in better]))
# print('length {} and sum {}'.format(len(better), sum(better)))
print()

print('worse lis: ')
print(worse[:10])
print('worse sum:')
print(sum([x for (_, x, _) in worse]))
# print('length {} and sum {}'.format(len(worse), sum(worse)))
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
