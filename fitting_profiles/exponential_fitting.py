import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import fsolve, minimize
import math

profile = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/neat_profiles.csv')
customer = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/unique_meetdata.csv', index_col = 0, nrows = 200)

'''

This function is giving to the neat profiles the correct indexing and is removing the exciding rows,
i.e. the one due to the fact that 2020 was a leaf year.

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

''' Now the data has been cleaned and we can start the exponential fit. '''
# Here is the first method tried

profile_np = profile.to_numpy()
customer_np = customer.to_numpy()

def optimal_parameters(profile_np, customer_np):
    
    def equations(p):
        a, b = p
        return (np.sum(a*np.sum(customer_np)*profile_np**b) - np.sum(customer_np), np.amax(a*np.sum(customer_np)*profile_np**b) - np.amax(customer_np))

    a, b =  fsolve(equations, (1, 1))

    print(equations((a, b)))

    return a, b

#Here is the second method that I tried

def optimal_parameters_1(profile_df, customer_df):
    
    profile_np = profile_df.to_numpy()
    customer_np = customer_df.to_numpy()
    
    def equation_1(p):
        a, b = p
        return (np.sum(a*np.sum(customer_np)*profile_np**b) - np.sum(customer_np))
        #, np.amax(a*profile_np**b) - np.amax(customer_np))

    def equation_2(p):
        a, b = p
        return (np.amax(a*np.sum(customer_np)*profile_np**b) - np.amax(customer_np))

    x0 = [1,1]
    opt_1 = minimize(equation_1,x0) #, args = (p0,c0))
    res_1 = opt_1.x

    opt_2 = minimize(equation_2,x0) #, args = (p0,c0))
    res_2 = opt_2.x

    res = (res_1 + res_2)/2
    return res

#Here is the third method that I tried

def optimal_parameters_2(profile_df, customer_df):
    
    profile_np = profile_df.to_numpy()
    customer_np = customer_df.to_numpy()
    
    def fun_1(x,a,b):
        y = np.sum(a*x*profile_np**b) - x
        return y

    def fun_2(x,a,b):
        y = np.amax(a*x*profile_np**b) - np.amax(customer_np)
        return y

    parameters_1, _ = curve_fit(fun_1, np.sum(customer_np), 0)
    parameters_2, _ = curve_fit(fun_2, np.sum(customer_np), 0)

    fit_A_1 = parameters_1[0]
    fit_B_1 = parameters_1[1]
    
    fit_A_2 = parameters_2[0]
    fit_B_2 = parameters_2[1]

    return (fit_A_1+fit_A_2)/2 , (fit_B_1+fit_B_2)/2 


'''
   ATTENTION: after experimenting with the three methods turns out that the only
   one that is working is the first one, were we try to solve the system numerically
   starting with (1,1) point. However, after trying out many other profiles as well,
   turns out that such method converges to a useful solution just sometimes.
   The good news is that since I set (1,1), when it doesn't converge it remains 
   statically fixed to its starting point (1,1) which corresponds to the linear approximation.
   Note that this method may suffer of overflow problems when not converging.
   In general I see that when it does converge it gives a relative improvement on the error
   of about 0.5%/1%, if it doesn't converge it does not give any improvement because it gives
   back the linear approximation (old one) in the point (1,1).

'''

print("Results: ")

res = optimal_parameters(profile,customer)
print(res[0])
print(res[1])

fit_A, fit_B = optimal_parameters(profile_np, customer_np)

plt.plot(customer_np, label = r"real load")
plt.plot(res[0]*np.sum(customer_np)*profile_np**res[1], label = r"estimated load exponential")
#plt.plot(np.sum(customer_np)*profile_np[2000:4000], label = r"estimated load linear")
plt.legend()
plt.show()

#Compute absolute error

error_exp = np.abs(res[0]*np.sum(customer_np)*profile_np**res[1] - customer_np)
error_lin = np.abs(np.sum(customer_np)*profile_np - customer_np)

absolute_error_exp = np.sum(error_exp)
absolute_error_lin = np.sum(error_lin)

print("Absolute error exp fit: ")
print(absolute_error_exp)
print("Absolute error lin fit: ")
print(absolute_error_lin)

#Let us compute the relative error

def relative_error(estimated_load, real_load):

    #The two lines below do the same job, however gives error if there is 
    #even just a zero value in the real_load time series.

    #relative_error_vector = np.divide(np.abs(estimated_load - real_load),real_load)
    #print(relative_error_vector)

    relative_error_vector = np.zeros(len(estimated_load))

    for i in range(len(estimated_load)):
        if real_load[i] != 0:
            relative_error_vector[i] = np.abs(estimated_load[i] - real_load[i])/real_load[i]

    relative_error = np.mean(relative_error_vector)

    return relative_error

estimated_load_e = res[0]*np.sum(customer_np)*profile_np**res[1]
real_load = customer_np

r_error_e = relative_error(estimated_load_e,real_load)

print("The relative error for the exponential estimate is: ")
print(r_error_e)

estimated_load_l = np.sum(customer_np)*profile_np

r_error_l = relative_error(estimated_load_l,real_load)

print("The relative error for the linear estimate is: ")
print(r_error_l)