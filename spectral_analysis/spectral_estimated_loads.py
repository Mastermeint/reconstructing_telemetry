import numpy as np
from scipy import stats, signal, fft
from statsmodels.regression.linear_model import yule_walker
from functions import produce_differences_df, spec_pgram, spec_taper, spec_ci, spec_ar, plot_spec, get_quads, compute_df_frequencies
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.graphics.tsaplots as tsaplots
from statsmodels.tsa.arima_model import ARMA
from statsmodels.compat import lzip
from scipy import stats

df = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/spectral_frequencies_cluster_profiles.csv')
connect_df = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/sorted_connect.csv')
meetdata_df = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/unique_meetdata.csv')
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
profile = profile['KVKSEGMENT_6']

customer = customer.iloc[4]
print('Customer 4:')
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

''' Now the data has been cleaned and we can start the exponential fit. '''
# Here is the first method tried

profile_np = profile.to_numpy()
customer_np = customer.to_numpy()


number_costumer = df['Customer']
print(number_costumer)
number_costumer = number_costumer.to_numpy()
print(number_costumer)
profiles = []

def estimated_load_cluster(df, number_cluster, meetdata_df):
    df =df[df['K-means cluster'] == number_cluster]
    print(df)
    
    numbers_customers = df["Customer"]
    print(numbers_customers)

    real_loads = []
    
    meetdata_df.rename(columns={ meetdata_df.columns[0]: "Placeholder" }, inplace = True)
    index = meetdata_df["Placeholder"]
    index_np = index.to_numpy()
    print(index.to_numpy())
    meetdata_df.index = index_np
    
    meetdata_df.drop(columns = "Placeholder", inplace= True)

    print("Meetdata: ")
    print(meetdata_df)
    print("Index meetdata: ")
    print(meetdata_df.index)
    
    #meetdata_df.to_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/trial.csv')
    #print(numbers_customers.values)
    for i in numbers_customers.values: 
        #print(i)  
        #print(meetdata_df.loc[i])
        real_loads.append(meetdata_df.loc[i])

    real_loads = pd.DataFrame(data = real_loads)
    
    #real_loads.to_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/estimated_loads.csv')
    real_loads_np = real_loads.to_numpy()
    print(real_loads_np)
    estimated_load = np.around(np.mean(real_loads_np, axis = 0),2)
        
    return estimated_load

estimated_load_cluster_1 = estimated_load_cluster(df,18,meetdata_df)
#print(estimated_load_cluster(df,1,meetdata_df))

print(estimated_load_cluster_1)
#estimated_load_cluster_1 = estimated_load_cluster(df,1,meetdata_df)

#Compute absolute error
#customer_np = meetdata_df.loc[9]
print(customer_np)
error_exp = np.abs(estimated_load_cluster_1 - customer_np)
error_lin = np.abs(np.sum(customer_np)*profile_np - customer_np)

absolute_error_spec = np.sum(error_exp)
absolute_error_lin = np.sum(error_lin)

print("Absolute error spec fit: ")
print(absolute_error_spec)
print("Absolute error lin fit: ")
print(absolute_error_lin)


plt.plot(customer_np[1000:2000], label = r"real load")
plt.plot(estimated_load_cluster_1[1000:2000], label = r"estimated load spectral")
plt.plot(np.sum(customer_np)*profile_np[1000:2000], label = r"estimated load linear")
plt.legend()
plt.show()
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


estimated_load_s = estimated_load_cluster_1
real_load = customer_np

r_error_s = relative_error(estimated_load_s,real_load)

print("The relative error for the spectral estimate is: ")
print(r_error_s)

estimated_load_l = np.sum(customer_np)*profile_np

r_error_l = relative_error(estimated_load_l,real_load)

print("The relative error for the linear estimate is: ")
print(r_error_l)
