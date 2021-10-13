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

df = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/spectral_multiple_frequencies_cluster_profiles.csv')
connect_df = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/sorted_connect.csv')
meetdata_df = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/unique_meetdata.csv')
first_column = meetdata_df.columns[0]
customer = meetdata_df#.drop([first_column], axis=1)
profile = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/neat_profiles.csv')

print("The data has been fully loaded.")

number_samples_to_plot = 20

def set_correct_index_profiles(profile_df, meetdata_df):
    
    index = pd.to_datetime(meetdata_df.iloc[3].index)

    print(index)
    profile_df = profile_df.drop(index = np.arange(5664,5760,1))

    print('profile shape: ')
    print(profile_df.shape)
    profile_df.index = index
    first_column = profile_df.columns[0]

    profile_df = profile_df.drop([first_column], axis=1)
    
    profile_df.to_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/neat_profiles_fitted.csv')
    return profile_df

meetdata_df_dropped_first_column = meetdata_df.drop([first_column], axis=1)
profile = set_correct_index_profiles(profile,meetdata_df_dropped_first_column)

def estimated_load_cluster(df_work, number_cluster, meetdata_df_work):
    df_work =df_work[df_work['K-means cluster'] == number_cluster]
    
    numbers_customers = df_work["Customer"]
    real_loads = []
    
    meetdata_df_work.rename(columns={ meetdata_df_work.columns[0]: "Placeholder" }, inplace = True)
    index = meetdata_df_work["Placeholder"]
    index_np = index.to_numpy()
    meetdata_df_work.index = index_np
    
    for i in numbers_customers.values: 
        real_loads.append(meetdata_df_work.loc[i])

    real_loads = pd.DataFrame(data = real_loads)
    
    real_loads_np = real_loads.to_numpy()
    estimated_load = np.around(np.mean(real_loads_np, axis = 0),2)
        
    return estimated_load

def relative_error(estimated_load, real_load):

    relative_error_vector = np.zeros(len(estimated_load))

    for i in range(len(estimated_load)):
        if real_load[i] != 0:
            relative_error_vector[i] = np.abs(estimated_load[i] - real_load[i])/real_load[i]

    relative_error = np.mean(relative_error_vector)

    return relative_error


def compute_loads_linear(meetdata_df, df, number_estimations,profile ):
    estimated_loads_matrix = np.zeros((number_estimations,meetdata_df.shape[1]-1))

    for i in range(number_estimations):
        customer_work = meetdata_df.iloc[i]
        customer_work = customer_work.drop([first_column])
        segment = df['Profiles'][i+1]
        segment = segment.replace('[','')
        segment = segment.replace(']','')
        segment = segment.replace("'","")
        profile_work = profile[segment]
        profile_np = profile_work.to_numpy()
        customer_work_np = customer_work.to_numpy()
        estimated_loads_matrix[i,:] = np.sum(customer_work_np)*profile_np

    return estimated_loads_matrix


loads_linear = compute_loads_linear(meetdata_df,df,number_samples_to_plot, profile)

print("The linear loads have been computed successfully.")

def compute_loads_spectral(meetdata_df, df, number_estimations ):
    estimated_loads_matrix = np.zeros((number_estimations,meetdata_df.shape[1]))
    #meetdata_df_dropped_first_column = meetdata_df.drop([first_column], axis=1)
    for i in range(number_estimations):
        number_cluster = df['K-means cluster'][i+1]
        estimated_loads_matrix[i,:] = estimated_load_cluster(df, number_cluster, meetdata_df)
    return estimated_loads_matrix

loads_spectral = compute_loads_spectral(meetdata_df,df,number_samples_to_plot)

loads_spectral = loads_spectral[:,1:]

print("The spectral loads have been computed successfully. Now we start plotting.")

print(loads_spectral)
print(loads_spectral.shape)
print(loads_linear)
print(loads_linear.shape)
meetdata_df_np = meetdata_df.to_numpy()

for i in range(number_samples_to_plot):
    plt.plot(meetdata_df_np[i,1:], label = r"real load",color = 'blue',alpha = 0.3)
    plt.plot(loads_spectral[i,:], label = r"estimated load spectral",color = 'red',alpha = 0.8)
    plt.plot(loads_linear[i,:], label = r"estimated load linear",color = 'green', alpha = 0.5)
    plt.legend()
    plt.show()

