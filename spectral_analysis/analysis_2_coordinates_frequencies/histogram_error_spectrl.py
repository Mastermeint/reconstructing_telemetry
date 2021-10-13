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

number_bins_histograms = 450
number_samples_histograms = 400

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


def compute_error_customer_real(meetdata_df, df, number_estimations,profile ):
    error_customer = np.zeros(number_estimations)

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
        estimated_load_l = np.sum(customer_work_np)*profile_np

        error_customer[i] = relative_error(estimated_load_l,customer_work_np)

    return error_customer


errors_real = compute_error_customer_real(meetdata_df,df,number_samples_histograms, profile)
counts, bins = np.histogram(errors_real, bins = number_bins_histograms)
plt.hist(bins[:-1], bins, weights=counts, label = 'Distribution Relative Errors Linear Estimate')
plt.legend()
plt.xlabel('Relative Errors')
plt.ylabel('Number customers')
plt.show()


def compute_error_customer(meetdata_df, df, number_estimations ):
    error_customer = np.zeros(number_estimations)
    for i in range(number_estimations):
        number_cluster = df['K-means cluster'][i+1]
        estimated_load = estimated_load_cluster(df, number_cluster, meetdata_df)
        real_load = customer.iloc[i].to_numpy()#.drop([first_column])
        error_customer[i] = relative_error(estimated_load, real_load)
    return error_customer

errors_spectral = compute_error_customer(meetdata_df,df,number_samples_histograms)
counts, bins = np.histogram(errors_spectral, bins = number_bins_histograms)
plt.hist(bins[:-1], bins, weights=counts, label = 'Distribution Relative Errors Spectral Clustering')
plt.legend()
plt.xlabel('Relative Errors')
plt.ylabel('Number customers')
plt.show()

counts1, bins1 = np.histogram(errors_real, bins = number_bins_histograms)
counts2, bins2 = np.histogram(errors_spectral, bins = number_bins_histograms)
plt.hist(bins1[:-1], bins1, weights=counts1, label = 'Distribution Relative Errors Linear Estimate', alpha = 0.6)
plt.hist(bins2[:-1], bins2, weights=counts2, label = 'Distribution Relative Errors Spectral Clustering', alpha = 0.6)
plt.legend()
plt.xlabel('Relative Errors')
plt.ylabel('Number customers')  
plt.show()