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
import seaborn as sns

import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels as sm
import matplotlib
import matplotlib.pyplot as plt
import csv


print("starting to load all the data")
profile_df = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/profielen.csv')
edsn_profile_df = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/edsn_profielen.csv')
neat_profiles_df = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/neat_profiles.csv')
connect_df = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/sorted_connect.csv')
meetdata_df = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/gv_meetdata_select.csv',index_col=0)

meetdata_df.sort_index(inplace=True)
print("loaded all the databases")
# print(meetdata_df.head())
columns_of_interest = meetdata_df.columns[1:]
print(columns_of_interest)


''' Start with the analysis of the meetdata'''


def total_consumption_meetdata(data):
    column_list = list(data)

    data["sum"] = data[column_list].sum(axis=1)

    return data

#print(meetdata_df)
sum_meetdata = total_consumption_meetdata(meetdata_df)
print(sum_meetdata.head())

sums_pd = sum_meetdata['sum']

sums_np = sums_pd.to_numpy()



def select_profile_costumer(connect_df,number_costumer):
    connect_df.sort_values(by = 'RND_ID')
    random_ids = connect_df['RND_ID']
    random_ids_np = random_ids.to_numpy()
    profile = []
    for i in range(len(random_ids_np)):
        if random_ids_np[i] == number_costumer:
            baseload_profiles = connect_df['BASELOAD_PROFILE']

            if baseload_profiles[i].isdigit() == True:
                baseload_profiles[i] = "KVKSEGMENT_{}".format(baseload_profiles[i])

            profile.append(baseload_profiles[i]) 

    return profile

print(select_profile_costumer(connect_df,1))

'''In this function I will use the previous one to compute the estimated loads for each customer '''

def compute_estimated_loads(sums_pd, connect_df, normalized_df):
    sums_np = sums_pd.to_numpy()
    number_costumer = np.arange(1,5000,1)
    estimated_loads = []
    profiles = []
    id_costumers = []
    for i in number_costumer-1:
        profile_costumer = select_profile_costumer(connect_df,number_costumer[i])
        for j in range(len(profile_costumer)):
            consumption_from_df = normalized_df[profile_costumer[j]]
            consumption_from_df_np = consumption_from_df.to_numpy()
            estimated_load_profile = np.dot(sums_np[i],consumption_from_df_np)
            

            #print(estimated_load_profile)
            #estimated_load_profile = np.asarray(estimated_load_profile)
            #array = np.asarray([[i+1 , profile_costumer[j] , estimated_load_profile]])
            id_costumers.append(i+1)
            profiles.append(profile_costumer[j])
            estimated_loads.append(estimated_load_profile)

    return estimated_loads, profiles, id_costumers



estimated_loads, profiles, id_costumers = compute_estimated_loads(sums_pd,connect_df,neat_profiles_df)
#print(estimated_loads)
'''
column_list = list(meetdata_df.columns)
column_list.remove("NUM")
column_list.remove("sum")
column_list_final = ['Costumer', 'Profile'] + column_list
'''
estimated_loads_df = pd.DataFrame(data = estimated_loads)
estimated_loads_df.insert(0, 'ID_customer', id_costumers)
estimated_loads_df.insert(1, 'Profiles', profiles)
#estimated_loads_df['id_customer'] = id_costumers
#estimated_loads_df['profile'] = profiles


print(estimated_loads_df.head())

estimated_loads_df.to_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/estimated_loads.csv')