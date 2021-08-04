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
# from sklearn import preprocessing
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates


# load data - be mindfull of size gv_meetdata_select.csv!
print("starting to load all the data")
profile_df = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/profielen.csv')
edsn_profile_df = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/edsn_profielen.csv')
connect_df = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/aansluiting_attributen.csv')
connect_df.sort_values(by=['RND_ID'], inplace=True)
meetdata_df = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/gv_meetdata_select.csv')
print("loaded all the databases")
# print(meetdata_df.head())
columns_of_interest = meetdata_df.columns[1:]
print(columns_of_interest)
# select all profiles from 'aansluiting_attributen' to get the
# relevant baseload profiles
connect_types = connect_df[["BASELOAD_PROFILE"]].values
profile_of_interest = np.unique(connect_types)

# convert names in profile_of_interest to match with column names in
# profielen.csv
for i in range(len(profile_of_interest)):
    try:
        profile_of_interest[i] = "KVKSEGMENT_{}".format(
            str(int(profile_of_interest[i])))
    except ValueError:
        pass




# select profiles of interest mentioned in the 'aansluiting_attributen' file
# from the big and intermediate profiles
big_interest_df = profile_df[
    profile_df.columns.intersection(profile_of_interest)]

edsn_interest_df = edsn_profile_df[
    edsn_profile_df.columns.intersection(profile_of_interest)]

interest_df = pd.concat([big_interest_df, edsn_interest_df],
                        axis=1)

print("interested profile matrix:")
print(interest_df.head(10))

interest_df_np = interest_df.to_numpy()

# recall that the column 0 is PV hence interest_df[1,1] = 0.047846 

# Let us compute the sum of all the profiles in order to understand how to normalize them


def compute_sum_columns(matrix):

    shape = np.shape(matrix)
    shape = np.asarray(shape)
    vector_sums = np.zeros(shape[1])
    
    for i in range(shape[1]):
        vector_sums[i] = np.sum(matrix[:,i])
    
    return vector_sums


vector_sums = compute_sum_columns(interest_df_np)

print(vector_sums)

''' From this we understand that all sectors are approximately normalized to 1 except PV and WIND which respectively sum up to: '''

print("PV sums up to: ", vector_sums[0])
print("WIND sums up to: ", vector_sums[1])

''' All the other sectors sum up approximately to 1, hence they are normalized with respect to their sum.
    Hence let us normalize also PV and WIND to their sums in order to compute the estimated loads later.
'''

def normalize_data_wrt_sum(matrix, vector_sums):
    shape = np.shape(matrix)
    shape = np.asarray(shape)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            matrix[i,j] = np.abs(matrix[i,j]/vector_sums[j])
        
    
    return matrix

interest_df_norm = normalize_data_wrt_sum(interest_df_np,vector_sums)

vector_sums_norm = compute_sum_columns(interest_df_norm)

'''Here we check that all the columns of the normalized df sum up to 1'''
print("The sum of the columns is:")
print(vector_sums_norm)
print("The new normalized matrix is:")
print(interest_df_norm)

''' Let us recreate the dataframe '''

column_names = list(interest_df.columns.values)

normalized_df = pd.DataFrame(data = interest_df_norm,
                  columns = column_names)

print(normalized_df.head())

''' So now we both have the dataframe in pandas and numpy both normalized wrt the sum 
    Hence we are ready to compute the estimated load
'''

print(connect_df.head())


def total_consumption_meetdata(data):
    column_list = list(data)
    column_list.remove("NUM")

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



estimated_loads, profiles, id_costumers = compute_estimated_loads(sums_pd,connect_df,normalized_df)
#print(estimated_loads)
'''
column_list = list(meetdata_df.columns)
column_list.remove("NUM")
column_list.remove("sum")
column_list_final = ['Costumer', 'Profile'] + column_list
'''
estimated_loads_df = pd.DataFrame(data = estimated_loads)
estimated_loads_df['id_customer'] = id_costumers
estimated_loads_df['profile'] = profiles


print(estimated_loads_df.head())

#print(profiles)
#print(id_costumers)


#print(estimated_loads_df.head())

'''
def remove_double_profiles(df):
    shape = np.shape(df)
    shape = np.asarray(shape)
    costumer_column_pd = df['Costumer']
    costumer_column_np = costumer_column_pd.to_numpy()
    cleaned_df_np = []
    for i in range(shape[0]):
        condition = False
        for j in range(shape[0]):
            if i != j:
                if costumer_column_np[i] == costumer_column_np[j]:
                    condition = True
        
        if condition == False:
            df_np = df.to_numpy()
            cleaned_df_np.append(df_np[i,:])
    
    cleaned_df = pd.DataFrame(data = cleaned_df_np, columns = ['Costumer', 'Profile', 'Estimated Load'])

    return cleaned_df


cleaned_df = remove_double_profiles(estimated_loads_df)
print(cleaned_df['Estimated Load'][2])
meetdata_df_np = meetdata_df.to_numpy()
estimated_load_company_1 = cleaned_df['Estimated Load'][2]
plt.plot(estimated_load_company_1[1:1000])
plt.plot(meetdata_df_np[2,1:1000])
plt.show()

#time_series_estimated_loads = pd.DataFrame(data = cleaned_df['Estimated Load'].to_numpy())

#print(time_series_estimated_loads)

def create_csv_estimated_loads(df, meetdata):
    #f = open('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/estimated_loads.csv', 'w')
    column_list = list(meetdata.columns)
    column_list.remove("NUM")
    column_list.remove("sum")
    #print(column_list)
    # create the csv writer
    shape = np.shape(df)
    shape = np.asarray(shape)
    column_list_final = ['Costumer', 'Profile'] + column_list

    
    for i in range(len(column_list)):
        for j in range(shape[0]):
            estimated_load_company_j = df['Estimated Load'][j]
            df.loc[j,column_list[i]] = estimated_load_company_j[i] 

    
    
    df_estimated_loads = pd.DataFrame(data = df, columns = column_list_final)
    
    
    #estimated_load_company_j = df['Estimated Load'][0]
    #df_estimated_loads.loc[0,column_list[1]] = estimated_load_company_j[1]

    #print(df_estimated_loads.head())
    df_estimated_loads.to_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/estimated_loads.csv')
    
    
    return

#create_csv_estimated_loads(estimated_loads_df,meetdata_df)



def create_csv_cleaned_estimated_loads(df):
    # open the file in the write mode
    f = open('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/cleaned_estimated_loads.csv', 'w')
    
    # create the csv writer
    writer = csv.writer(f)

    writer.writerow(['Costumer','Profile', 'Estimated Load'])

    # write multiple rows
    writer.writerows(df.to_numpy())

    # close the file
    f.close()

    return 

#create_csv_cleaned_estimated_loads(cleaned_df)
'''