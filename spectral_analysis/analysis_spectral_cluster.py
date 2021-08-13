import numpy as np
from scipy import stats, signal, fft
from statsmodels.regression.linear_model import yule_walker
from functions import produce_differences_df, spec_pgram, spec_taper, spec_ci, spec_ar, plot_spec, select_profile_costumer
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

df = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/spectral_frequencies_cluster.csv')
connect_df = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/sorted_connect.csv')

''' Now we want to compute some statistics concerning the quality of the cluster obtained 
'''

print(df.count())

count = df.count()


number_costumer = df['Customer']
print(number_costumer)
number_costumer = number_costumer.to_numpy()
print(number_costumer)
profiles = []

def select_profile_costumer_cluster(connect_df,number_costumer):
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

for i in range(len(number_costumer)):
    #print(number_costumer[i])
    profile_costumer = select_profile_costumer_cluster(connect_df,number_costumer[i])
    #print(profile_costumer)
    profiles.append(profile_costumer)


df['Profiles'] = profiles
first_column = df.columns[0]

# Delete first
df = df.drop([first_column], axis=1)
df.to_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/spectral_frequencies_cluster_profiles.csv', index = False)


def histogram_spec_cluster(df,number_cluster, connect_df):
    df =df[df['K-means cluster'] == number_cluster]
    print(df)
    #count = df.count()
    numbers_customers = df["Customer"]
    #range_loop = np.arange(1,count.iloc[0]+1,1)
    
    nb_PV = 0
    nb_WIND = 0
    nb_001 = 0
    nb_002 = 0
    nb_003 = 0
    nb_004 = 0
    nb_005 = 0
    nb_006 = 0
    nb_007 = 0
    nb_008 = 0
    nb_009 = 0
    nb_010 = 0
    nb_011 = 0
    nb_012 = 0
    nb_013 = 0
    nb_014 = 0
    nb_015 = 0
    nb_016 = 0
    nb_017 = 0
    nb_018 = 0
    nb_019 = 0
    nb_020 = 0
    nb_E3A = 0
    nb_E3B = 0
    nb_E3C = 0
    nb_E3D = 0

    
    for i in range(len(numbers_customers)):
        profile_costumer = select_profile_costumer_cluster(connect_df,number_costumer[i])
        for j in range(len(profile_costumer)):
            if profile_costumer[j] == 'KVKSEGMENT_1': nb_001 += 1
            if profile_costumer[j] == 'KVKSEGMENT_2': nb_002 += 1
            if profile_costumer[j] == 'KVKSEGMENT_3': nb_003 += 1
            if profile_costumer[j] == 'KVKSEGMENT_4': nb_004 += 1
            if profile_costumer[j] == 'KVKSEGMENT_5': nb_005 += 1
            if profile_costumer[j] == 'KVKSEGMENT_6': nb_006 += 1
            if profile_costumer[j] == 'KVKSEGMENT_7': nb_007 += 1
            if profile_costumer[j] == 'KVKSEGMENT_8': nb_008 += 1
            if profile_costumer[j] == 'KVKSEGMENT_9': nb_009 += 1
            if profile_costumer[j] == 'KVKSEGMENT_10': nb_010 += 1
            if profile_costumer[j] == 'KVKSEGMENT_11': nb_011 += 1
            if profile_costumer[j] == 'KVKSEGMENT_12': nb_012 += 1
            if profile_costumer[j] == 'KVKSEGMENT_13': nb_013 += 1
            if profile_costumer[j] == 'KVKSEGMENT_14': nb_014 += 1
            if profile_costumer[j] == 'KVKSEGMENT_15': nb_015 += 1
            if profile_costumer[j] == 'KVKSEGMENT_16': nb_016 += 1
            if profile_costumer[j] == 'KVKSEGMENT_17': nb_017 += 1
            if profile_costumer[j] == 'KVKSEGMENT_18': nb_018 += 1
            if profile_costumer[j] == 'KVKSEGMENT_19': nb_019 += 1
            if profile_costumer[j] == 'KVKSEGMENT_20': nb_020 += 1
            if profile_costumer[j] == 'PV': nb_PV += 1
            if profile_costumer[j] == 'WIND': nb_WIND += 1
            if profile_costumer[j] == 'E3A': nb_E3A += 1
            if profile_costumer[j] == 'E3B': nb_E3B += 1
            if profile_costumer[j] == 'E3C': nb_E3C += 1
            if profile_costumer[j] == 'E3D': nb_E3D += 1
    
    dataframe_histogram = pd.Series(data = [nb_PV, nb_WIND, nb_001, nb_002, nb_003, nb_004, nb_005, nb_006, nb_007, nb_008, nb_009, nb_010, nb_011, nb_012, 
    nb_013, nb_014, nb_015, nb_016, nb_017, nb_018, nb_019, nb_020, nb_E3A, nb_E3B, nb_E3C, nb_E3D], index = ['PV', 'WIND', 'KVKSEGMENT_1','KVKSEGMENT_2',
    'KVKSEGMENT_3','KVKSEGMENT_4','KVKSEGMENT_5','KVKSEGMENT_6','KVKSEGMENT_7','KVKSEGMENT_8','KVKSEGMENT_9','KVKSEGMENT_10','KVKSEGMENT_11',
    'KVKSEGMENT_12','KVKSEGMENT_13','KVKSEGMENT_14','KVKSEGMENT_15','KVKSEGMENT_16','KVKSEGMENT_17','KVKSEGMENT_18','KVKSEGMENT_19','KVKSEGMENT_20',
    'E3A','E3B','E3C','E3D'])

    total_customers_in_cluster = (nb_PV + nb_WIND + nb_001 + nb_002 + nb_003 + nb_004 + nb_005 + nb_006 + nb_007 + nb_008 + nb_009 
        + nb_010 + nb_011 + nb_012 + nb_013 + nb_014 + nb_015 + nb_016 + nb_017 + nb_018 + nb_019 + nb_020 + nb_E3A + nb_E3B  +nb_E3C + nb_E3D)

    return dataframe_histogram, number_cluster, total_customers_in_cluster
    
dataframe_histogram, number_cluster, total_customers_in_cluster = histogram_spec_cluster(df,0,connect_df)
#print(dataframe_histogram.head())


dataframe_histogram_np = dataframe_histogram.to_numpy()
plt.bar(['PV', 'WIND', '001','002',
    '003','004','005','006','007','008','009','010','011',
    '012','013','014','015','016','017','018','019','020',
    'E3A','E3B','E3C','E3D'],dataframe_histogram_np)
plt.title('Spectral cluster number: {}. Total customers in cluster: {}'.format(number_cluster,total_customers_in_cluster))
plt.show()


dataframe_histogram_0, number_cluster_0, total_customers_in_cluster_0 = histogram_spec_cluster(df,8,connect_df)
dataframe_histogram_1, number_cluster_1, total_customers_in_cluster_1 = histogram_spec_cluster(df,9,connect_df)
dataframe_histogram_2, number_cluster_2, total_customers_in_cluster_2 = histogram_spec_cluster(df,0,connect_df)
dataframe_histogram_3, number_cluster_3, total_customers_in_cluster_3 = histogram_spec_cluster(df,1,connect_df)
#dataframe_histogram_4, number_cluster_4 = histogram_spec_cluster(df,0,connect_df)

dataframe_histogram_np_0 = dataframe_histogram_0.to_numpy()
dataframe_histogram_np_1 = dataframe_histogram_1.to_numpy()
dataframe_histogram_np_2 = dataframe_histogram_2.to_numpy()
dataframe_histogram_np_3 = dataframe_histogram_3.to_numpy()
#dataframe_histogram_np_4 = dataframe_histogram_4.to_numpy()


fig, axs = plt.subplots(2, 2)
axs[0, 0].bar(['PV', 'WD', '1','2',
    '3','4','5','6','7','8','9','10','11',
    '12','13','14','15','16','17','18','19','20',
    'A','B','C','D'],dataframe_histogram_np_0)
axs[0, 0].set_title('Spectral cluster number: {}. Total customers in cluster: {}'.format(number_cluster_0,total_customers_in_cluster_0))
axs[0, 1].bar(['PV', 'WD', '1','2',
    '3','4','5','6','7','8','9','10','11',
    '12','13','14','15','16','17','18','19','20',
    'A','B','C','D'],dataframe_histogram_np_1)
axs[0, 1].set_title('Spectral cluster number: {}. Total customers in cluster: {}'.format(number_cluster_1,total_customers_in_cluster_1))
axs[1, 0].bar(['PV', 'WD', '1','2',
    '3','4','5','6','7','8','9','10','11',
    '12','13','14','15','16','17','18','19','20',
    'A','B','C','D'],dataframe_histogram_np_2)
axs[1, 0].set_title('Spectral cluster number: {}. Total customers in cluster: {}'.format(number_cluster_2,total_customers_in_cluster_2))
axs[1, 1].bar(['PV', 'WD', '1','2',
    '3','4','5','6','7','8','9','10','11',
    '12','13','14','15','16','17','18','19','20',
    'A','B','C','D'],dataframe_histogram_np_3)
axs[1, 1].set_title('Spectral cluster number: {}. Total customers in cluster: {}'.format(number_cluster_3,total_customers_in_cluster_3))

for ax in axs.flat:
    ax.set(xlabel='x-label', ylabel='y-label')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.show()

