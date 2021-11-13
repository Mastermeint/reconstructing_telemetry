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


'''
    1) Load the unique_data and from the first column read the number that it is displayed. This number will 
    give you the ID_number of the customer
    2) Use the ID_number to read from the estimated_loads file the row of the costumer. The number will correspond 
    if taken from the first column and it should match with the number in the column ID_customer of the estimated_loads
    3) Plot the time series obtained both from the unique_data file and from the estimated_loads file
'''
print("Loading profiles")
unique_meetdata_df = pd.read_csv('../Alliander_data/unique_meetdata.csv', nrows= 50)
estimated_loads_df = pd.read_csv('../Alliander_data/estimated_loads.csv', nrows= 70)
ID_number = unique_meetdata_df.iloc[:,0]
# remember that row 0 corresponds to customer 2 in this case
print("ID customer from the unique_meetdata: ")
print(ID_number)
print("First costumer to be considered unique_meetdata: ")
print(ID_number[0])
print("Let us verify that it coincides with the one from the estimated loads: ")
ID_customer = estimated_loads_df['ID_customer']
print(ID_customer)
print("Corresponding first costumer from the estimated_loads: ")
estimated_costumer_profile = estimated_loads_df[estimated_loads_df["ID_customer"].isin([ID_number[0]])] 
print(estimated_costumer_profile)
print("Select the corresponding profile from the unique_meetdata: ")
real_costumer_profile = unique_meetdata_df[unique_meetdata_df.iloc[:,0].isin([ID_number[0]])]
print(real_costumer_profile)

''' Now that I have developed a method to select the profiles from the data I 
    will create a function that given an input profile will give back the 
    two: estimated_loads and unique_meetdata profiles
'''

'''Note that the number of customer must be loaded at the beginning otherwise it will not work'''

def select_time_series(number_customer, unique_meetdata_df, estimated_loads_df):
   
    estimated_costumer_profile = estimated_loads_df[estimated_loads_df["ID_customer"].isin([number_customer])] 
    print(estimated_costumer_profile)
    print("Select the corresponding profile from the unique_meetdata: ")
    real_costumer_profile = unique_meetdata_df[unique_meetdata_df.iloc[:,0].isin([number_customer])]    
    print(real_costumer_profile)

    estimated_time_series = estimated_costumer_profile.iloc[0,3:35002]
    real_time_series = real_costumer_profile.iloc[0,1:35000]

    return estimated_time_series, real_time_series

estimated_time_series, real_time_series = select_time_series(2,unique_meetdata_df,estimated_loads_df)

print(estimated_time_series)
print(real_time_series)

''' Now that they have the same length we can plot them '''
estimated_time_series_plot = estimated_time_series.iloc[:1000]
real_time_series_plot = real_time_series.iloc[:1000]

plt.plot(real_time_series_plot.to_numpy(), label = r'real load')
plt.plot(estimated_time_series_plot.to_numpy(), label = r'estimated load')
plt.ylabel('Time series loads')
plt.xlabel('Number of time steps 15 minutes')
plt.title('Comparison estimated-real load customer 2')
plt.legend()
plt.show()


estimated_time_series_3, real_time_series_3 = select_time_series(3,unique_meetdata_df,estimated_loads_df)
estimated_time_series_4, real_time_series_4 = select_time_series(4,unique_meetdata_df,estimated_loads_df)
estimated_time_series_5, real_time_series_5 = select_time_series(5,unique_meetdata_df,estimated_loads_df)
estimated_time_series_6, real_time_series_6 = select_time_series(6,unique_meetdata_df,estimated_loads_df)

estimated_time_series_plot_3 = estimated_time_series_3.iloc[:1000]
real_time_series_plot_3 = real_time_series_3.iloc[:1000]
estimated_time_series_plot_4 = estimated_time_series_4.iloc[:1000]
real_time_series_plot_4 = real_time_series_4.iloc[:1000]
estimated_time_series_plot_5 = estimated_time_series_5.iloc[:1000]
real_time_series_plot_5 = real_time_series_5.iloc[:1000]
estimated_time_series_plot_6 = estimated_time_series_6.iloc[:1000]
real_time_series_plot_6 = real_time_series_6.iloc[:1000]


fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(real_time_series_plot_3.to_numpy(), label = r'real load')
axs[0, 0].plot(estimated_time_series_plot_3.to_numpy(), label = r'estimated load')
axs[0, 0].set_title('third company')
axs[0, 1].plot(real_time_series_plot_4.to_numpy(), label = r'real load')
axs[0, 1].plot(estimated_time_series_plot_4.to_numpy(), label = r'estimated load')
axs[0, 1].set_title('fourth company')
axs[1, 0].plot(real_time_series_plot_5.to_numpy(), label = r'real load')
axs[1, 0].plot(estimated_time_series_plot_5.to_numpy(), label = r'estimated load')
axs[1, 0].set_title('fifth company')
axs[1, 1].plot(real_time_series_plot_6.to_numpy(), label = r'real load')
axs[1, 1].plot(estimated_time_series_plot_6.to_numpy(), label = r'estimated load')
axs[1, 1].set_title('sixth company')

for ax in axs.flat:
    ax.set(xlabel='x-label', ylabel='y-label')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.legend()
plt.show()

'''
    Now I will compute and plot the error
'''

error_3 = real_time_series_plot_3.to_numpy() - estimated_time_series_plot_3.to_numpy()
error_4 = real_time_series_plot_4.to_numpy() - estimated_time_series_plot_4.to_numpy()
error_5 = real_time_series_plot_5.to_numpy() - estimated_time_series_plot_5.to_numpy()
error_6 = real_time_series_plot_6.to_numpy() - estimated_time_series_plot_6.to_numpy()

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(error_3)
axs[0, 0].set_title('error third company')
axs[0, 1].plot(error_4)
axs[0, 1].set_title('error fourth company')
axs[1, 0].plot(error_5)
axs[1, 0].set_title('error fifth company')
axs[1, 1].plot(error_6)
axs[1, 1].set_title('error sixth company')

for ax in axs.flat:
    ax.set(xlabel='x-label', ylabel='y-label')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.show()
