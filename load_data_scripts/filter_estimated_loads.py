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
estimated_loads_df = pd.read_csv('../Alliander_data/estimated_loads.csv',nrows = 100)
#meetdata_df = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/gv_meetdata_select.csv',nrows = 10, index_col=0)

#meetdata_df.sort_index(inplace=True)
print("loaded all the databases")


def remove_double_profiles(df):
    shape = np.shape(df)
    shape = np.asarray(shape)
    costumer_column_pd = df['ID_customer']
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

    cleaned_df = pd.DataFrame(data = cleaned_df_np)

    return cleaned_df


cleaned_df = remove_double_profiles(estimated_loads_df)
cleaned_df = cleaned_df.rename(columns={cleaned_df.columns[0] : "x", cleaned_df.columns[1]: "ID_customer", cleaned_df.columns[2]: "Profile"}) #, inplace=True)

cleaned_df = cleaned_df.drop(["x"], axis=1)
print(cleaned_df.head())
cleaned_df.to_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/neat_estimated_loads.csv')

#meetdata_df_np = meetdata_df.to_numpy()
#estimated_load_company_1 = cleaned_df['Estimated Load'][2]
#plt.plot(estimated_load_company_1[1:1000])
#plt.plot(meetdata_df_np[2,1:1000])
#plt.show()

#time_series_estimated_loads = pd.DataFrame(data = cleaned_df['Estimated Load'].to_numpy())

#print(time_series_estimated_loads)
