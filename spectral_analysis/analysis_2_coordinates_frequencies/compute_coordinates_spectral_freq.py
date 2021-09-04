import numpy as np
from scipy import stats, signal, fft
from statsmodels.regression.linear_model import yule_walker
from functions import compute_df_coordinates_frequencies, normalize_meetdata
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

meetdata_df = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/unique_meetdata.csv', index_col=0, nrows = 1500)

#meetdata_df = normalize_meetdata(meetdata_df)
meetdata_df.sort_index(inplace=True)
print("loaded all the databases")
print(meetdata_df.head())
#meetdata_df.T.plot()
#plt.show()
meetdata_df.dropna(inplace=True)

print("Started computing the spectral frequencies.")
df_frequencies_np = compute_df_coordinates_frequencies(meetdata_df)
print("Finished computing the spectral frequencies.")


df_frequecies_pd = pd.DataFrame(data = df_frequencies_np, columns = ['Spectral Frequency 1 X', 'Spectral Frequency 1 Y', 'Spectral Frequency 2 X', 'Spectral Frequency 2 Y'])
#df_frequecies_pd.drop(columns = 'Placeholder')
df_frequecies_pd.insert(loc = 0, column = 'Customer', value = meetdata_df.index) 

print(df_frequecies_pd)

import csv

df_frequecies_pd.to_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/spectral_multiple_frequencies.csv', chunksize= 100, index = False)


