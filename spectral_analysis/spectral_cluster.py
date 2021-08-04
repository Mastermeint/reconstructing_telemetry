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

meetdata_df = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/gv_meetdata_select.csv', nrows=1000)
print(meetdata_df.head())
#meetdata_df.T.plot()
#plt.show()
meetdata_df.dropna(inplace=True)

df_frequencies_np = compute_df_frequencies(meetdata_df)

df_frequecies_pd = pd.DataFrame(data = df_frequencies_np, columns = ['Costumer', 'Spectral Frequency'])

print(df_frequecies_pd)

import csv

# open the file in the write mode
f = open('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/spectral_frequencies.csv', 'w')

# create the csv writer
writer = csv.writer(f)

writer.writerow(['Costumer', 'Spectral Frequency'])

# write multiple rows
writer.writerows(df_frequecies_pd.to_numpy())

# close the file
f.close()