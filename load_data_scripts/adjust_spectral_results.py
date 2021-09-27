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

df = pd.read_csv('../Alliander_data/spectral_frequencies_minutes.csv')
df.to_csv('../Alliander_data/spectral_frequencies_no_gap.csv', index=False)

spectral_frequencies_nogaps = pd.read_csv('../Alliander_data/spectral_frequencies_no_gap.csv')

spectral_frequencies_nogaps["Spectral Frequency"] = 4*spectral_frequencies_nogaps["Spectral Frequency"]

spectral_frequencies_nogaps.to_csv('../Alliander_data/spectral_frequencies_hours.csv')

df = pd.read_csv('../Alliander_data/spectral_frequencies_hours.csv')
# If you know the name of the column skip this
first_column = df.columns[0]
# Delete first
df = df.drop([first_column], axis=1)
df.to_csv('../Alliander_data/spectral_frequencies_hours.csv', index=False)
