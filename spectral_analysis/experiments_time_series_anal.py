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


shape = np.shape(meetdata_df)
shape = np.asarray(shape)

meetdata_df_np = np.asarray(meetdata_df.dropna())

''' Here I want to compare two companies with similar time series, analyse the periodograms of the two companies which should 
estimate the spectral density, and later estimate an autoregressive model in order to predict the behaviour of the time series:
finally we want to be able to classify the companies based on their estimated spectral density.'''

first_company = meetdata_df_np[8,1:]
second_company = meetdata_df_np[7,1:]

plt.figure(figsize=(12, 4))
plt.plot(second_company)
plt.plot(first_company)
plt.show()

'''Here below I performed the analysis on the two companies without taking the differences, 
   in order to see how it goes even without clear stationarity in the time series.
'''

plt.figure(figsize=(12, 4))

s1 = spec_pgram(first_company, spans = [51,51], taper = 0.1, plot=True)
plt.show()

print('1 / Peak: ', round(1/s1['freq'][np.argmax(s1['spec'])]/15, 2), 'minutes')

plt.figure(figsize=(12, 4))
s2 = spec_ar(first_company, plot=True)
plt.show()

print('1 / Peak: ', round(1/s2['freq'][np.argmax(s2['spec'])]/15, 2), 'minutes')

plt.figure(figsize=(12, 4))
plt.plot(s1['freq'], s1['spec'])
plt.plot(s2['freq'], s2['spec'])
plt.xlabel('Frequency')
plt.ylabel('Log Spectrum')
plt.yscale('log')
plt.show()

plt.figure(figsize=(12, 4))

s1_2 = spec_pgram(second_company, spans = [51,51], taper = 0.1, plot=True)
plt.show()

print('1 / Peak: ', round(1/s1_2['freq'][np.argmax(s1_2['spec'])]/15, 2), 'minutes')

plt.figure(figsize=(12, 4))
s2_2 = spec_ar(second_company, plot=True)
plt.show()

print('1 / Peak: ', round(1/s2_2['freq'][np.argmax(s2_2['spec'])]/15, 2), 'minutes')

plt.figure(figsize=(12, 4))
plt.plot(s1_2['freq'], s1_2['spec'])
plt.plot(s2_2['freq'], s2_2['spec'])
plt.xlabel('Frequency')
plt.ylabel('Log Spectrum')
plt.yscale('log')
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(s1['freq'], s1['spec'])
plt.plot(s2['freq'], s2['spec'])
plt.plot(s1_2['freq'], s1_2['spec'])
plt.plot(s2_2['freq'], s2_2['spec'])
plt.xlabel('Frequency')
plt.ylabel('Log Spectrum')
plt.yscale('log')
plt.show()

'''Here I finished the analysis without taking the differences, the results are not bad but 
   there is space for improvements, so now I will try with taking the differences.     
'''

diff_meetdata_pd = produce_differences_df(meetdata_df)

print(meetdata_df.head())
print(diff_meetdata_pd.head())

shape = np.shape(diff_meetdata_pd)
shape = np.asarray(shape)

diff_meetdata_np = np.asarray(diff_meetdata_pd)

diff_first_company = diff_meetdata_np[8,1:] #567
diff_second_company = diff_meetdata_np[7,1:]

plt.figure(figsize=(12, 4))
plt.plot(diff_second_company)
plt.plot(diff_first_company)
plt.show()

plt.figure(figsize=(12, 4))

s1_diff = spec_pgram(diff_first_company, spans = [51,51], taper = 0.1, plot=True)
plt.show()

print('1 / Peak: ', round(1/s1_diff['freq'][np.argmax(s1_diff['spec'])]/15, 2), 'minutes')

plt.figure(figsize=(12, 4))
s2_diff = spec_ar(diff_first_company, plot=True)
plt.show()

print('1 / Peak: ', round(1/s2_diff['freq'][np.argmax(s2_diff['spec'])]/15, 2), 'minutes')

plt.figure(figsize=(12, 4))
plt.plot(s1_diff['freq'], s1_diff['spec'])
plt.plot(s2_diff['freq'], s2_diff['spec'])
plt.xlabel('Frequency')
plt.ylabel('Log Spectrum')
plt.yscale('log')
plt.show()

''' Let us check stationarity'''

#
# Check for stationarity of the time-series data
# We will look for p-value. In case, p-value is less than 0.05, the time series
# data can said to have stationarity
#
from statsmodels.tsa.stattools import adfuller
#
# Run the test
diff_first_company_df = pd.DataFrame(data = diff_first_company)
print(diff_first_company_df)
#
df_stationarityTest = adfuller(diff_first_company_df, autolag='AIC')
print(df_stationarityTest)
#
# Check the value of p-value
#
print("P-value: ", df_stationarityTest[1])

first_company_df = pd.DataFrame(data = first_company)
df_stationarityTest_1 = adfuller(first_company_df, autolag='AIC')
print(df_stationarityTest_1)
#
# Check the value of p-value
#
print("P-value: ", df_stationarityTest_1[1])

#
# Next step is to find the order of AR model to be trained
# for this, we will plot partial autocorrelation plot to assess
# the direct effect of past data on future data
#

''' Here I performed a standard time series analysis '''

from statsmodels.graphics.tsaplots import plot_pacf
fig,ax = plt.subplots(nrows=1, ncols=2, figsize=(16,6));
ax[0].plot(diff_first_company_df);
ax[0].set_title('Differences first company')

tsaplots.plot_acf(diff_first_company_df.dropna(), lags=50, zero=False, ax = ax[1])
tsaplots.plot_pacf(diff_first_company_df.dropna(), lags=50, zero=False, ax=None)

ax[1].set_ylabel(r'$\rho_\ell$')
ax[1].set_xlabel(r'$\ell$');
#plt.show()

p = 5
q = 5

model = ARMA(diff_first_company_df.dropna(), order=(p,q))  # dropna() s.t. no missing values, and xr not overwritten
res = model.fit()
print('BIC: ', res.bic)
print(res.summary2())


from matplotlib import pyplot

diff_first_company_df.hist()
pyplot.show()

''' I implement all the tests that I know to check stationarity of the time series: '''

''' Here I apply the ADFuller test tests H0: non-stationarity '''

from statsmodels.tsa.stattools import adfuller

result = adfuller(diff_first_company_df.values, autolag='AIC')
t_stat, p_value, _, _, critical_values, _  = adfuller(diff_first_company_df.values, autolag='AIC')
print(f'ADF Statistic: {t_stat:.2f}')
for key, value in critical_values.items():
     print('Critial Values:')
     print(f'   {key}, {value:.2f}')

print(f'\np-value: {p_value:.2f}')
print("Non-Stationary") if p_value > 0.05 else print("Stationary")

''' Here I can look at the most significant statistics for the dataset '''

print(get_quads(diff_first_company_df))

''' Here I apply the Kwiakowski-Phillips-Schmidt-Shin (KPSS) test that tests H0: stationarity '''

from statsmodels.tsa.stattools import kpss
t_stat, p_value, _, critical_values = kpss(diff_first_company_df.values, nlags='auto')

print(f'ADF Statistic: {t_stat:.2f}')
for key, value in critical_values.items():
     print('Critial Values:')
     print(f'   {key}, {value:.2f}')

print(f'\np-value: {p_value:.2f}')
print("Stationary") if p_value > 0.05 else print("Non-Stationary")

''' Here I apply the Zivot and Andrews test that tests the presence of structural breaks '''

from statsmodels.tsa.stattools import zivot_andrews
t_stat, p_value, critical_values, _, _ = zivot_andrews(diff_first_company_df.values)
print(f'Zivot-Andrews Statistic: {t_stat:.2f}')
for key, value in critical_values.items():
     print('Critial Values:')
     print(f'   {key}, {value:.2f}')

print(f'\np-value: {p_value:.2f}')
print("Non-Stationary") if p_value > 0.05 else print("Stationary")

