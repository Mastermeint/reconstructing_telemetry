import pandas as pd
import numpy as np
from pandas.core.indexes import base
# from sklearn import preprocessing
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from scipy import stats, signal, fft
from statsmodels.regression.linear_model import yule_walker
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
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import zivot_andrews


def spec_taper(x, p=0.1):
    """
    Computes a tapered version of x, with tapering p.
    
    Adapted from R's stats::spec.taper.
    """
    
    p = np.r_[p]
    assert np.all((p >= 0) & (p < 0.5)), "'p' must be between 0 and 0.5"
    
    x = np.r_[x].astype('float64')
    original_shape = x.shape
    
    assert len(original_shape) <= 2, "'x' must have at most 2 dimensions"
    while len(x.shape) < 2:
        x = np.expand_dims(x, axis=1)
    
    nr, nc = x.shape
    if len(p) == 1:
        p = p * np.ones(nc)
    else:
        assert len(p) == nc, "length of 'p' must be 1 or equal the number of columns of 'x'"
    
    for i in range(nc):
        m = int(np.floor(nr * p[i]))
        if m == 0:
            continue
        w = 0.5 * (1 - np.cos(np.pi * np.arange(1, 2 * m, step=2)/(2 * m)))
        x[:, i] = np.r_[w, np.ones(nr - 2 * m), w[::-1]] * x[:, i]
    
    x = np.reshape(x, original_shape)
    return x

def spec_ci(df, coverage=0.95):
    """
    Computes the confidence interval for a spectral fit, based on the number of degrees of freedom.
    
    Adapted from R's stats::plot.spec.
    """
    
    assert coverage >= 0 and coverage < 1, "coverage probability out of range [0, 1)"
    
    tail = 1 - coverage
    
    phi = stats.chi2.cdf(x=df, df=df)
    upper_quantile = 1 - tail * (1 - phi)
    lower_quantile = tail * phi
    
    return df / stats.chi2.ppf([upper_quantile, lower_quantile], df=df)

def spec_pgram(x, xfreq=1, spans=None, kernel=None, taper=0.1, pad=0, fast=True, demean=False, detrend=True, 
               plot=True, **kwargs):
    """
    Computes the spectral density estimate using a periodogram.  Optionally, it also:
    - Uses a provided kernel window, or a sequence of spans for convoluted modified Daniell kernels.
    - Tapers the start and end of the series to avoid end-of-signal effects.
    - Pads the provided series before computation, adding pad*(length of series) zeros at the end.
    - Pads the provided series before computation to speed up FFT calculation.
    - Performs demeaning or detrending on the series.
    - Plots results.
    
    Implemented to ensure compatibility with R's spectral functions, as opposed to reusing scipy's periodogram.
    
    Adapted from R's stats::spec.pgram.
    """
    def daniell_window_modified(m):
        """ Single-pass modified Daniell kernel window.
        
        Weight is normalized to add up to 1, and all values are the same, other than the first and the
        last, which are divided by 2.
        """
        def w(k):
            return np.where(np.abs(k) < m, 1 / (2*m), np.where(np.abs(k) == m, 1/(4*m), 0))

        return w(np.arange(-m, m+1))

    def daniell_window_convolve(v):
        """ Convolved version of multiple modified Daniell kernel windows.
        
        Parameter v should be an iterable of m values.
        """
        
        if len(v) == 0:
            return np.r_[1]

        if len(v) == 1:
            return daniell_window_modified(v[0])

        return signal.convolve(daniell_window_modified(v[0]), daniell_window_convolve(v[1:]))
    
    # Ensure we can store non-integers in x, and that it is a numpy object
    x = np.r_[x].astype('float64')
    original_shape = x.shape
    
    # Ensure correct dimensions
    assert len(original_shape) <= 2, "'x' must have at most 2 dimensions"
    while len(x.shape) < 2:
        x = np.expand_dims(x, axis=1)
        
    N, nser = x.shape
    N0 = N
    
    # Ensure only one of spans, kernel is provided, and build the kernel window if needed
    assert (spans is None) or (kernel is None), "must specify only one of 'spans' or 'kernel'"
    if spans is not None:
        kernel = daniell_window_convolve(np.floor_divide(np.r_[spans], 2))
        
    # Detrend or demean the series
    if detrend:
        t = np.arange(N) - (N - 1)/2
        sumt2 = N * (N**2 - 1)/12
        x -= (np.repeat(np.expand_dims(np.mean(x, axis=0), 0), N, axis=0) + np.outer(np.sum(x.T * t, axis=1), t/sumt2).T)
    elif demean:
        x -= np.mean(x, axis=0)
        
    # Compute taper and taper adjustment variables
    x = spec_taper(x, taper)
    u2 = (1 - (5/8) * taper * 2)
    u4 = (1 - (93/128) * taper * 2)
         
    # Pad the series with copies of the same shape, but filled with zeroes
    if pad > 0:
        x = np.r_[x, np.zeros((pad * x.shape[0], x.shape[1]))]
        N = x.shape[0]
        
    # Further pad the series to accelerate FFT computation
    if fast:
        newN = fft.next_fast_len(N, True)
        x = np.r_[x, np.zeros((newN - N, x.shape[1]))]
        N = newN
        
    # Compute the Fourier frequencies (R's spec.pgram convention style)
    Nspec = int(np.floor(N/2))
    freq = (np.arange(Nspec) + 1) * xfreq / N
    
    # Translations to keep same row / column convention as stats::mvfft
    xfft = fft.fft(x.T).T
    
    # Compute the periodogram for each i, j
    pgram = np.empty((N, nser, nser), dtype='complex')
    for i in range(nser):
        for j in range(nser):
            pgram[:, i, j] = xfft[:, i] * np.conj(xfft[:, j]) / (N0 * xfreq)
            pgram[0, i, j] = 0.5 * (pgram[1, i, j] + pgram[-1, i, j])
       
    if kernel is None:    
        # Values pre-adjustment
        df = 2
        bandwidth = np.sqrt(1 / 12)
    else:
        def conv_circular(signal, kernel):
            """
            Performs 1D circular convolution, in the same style as R::kernapply,
            assuming the kernel window is centered at 0.
            """
            pad = len(signal) - len(kernel)
            half_window = int((len(kernel) + 1) / 2)
            indexes = range(-half_window, len(signal) - half_window)
            orig_conv = np.real(fft.ifft(fft.fft(signal) * fft.fft(np.r_[np.zeros(pad), kernel])))
            return orig_conv.take(indexes, mode='wrap')
                
        # Convolve pgram with kernel with circular conv
        for i in range(nser):
            for j in range(nser):
                pgram[:, i, j] = conv_circular(pgram[:, i, j], kernel)
        
        df = 2 / np.sum(kernel**2)
        m = (len(kernel) - 1)/2
        k = np.arange(-m, m+1)
        bandwidth = np.sqrt(np.sum((1/12 + k**2) * kernel))
    
    df = df/(u4/u2**2)*(N0/N)
    bandwidth = bandwidth * xfreq/N
    
    # Remove padded results
    pgram = pgram[1:(Nspec+1), :, :]
    
    spec = np.empty((Nspec, nser))
    for i in range(nser):
        spec[:, i] = np.real(pgram[:, i, i])
    
    if nser == 1:
        coh = None
        phase = None
    else:
        coh = np.empty((Nspec, int(nser * (nser - 1)/2)))
        phase = np.empty((Nspec, int(nser * (nser - 1)/2)))
        for i in range(nser):
            for j in range(i+1, nser):
                index = int(i + j*(j-1)/2)
                coh[:, index] = np.abs(pgram[:, i, j])**2 / (spec[:, i] * spec[:, j])
                phase[:, index] = np.angle(pgram[:, i, j])
            
    spec = spec / u2
    spec = spec.squeeze()
    
    results = {
        'freq': freq,
        'spec': spec,
        'coh': coh,
        'phase': phase,
        'kernel': kernel,
        'df': df,
        'bandwidth': bandwidth,
        'n.used': N,
        'orig.n': N0,
        'taper': taper,
        'pad': pad,
        'detrend': detrend,
        'demean': demean,
        'method': 'Raw Periodogram' if kernel is None else 'Smoothed Periodogram'
    }
    
    if plot:
        plot_spec(results, coverage=0.95, **kwargs)
    
    return results

def spec_ar(x, x_freq=1, n_freq=500, order_max=None, plot=True, **kwargs):
    x = np.r_[x]
    N = len(x)
    if order_max is None:
        order_max = min(N - 1, int(np.floor(10 * np.log10(N))))

    # Use Yule-Walker to find best AR model via AIC
    def aic(sigma2, df_model, nobs):
        return np.log(sigma2) + 2 * (1 + df_model) / nobs
    
    best_results = None
    
    for lag in range(order_max+1):
        ar, sigma = yule_walker(x, order=lag, method='mle')
        model_aic = aic(sigma2=sigma**2, df_model=lag, nobs=N-lag)
        if best_results is None or model_aic < best_results['aic']:
            best_results = {
                'aic': model_aic,
                'order': lag,
                'ar': ar,
                'sigma2': sigma**2
            }
        
    order = best_results['order']
    freq = np.arange(0, n_freq) / (2 * (n_freq - 1))
      
    if order >= 1:
        ar, sigma2 = best_results['ar'], best_results['sigma2']
    
        outer_xy = np.outer(freq, np.arange(1, order+1))
        cs = np.cos(2 * np.pi * outer_xy) @ ar
        sn = np.sin(2 * np.pi * outer_xy) @ ar

        spec = sigma2 / (x_freq*((1 - cs)**2 + sn**2))
        
    else:
        sigma2 = best_results['sigma2']
        spec = (sigma2 / x_freq) * np.ones(len(freq))
    
    results = {
        'freq': freq,
        'spec': spec,
        'coh': None,
        'phase': None,
        'n.used': len(x),
        'method': 'AR(' + str(order) + ') spectrum'
    } 
    
    if plot:
        plot_spec(results, coverage=None, **kwargs)
    
    return results

def plot_spec(spec_res, coverage=None, ax=None, title=None):
    """Convenience plotting method, also includes confidence cross in the same style as R.
    
    Note that the location of the cross is irrelevant; only width and height matter."""
    f, Pxx = spec_res['freq'], spec_res['spec']
    
    if coverage is not None:
        ci = spec_ci(spec_res['df'], coverage=coverage)
        conf_x = (max(spec_res['freq']) - spec_res['bandwidth']) + np.r_[-0.5, 0.5] * spec_res['bandwidth']
        conf_y = max(spec_res['spec']) / ci[1]

    if ax is None:
        ax = plt.gca()
    
    ax.plot(f, Pxx, color='C0')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Log Spectrum')
    ax.set_yscale('log')
    if coverage is not None:
        ax.plot(np.mean(conf_x) * np.r_[1, 1], conf_y * ci, color='red')
        ax.plot(conf_x, np.mean(conf_y) * np.r_[1, 1], color='red')

    ax.set_title(spec_res['method'] if title is None else title)
        

def produce_differences_df(df):
    column_list = list(df)
    #column_list.remove("NUM")

    df = df[column_list].diff(axis=1, periods = 1)
    df.fillna(value = 0, axis = 0, inplace = True)

    return df

def specific_profile_telem(profiles):
    profile_df = pd.read_csv('../Alliander_data/profielen.csv')
    profile_df.loc[profile_df['BASELOAD_PROFILE'].isin(profiles)]
    indexes = profile_df['RND_ID'].sort_values(by='RND_ID').values

    meetdata_df = pd.read_csv('../Alliander_data/gv_meetdata_select.csv',
                              skiprows=lambda x: x not in indexes)
    return meetdata_df


def compute_sum_columns(matrix):

    shape = np.shape(matrix)
    shape = np.asarray(shape)
    vector_sums = np.zeros(shape[1])
    
    for i in range(shape[1]):
        vector_sums[i] = np.sum(matrix[:,i])
    
    return vector_sums

def normalize_data_wrt_sum(matrix, vector_sums):
    shape = np.shape(matrix)
    shape = np.asarray(shape)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            matrix[i,j] = np.abs(matrix[i,j]/vector_sums[j])
        
    
    return matrix

def normalize_meetdata(df):

    df = df.div(df.sum(axis=1), axis=0)

    df = df.round(7)
    
    return df

def total_consumption_meetdata(data):
    column_list = list(data)
    column_list.remove("NUM")

    data["sum"] = data[column_list].sum(axis=1)

    return data

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

def compute_estimated_loads(sums_pd, connect_df, normalized_df):
    sums_np = sums_pd.to_numpy()
    number_costumer = np.arange(1,5000,1)
    estimated_loads = []
    for i in number_costumer-1:
        profile_costumer = select_profile_costumer(connect_df,number_costumer[i])
        for j in range(len(profile_costumer)):
            consumption_from_df = normalized_df[profile_costumer[j]]
            consumption_from_df_np = consumption_from_df.to_numpy()
            estimated_load_profile = consumption_from_df_np*sums_np[i]
            estimated_loads.append([i+1 , profile_costumer[j] , estimated_load_profile])

    return estimated_loads

def get_quads(df):
    quadlen = int(len(df) * 0.25)
    ss = df[:quadlen].describe()
    ss[1] = df[quadlen:quadlen*2].describe()
    ss[2] = df[quadlen*2:quadlen*3].describe()
    ss[3] = df[quadlen*3:].describe()
    return ss

def ADF_test(df):
    result = adfuller(df, autolag='AIC')
    t_stat, p_value, _, _, critical_values, _  = adfuller(df, autolag='AIC')
    #print(f'ADF Statistic: {t_stat:.2f}')
    #for key, value in critical_values.items():
    #     print('Critial Values:')
    #     print(f'   {key}, {value:.2f}')

    #print(f'\np-value: {p_value:.2f}')
    if p_value > 0.05:
        result_test = "Non-Stationary"  
    else:
        result_test = "Stationary"

    return result_test

def KPSS_test(df):
    t_stat, p_value, _, critical_values = kpss(df, nlags='auto')

    #print(f'ADF Statistic: {t_stat:.2f}')
    #for key, value in critical_values.items():
    #     print('Critial Values:')
    #     print(f'   {key}, {value:.2f}')

    #print(f'\np-value: {p_value:.2f}')

    if p_value > 0.05:
        result_test = "Stationary"  
    else:
        result_test = "Non-Stationary"

    return result_test

def ZIVOT_ANDREWS_test(df):
    t_stat, p_value, critical_values, _, _ = zivot_andrews(df)
    #print(f'Zivot-Andrews Statistic: {t_stat:.2f}')
    #for key, value in critical_values.items():
    #     print('Critial Values:')
    #     print(f'   {key}, {value:.2f}')

    #print(f'\np-value: {p_value:.2f}')

    if p_value > 0.05:
        result_test = "Non-Stationary"  
    else:
        result_test = "Stationary"

    return result_test

def compute_frequency_peak(diff_df_np):
    
    s1_diff = spec_pgram(diff_df_np, spans = [51,51], taper = 0.1, plot=True)

    #print('1 / Peak: ', round(1/s1_diff['freq'][np.argmax(s1_diff['spec'])]/4, 2), 'hour')
    frequency = round(1/s1_diff['freq'][np.argmax(s1_diff['spec'])]/4, 2)

    return frequency

def compute_coordinate_frequency_peak(diff_df_np):
    
    s1_diff = spec_pgram(diff_df_np, spans = [51,51], taper = 0.1, plot=True)
    
    #print("Frequency results: ")
    #print(np.argmax(s1_diff['spec']))
    #print(type(s1_diff['spec']))
    
    maximum1 = max(s1_diff['spec'])
    index_maximum1 = np.argmax(s1_diff['spec'])
    maximum2 = max(s1_diff['spec'], key = lambda x: min(s1_diff['spec'])-1 if (x == maximum1) else x)
    index_maximum2_array, = np.where(s1_diff['spec'] == maximum2)
    index_maximum2 = index_maximum2_array[0]
    
    '''
    print('Maximum 1: ')
    print(maximum1)
    print('Maximum 2: ')
    print(maximum2)
    print('Index Maximum 1: ')
    print(index_maximum1)
    print('Index Maximum 2: ')
    print(index_maximum2)
    #print(s1_diff['freq'][s1_diff['spec'].nlargest(3)])
    print("Spectral results: ")
    print(s1_diff['spec'])
    '''
    #print('1 / Peak: ', round(1/s1_diff['freq'][np.argmax(s1_diff['spec'])]/4, 2), 'hour')
    frequency_1 = round(1/s1_diff['freq'][index_maximum1]/4, 4)
    frequency_1_x = round(s1_diff['spec'][index_maximum1], 4)
    frequency_2 = round(1/s1_diff['freq'][index_maximum2]/4, 4)
    frequency_2_x = round(s1_diff['spec'][index_maximum2], 4)
    '''
    print('Frequency 1: ')
    print(frequency_1)
    print('Frequency 1 x: ')
    print(frequency_1_x)
    print('Frequency 2: ')
    print(frequency_2)
    print('Frequency 2 x: ')
    print(frequency_2_x)
    '''

    return frequency_1, frequency_1_x, frequency_2, frequency_2_x

def compute_df_frequencies(df):
    diff_meetdata_pd = produce_differences_df(df)

    shape = np.shape(diff_meetdata_pd)
    shape = np.asarray(shape)

    diff_meetdata_np = np.asarray(diff_meetdata_pd)

    estimated_peaks = []

    for i in range(shape[0]):
        diff_company_i = diff_meetdata_np[i,1:]

        condition = False
        #check stationarity
        if i%100 == 0:
            print(i)

        try:
            ADF = ADF_test(diff_company_i)
            #KPSS = KPSS_test(diff_company_i)
            if ADF == "Stationary":
                condition = True
            else:
                condition = False
        except ValueError:
            pass
        
        if condition:
            peak = compute_frequency_peak(diff_company_i)
            estimated_peaks.append([peak])
        else:
            estimated_peaks.append([0])

    return estimated_peaks

def compute_df_coordinates_frequencies(df):
    diff_meetdata_pd = produce_differences_df(df)

    shape = np.shape(diff_meetdata_pd)
    shape = np.asarray(shape)

    diff_meetdata_np = np.asarray(diff_meetdata_pd)

    estimated_peaks = []

    for i in range(shape[0]):
        diff_company_i = diff_meetdata_np[i,1:]

        condition = False
        #check stationarity
        if i%100 == 0:
            print(i)

        try:
            ADF = ADF_test(diff_company_i)
            #KPSS = KPSS_test(diff_company_i)
            if ADF == "Stationary":
                condition = True
            else:
                condition = False
        except ValueError:
            pass
        
        if condition:
            peak1, peak1_x, peak2, peak2_x = compute_coordinate_frequency_peak(diff_company_i)
            estimated_peaks.append([peak1, peak1_x, peak2, peak2_x])
        else:
            estimated_peaks.append([0, 0, 0, 0])

    return estimated_peaks



def normalize_df(df):
    matrix = df.to_numpy()

    shape = np.shape(matrix)
    shape = np.asarray(shape)
    vector_sums = np.zeros(shape[1])

    for i in range(shape[1]):
        vector_sums[i] = np.sum(matrix[:, i])

    # normalize_data_wrt_sum
    for i in range(shape[0]):
        for j in range(shape[1]):
            matrix[i, j] = np.abs(matrix[i, j]/vector_sums[j])

    normalized_df = pd.DataFrame(data=matrix,
                                 index=df.index,
                                 columns=df.columns)

    return normalized_df


def total_consumption_meetdata(df):
    column_list = list(df)

    df["SUM"] = df[column_list].sum(axis=1)

    return df

def load_profiles(reload=0, Alliander_path="../Alliander_data/"):
    try:
        if reload: raise FileNotFoundError
        connect_df = pd.read_csv(
            Alliander_path + "sorted_connect.csv", index_col=0)
        neat_profile_df = pd.read_csv(
            Alliander_path + "neat_profiles.csv", index_col=0)
    except FileNotFoundError:
        connect_df = pd.read_csv(
            Alliander_path + "aansluiting_attributen.csv", index_col=0)
        connect_df.sort_values(by=['RND_ID'], inplace=True)
        connect_df.to_csv(
            Alliander_path + 'sorted_connect.csv')

        # read all profiles
        profile_df = pd.read_csv(
            Alliander_path + 'profielen.csv', index_col=0)
        edsn_profile_df = pd.read_csv(
            Alliander_path + "edsn_profielen.csv", index_col=0)

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

        strp_profile_df = pd.concat([big_interest_df, edsn_interest_df],
                                axis=1)
        # normalize all profiles
        norm_strp_profiles_df = normalize_df(strp_profile_df)

        # append 'sum' column for all customers
        neat_profile_df = total_consumption_meetdata(
            norm_strp_profiles_df)


        neat_profile_df.to_csv(
            Alliander_path + "neat_profiles.csv")

    return connect_df, neat_profile_df


# TODO: adjust to load_data function
# only load parts of the meet_data have the desired profiles
def specific_profile_telem(profiles):
    profile_df = pd.read_csv('../Alliander_data/profielen.csv')
    profile_df.loc[profile_df['BASELOAD_PROFILE'].isin(profiles)]
    indexes = profile_df['RND_ID'].sort_values(by='RND_ID').values

    meetdata_df = pd.read_csv('../Alliander_data/gv_meetdata_select.csv',
                              skiprows=lambda x: x not in indexes)
    return meetdata_df

'''

Here below there is an example of how to use the set_correct_index_profiles function

profile = pd.read_csv('../Alliander_data/neat_profiles.csv')
customer = pd.read_csv('../Alliander_data/unique_meetdata.csv', index_col = 0, nrows = 100)

print(set_correct_index_profiles(profile,customer).head())    


This function is giving to the neat profiles the correct indexing and is removing the exciding rows.

profile_df is the one extracted from neat_profiles, which has been created through profielen.csv and edsn_profielen.csv
See load_save_profiles.py to learn how neat_profiles.csv is created.

meetdata_df can be extracted directly from gv_meetdata.csv or it is more convenient to use unique_meetdata.csv
See save_unique_profile_customer.py to learn how unique_meetdata.csv is created.

'''
def set_correct_index_profiles(profile_df, meetdata_df):
    
    index = pd.to_datetime(meetdata_df.iloc[3].index)

    print(index)
    profile_df = profile_df.drop(index = np.arange(5664,5760,1))

    print('profile shape: ')
    print(profile_df.shape)
    profile_df.index = index
    first_column = profile_df.columns[0]
# Delete first
    profile_df = profile_df.drop([first_column], axis=1)
    
    profile_df.to_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/neat_profiles_fitted.csv')
    return profile_df
