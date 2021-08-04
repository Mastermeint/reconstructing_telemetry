import pandas as pd
import numpy as np
# from sklearn import preprocessing
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates


# load data - be mindfull of size gv_meetdata_select.csv!
print("starting to load all the data")
profile_df = pd.read_csv("../Alliander_data/profielen.csv")
edsn_profile_df = pd.read_csv("../Alliander_data/edsn_profielen.csv")
connect_df = pd.read_csv("../Alliander_data/aansluiting_attributen.csv")
connect_df.sort_values(by=['RND_ID'], inplace=True)
meetdata_df = pd.read_csv("../Alliander_data/gv_meetdata_select.csv",
                          nrows=1)
print("loaded all the databases")
# print(meetdata_df.head())


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


# only load parts of the meet_data have the desired profiles
def specific_profile_telem(profiles):

    profile_df.loc[profile_df['BASELOAD_PROFILE'].isin(profiles)]
    indexes = profile_df['RND_ID'].sort_values(by='RND_ID').values

    meetdata_df = pd.read_csv("../Alliander_data/gv_meetdata_select.csv",
                              skiprows=lambda x: x not in indexes)
    return meetdata_df

# ############################################################
# # plot stuff
# ############################################################
# # Major ticks every 6 months.
# fig, axes = plt.subplots(2)
# fig.suptitle('regular plot and normalized plot')
#
# fmt_half_year = mdates.MonthLocator(interval=6)
# axes[0].xaxis.set_major_locator(fmt_half_year)
# axes[1].xaxis.set_major_locator(fmt_half_year)
#
# meetdata_df.T.plot(ax=axes[0])
#
# # normalize the rows/customers:
# scaler = preprocessing.MaxAbsScaler()
# meetdata_scaled = scaler.fit_transform(meetdata_df)
# scaled_meet_df = pd.DataFrame(meetdata_scaled, columns=meetdata_df.columns,
#                   index=meetdata_df.index)
# scaled_meet_df = scaled_meet_df.rdiv(-1)
# scaled_meet_df.T.plot(ax=axes[1])
# plt.show()


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

print('interested numpy: ')

print(interest_df.to_numpy())
