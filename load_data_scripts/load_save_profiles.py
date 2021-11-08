import pandas as pd
import numpy as np


def normalize_df(df):

    for column in df.columns:
        df[column] = df[column]/df[column].sum()

    return df


# drop leapday '2020-02-29'
def drop_leap_day(df):
    df = df[~((df.index.month == 2) & (df.index.day == 29))]
    return df


def convert_to_kvk(profile_name):
    # convert names in profile_of_interest to match with column names in
    # profielen.csv
    try:
        profile_name = "KVKSEGMENT_{}".format(
            str(int(profile_name)))
    except ValueError:
        pass
    return profile_name


def load_profiles(reload=0, Alliander_path="../Alliander_data/"):
    try:
        if reload:
            raise FileNotFoundError
        connect_df = pd.read_csv(
            Alliander_path + "sorted_connect.csv", index_col=0)
        neat_profile_df = pd.read_csv(
            Alliander_path + "neat_profiles.csv", index_col=0)
        neat_profile_df.index = pd.to_datetime(neat_profile_df.index)
    except FileNotFoundError:
        # create new, sorted connection file
        connect_df = pd.read_csv(
            Alliander_path + "aansluiting_attributen.csv",
            usecols=['BASELOAD_PROFILE', 'AANSLUITCATEGORIE', 'RND_ID'],
            index_col='RND_ID').sort_index()

        # put correct labels in new dataframe
        connect_df['BASELOAD_PROFILE'] = connect_df['BASELOAD_PROFILE'].apply(
            convert_to_kvk)

        connect_df.to_csv(
            Alliander_path + 'sorted_connect.csv')

        # create new concatenated profile dataframe
        profile_df = pd.read_csv(
            Alliander_path + 'profielen.csv',
            index_col='DATUM_TIJDSTIP')
        profile_df.index = pd.to_datetime(profile_df.index)

        edsn_profile_df = pd.read_csv(
            Alliander_path + "edsn_profielen.csv",
            index_col='DATUM_TIJD_VAN_STRING')
        edsn_profile_df.index = pd.to_datetime(edsn_profile_df.index)

        # select all profiles from 'aansluiting_attributen' to get the
        # relevant baseload profiles
        connect_types = connect_df[["BASELOAD_PROFILE"]].values
        profile_of_interest = np.unique(connect_types)

        # select profiles of interest mentioned in the
        # 'aansluiting_attributen' file
        # from the big and intermediate profiles
        big_interest_df = profile_df[
            profile_df.columns.intersection(profile_of_interest)].sort_index()

        edsn_interest_df = edsn_profile_df[
            edsn_profile_df.columns.intersection(
                profile_of_interest)].sort_index()

        strp_profile_df = pd.concat([big_interest_df, edsn_interest_df],
                                    axis=1)
        # normalize all profiles
        neat_profile_df = normalize_df(strp_profile_df)
        # drop the leap day
        neat_profile_df = drop_leap_day(neat_profile_df)

        neat_profile_df.to_csv(
            Alliander_path + "neat_profiles.csv")

    return connect_df, neat_profile_df


if __name__ == "__main__":
    connect_df, profile_df = load_profiles(reload=True)
    print("profiles: ")
    print(profile_df.head())
    print('connect: ')
    print(connect_df.head(6))

    print("done")
