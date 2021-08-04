import pandas as pd
import numpy as np


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
        if reload:
            raise FileNotFoundError
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

        # select profiles of interest mentioned in the
        # 'aansluiting_attributen' file
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
    profile_df = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/profielen.csv')
    profile_df.loc[profile_df['BASELOAD_PROFILE'].isin(profiles)]
    indexes = profile_df['RND_ID'].sort_values(by='RND_ID').values

    meetdata_df = pd.read_csv('C:/Users/tratt/OneDrive/Desktop/Internship Alliander/Alliander_data/gv_meetdata_select.csv',
                              skiprows=lambda x: x not in indexes)
    return meetdata_df


if __name__ == "__main__":
    profile_df, meetdata_df = load_profiles(reload=True)
    SJV = total_consumption_meetdata(meetdata_df)
    print("SJV: ")
    print(SJV.head())

    print("done")
