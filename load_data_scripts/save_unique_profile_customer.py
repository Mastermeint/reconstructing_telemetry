import pandas as pd
from .load_save_profiles import load_profiles


# drop leapday '2020-02-29'
def drop_leap_day(df):
    column_drop_bool = [df.columns.month == 2 and df.columns.day == 29]
    drop_column = [i for i, x in enumerate(column_drop_bool) if x]
    df = df.drop(drop_column, axis=1)
    return df


# remove customers that have 0s the first or last month
# note: there are roughly 4*24*30 timeframes in a month
def remove_incomplete_customers(df):
    df = df[(df.iloc[:, :2880].T != 0).any()]
    df = df[(df.iloc[:, -2880:].T != 0).any()]
    return df


def unique_customer(reload=0, Alliander_path="../Alliander_data/",
                    nrows=None):
    try:
        if reload:
            raise FileNotFoundError
        columns = pd.read_csv(Alliander_path + 'unique_meetdata.csv',
                              nrows=1).columns.tolist()
        raw_dtypes = dict((column, 'float32') for column in columns)
        unique_df = pd.read_csv(
            Alliander_path + 'unique_meetdata.csv',
            index_col=0,
            nrows=nrows,
            dtype=raw_dtypes)
        unique_df.columns = pd.to_datetime(unique_df.columns)
    except FileNotFoundError:
        connect_df, _ = load_profiles(reload=0,
                                      Alliander_path=Alliander_path)
        get_indexes = connect_df.index.drop_duplicates(
            keep=False).append(pd.Index([0]))

        columns = pd.read_csv(Alliander_path + 'gv_meetdata_select.csv',
                              nrows=1).columns.tolist()
        raw_dtypes = dict((column, 'float32') for column in columns)

        unique_df = pd.read_csv(Alliander_path + 'gv_meetdata_select.csv',
                                skiprows=lambda x: x not in get_indexes,
                                index_col=0,
                                dtype=raw_dtypes,
                                nrows=nrows)

        unique_df.columns = pd.to_datetime(unique_df.columns)
        # unique_df = drop_leap_day(unique_df)
        unique_df = unique_df.dropna()
        unique_df = remove_incomplete_customers(unique_df)
        unique_df.to_csv(Alliander_path + 'unique_meetdata.csv')
    return unique_df


if __name__ == "__main__":
    print('reloading everything')
    unique_df = unique_customer(reload=1)
    print('number of customers left: ', len(unique_df.index))
    print('unique meetdata: ')
    print(unique_df[:5].head(10))
    print("done")
