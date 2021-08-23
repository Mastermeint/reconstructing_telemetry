import pandas as pd
from .load_save_profiles import load_profiles


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
        connect_df, _ = load_profiles(reload=reload,
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
        unique_df = unique_df.dropna()
        unique_df.to_csv(Alliander_path + 'unique_meetdata.csv')
    return unique_df


if __name__ == "__main__":
    print('reloading everything')
    unique_df = unique_customer(reload=1)

    print('unique meetdata: ')
    print(unique_df[:5].head(10))
    print("done")
