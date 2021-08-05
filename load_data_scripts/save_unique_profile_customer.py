import pandas as pd
from load_save_profiles import load_profiles


def unique_profile(reload=0, Alliander_path="../Alliander_data/",
                   nrows=100):
    try:
        if reload:
            raise FileNotFoundError
        unique_df = pd.read_csv(
            Alliander_path + 'unique_meetdata.csv',
            index_col=0,
            nrows=nrows)
    except FileNotFoundError:
        connect_df, _ = load_profiles(reload=reload,
                                      Alliander_path=Alliander_path)
        get_indexes = connect_df.index.drop_duplicates(
            keep=False).append(pd.Index([0]))

        unique_df = pd.read_csv(Alliander_path + 'gv_meetdata_select.csv',
                                skiprows=lambda x: x not in get_indexes,
                                index_col=0,
                                nrows=100)
        unique_df.to_csv(Alliander_path + 'unique_meetdata.csv')
    return unique_df


if __name__ == "__main__":
    print('reloading everything')
    unique_df = unique_profile(reload=1)

    print('unique meetdata: ')
    print(unique_df[:5].head(10))
    print("done")
