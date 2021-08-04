import pandas as pd
# import numpy as np
from load_save_profiles import load_profiles


Alliander_path = "../Alliander_data/"

connect_df, profile_df = load_profiles()

if __name__ == "__main__":

    print('loading connect and profile dfs')
    connect_df, _ = load_profiles()
    print('finished loading dfs')
    print('connect head: ')
    print(connect_df.head(15))

    print('now loading meetdata_df')
    get_indexes = connect_df.index.drop_duplicates(keep=False)
#  meet_data_df = pd.read_csv(Alliander_path + 'gv_meetdata_select.csv',
#                             nrows=30, index_col=0)
    meet_data_df = pd.read_csv(Alliander_path + 'gv_meetdata_select.csv',
                               skiprows=lambda x: x not in get_indexes,
                               index_col=0,
                               nrows=500)
    meet_data_df.sort_index(inplace=True)

    meet_data_nodup_df = meet_data_df[
        connect_df.index.drop_duplicates(keep=False)]
    # meet_data_nodup_df = meet_data_df[~no_dup_idx]
    print('length of shortened meetdata')
    print(len(meet_data_nodup_df[0]))
    print('finished loading meetdata_df')
    print(meet_data_df[:5].head(5))
    print(meet_data_df.index)
    print("done")
