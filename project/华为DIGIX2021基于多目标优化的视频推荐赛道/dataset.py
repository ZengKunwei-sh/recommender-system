import pandas as pd
import numpy as np
from config import DataConfig
import gc, glob

def read_raw_data() -> list([pd.DataFrame]):
    """
    Returns:
        user_features, video_features, history_behavior
    """
    config = DataConfig()
    file_path = config.file_path
    user_features = pd.read_csv(f'{file_path}/traindata/user_features_data/user_features_data.csv', sep='\t')
    video_features = pd.read_csv(f'{file_path}/traindata/video_features_data/video_features_data.csv', sep='\t')
    history_behavior = pd.concat([
        reduce_mem(pd.read_csv(x, sep='\t'), 'history_behavior') for x in glob.glob(f'{file_path}/traindata/history_behavior_data/*/*')
    ])
    history_behavior = history_behavior.sort_values(by=['pt_d', 'user_id'])

    user_features = reduce_mem(user_features, 'user_features')
    video_features = reduce_mem(video_features, 'video_features')

    # pip install pytables 出错，使用 conda install pytables解决了这个问题
    user_features.to_hdf(f'{file_path}/traindata/history_behavior_data/digix-data.hdf', 'user_features')
    video_features.to_hdf(f'{file_path}/traindata/user_features_data/digix-data.hdf', 'video_features')
    history_behavior.to_hdf(f'{file_path}/traindata/video_features_data/digix-data.hdf', 'history_behavior')

    return user_features, video_features, history_behavior

def read_data() -> list([pd.DataFrame]):
    """
    Returns:
        user_features, video_features, history_behavior
    """
    config = DataConfig()
    file_path = config.file_path
    
    user_features = pd.read_hdf(f'{file_path}/traindata/history_behavior_data/digix-data.hdf', 'user_features')
    video_features = pd.read_hdf(f'{file_path}/traindata/user_features_data/digix-data.hdf', 'video_features')
    history_behavior = pd.read_hdf(f'{file_path}/traindata/video_features_data/digix-data.hdf', 'history_behavior')
    return user_features, video_features, history_behavior

def reduce_mem(df, df_name='DataFrame'):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    df.name = df_name
    print('{} from {:.2f} Mb reduce to {:.2f} Mb, reduce ({:.2f} %)'.format(df_name, start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df

user, video, history = read_raw_data()
print(user.dtypes)
print(video.dtypes)
print(history.dtypes)
user, video, history = read_data()
print(user.dtypes)
print(video.dtypes)
print(history.dtypes)