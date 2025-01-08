import sys
from pathlib import Path
import os

# 引入项目根目录
home_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(home_dir))

import utils.config_util as config_util

'''
获取项目中各个文件和目录的地址
'''

def get_home_fp():
    return config_util.get_home_path()

def get_folder_path_by_name(name):
    home_path = get_home_fp()
    return os.path.join(home_path, name)

def get_log_fp():
    return get_folder_path_by_name('log')

def get_model_fp():
    return get_folder_path_by_name('model')

def get_model_fp_by_name(name):
    return os.path.join(get_model_fp(), name)

def get_scripts_fp():
    return get_folder_path_by_name('scripts')

def get_utils_fp():
    return get_folder_path_by_name('utils')

def get_data_fp():
    return get_folder_path_by_name('data')

def get_movielens_fp():
    return os.path.join(get_data_fp(), 'MovieLens')

def get_movielens_raw_data_path():
    return os.path.join(get_movielens_fp(), 'ratings.csv')

def get_movielens_preprocess_fp():
    return os.path.join(get_movielens_fp(), 'preprocess')

def get_amazon_fp():
    return os.path.join(get_data_fp(), 'Amazon')

def get_amazon_raw_data_path():
    return os.path.join(get_amazon_preprocess_fp(), 'ratings.csv')

def get_amazon_preprocess_fp():
    return os.path.join(get_amazon_fp(), 'preprocess')
