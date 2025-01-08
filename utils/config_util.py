import os
import yaml
import sys
from pathlib import Path

# 引入项目根目录
home_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(home_dir))

'''
获取全局config.yml中的配置信息
'''

def get_config_by_name(name):
    # 获取当前 Python 文件的绝对路径
    script_path = os.path.abspath(__file__)
    # 获取当前 Python 文件所在的目录
    script_directory = os.path.dirname(script_path)
    # 获取当前 Python 文件所在目录的上一级目录
    parent_directory = os.path.dirname(script_directory)
    with open(os.path.join(parent_directory, 'config.yml'), "r") as file:
        config = yaml.safe_load(file)
    # 访问变量
    variable = config[name]
    return variable

def get_home_path():
    return get_config_by_name('home_path')

def get_movielens_timestamp():
    return get_config_by_name('movielens_timestamp')

def get_amazon_timestamp():
    return get_config_by_name('amazon_timestamp')

def get_random_seed():
    return get_config_by_name('random_seed')

def get_num_task():
    return get_config_by_name('num_task')

def get_num_recall_item_pairs():
    return get_config_by_name('num_recall_item_pairs')

def get_popular_item_pairs_ratio():
    return get_config_by_name('popular_item_pairs_ratio')

def get_one_class_auc():
    return get_config_by_name('one_class_auc')  

def get_random_mask_ratio():
    return get_config_by_name('random_mask_ratio')  
