import sys
from pathlib import Path
import pandas as pd
import json
import os
import numpy as np
import torch

# 引入项目根目录
home_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(home_dir))

import utils.config_util as config_util
import utils.path_util as path_util


'''
处理amazon数据集的工具
'''

# 获取原始数据，并根据时间排序
def get_raw_data():
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data_path = path_util.get_amazon_raw_data_path()
    data = pd.read_csv(data_path, header=0, names=names)
    # 按用户分组并根据 timestamp 排序
    data_sorted = data.sort_values(by=['user_id', 'timestamp'])
    return data_sorted

raw_data = get_raw_data()

# 获取每个用户按时间排序过的原始数据
def get_user_raw_data(raw_data, user):
    return raw_data[raw_data['user_id'] == user]

# 获取每个用户训练集的原始数据
def get_user_train_data(raw_data, user):
    user_data = get_user_raw_data(raw_data, user)
    timestamp = config_util.get_amazon_timestamp()
    return user_data[user_data['timestamp'] < timestamp]

# 获取每个用户测试集的原始数据
def get_user_test_data(raw_data, user):
    user_data = get_user_raw_data(raw_data, user)
    timestamp = config_util.get_amazon_timestamp()
    return user_data[user_data['timestamp'] >= timestamp]

# 获取所有用户id
def get_all_users(raw_data):
    return raw_data['user_id'].unique()

# 获取所有有训练集的用户
def get_user_with_train(raw_data):
    user_with_train = []
    all_users = get_all_users(raw_data)
    for user in all_users:
        train_data = get_user_train_data(raw_data, user)
        
        # 如果存在训练集记录，加入结果集合
        if len(train_data) != 0:
            user_with_train.append(user)
    
    return user_with_train


# 将列表对象存储到json文件中
def save_json_file(list, json_path):
    with open(json_path, "w") as json_file:
        json.dump(list, json_file, indent=4)
    return None

# 加载recall_item_pairs.json
def get_recall_item_pairs():
    json_path = os.path.join(path_util.get_amazon_preprocess_fp(), 'recall_item_pairs.json')
    with open(json_path, 'r') as file:
        item_pairs = json.load(file)
    return item_pairs

# 加载user_mapping.json
def get_user_mapping():
    json_path = os.path.join(path_util.get_amazon_preprocess_fp(), 'user_mapping.json')
    with open(json_path, 'r') as file:
        user_mapping = json.load(file)
    return user_mapping

user_mapping = get_user_mapping()

# 加载item_mapping.json
def get_item_mapping():
    json_path = os.path.join(path_util.get_amazon_preprocess_fp(), 'item_mapping.json')
    with open(json_path, 'r') as file:
        item_mapping = json.load(file)
    return item_mapping

item_mapping = get_item_mapping()

# 根据user_mapping获取用户的映射id
def get_user_mapping_id(user_mapping, user):
    if isinstance(user, str):  # 如果是单个值
        return user_mapping.get(str(user))
    elif isinstance(user, (list, np.ndarray)):  # 如果是列表或 NumPy 数组
        return [user_mapping.get(str(uid)) for uid in user]
    elif isinstance(user, torch.Tensor):  # 如果是 Tensor
        user = user.tolist() 
        return torch.tensor([user_mapping.get(str(uid)) for uid in user])

# 根据item_mapping获取物品的映射id
def get_item_mapping_id(item_mapping, item):
    if isinstance(item, str):  # 如果是单个值
        return item_mapping.get(str(item))
    elif isinstance(item, (list, np.ndarray)):  # 如果是列表或 NumPy 数组
        return [item_mapping.get(str(iid)) for iid in item]
    elif isinstance(item, torch.Tensor):  # 如果是 Tensor
        item = item.tolist() 
        return torch.tensor([item_mapping.get(str(iid)) for iid in item])

# 加载user_with_train.json
def get_user_with_train():
    json_path = os.path.join(path_util.get_amazon_preprocess_fp(), 'user_with_train.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


# 加载user_with_train_and_test.json
def get_user_with_train_and_test():
    json_path = os.path.join(path_util.get_amazon_preprocess_fp(), 'user_with_train_and_test.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

# 获取全局训练集和测试集
def get_train_test_data():
    data = get_raw_data()
    timestamp = config_util.get_amazon_timestamp()
    trainset = data[data['timestamp'] < timestamp]
    testset = data[data['timestamp'] >= timestamp]
    return trainset, testset


# 获取用户交互物品列表
def get_trainset_item_interaction(raw_data, user):
    train_data = get_user_train_data(raw_data, user)
    return train_data['item_id'].unique()
