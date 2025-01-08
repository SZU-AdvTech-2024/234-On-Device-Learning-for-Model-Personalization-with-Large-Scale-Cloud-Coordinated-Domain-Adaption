import sys
from pathlib import Path
import torch
import os
import importlib.util
import datetime
from torch.utils.data import DataLoader
import copy
from openpyxl import Workbook
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 引入项目根目录
home_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(home_dir))

from scripts.metric import cal_auc
import utils.path_util as path_util
import utils.movielens_data_util as movielens_data_util
import utils.amazon_data_util as amazon_data_util

'''
模型训练和测试工具
'''

def get_model_class_by_name(name, device):
    file_path = os.path.join(path_util.get_model_fp(), name, 'model.py')
    # 动态加载模块
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_class =  getattr(module, name)
    return model_class(device)

def get_init_model_by_name(model_name, dataset_name, device):
    saved_model_path = os.path.join(path_util.get_model_fp_by_name(model_name), f'{dataset_name}_model.pth')
    model = get_model_class_by_name(model_name, device)
    model.load_state_dict(torch.load(saved_model_path))
    return model

def train_model_with_dataset(model, criterion, optimizer, train_loader, device):
    model.train()
    for index, (batch_user, batch_item, batch_label) in enumerate(train_loader):
        batch_user = batch_user.to(device)
        batch_item = batch_item.to(device)
        batch_label = batch_label.to(device).unsqueeze(1)
        # 前向传播
        preds = model(batch_user, batch_item)
        # 计算损失
        loss = criterion(preds, batch_label)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return None

def test_model_with_dataset(model, test_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for index, (batch_user, batch_item, batch_label) in enumerate(test_loader):
            batch_user = batch_user.to(device)
            batch_item = batch_item.to(device)
            batch_label = batch_label.to(device).unsqueeze(1)
            
            preds = model(batch_user, batch_item)
            all_labels.extend(batch_label.cpu().detach().numpy())
            all_preds.extend(preds.cpu().detach().numpy())

    return cal_auc(all_labels, all_preds)

def get_data_loader(data, dataset_name, batch_size, is_train):
    if dataset_name == 'movielens':
        from scripts.movielens_dataset import MovielensDataset
        dataset = MovielensDataset(data)
        data_loader = DataLoader(dataset, batch_size, shuffle=True) if is_train == True else DataLoader(dataset, batch_size, shuffle=False)
    elif dataset_name == 'amazon':
        from scripts.amazon_dataset import AmazonDataset
        dataset = AmazonDataset(data)
        data_loader = DataLoader(dataset, batch_size, shuffle=True) if is_train == True else DataLoader(dataset, batch_size, shuffle=False)
    return data_loader

def incremental_training(model, criterion, dataset_name, raw_data, lr, test_loader, augumented_users, device, batch_size, best_auc):
    current_model = copy.deepcopy(model)  # 当前模型
    print(f"Initial AUC: {best_auc}")
    num_selected_users = 0

    # 导入对应的datautil
    if dataset_name == 'movielens':
        import utils.movielens_data_util as data_util
    elif dataset_name == 'amazon':
        import utils.amazon_data_util as data_util

    for index, augumented_user in enumerate(augumented_users):
        print(f"Training on augumented user {augumented_user} {index}/{len(augumented_users)} current_auc = {best_auc}")
        augumented_data = data_util.get_user_train_data(raw_data, augumented_user)
        augumented_loader = get_data_loader(augumented_data, dataset_name, batch_size, True)
        temp_model = copy.deepcopy(current_model)  # 临时模型
        temp_optimizer = torch.optim.Adam(temp_model.parameters(), lr=lr)

        # 在增强用户的数据上训练一个 epoch
        train_model_with_dataset(temp_model, criterion, temp_optimizer, augumented_loader, device)

        # 评估增强后的模型
        temp_auc = test_model_with_dataset(temp_model, test_loader, device)
        print(f"User {augumented_user} AUC after training: {temp_auc}")

        if temp_auc > best_auc:
            # 如果 AUC 提升，保留增强的模型
            print(f"User {augumented_user} improves AUC. Keeping the model.")
            num_selected_users += 1
            current_model = temp_model
            best_auc = temp_auc
        else:
            # 如果 AUC 没有提升，回退模型
            print(f"User {augumented_user} does not improve AUC. Reverting changes.")

    return current_model, best_auc, num_selected_users


def result_to_xlsx(user_id_list, num_selected_users_list, num_train_samples_list, 
    num_test_samples_list, local_plus_list, mpda_minus_list, mpda_list, log_fp, task_index):
        if not os.path.exists(log_fp):
            os.makedirs(log_fp)
            print(f"log_fp = {log_fp} created")

        names = ['user_id', 'num_selected_users', 'num_train_samples', 'num_test_samples', 'Local+', 'MPDA-', 'MPDA']
        # 创建一个工作簿和工作表
        wb = Workbook()
        ws = wb.active
        
        # 设置列标题
        ws.append(names)
        
        # 将数据写入表格
        for user_id, selected_users, train_samples, test_samples, local_plus, mpda_minus, mpda in zip(user_id_list, num_selected_users_list, 
        num_train_samples_list, num_test_samples_list, local_plus_list, mpda_minus_list, mpda_list):
            ws.append([user_id, selected_users, train_samples, test_samples, local_plus, mpda_minus, mpda])
        
        # 保存 Excel 文件
        file_path = os.path.join(log_fp, str(task_index) + '.xlsx')
        wb.save(file_path)
        print(f"文件已保存为 {file_path}")
        return None

def get_user_vectors_by_item_interaction(init_model, raw_data, dataset_name, test_users, device):
    user_vectors = {}
    print(f'model device = {next(init_model.parameters()).device}')
    for index, user in enumerate(test_users):
        history_items = movielens_data_util.get_trainset_item_interaction(raw_data, user) if dataset_name == 'movielens' else amazon_data_util.get_trainset_item_interaction(raw_data, user)
        item_features = init_model.get_item_embedding(torch.tensor(history_items).to(device))
        user_feature = torch.mean(item_features, dim=0)
        user_vectors[user] = user_feature.detach().cpu().numpy()
        print(f'[{datetime.now()}] user{user} feature vector has been loaded {index}/{len(test_users)}')
    return user_vectors

def cosine_similarity_vecs(vec1, vec2):
    vec1 = np.array(vec1).reshape(1, -1)  # 转为二维数组
    vec2 = np.array(vec2).reshape(1, -1)  # 转为二维数组
    return cosine_similarity(vec1, vec2)[0][0]

def get_user_feature_by_recall_item_pairs(model, recall_item_pairs):
    item_pair_sim = []
    # 使用embedding输出物品对的嵌入
    model.eval()
    with torch.no_grad():
        for item_pair in recall_item_pairs:
            item1, item2 = item_pair[0], item_pair[1]
            item_embedding = model.get_item_embedding(torch.tensor([item1, item2]))
            # 计算两个item的余弦相似度
            similarity = cosine_similarity_vecs(item_embedding[0].detach().cpu().numpy(), item_embedding[1].detach().cpu().numpy())
            item_pair_sim.append(similarity)
    return item_pair_sim
