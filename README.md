# Implementation MPDA with pytorch

This is my implementation of the paper MPDA(On-Device Learning for Model Personalization with Large-Scale Cloud-Coordinated Domain Adaption) with pytorch. 

## Download Dataset

Download [Movielens-20m](https://grouplens.org/datasets/movielens/20m/) to /data/MovieLens

Download [Amazon Electronics](https://jmcauley.ucsd.edu/data/amazon/) to /data/Amazon

## Config
First you need to config the root path in /config.yml

## MovieLens Dataset Preprocess
generate users with train json file and users with train and test data json file
```shell
nohup python -u scripts/preprocess/movielens/generate_user_with_train_and_test.py > ./log/generate_user_with_train_and_test.log 2>&1 &
```

generate user and item mapping
```shell
python scripts/preprocess/movielens/generate_mapping_file.py
```

generate recall item pairs
```shell
python scripts/preprocess/movielens/generate_recall_item_pairs.py
```

## Amazon Dataset Preprocess
generate rating.csv with raw_data.json
```shell
python scripts/preprocess/amazon/generate_raw_data.py
```


## Initial Model
train global model NCF on MovieLens
```shell
nohup python -u scripts/train_global_model.py -model=NCF -epochs=10 -dataset=movielens -device=cuda:2 > ./log/train_global_model.log 2>&1 &
```

train mask model on MovieLens
```
nohup python -u scripts/train_mask_model.py -device=cuda:2 > ./log/train_mask_model.log 2>&1 &
```

transfer model NCF
```shell
bash ./commands/ncf_movielens_50_random.sh
```