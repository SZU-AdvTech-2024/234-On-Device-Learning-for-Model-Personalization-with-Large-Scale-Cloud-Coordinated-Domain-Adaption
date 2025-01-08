home_path="/home/chao/workspace/MPDA-implementation"

run_name="transfer_movielens_NCF_100_recall_item_pair_similarity"

mkdir -p "${home_path}/log/${run_name}/running_logs"

recall_num=100
recall_alg="recall_item_pair_similarity"
device="cuda:2"
dataset="movielens"

for ti in $(seq 0 1 9)
do
  nohup python -u ${home_path}/scripts/transfer.py -task_index=${ti} -recall_num=${recall_num} -recall_alg=${recall_alg} -device=${device} -dataset=${dataset} > ${home_path}/log/${run_name}/running_logs/${ti}.log 2>&1 &
done