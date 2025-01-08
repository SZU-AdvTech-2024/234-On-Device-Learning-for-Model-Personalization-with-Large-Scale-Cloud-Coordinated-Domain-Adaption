home_path="/home/chao/workspace/MPDA-implementation"

run_name="movielens_NCF_Local"

mkdir -p "${home_path}/log/${run_name}/running_logs"


device="cuda:2"
dataset="movielens"
model="NCF"

for ti in $(seq 0 1 9)
do
  nohup python -u ${home_path}/scripts/local.py -task_index=${ti} -device=${device} -dataset=${dataset} -model=${model} > ${home_path}/log/${run_name}/running_logs/${ti}.log 2>&1 &
done