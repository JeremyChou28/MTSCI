#!/bin/bash
###
 # @Description: 
 # @Author: Jianping Zhou
 # @Email: jianpingzhou0927@gmail.com
 # @Date: 2024-05-02 12:05:26
### 
version='v2'    
method='intra+inter'
python_script="../src/main_aqi36.py"

dataset='AQI36'
feature_num=36
seq_len=36
batch_size=16
loss_weight=0.5
num_steps=100

scratch=True
cuda='cuda:0'

if [ $scratch = True ]; then
    folder_path="../logs/main_aqi36"
else
    folder_path="../logs/test"
fi

# 检查文件夹是否存在
if [ ! -d "$folder_path" ]; then
    # 如果文件夹不存在，则创建
    mkdir -p "$folder_path"
    echo "Folder created: $folder_path"
else
    echo "Folder already exists: $folder_path"
fi

for i in {5..5}; do
    seed=$i
    echo "seed: $seed"
    nohup python -u $python_script \
        --scratch \
        --device $cuda \
        --seed $seed \
        --dataset $dataset \
        --dataset_path ../datasets/$dataset/raw_data \
        --seq_len $seq_len \
        --feature $feature_num \
        --loss_weight $loss_weight \
        --num_steps $num_steps \
        --batch_size $batch_size \
        > $folder_path/${method}_${dataset}_seed${seed}_bs${batch_size}_lweight${loss_weight}_step${num_steps}.log 2>&1 &
    wait
done