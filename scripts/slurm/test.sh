#!/bin/bash
single_gpus=1
node_num=1
host="SH-IDC1-10-140-24-122"
quotatype="auto"

version='v2'    
method='intra+inter'
python_script="../src/main.py"

dataset='Weather'
feature_num=21
seq_len=24
missing_pattern='block'
missing_ratio=0.2

scratch=True
# checkpoint_path="../saved_models/test/ETTm1/block/0.2/model_2024-04-27-19-52-01.pth"
cuda='cuda:0'
seed=1

if [ $scratch = True ]; then
    folder_path="../logs/scratch"
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

if [ $scratch = True ]; then
    nohup srun -p ai4earth -w $host --kill-on-bad-exit=1 --quotatype=$quotatype --ntasks-per-node=$single_gpus -N $node_num --gres=gpu:$single_gpus python -u $python_script \
        --scratch \
        --device $cuda \
        --seed $seed \
        --dataset $dataset \
        --dataset_path ../datasets/$dataset/raw_data \
        --seq_len $seq_len \
        --feature $feature_num \
        --missing_pattern $missing_pattern \
        --missing_ratio $missing_ratio \
        --scratch \
        > $folder_path/${method}_${dataset}_${missing_pattern}_ms${missing_ratio}_seed${seed}_${version}.log 2>&1 &
else
    nohup srun -p ai4earth -w $host --kill-on-bad-exit=1 --quotatype=spot --ntasks-per-node=$single_gpus -N $node_num --gres=gpu:$single_gpus python -u $python_script \
        --checkpoint_path $checkpoint_path \
        --device $cuda \
        --seed $seed \
        --dataset $dataset \
        --dataset_path ../datasets/$dataset/raw_data \
        --seq_len $seq_len \
        --feature $feature_num \
        --missing_pattern $missing_pattern \
        --missing_ratio $missing_ratio \
        > $folder_path/${method}_${dataset}_${missing_pattern}_ms${missing_ratio}_seed${seed}_${version}.log 2>&1 &
fi