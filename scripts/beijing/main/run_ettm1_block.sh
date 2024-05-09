#!/bin/bash
###
 # @Description: 
 # @Author: Jianping Zhou
 # @Email: jianpingzhou0927@gmail.com
 # @Date: 2024-05-02 12:05:26
### 
version='v2'    
method='intra+inter'
python_script="../src/main.py"

dataset='ETTm1'
feature_num=7
seq_len=24
missing_pattern='block'
missing_ratio=0.2

scratch=True
cuda='cuda:2'
seed=5

if [ $scratch = True ]; then
    folder_path="../logs/ettm1_hyper"
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
    nohup python -u $python_script \
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
    nohup python -u $python_script \
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
