#!/bin/bash
###
 # @Description: 
 # @Author: Jianping Zhou
 # @Email: jianpingzhou0927@gmail.com
 # @Date: 2024-05-02 12:05:26
### 
version='v3'    
method='intra+inter'
python_script="../src/main_iid.py"

dataset='Weather'
feature_num=21
seq_len=24
missing_pattern='point'

scratch=True
cuda='cuda:3'
seed=5

if [ $scratch = True ]; then
    folder_path="../logs/sensitivity_iid"
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

for ((i=9; i<=9; i++))
do

    missing_ratio=0.$i

    echo "Running with missing ratio $missing_ratio"

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

    wait

done