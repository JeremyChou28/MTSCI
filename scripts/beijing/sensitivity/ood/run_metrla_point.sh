#!/bin/bash
###
 # @Description: 
 # @Author: Jianping Zhou
 # @Email: jianpingzhou0927@gmail.com
 # @Date: 2024-05-02 12:05:26
### 
version='v2'    
method='intra+inter'
python_script="../src/main_ood.py"

dataset='METR-LA'
feature_num=207
seq_len=24
missing_pattern='point'

scratch=False
cuda='cuda:0'
seed=5
checkpoint_path='../saved_models/test/METR-LA/point/0.2/model_2024-05-04-12-03-28.pth'
folder_path="../logs/sensitivity_ood"


# 检查文件夹是否存在
if [ ! -d "$folder_path" ]; then
    # 如果文件夹不存在，则创建
    mkdir -p "$folder_path"
    echo "Folder created: $folder_path"
else
    echo "Folder already exists: $folder_path"
fi

for ((i=7; i<=9; i++))
do

    missing_ratio=0.2
    val_miss_ratio=0.$i

    echo "Running with val missing ratio $val_miss_ratio"

    nohup python -u $python_script \
        --ood \
        --device $cuda \
        --seed $seed \
        --dataset $dataset \
        --dataset_path ../datasets/$dataset/raw_data \
        --checkpoint_path $checkpoint_path \
        --seq_len $seq_len \
        --feature $feature_num \
        --missing_pattern $missing_pattern \
        --missing_ratio $missing_ratio \
        --val_missing_ratio $val_miss_ratio \
        --test_missing_ratio $val_miss_ratio \
        > $folder_path/${method}_${dataset}_${missing_pattern}_valms${val_miss_ratio}_seed${seed}_${version}.log 2>&1 &
    wait

done