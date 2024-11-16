#!/bin/bash
###
 # @Description: 
 # @Author: Jianping Zhou
 # @Email: jianpingzhou0927@gmail.com
 # @Date: 2024-11-16 18:31:07
### 
cd ../src

python_script="main.py"

scratch=False
cuda='cuda:4'
dataset='METR-LA'
feature_num=207
seq_len=24
missing_pattern='block'
missing_ratio=0.05
val_missing_ratio=0.05
test_missing_ratio=0.05
dataset_path="../datasets/$dataset/"
checkpoint_path="../saved_models/METR-LA/block/0.2/model.pth"

if [ $scratch = True ]; then
    log_path="../logs/scratch"
else
    log_path="../logs/test"
fi

if [ ! -d "$log_path" ]; then
    mkdir -p "$log_path"
    echo "Folder created: $log_path"
else
    echo "Folder already exists: $log_path"
fi

for ((i=1; i<=5; i++))
do
    seed=$i

    echo "Running iteration $i with seed $seed on device $cuda"

    if [ $scratch = True ]; then
        nohup python -u $python_script \
            --scratch \
            --device $cuda \
            --seed $seed \
            --dataset $dataset \
            --dataset_path $dataset_path \
            --seq_len $seq_len \
            --feature $feature_num \
            --missing_pattern $missing_pattern \
            --missing_ratio $missing_ratio \
            --val_missing_ratio $val_missing_ratio \
            --test_missing_ratio $test_missing_ratio \
            > $log_path/${dataset}_${missing_pattern}_ms${missing_ratio}_seed${seed}.log 2>&1 &
    else
        nohup python -u $python_script \
            --device $cuda \
            --seed $seed \
            --dataset $dataset \
            --dataset_path $dataset_path \
            --seq_len $seq_len \
            --feature $feature_num \
            --missing_pattern $missing_pattern \
            --missing_ratio $missing_ratio \
            --val_missing_ratio $val_missing_ratio \
            --test_missing_ratio $test_missing_ratio \
            --checkpoint_path $checkpoint_path \
            --nsample 100 \
            > $log_path/${dataset}_${missing_pattern}_ms${missing_ratio}_seed${seed}.log 2>&1 &
    fi

    wait

    echo ""
done
