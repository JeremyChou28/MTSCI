#!/bin/bash
cd ../src

method='MTSCI'
python_script="main.py"

scratch=True
cuda='cuda:2'
dataset='ETT'
feature_num=7
seq_len=24
missing_pattern='block'
missing_ratio=0.2

if [ $scratch = True ]; then
    folder_path="../logs/scratch"
else
    folder_path="../logs/test"
fi

if [ ! -d "$folder_path" ]; then
    mkdir -p "$folder_path"
    echo "Folder created: $folder_path"
else
    echo "Folder already exists: $folder_path"
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
            --dataset_path ../../../../KDD2024/datasets/$dataset/raw_data \
            --seq_len $seq_len \
            --feature $feature_num \
            --missing_pattern $missing_pattern \
            --missing_ratio $missing_ratio \
            > $folder_path/${method}_${dataset}_${missing_pattern}_ms${missing_ratio}_seed${seed}.log 2>&1 &
    else
        nohup python -u $python_script \
            --device $cuda \
            --seed $seed \
            --dataset $dataset \
            --dataset_path ../../../../KDD2024/datasets/$dataset/raw_data \
            --seq_len $seq_len \
            --feature $feature_num \
            --missing_pattern $missing_pattern \
            --missing_ratio $missing_ratio \
            > $folder_path/${method}_${dataset}_${missing_pattern}_ms${missing_ratio}_seed${seed}.log 2>&1 &
    fi

    wait

    # 添加一个空行，以便输出更清晰
    echo ""
done
