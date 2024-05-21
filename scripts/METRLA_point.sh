#!/bin/bash
cd ../src_copy

method='CSDI+con+inter'
python_script="main.py"
scratch=True
cuda='cuda:3'

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

dataset='METR-LA'
feature_num=207
seq_len=24
missing_pattern='point'
missing_ratio=0.2

# 设置循环次数
for ((i=1; i<=1; i++))
do
    # 生成随机 seed 参数，可以根据需要修改
    seed=$i

    
    # 打印当前运行的信息
    echo "Running iteration $i with seed $seed on device $cuda"

    if [ $scratch = True ]; then
        nohup python -u $python_script \
            --scratch \
            --device $cuda \
            --seed $seed \
            --dataset $dataset \
            --dataset_path ../../KDD2024/datasets/$dataset/raw_data \
            --seq_len $seq_len \
            --feature $feature_num \
            --missing_pattern $missing_pattern \
            --missing_ratio $missing_ratio \
            --scratch \
            > $folder_path/${method}_${dataset}_${missing_pattern}_ms${missing_ratio}_seed${seed}.log 2>&1 &
    else
        nohup python -u $python_script \
            --device $cuda \
            --seed $seed \
            --dataset $dataset \
            --dataset_path ../../KDD2024/datasets/$dataset/raw_data \
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
