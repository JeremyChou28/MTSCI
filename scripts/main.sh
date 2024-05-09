#!/bin/bash
###
 # @Description: 
 # @Author: Jianping Zhou
 # @Email: jianpingzhou0927@gmail.com
 # @Date: 2023-11-25 20:47:57
### 
cd src

method='main'
python_script="main.py"

folder_path="../logs/test"

# 检查文件夹是否存在
if [ ! -d "$folder_path" ]; then
    # 如果文件夹不存在，则创建
    mkdir -p "$folder_path"
    echo "Folder created: $folder_path"
else
    echo "Folder already exists: $folder_path"
fi


# dataset='ETTm1'
# feature_num=7

dataset='Weather'
feature_num=21

# dataset='METR-LA'
# feature_num=207

# dataset='AQI36'
# feature_num=36
# seq_len=36

seq_len=24
missing_pattern='point'
missing_ratio=0.2

# 设置循环次数
for ((i=4; i<=4; i++))
do
    # 生成随机 seed 参数，可以根据需要修改
    seed=$i

    
    # 打印当前运行的信息
    echo "Running iteration $i with seed $seed: $dataset $missing_pattern $missing_ratio"


    nohup python -u $python_script \
        --scratch \
        --device 'cuda:1' \
        --seed $seed \
        --dataset $dataset \
        --dataset_path ../../KDD2024/datasets/$dataset/raw_data \
        --seq_len $seq_len \
        --feature $feature_num \
        --missing_pattern $missing_pattern \
        --missing_ratio $missing_ratio \
        > $folder_path/${method}_${dataset}_${missing_pattern}_ms${missing_ratio}_seed${seed}.log 2>&1 &

    wait

    # 添加一个空行，以便输出更清晰
    echo ""
done
