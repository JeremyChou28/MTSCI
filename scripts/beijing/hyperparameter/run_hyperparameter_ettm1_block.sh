#!/bin/bash
###
 # @Description: 
 # @Author: Jianping Zhou
 # @Email: jianpingzhou0927@gmail.com
 # @Date: 2024-05-02 12:02:00
### 
version='v3'    
method='intra+inter'
python_script="../src/main_hyperparamter.py"

dataset='ETTm1'
feature_num=7
seq_len=24
missing_pattern='block'
missing_ratio=0.2

scratch=True
cuda='cuda:0'
seed=5

# 调channel
# channel_list=(16 32 128)
# layer_list=(4)
# beta_list=(0.2)

# 调layer
# channel_list=(64)
# layer_list=(1)
# beta_list=(0.2)

# 调beta
# channel_list=(64)
# layer_list=(4)
# beta_list=(0.1 0.3 0.4)

# 调lambda
lambda_list=(0.5 1 2)
channel_list=(64)
layer_list=(4)
beta_list=(0.2)
# lambda_list=(0.5 1 2 5 10)

if [ $scratch = True ]; then
    folder_path="../logs/hyperparamter"
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

for loss_weight in ${lambda_list[@]}; do
    for channel in ${channel_list[@]}; do
        for layer in ${layer_list[@]}; do
            for beta_end in ${beta_list[@]}; do
                echo "channel: $channel; layer: $layer; beta_end: $beta_end; loss_weight: $loss_weight;"
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
                    --channel $channel \
                    --layer $layer \
                    --beta_end $beta_end \
                    --loss_weight $loss_weight \
                    > $folder_path/${method}_${dataset}_${missing_pattern}_channel${channel}_layer${layer}_beta${beta_end}_lweight${loss_weight}.log 2>&1 &
                wait
            done
        done
    done
done