'''
Description: 
Author: Jianping Zhou
Email: jianpingzhou0927@gmail.com
Date: 2023-11-28 17:17:37
'''
import sys
import numpy as np
import pickle as pk
import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset
sys.path.append('../utils')
from timefeatures import time_features


def generate_val_dataloader(dataset_path,
                            seq_len,
                            batch_size=4,
                            timefeature=None):
    with open(dataset_path + '/valid_data.pkl', 'rb') as fb:
        data_dict = pk.load(fb)

    X_list = data_dict['X']
    X_Tilde_list = data_dict['X_Tilde']
    X_mask_list = data_dict['X_Mask']
    X_Tilde_mask_list = data_dict['X_Tilde_Mask']
    timestamp_list = data_dict['timestamps']

    input_X_list, input_mask_list, eval_mask, output_gt_list,timepoints = [], [], [], [],[]
    for index in range(len(X_list)):
        X, mask, X_Tilde, gt_mask, timestamp = np.nan_to_num(
            X_list[index]), X_mask_list[index], np.nan_to_num(
                X_Tilde_list[index]
            ), X_Tilde_mask_list[index], timestamp_list[index]
        indicating_mask = gt_mask * (1 - mask)

        sample_nums = X.shape[0] - seq_len + 1
        for i in range(sample_nums):
            input_X_list.append(X[i:i + seq_len])
            input_mask_list.append(mask[i:i + seq_len])
            eval_mask.append(indicating_mask[i:i + seq_len])
            output_gt_list.append(X_Tilde[i:i + seq_len])
            timepoints.append(
                time_features(pd.to_datetime(timestamp[i:i + seq_len]),
                              freq='m').transpose(1, 0))

    print("val samples: {}".format(len(input_X_list)))

    X_tensor = torch.from_numpy(np.array(input_X_list)).float()
    mask_tensor = torch.from_numpy(np.array(input_mask_list)).float()
    eval_mask_tensor = torch.from_numpy(np.array(eval_mask)).float()
    X_Tilde_tensor = torch.from_numpy(np.array(output_gt_list)).float()
    timepoint_tensor = torch.from_numpy(np.array(timepoints)).float()

    if timefeature is not None:
        tensor_dataset = TensorDataset(X_tensor, mask_tensor, X_Tilde_tensor,
                                       eval_mask_tensor, timepoint_tensor)
    else:
        tensor_dataset = TensorDataset(X_tensor, mask_tensor, X_Tilde_tensor,
                                       eval_mask_tensor)
    dataloader = DataLoader(tensor_dataset,
                            batch_size=batch_size,
                            shuffle=False)
    return dataloader


def generate_test_dataloader(dataset_path,
                             seq_len,
                             batch_size=4,
                             timefeature=None):
    with open(dataset_path + '/test_data.pkl', 'rb') as fb:
        data_dict = pk.load(fb)

    X_list = data_dict['X']
    X_Tilde_list = data_dict['X_Tilde']
    X_mask_list = data_dict['X_Mask']
    X_Tilde_mask_list = data_dict['X_Tilde_Mask']
    timestamp_list = data_dict['timestamps']

    input_X_list, input_mask_list, eval_mask, output_gt_list,timepoints = [], [], [], [],[]
    for index in range(len(X_list)):
        X, mask, X_Tilde, gt_mask, timestamp = np.nan_to_num(
            X_list[index]), X_mask_list[index], np.nan_to_num(
                X_Tilde_list[index]
            ), X_Tilde_mask_list[index], timestamp_list[index]
        indicating_mask = gt_mask * (1 - mask)

        sample_nums = X.shape[0] // seq_len
        for i in range(sample_nums):
            input_X_list.append(X[i * seq_len:(i + 1) * seq_len])
            input_mask_list.append(mask[i * seq_len:(i + 1) * seq_len])
            eval_mask.append(indicating_mask[i * seq_len:(i + 1) * seq_len])
            output_gt_list.append(X_Tilde[i * seq_len:(i + 1) * seq_len])
            timepoints.append(
                time_features(pd.to_datetime(timestamp[i:i + seq_len]),
                              freq='m').transpose(1, 0))

    print("test samples: {}".format(len(input_X_list)))

    X_tensor = torch.from_numpy(np.array(input_X_list)).float()
    mask_tensor = torch.from_numpy(np.array(input_mask_list)).float()
    eval_mask_tensor = torch.from_numpy(np.array(eval_mask)).float()
    X_Tilde_tensor = torch.from_numpy(np.array(output_gt_list)).float()
    timepoint_tensor = torch.from_numpy(np.array(timepoints)).float()

    if timefeature is not None:
        tensor_dataset = TensorDataset(X_tensor, mask_tensor, X_Tilde_tensor,
                                       eval_mask_tensor, timepoint_tensor)
    else:
        tensor_dataset = TensorDataset(X_tensor, mask_tensor, X_Tilde_tensor,
                                       eval_mask_tensor)
    dataloader = DataLoader(tensor_dataset,
                            batch_size=batch_size,
                            shuffle=False)
    return dataloader


def generate_train_dataloader(dataset_path,
                              seq_len,
                              batch_size=4,
                              timefeature=None):
    with open(dataset_path + '/train_data.pkl', 'rb') as fb:
        train_data_dict = pk.load(fb)

    X_list = train_data_dict['X']
    X_Tilde_list = train_data_dict['X_Tilde']
    X_mask_list = train_data_dict['X_Mask']
    X_Tilde_mask_list = train_data_dict['X_Tilde_Mask']
    timestamp_list = train_data_dict['timestamps']

    input_X_list,input_mask_list,eval_mask,output_gt_list,pred_gt_list,pred_gt_mask,timepoints=[],[],[],[],[],[],[]
    for index in range(len(X_list)):
        X, mask, X_Tilde, gt_mask, train_timestamp = np.nan_to_num(
            X_list[index]), X_mask_list[index], np.nan_to_num(
                X_Tilde_list[index]
            ), X_Tilde_mask_list[index], timestamp_list[index]
        indicating_mask = gt_mask * (1 - mask)
        train_nums = X.shape[0] - 2 * seq_len + 1
        for i in range(train_nums):
            input_X_list.append(X[i:i + seq_len])
            input_mask_list.append(mask[i:i + seq_len])
            eval_mask.append(indicating_mask[i:i + seq_len])
            output_gt_list.append(X_Tilde[i:i + seq_len])
            pred_gt_list.append(X_Tilde[i + seq_len:i + 2 * seq_len])
            pred_gt_mask.append(gt_mask[i + seq_len:i + 2 * seq_len])
            timepoints.append(
                time_features(pd.to_datetime(train_timestamp[i:i + seq_len]),
                              freq='m').transpose(1, 0))

    print("train samples: {}".format(len(input_X_list)))

    X_tensor = torch.from_numpy(np.array(input_X_list)).float()
    mask_tensor = torch.from_numpy(np.array(input_mask_list)).float()
    eval_mask_tensor = torch.from_numpy(np.array(eval_mask)).float()
    X_Tilde_tensor = torch.from_numpy(np.array(output_gt_list)).float()
    pred_gt_tensor = torch.from_numpy(np.array(pred_gt_list)).float()
    pred_gt_mask_tensor = torch.from_numpy(np.array(pred_gt_mask)).float()
    timepoint_tensor = torch.from_numpy(np.array(timepoints)).float()

    if timefeature is not None:
        train_dataset = TensorDataset(X_tensor, mask_tensor, eval_mask_tensor,
                                      X_Tilde_tensor, pred_gt_tensor,
                                      pred_gt_mask_tensor, timepoint_tensor)
    else:
        train_dataset = TensorDataset(X_tensor, mask_tensor, eval_mask_tensor,
                                      X_Tilde_tensor, pred_gt_tensor,
                                      pred_gt_mask_tensor)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
    return train_dataloader


if __name__ == "__main__":
    dataset_path = '../datasets/AQI36/raw_data/'
    seq_len = 36
    batch_size = 4
    train_loader = generate_train_dataloader(dataset_path,
                                             seq_len,
                                             batch_size=batch_size)
    val_loader = generate_val_dataloader(dataset_path,
                                         seq_len,
                                         batch_size=batch_size)
    test_loader = generate_test_dataloader(dataset_path,
                                           seq_len,
                                           batch_size=batch_size)
    print('len train dataloader: ', len(train_loader))
    print('len val dataloader: ', len(val_loader))
    print('len test dataloader: ', len(test_loader))
