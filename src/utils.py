import os
import random

import torch
import numpy as np


class DotDict:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, DotDict(value))
            else:
                setattr(self, key, value)

    def __getattr__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float32)


"""
def random_mask(observed_data, mask, p, rng=None):
    choice = rng.choice
    masked_data = observed_data.copy()
    indicating_mask = np.zeros_like(mask)
    ones_indices = np.argwhere(mask == 1)

    num_ones = len(ones_indices)
    num_elements_to_replace = int(p * num_ones)
    replace_indices = choice(len(ones_indices),
                             size=num_elements_to_replace,
                             replace=False)

    for index in replace_indices:
        i, j = ones_indices[index]
        indicating_mask[i, j] = 1
        masked_data[i, j] = np.nan
    return masked_data, indicating_mask
"""


def random_obs_mask(X, M, p):
    X_copy = X.clone()
    I = torch.zeros_like(M)
    obs_indices = torch.nonzero(M == 1, as_tuple=False)
    num_obs = len(obs_indices)
    num_rnd_obs_mask = int(num_obs * p)
    rnd_indices = torch.randperm(num_obs)[:num_rnd_obs_mask]
    for index in rnd_indices:
        i, j, k = obs_indices[index]
        I[i, j, k] = 1
        X_copy[i, j, k] = 0
    return X_copy, I


def sample_mask(
    shape, p=0.0015, p_noise=0.05, max_seq=1, min_seq=1, rng=None
):  # 制造缺失，block missing和point missing
    if rng is None:
        rand = np.random.random
        randint = np.random.randint
    else:
        rand = rng.random
        randint = rng.integers
    mask = rand(shape) < p
    for col in range(mask.shape[1]):
        idxs = np.flatnonzero(mask[:, col])
        if not len(idxs):
            continue
        fault_len = min_seq
        if max_seq > min_seq:
            fault_len = fault_len + int(randint(max_seq - min_seq))
        idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
        idxs = np.unique(idxs_ext)
        idxs = np.clip(idxs, 0, shape[0] - 1)
        mask[idxs, col] = True
    mask = mask | (rand(mask.shape) < p_noise)
    return mask.astype("uint8")


def get_randmask(observed_mask, min_miss_ratio=0.0, max_miss_ratio=1.0):
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask
    rand_for_mask = rand_for_mask.reshape(-1)
    sample_ratio = np.random.rand()
    sample_ratio = sample_ratio * (max_miss_ratio - min_miss_ratio) + min_miss_ratio
    num_observed = observed_mask.sum().item()
    num_masked = round(num_observed * sample_ratio)
    rand_for_mask[rand_for_mask.topk(num_masked).indices] = -1

    cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
    return cond_mask


def get_block_mask(observed_mask, target_strategy="block"):
    rand_sensor_mask = torch.rand_like(observed_mask)
    randint = np.random.randint
    sample_ratio = np.random.rand()
    sample_ratio = sample_ratio * 0.15
    mask = rand_sensor_mask < sample_ratio
    min_seq = 12
    max_seq = 24
    for col in range(observed_mask.shape[1]):
        idxs = np.flatnonzero(mask[:, col])
        if not len(idxs):
            continue
        fault_len = min_seq
        if max_seq > min_seq:
            fault_len = fault_len + int(randint(max_seq - min_seq))
        idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
        idxs = np.unique(idxs_ext)
        idxs = np.clip(idxs, 0, observed_mask.shape[0] - 1)
        mask[idxs, col] = True
    rand_base_mask = torch.rand_like(observed_mask) < 0.05
    reverse_mask = mask | rand_base_mask
    block_mask = 1 - reverse_mask.to(torch.float32)

    cond_mask = observed_mask.clone()
    mask_choice = np.random.rand()
    if target_strategy == "hybrid" and mask_choice > 0.7:
        cond_mask = get_randmask(observed_mask, 0.0, 1.0)
    else:
        cond_mask = block_mask * cond_mask

    return cond_mask


def masked_mae_loss(predict, true, mask):
    mae = torch.sum(torch.absolute(predict - true) * (1 - mask)) / (
        torch.sum(1 - mask) + 1e-5
    )
    return mae


def masked_mse_loss(predict, true, mask):
    mse = torch.sum((predict - true) ** 2 * (1 - mask)) / (torch.sum(1 - mask) + 1e-5)
    return mse


def masked_rmse_loss(predict, true, mask):
    rmse = torch.sqrt(
        torch.sum((predict - true) ** 2 * (1 - mask)) / (torch.sum(1 - mask)) + 1e-5
    )
    return rmse


def masked_mape_loss(predict, true, mask):
    mape = torch.sum(torch.absolute((predict - true) * (1 - mask))) / (
        torch.sum(true * (1 - mask)) + 1e-5
    )
    return mape


def missed_eval_torch(predict, true, mask):
    mae = torch.sum(torch.absolute(predict - true) * (1 - mask)) / torch.sum(1 - mask)
    rmse = torch.sqrt(
        torch.sum((predict - true) ** 2 * (1 - mask)) / torch.sum(1 - mask)
    )
    mape = torch.sum(torch.absolute((predict - true) * (1 - mask))) / (
        torch.sum(torch.absolute(true * (1 - mask))) + 1e-5
    )
    # mape_list = []
    # for i in range(predict.shape[0]):
    #     mape = torch.sum(torch.absolute(
    #         (predict[i] - true[i]) * (1 - mask[i]))) / (
    #             torch.sum(torch.absolute(true[i] * (1 - mask[i]))) + 1e-5)
    #     mape_list.append(mape.item())
    return mae, rmse, mape


def missed_eval_np(predict, true, mask):
    predict, true = np.asarray(predict), np.asarray(true)
    mae = np.sum(np.absolute(predict - true) * (1 - mask)) / (np.sum(1 - mask) + 1e-5)
    rmse = np.sqrt(
        np.sum((predict - true) ** 2 * (1 - mask)) / (np.sum(1 - mask) + 1e-5)
    )
    mape = np.sum(np.absolute((predict - true) * (1 - mask))) / (
        np.sum(np.absolute(true * (1 - mask))) + 1e-5
    )
    return mae, rmse, mape


def missed_eval_np(predict, true, mask):
    """
    predict: [samples, seq_len, feature_dim]
    true: [samples, seq_len, feature_dim]
    mask: [samples, seq_len, feature_dim]
    """
    predict, true = np.asarray(predict), np.asarray(true)
    mae = np.sum(np.absolute(predict - true) * (1 - mask)) / (np.sum(1 - mask) + 1e-5)
    mse = np.sum((predict - true) ** 2 * (1 - mask)) / (np.sum(1 - mask) + 1e-5)
    rmse = np.sqrt(
        np.sum((predict - true) ** 2 * (1 - mask)) / (np.sum(1 - mask) + 1e-5)
    )
    mape = np.sum(np.absolute((predict - true) * (1 - mask))) / (
        np.sum(np.absolute(true * (1 - mask))) + 1e-5
    )
    R2 = 1 - mse / (
        np.sum((true - np.mean(true)) ** 2 * (1 - mask)) / (np.sum(1 - mask) + 1e-5)
    )
    return mae, rmse, mape, mse, R2
