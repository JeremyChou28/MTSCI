import argparse
import json
import yaml
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import wandb

from utils import *
sys.path.append('../models')
sys.path.append('../dataloader')
from dataloader import *
from model_wointra import CSDI_Imp


def train(
    model,
    config,
    args,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=5,
    foldername="",
    current_time=None,
):
    wandb.init(
        project="CIKM2024",
        name="test_{}_{}".format(args.dataset, current_time),
        config=args,
    )

    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model_{}.pth".format(current_time)

    p1 = int(0.5 * config["epochs"])
    p2 = int(0.75 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss, avg_loss_noise = 0.0, 0.0
        model.train()
        # with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
        for batch_no, train_batch in enumerate(train_loader):
            optimizer.zero_grad()

            loss_noise = model(train_batch)
            loss = loss_noise   # compute total loss
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            avg_loss_noise += loss_noise.item()
        lr_scheduler.step()
        train_loss = avg_loss / batch_no
        train_loss_noise = avg_loss_noise / batch_no

        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                for batch_no, valid_batch in enumerate(valid_loader):
                    loss = model(valid_batch, is_train=0)
                    avg_loss_valid += loss.item()
                valid_loss = avg_loss_valid / batch_no
                print(
                    "Epoch {}: train loss = {} train_loss_noise = {} valid loss = {}".format(
                        epoch_no + 1,
                        train_loss,
                        train_loss_noise,
                        valid_loss,
                    )
                )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )
                torch.save(model.state_dict(), output_path)
            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_loss_noise": train_loss_noise,
                    "valid_loss": valid_loss,
                }
            )
        else:
            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_loss_noise": train_loss_noise,
                }
            )
            print(
                "Epoch {}: train loss = {} train_loss_noise = {}".format(
                    epoch_no + 1, train_loss, train_loss_noise
                )
            )


def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):

    with torch.no_grad():
        model.eval()

        imputed_data = []
        groundtruth = []
        eval_mask = []
        results = {}
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time = (
                    output  # imputed results
                )
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)  # (B,L,K)
                observed_points = observed_points.permute(0, 2, 1)  # (B,L,K)

                samples_median = samples.median(
                    dim=1
                )  # use median as prediction to calculate the RMSE and MAE, include the median values and the indices

                output = samples_median.values * scaler + mean_scaler
                X_Tilde = c_target * scaler + mean_scaler
                eval_M = eval_points
                imputed_data.append(output.cpu().numpy())
                groundtruth.append(X_Tilde.cpu().numpy())
                eval_mask.append(eval_M.cpu().numpy())

            results["imputed_data"] = np.concatenate(imputed_data, axis=0)
            results["groundtruth"] = np.concatenate(groundtruth, axis=0)
            results["eval_mask"] = np.concatenate(eval_mask, axis=0)

            mae, rmse, mape, mse, r2 = missed_eval_np(
                results["imputed_data"],
                results["groundtruth"],
                1 - results["eval_mask"],
            )

            print(
                "mae = {:.3f}, rmse = {:.3f}, mape = {:.3f}%".format(
                    mae, rmse, mape * 100
                )
            )
            return results


def main(args):
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(current_time)

    seed_torch(args.seed)
    path = "../config/{}_{}.yaml".format(args.dataset, args.missing_pattern)
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    config["model"]["is_unconditional"] = args.unconditional

    print(json.dumps(config, indent=4))

    # load args
    dataset = args.dataset
    dataset_path = args.dataset_path
    seq_len = args.seq_len
    miss_rate = args.missing_ratio
    missing_pattern = args.missing_pattern
    batch_size = config["train"]["batch_size"]

    saving_path = args.saving_path + "/test/{}/{}/{}".format(
        dataset, missing_pattern, miss_rate
    )
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    print("model folder:", saving_path)

    save_result_path = args.save_result_path + "test/{}/{}/{}".format(
        dataset, missing_pattern, miss_rate
    )
    if not os.path.exists(save_result_path):
        os.makedirs(save_result_path)
    print("save result path: ", save_result_path)

    # load data
    train_loader = generate_train_dataloader(
        dataset_path,
        seq_len,
        missing_ratio=miss_rate,
        missing_pattern=missing_pattern,
        batch_size=batch_size,
    )
    val_loader = generate_val_test_dataloader(
        dataset_path,
        seq_len,
        missing_ratio=miss_rate,
        missing_pattern=missing_pattern,
        batch_size=batch_size,
        mode="val",
    )
    test_loader = generate_val_test_dataloader(
        dataset_path,
        seq_len,
        missing_ratio=miss_rate,
        missing_pattern=missing_pattern,
        batch_size=batch_size,
        mode="test",
    )
    print("len train dataloader: ", len(train_loader))
    print("len val dataloader: ", len(val_loader))
    print("len test dataloader: ", len(test_loader))
    with open(dataset_path + "/scaler.pkl", "rb") as fb:
        mean, std = pk.load(fb)
    mean = torch.from_numpy(mean).to(args.device)
    std = torch.from_numpy(std).to(args.device)

    model = CSDI_Imp(
        config, args.device, target_dim=args.feature, seq_len=args.seq_len
    ).to(args.device)

    if args.scratch:
        train(
            model,
            config["train"],
            args,
            train_loader,
            valid_loader=val_loader,
            foldername=saving_path,
            current_time=current_time,
        )
        print("load model from", saving_path)
        model.load_state_dict(
            torch.load(saving_path + "/model_{}.pth".format(current_time))
        )
    else:
        print("load model from", args.checkpoint_path)

        model.load_state_dict(torch.load(args.checkpoint_path))

    results = evaluate(
        model,
        test_loader,
        nsample=args.nsample,
        scaler=std,
        mean_scaler=mean,
        foldername=saving_path,
    )
    np.save(save_result_path + "/result_{}.npy".format(current_time), results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSDI")
    parser.add_argument("--device", default="cuda:0", help="Device for Attack")
    parser.add_argument("--dataset", default="ETTm1", type=str, help="dataset name")
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="parent dir of generated dataset",
        default="../../../../KDD2024/datasets/ETTm1/raw_data/",
    )
    parser.add_argument(
        "--save_result_path",
        help="the save path of imputed data",
        type=str,
        default="../results/",
    )
    parser.add_argument(
        "--saving_path", type=str, help="saving model pth", default="../saved_models"
    )
    parser.add_argument("--seq_len", help="sequence length", type=int, default=24)
    parser.add_argument(
        "--missing_pattern", help="missing pattern", type=str, default="point"
    )
    parser.add_argument(
        "--missing_ratio", help="missing ratio", type=float, default=0.2
    )
    parser.add_argument("--feature", help="feature nums", type=int, default=7)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="../saved_models/test/ETTm1/block/0.2/model_2024-04-27-15-58-21.pth",
    )
    parser.add_argument("--num_workers", type=int, default=0, help="Device for Attack")
    parser.add_argument("--scratch", action="store_true", help="test or scratch")
    parser.add_argument("--nsample", type=int, default=100)
    parser.add_argument("--unconditional", action="store_true")
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()
    print(args)

    start_time = time.time()
    main(args)
    print("Spend Time: ", time.time() - start_time)
