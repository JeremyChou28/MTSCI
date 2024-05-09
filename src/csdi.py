import argparse
import torch
import json
import yaml
import os
import time
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

from datetime import datetime
from torch.optim import Adam
from tqdm import tqdm
import pickle
import wandb

from utils import *

sys.path.append("../dataloader")
from dataloader import *


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(
            0
        )  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(
            -1
        )  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip


class CSDI_base(nn.Module):

    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = (
                np.linspace(
                    config_diff["beta_start"] ** 0.5,
                    config_diff["beta_end"] ** 0.5,
                    self.num_steps,
                )
                ** 2
            )
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = (
            torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
        )

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp, cond_mask, X_Tilde, pred, pred_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb_feature)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat(
            [time_embed, feature_embed], dim=-1
        )  # (B,L,K,*=emb+emb_feature)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat(
                [side_info, side_mask], dim=1
            )  # (B,*=emb+emb_feature+1,K,L)

        return side_info

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self,
        observed_data,
        cond_mask,
        observed_mask,
        side_info,
        is_train,
        set_t=-1,
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha**0.5) * observed_data + (
            1.0 - current_alpha
        ) ** 0.5 * noise  # 加噪

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual**2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[
                        t
                    ] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional == True:
                    diff_input = (
                        cond_mask * noisy_cond_history[t]
                        + (1.0 - cond_mask) * current_sample
                    )
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                predicted = self.diffmodel(
                    diff_input, side_info, torch.tensor([t]).to(self.device)
                )

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1):
        if is_train:
            (
                X_Tilde,
                X_Tilde_mask,
                observed_tp,
                X_mask,
                indicating_mask,
                pred,
                pred_mask,
            ) = self.process_data(batch, is_train)
        else:
            (X_Tilde, X_Tilde_mask, observed_tp, X_mask, indicating_mask) = (
                self.process_data(batch, is_train)
            )
            pred, pred_mask, pred_side_info = None, None, None

        cond_mask = X_mask

        side_info = self.get_side_info(
            observed_tp, cond_mask, X_Tilde, pred, pred_mask
        )  # 获取边信息，即位置编码

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(X_Tilde, cond_mask, X_Tilde_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
        ) = self.process_data(batch, istrain=0)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

        return samples, observed_data, target_mask, observed_mask, observed_tp


class CSDI_Imp(CSDI_base):

    def __init__(self, config, device, target_dim=36, seq_len=24):
        super(CSDI_Imp, self).__init__(target_dim, config, device)
        self.seq_len = seq_len

    def process_data(self, batch, istrain=1):
        if istrain:
            (
                X_tensor,
                mask_tensor,
                indicating_mask_tensor,
                X_Tilde_tensor,
                pred_tensor,
                pred_mask_tensor,
            ) = batch
        else:
            X_tensor, mask_tensor, X_Tilde_tensor, indicating_mask_tensor = batch

        X_Tilde = X_Tilde_tensor.to(self.device).float()  # B,L,F，带天然缺失的数据
        X_mask = mask_tensor.to(
            self.device
        ).float()  # 天然缺失+人为缺失后的mask，缺失为0，不缺失为1
        indicating_mask = indicating_mask_tensor.to(
            self.device
        ).float()  # indicating mask
        X_Tilde_mask = X_mask + indicating_mask  # 天然缺失的mask，缺失为0，不缺失为1

        batch_size = X_Tilde.shape[0]
        observed_tp = (
            torch.from_numpy(
                np.tile(
                    np.arange(self.seq_len), batch_size
                ).reshape(  # [0, 1, 2, 3, seq_len - 1] * batch_size
                    batch_size, self.seq_len
                )
            )
            .to(self.device)
            .float()
        )

        X_Tilde = X_Tilde.permute(0, 2, 1)  # B,F,L
        X_Tilde_mask = X_Tilde_mask.permute(0, 2, 1)
        X_mask = X_mask.permute(0, 2, 1)
        indicating_mask = indicating_mask.permute(0, 2, 1)

        if istrain:
            pred = pred_tensor.to(self.device).float()
            pred_mask = pred_mask_tensor.to(self.device).float()
            pred = pred.permute(0, 2, 1)
            pred_mask = pred_mask.permute(0, 2, 1)

            return (
                X_Tilde,
                X_Tilde_mask,
                observed_tp,
                X_mask,
                indicating_mask,
                pred,
                pred_mask,
            )
        else:
            return (X_Tilde, X_Tilde_mask, observed_tp, X_mask, indicating_mask)


def train(
    model,
    config,
    args,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=5,
    foldername="",
):
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(current_time)
    wandb.init(
        project="CSDI",
        name="csdi_{}_{}".format(args.dataset, current_time),
        config=args,
    )

    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model_seed{}.pth".format(args.seed)

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        # with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
        for batch_no, train_batch in enumerate(train_loader):
            optimizer.zero_grad()

            loss = model(train_batch)
            loss.backward()
            avg_loss += loss.item()
            optimizer.step()
            # it.set_postfix(
            #     ordered_dict={
            #         "avg_epoch_loss": avg_loss / batch_no,
            #         "epoch": epoch_no,
            #     },
            #     refresh=False,
            # )
        lr_scheduler.step()
        train_loss = avg_loss / batch_no
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                # with tqdm(valid_loader, mininterval=5.0,
                #   maxinterval=50.0) as it:
                for batch_no, valid_batch in enumerate(valid_loader):
                    loss = model(valid_batch, is_train=0)
                    avg_loss_valid += loss.item()
                    # it.set_postfix(
                    #     ordered_dict={
                    #         "valid_avg_epoch_loss":
                    #         avg_loss_valid / batch_no,
                    #         "epoch": epoch_no,
                    #     },
                    #     refresh=False,
                    # )
                valid_loss = avg_loss_valid / batch_no
                print(
                    "Epoch {}: train loss = {} valid loss = {}".format(
                        epoch_no + 1, train_loss, valid_loss
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
            wandb.log({"train_loss": train_loss, "valid_loss": valid_loss})
        else:
            wandb.log({"train_loss": train_loss})
            print(
                "Epoch {}: train loss = {}".format(
                    epoch_no + 1,
                    train_loss,
                )
            )

    if foldername != "":
        torch.save(model.state_dict(), output_path)
    wandb.finish()


def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):

    with torch.no_grad():
        model.eval()
        mae_list, rmse_list, mape_list = [], [], []
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
            print(
                results["imputed_data"].shape,
                results["groundtruth"].shape,
            )
            mae, rmse, mape, mse, r2 = missed_eval_np(
                results["imputed_data"],
                results["groundtruth"],
                1 - results["eval_mask"],
            )
            print(
                "mae = {:.3f}, rmse = {:.3f}, mape = {:.3f}%, mse = {:.3f}, r2 = {:.3f}".format(
                    mae, rmse, mape * 100, mse, r2
                )
            )
            return results


def main(args):
    seed_torch(args.seed)

    # load args
    dataset = args.dataset
    dataset_path = args.dataset_path
    seq_len = args.seq_len
    miss_rate = args.missing_ratio
    val_miss_rate, test_miss_rate = args.val_missing_ratio, args.test_missing_ratio
    missing_pattern = args.missing_pattern

    path = "../csdi_config/base.yaml"
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    batch_size = config["train"]["batch_size"]
    config["model"]["is_unconditional"] = args.unconditional
    config["model"]["target_strategy"] = args.targetstrategy

    print(json.dumps(config, indent=4))

    saving_path = args.saving_path + "/CSDI/{}/{}/{}".format(
        dataset, missing_pattern, miss_rate
    )
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    save_result_path = args.save_result_path + "csdi/{}/{}/{}".format(
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
        missing_ratio=val_miss_rate,
        missing_pattern=missing_pattern,
        batch_size=batch_size,
        mode="val",
    )
    test_loader = generate_val_test_dataloader(
        dataset_path,
        seq_len,
        missing_ratio=test_miss_rate,
        missing_pattern=missing_pattern,
        batch_size=batch_size,
        mode="test",
    )
    print("len train dataloader: ", len(train_loader))
    print("len val dataloader: ", len(val_loader))
    print("len test dataloader: ", len(test_loader))
    with open(dataset_path + "scaler.pkl", "rb") as fb:
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
        )

    print("load model from", saving_path)
    model.load_state_dict(
        torch.load(saving_path + "/model_seed{}.pth".format(args.seed))
    )

    results = evaluate(
        model,
        test_loader,
        nsample=args.nsample,
        scaler=std,
        mean_scaler=mean,
        foldername=saving_path,
    )
    if args.ood:
        np.save(save_result_path + "/result_ood_{}.npy".format(test_miss_rate), results)
    else:
        np.save(save_result_path + "/result_seed{}.npy".format(args.seed), results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSDI")
    parser.add_argument("--config", type=str, default="base.yaml")
    parser.add_argument("--device", default="cuda:0", help="Device for Attack")
    parser.add_argument("--dataset", default="ETTm1", type=str, help="dataset name")
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="parent dir of generated dataset",
        default="../datasets/ETTm1/raw_data/",
    )
    parser.add_argument(
        "--saving_path", help="saving model pth", type=str, default="../saved_models"
    )
    parser.add_argument(
        "--save_result_path",
        help="the save path of imputed data",
        type=str,
        default="../results/",
    )
    parser.add_argument("--seq_len", help="sequence length", type=int, default=24)
    parser.add_argument(
        "--missing_pattern", help="missing pattern", type=str, default="point"
    )
    parser.add_argument(
        "--missing_ratio", help="missing ratio in train", type=float, default=0.2
    )
    parser.add_argument(
        "--val_missing_ratio", help="missing ratio in valid", type=float, default=0.2
    )
    parser.add_argument(
        "--test_missing_ratio", help="missing ratio in test", type=float, default=0.2
    )
    parser.add_argument("--feature", help="feature nums", type=int, default=7)
    parser.add_argument(
        "--targetstrategy",
        type=str,
        default="mix",
        choices=["mix", "random", "historical"],
    )
    parser.add_argument("--num_workers", type=int, default=0, help="Device for Attack")
    parser.add_argument("--nsample", type=int, default=100)
    parser.add_argument("--unconditional", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scratch", action="store_true", help="test or scratch")
    parser.add_argument("--ood", action="store_true", help="i.i.d or OoD")

    args = parser.parse_args()
    print(args)

    start_time = time.time()
    main(args)
    print("Spend Time: ", time.time() - start_time)

    os._exit(0)
