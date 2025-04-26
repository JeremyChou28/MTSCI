import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from info_nce import InfoNCE, info_nce
from diff_block import denoising_network


class MTSCI_base(nn.Module):

    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim  # target_dim = number of features

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim + 1
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1
        self.diffmodel = denoising_network(config_diff, input_dim)

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

        self.calc_intra_cons_loss = InfoNCE()

    def time_embedding(self, pos: torch.Tensor, d_model: int = 128) -> torch.Tensor:
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(
        self, observed_tp: torch.Tensor, cond_mask: torch.Tensor
    ) -> torch.Tensor:
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(
            observed_tp, self.emb_time_dim
        )  # (B, L, emb_time)
        time_embed = time_embed.unsqueeze(2).expand(
            -1, -1, K, -1
        )  # (B, L, K, emb_time)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K, emb_feature)
        feature_embed = (
            feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        )  # (B, L, K, emb_feature)

        side_info = torch.cat(
            [time_embed, feature_embed], dim=-1
        )  # (B, L, K, emb_total)
        side_info = side_info.permute(0, 3, 2, 1)  # (B, emb_total, K, L)

        side_info = torch.cat(
            [side_info, cond_mask.unsqueeze(1)], dim=1
        )  # (B, emb_total + 1, K, L)

        return side_info

    def calc_loss_valid(
        self,
        X_Tilde,
        cond_mask,
        X_Tilde_mask,
        indicating_mask,
        X_pred,
        side_info,
        reverse_side_info,
        observed_cond_info,
        noisy_cond_info,
        is_train,
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss, _ = self.calc_loss(
                X_Tilde,
                cond_mask,
                X_Tilde_mask,
                indicating_mask,
                X_pred,
                side_info,
                reverse_side_info,
                observed_cond_info,
                noisy_cond_info,
                is_train=is_train,
                set_t=t,
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self,
        X_Tilde,
        cond_mask,
        X_Tilde_mask,
        indicating_mask,
        X_pred,
        side_info,
        reverse_side_info,
        observed_cond_info,
        noisy_cond_info,
        is_train=1,
        set_t=-1,
    ):
        B, K, L = X_Tilde.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)

        # add noise to observed data
        current_alpha = self.alpha_torch[t]  # (B, 1, 1)
        noise = torch.randn_like(X_Tilde)
        noisy_data = (current_alpha**0.5) * X_Tilde + (
            1.0 - current_alpha
        ) ** 0.5 * noise  # (B, K, L)
        reverse_cond_mask = (
            X_Tilde_mask * indicating_mask
        )  # observed but served as prediction target

        # get the input to the diffusion model
        (
            total_input,
            reverse_total_input,
            all_observed_input,
            all_noisy_input,
        ) = self.set_input_to_diffmodel(
            X_original=X_Tilde,
            noisy_data=noisy_data,
            cond_mask=cond_mask,
            reverse_cond_mask=reverse_cond_mask,
        )

        # get the output of diffusion model
        (
            forward_pred_noise,
            forward_hidden,
            reverse_hidden,
            negative_hidden,  # anchor: all observed data, negative_input: all filled with noise
        ) = self.diffmodel(
            total_input,
            side_info,
            reverse_total_input,
            reverse_side_info,
            all_noisy_input,
            noisy_cond_info,
            X_pred,
            t,
        )  # if in validation mode, pred_total_input and pred_side_info are None

        # loss of noise
        target_mask = X_Tilde_mask - cond_mask
        residual = (noise - forward_pred_noise) * target_mask
        num_eval = target_mask.sum()
        loss_noise = (residual**2).sum() / (num_eval if num_eval > 0 else 1)

        # predict_hidden => first_hidden(missing) + reverse_hidden(not missing)
        predict_hidden = forward_hidden.reshape(B, -1, K, L) * (
            1 - cond_mask.unsqueeze(1)
        ) + reverse_hidden.reshape(B, -1, K, L) * (1 - reverse_cond_mask.unsqueeze(1))
        loss_cons = self.calc_intra_cons_loss(
            negative_hidden.reshape(B, -1), predict_hidden.reshape(B, -1)
        )
        loss_cons.detach()

        torch.cuda.empty_cache()

        return loss_noise, loss_cons

    def set_input_to_diffmodel(
        self,
        X_original,
        noisy_data,
        cond_mask,
        reverse_cond_mask,
    ):

        def get_noisy_total_input(X_original, X_noisy, mask) -> torch.Tensor:
            return (mask * X_original).unsqueeze(1) + ((1 - mask) * X_noisy).unsqueeze(
                1
            )

        total_input = get_noisy_total_input(X_original, noisy_data, cond_mask)
        reverse_total_input = get_noisy_total_input(
            X_original, noisy_data, reverse_cond_mask
        )
        return (
            total_input,
            reverse_total_input,
            X_original.unsqueeze(1),
            noisy_data.unsqueeze(1),
        )

    def impute(self, X_Tilde, cond_mask, side_info, n_samples):
        B, K, L = X_Tilde.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            current_sample = torch.randn_like(X_Tilde)

            for t in range(self.num_steps - 1, -1, -1):
                cond_obs = (cond_mask * X_Tilde).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                diff_input = cond_obs + noisy_target

                predicted = self.diffmodel.impute(
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
                X_pred,
                X_pred_mask,
            ) = self.process_data(batch, is_train)
        else:
            (X_Tilde, X_Tilde_mask, observed_tp, X_mask, indicating_mask) = (
                self.process_data(batch, is_train)
            )
            X_pred, X_pred_mask = None, None

        cond_mask = X_mask
        reverse_cond_mask = X_Tilde_mask * indicating_mask  
        negative_mask = torch.zeros_like(X_Tilde_mask)

        # all values are: natural missing + natural oberved (artificially masked + conditional info)
        side_info = self.get_side_info(
            observed_tp, cond_mask
        )  # cond_mask => natural missing + artificial missing
        reverse_side_info = self.get_side_info(
            observed_tp, reverse_cond_mask
        )  # reverse_cond_mask => natural missing + conditional info
        anchor_cond_info = self.get_side_info(
            observed_tp, X_Tilde_mask
        )  # observed_mask => natural missing
        negative_cond_info = self.get_side_info(
            observed_tp, negative_mask
        )  # negative_mask => all assumed missing

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(
            X_Tilde=X_Tilde,
            cond_mask=cond_mask,
            X_Tilde_mask=X_Tilde_mask,
            indicating_mask=indicating_mask,
            X_pred=X_pred,
            side_info=side_info,
            reverse_side_info=reverse_side_info,
            observed_cond_info=anchor_cond_info,
            noisy_cond_info=negative_cond_info,
            is_train=is_train,
        )

    def evaluate(self, batch, n_samples):
        (X_Tilde, X_Tilde_mask, observed_tp, X_mask, indicating_mask) = (
            self.process_data(batch, istrain=0)
        )

        with torch.no_grad():
            cond_mask = X_mask
            eval_mask = X_Tilde_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(X_Tilde, cond_mask, side_info, n_samples)

        return samples, X_Tilde, eval_mask, X_Tilde_mask, observed_tp


class MTSCI(MTSCI_base):

    def __init__(self, config, device, target_dim=36, seq_len=24):
        super(MTSCI, self).__init__(target_dim, config, device)
        self.seq_len = seq_len

    def process_data(self, batch, istrain=1):
        if istrain == 1:
            (
                X_tensor,
                mask_tensor,
                indicating_mask_tensor,
                X_Tilde_tensor,
                X_Tilde_mask_tensor,
                pred_tensor,
                pred_mask_tensor,
            ) = batch
        else:
            X_tensor, mask_tensor, X_Tilde_tensor, X_Tilde_mask_tensor,indicating_mask_tensor = batch

        X_Tilde = X_Tilde_tensor.to(
            self.device
        ).float()  # B,L,F, with original missing values
        X_mask = mask_tensor.to(
            self.device
        ).float()  # original missing + artifically missing (0: missing, 1: observed)
        indicating_mask = indicating_mask_tensor.to(
            self.device
        ).float()  # indicating mask
        X_Tilde_mask = X_Tilde_mask_tensor.to(
            self.device
        ).float()

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

        if istrain == 0:
            return (X_Tilde, X_Tilde_mask, observed_tp, X_mask, indicating_mask)

        else:
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
