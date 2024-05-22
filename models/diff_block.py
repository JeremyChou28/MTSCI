import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class ResidualBlock(nn.Module):

    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, channels, 1)
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
        y = x + diffusion_emb  # (B,channel,K,L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,channel,K*L)
        y = y + cond_info.reshape(B, channel, K * L)

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)  # (B,channel,K,L)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)  # (B,channel,K,L)
        return (x + residual) / math.sqrt(2.0), skip


class Encoder(nn.Module):
    def __init__(self, layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, cond_info, diffusion_emb):
        skips = []
        for layer in self.layers:
            x, skip = layer(x, cond_info, diffusion_emb)
            skips.append(skip)
        skip_concat = torch.sum(torch.stack(skips), dim=0) / math.sqrt(len(self.layers))
        B, _, K, L = skip_concat.shape
        return skip_concat.reshape(B, -1, K * L)


class Decoder(nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.output_projection1 = Conv1d_with_init(channels, channels, 1)
        self.output_projection2 = Conv1d_with_init(channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

    def forward(self, x_hidden, B, K, L):  # (B, channel, K*L) => (B, K, L)
        x = self.output_projection1(x_hidden)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x


class denoising_network(nn.Module):

    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.seqlen = config["seqlen"]

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)

        self.backward_projection = Conv1d_with_init(self.seqlen, self.seqlen, 1)
        self.cond_projection = Conv1d_with_init(self.channels + 1, self.channels, 1)

        self.encoder = Encoder(
            nn.ModuleList(
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
        )
        self.decoder = Decoder(self.channels)

    def forward(
        self,
        x,
        cond_info,
        reverse_x,
        reverse_cond_info,
        negative_input,
        negative_cond_info,
        X_pred,
        diffusion_step,
    ):

        B, inputdim, K, L = x.shape

        x_hidden = self.__embedding(x, B, K, L, inputdim)
        reverse_x_hidden = self.__embedding(reverse_x, B, K, L, inputdim)
        negative_hidden = self.__embedding(negative_input, B, K, L, inputdim)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        forward_noise_hidden = self.encoder(x_hidden, cond_info, diffusion_emb)
        reverse_noise_hidden = self.encoder(
            reverse_x_hidden, reverse_cond_info, diffusion_emb
        )
        negative_hidden = self.encoder(
            negative_hidden, negative_cond_info, diffusion_emb
        )  # B,D,K*L

        if X_pred is not None:
            random_mask = torch.rand(B, 1, K, L).to(X_pred.device)
            pred_cond = (
                self.backward_projection(X_pred.permute(0, 2, 1))
                .permute(0, 2, 1)
                .unsqueeze(1)
            )  # B,1,K,L
            new_cond = pred_cond * random_mask + x * (1 - random_mask)
            new_cond = new_cond.reshape(B, 1, K * L)
            forward_noise_hidden = torch.cat([forward_noise_hidden, new_cond], dim=1)
            forward_noise_hidden = self.cond_projection(forward_noise_hidden)
        else:
            new_cond = x
            new_cond = new_cond.reshape(B, 1, K * L)
            forward_noise_hidden = torch.cat([forward_noise_hidden, new_cond], dim=1)
            forward_noise_hidden = self.cond_projection(forward_noise_hidden)

        forward_noise = self.decoder(forward_noise_hidden, B, K, L)

        return (
            forward_noise,
            forward_noise_hidden,
            reverse_noise_hidden,
            negative_hidden,
        )

    def impute(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape

        x_enc = self.__embedding(x, B, K, L, inputdim)
        diffusion_emb = self.diffusion_embedding(diffusion_step)

        x_hidden = self.encoder(x_enc, cond_info, diffusion_emb)

        new_cond = x
        new_cond = new_cond.reshape(B, 1, K * L)
        x_hidden = torch.cat([x_hidden, new_cond], dim=1)
        x_hidden = self.cond_projection(x_hidden)

        x_noise = self.decoder(x_hidden, B, K, L)
        return x_noise

    # Private helper functions
    def __embedding(self, x, B, K, L, input_dim):
        if x is None:  # for pred_x in validation phase
            return None
        x = x.reshape(B, input_dim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)
        return x
