import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOGVAR_MIN = -10.0
LOGVAR_MAX = 10.0


def _downsampled_size(size, layers=4, kernel=3, stride=2, padding=1):
    for _ in range(layers):
        size = math.floor((size + 2 * padding - kernel) / stride) + 1
    return size


class SetEdgeEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64):
        super().__init__()
        self.hidden = int(hidden)
        self.edge = nn.Sequential(
            nn.Linear(in_channels, self.hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.hidden, self.hidden),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.out = nn.Linear(self.hidden * 5, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N, N) -> edge features (B, N, N, C)
        e = x.permute(0, 2, 3, 1).contiguous()
        h = self.edge(e)  # (B, N, N, H)

        global_mean = h.mean(dim=(1, 2))
        global_max = h.amax(dim=(1, 2))

        row_h = h.mean(dim=2)  # (B, N, H)
        col_h = h.mean(dim=1)  # (B, N, H)
        row_mean = row_h.mean(dim=1)
        col_mean = col_h.mean(dim=1)
        row_std = row_h.std(dim=1, unbiased=False)

        pooled = torch.cat([global_mean, global_max, row_mean, col_mean, row_std], dim=1)
        return self.out(pooled)


class CVAE(nn.Module):
    def __init__(
        self,
        in_channels=2,
        img_size=64,
        latent_dim=32,
        base_channels=32,
        cond_dim=0,
        encoder_type="set_edge",
        set_hidden=64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.cond_dim = cond_dim
        self.encoder_type = str(encoder_type).strip().lower()
        self.set_hidden = int(set_hidden)

        feat = _downsampled_size(img_size, layers=4)
        self.dec_feat = base_channels * 8 * feat * feat

        if self.encoder_type == "conv":
            self.enc = nn.Sequential(
                nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(base_channels),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
                nn.BatchNorm2d(base_channels * 2),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
                nn.BatchNorm2d(base_channels * 4),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1),
                nn.BatchNorm2d(base_channels * 8),
                nn.LeakyReLU(0.1, inplace=True),
            )
            self.enc_out_dim = self.dec_feat
        elif self.encoder_type == "set_edge":
            self.enc = SetEdgeEncoder(in_channels=in_channels, hidden=self.set_hidden)
            self.enc_out_dim = 256
        else:
            raise ValueError("encoder_type must be 'conv' or 'set_edge'")

        self.fc = nn.Linear(self.enc_out_dim + cond_dim, 256)
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

        self.fc_dec = nn.Linear(latent_dim + cond_dim, 256)
        self.fc_unflatten = nn.Linear(256, self.dec_feat)

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(base_channels, in_channels, 4, stride=2, padding=1),
        )

    def _cond(self, batch, cond):
        if self.cond_dim == 0:
            return None
        if cond is None:
            return torch.zeros(batch, self.cond_dim, device=next(self.parameters()).device)
        return cond

    def reparam(self, mu, logvar):
        logvar = torch.clamp(logvar, min=LOGVAR_MIN, max=LOGVAR_MAX)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _encode_hidden(self, x):
        if self.encoder_type == "conv":
            h = self.enc(x)
            return h.view(x.size(0), -1)
        return self.enc(x)

    def encode_stats(self, x, cond=None):
        h = self._encode_hidden(x)
        cond_vec = self._cond(x.size(0), cond)
        if cond_vec is not None:
            h = torch.cat([h, cond_vec], dim=1)
        h = F.leaky_relu(self.fc(h), 0.1, inplace=True)
        mu = self.mu(h)
        logvar = torch.clamp(self.logvar(h), min=LOGVAR_MIN, max=LOGVAR_MAX)
        return mu, logvar

    def encode_mu(self, x, cond=None):
        mu, _ = self.encode_stats(x, cond=cond)
        return mu

    def forward(self, x, cond=None):
        mu, logvar = self.encode_stats(x, cond=cond)
        z = self.reparam(mu, logvar)

        cond_vec = self._cond(x.size(0), cond)
        d = z
        if cond_vec is not None:
            d = torch.cat([z, cond_vec], dim=1)
        d = F.leaky_relu(self.fc_dec(d), 0.1, inplace=True)
        d = F.leaky_relu(self.fc_unflatten(d), 0.1, inplace=True)
        feat = _downsampled_size(self.img_size, layers=4)
        d = d.view(x.size(0), self.base_channels * 8, feat, feat)
        x_hat_logits = self.dec(d)
        return x_hat_logits, mu, logvar


def kl_divergence(mu, logvar):
    logvar = torch.clamp(logvar, min=LOGVAR_MIN, max=LOGVAR_MAX)
    return -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()


def loss_fn(x, x_hat_logits, mu, logvar, beta=1.0, w_contact=1.0, w_dist=1.0):
    x0 = x[:, 0:1]
    l0 = F.binary_cross_entropy_with_logits(x_hat_logits[:, 0:1], x0, reduction="mean")

    x1 = x[:, 1:2]
    x1_hat = torch.sigmoid(x_hat_logits[:, 1:2])
    l1 = F.mse_loss(x1_hat, x1, reduction="mean")

    kl = kl_divergence(mu, logvar)
    return w_contact * l0 + w_dist * l1 + beta * kl, {"bce": l0.item(), "mse": l1.item(), "kl": kl.item()}
