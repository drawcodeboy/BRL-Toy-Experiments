import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal
from einops import rearrange
import numpy as np
import matplotlib.pyplot as plt
import os, shutil

class GaussianModel(nn.Module):
    def __init__(self,
                 dim=2,
                 init_mu=[1., 1.],
                 n_samples=50):
        super().__init__()
        base = torch.tensor(init_mu, dtype=torch.float32).view(1, -1)  # (1,2)
        self.mu = nn.Parameter(base.repeat(n_samples, 1) + 1.*torch.randn((n_samples, dim)))
        self.logvar = nn.Parameter(torch.tensor([0.1, 0.1], dtype=torch.float32).repeat(n_samples).reshape(n_samples, -1))

        self.l2_norm = torch.norm

    def forward(self):
        # return L2-normalized mu and sigma
        return F.normalize(self.mu, p=2, dim=-1),  torch.exp(0.5 * self.logvar)

class LossFunc(nn.Module):
    def __init__(self, n_mc=64, loss_type='cont', M=2.0, a=1.0, b=0.0):
        super().__init__()
        self.n_mc = n_mc          # Monte Carlo samples per distribution
        self.loss_type = loss_type
        self.M = M
        self.a = a
        self.b = b

    def _split_abc(self, mu, sigma):
        assert mu.dim() == 2 and sigma.dim() == 2, "mu/sigma must be (3N, D)"
        assert mu.shape == sigma.shape, "mu and sigma must have same shape"
        B, D = mu.shape
        assert B % 3 == 0, "batch size must be multiple of 3"
        N = B // 3
        muA, muB, muC = mu[:N], mu[N:2*N], mu[2*N:3*N]
        sA, sB, sC = sigma[:N], sigma[N:2*N], sigma[2*N:3*N]
        return (muA, sA), (muB, sB), (muC, sC)

    def _rsample_batch(self, mu, sigma):
        # mu, sigma: (N, D) -> samples: (K, N, D)
        dist = Normal(loc=mu, scale=sigma)
        return dist.rsample((self.n_mc,))

    @staticmethod
    def _pairwise_dist(x, y):
        # x: (K, N, D), y: (K, M, D) -> dist: (K, N, M)
        # torch.cdist supports leading batch dims
        return torch.cdist(x, y)  # L2 norm

    def forward(self, mu_batch, sigma_batch):
        (muA, sA), (muB, sB), (muC, sC) = self._split_abc(mu_batch, sigma_batch)

        XA = self._rsample_batch(muA, sA)  # (K, N, D)
        XB = self._rsample_batch(muB, sB)  # (K, N, D)
        XC = self._rsample_batch(muC, sC)  # (K, N, D)

        dAC = self._pairwise_dist(XA, XC)  # (K, N, N)  positive
        dBA = self._pairwise_dist(XB, XA)  # (K, N, N)  negative
        dBC = self._pairwise_dist(XB, XC)  # (K, N, N)  negative

        if self.loss_type == 'cont':
            pos = (dAC ** 2).mean()
            neg = (F.relu(self.M - dBA) ** 2).mean() + (F.relu(self.M - dBC) ** 2).mean()
            return pos + neg

        elif self.loss_type == 'soft_cont':
            p_pos = torch.sigmoid(-self.a * dAC + self.b)
            p_ba  = torch.sigmoid(-self.a * dBA + self.b)
            p_bc  = torch.sigmoid(-self.a * dBC + self.b)

            pos = (-torch.log(p_pos + 1e-12)).mean()
            neg = (-torch.log(1.0 - p_ba + 1e-12)).mean() + (-torch.log(1.0 - p_bc + 1e-12)).mean()
            return pos + neg

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

def plot_diag_gaussian(mu, sigma, n_std=0.1, color='red', alpha=0.2):
    mu ,var = mu.detach().numpy(), sigma.detach().numpy() ** 2
    theta = np.linspace(0, 2*np.pi, 200)
    x = mu[0] + n_std * np.sqrt(var[0]) * np.cos(theta)
    y = mu[1] + n_std * np.sqrt(var[1]) * np.sin(theta)

    plt.fill(x, y, color=color, alpha=alpha)
    plt.plot(x, y, c=color)
    plt.scatter(mu[0], mu[1], c=color)

def main():
    loss_type = 'soft_cont'

    shutil.rmtree(f"experiments/expr6/assets/{loss_type}", ignore_errors=True)
    os.makedirs(f"experiments/expr6/assets/{loss_type}")

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'

    n_samples = 30
    model_1 = GaussianModel(init_mu=[1., 0.], n_samples=n_samples) # Red
    model_2 = GaussianModel(init_mu=[-1, 0.], n_samples=n_samples) # Blue
    model_3 = GaussianModel(init_mu=[0., 1.], n_samples=n_samples) # Green (Ambiguous, become Red or Blue alternatively.)
    num_iter = 1000

    loss_fn = LossFunc(loss_type=loss_type)

    optimizer = torch.optim.SGD(list(model_1.parameters())
                                +list(model_2.parameters())
                                +list(model_3.parameters())
                                +list(loss_fn.parameters()),
                                lr=2.)

    for iter in range(1, num_iter+1):
        plt.figure(figsize=(6, 6))
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        optimizer.zero_grad()

        mu_1, sigma_1 = model_1()
        mu_2, sigma_2 = model_2()
        mu_3, sigma_3 = model_3()

        for i in range(0, n_samples):
            plot_diag_gaussian(mu_1[i], sigma_1[i], color='red')
            plot_diag_gaussian(mu_2[i], sigma_2[i], color='blue')
            plot_diag_gaussian(mu_3[i], sigma_3[i], color='green')

        if iter % 2 == 1:
            mu_batch = torch.cat([mu_1, mu_2, mu_3], dim=0)
            sigma_batch = torch.cat([sigma_1, sigma_2, sigma_3], dim=0)
        else:
            mu_batch = torch.cat([mu_2, mu_1, mu_3], dim=0)
            sigma_batch = torch.cat([sigma_2, sigma_1, sigma_3], dim=0)  

        loss = loss_fn(mu_batch, sigma_batch)

        loss.backward()
        optimizer.step()      

        plt.tight_layout()
        plt.savefig(f'experiments/expr6/assets/{loss_type}/frame_{iter:05d}.png', dpi=200)
        plt.close()

        print(f'\rIteration [{iter:03d}/{num_iter:03d}]', end="")

if __name__ == '__main__':
    main()