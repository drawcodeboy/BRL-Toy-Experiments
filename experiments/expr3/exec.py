import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import os

class GaussianModel(nn.Module):
    def __init__(self,
                 dim=2,
                 init_mu=[1., 1.]):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(init_mu, dtype=torch.float32))
        self.logvar = nn.Parameter(torch.tensor([0.1, 0.1], dtype=torch.float32))

        self.l2_norm = torch.norm

    def forward(self):
        # return L2-normalized mu and sigma
        return F.normalize(self.mu, p=2, dim=-1),  torch.exp(0.5 * self.logvar)
    
class MultiModel(nn.Module):
    def __init__(self, model1, model2):
        super().__init__()

        self.model1 = model1
        self.model2 = model2

    def forward(self):
        return self.model1, self.model2
    
class LossFunc(nn.Module):
    def __init__(self, n_samples=2000, case='positive', loss_type='cont'):
        super().__init__()

        self.a = nn.Parameter(torch.tensor([1., ], dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor([0., ], dtype=torch.float32))
        self.n_samples = n_samples
        self.case = case
        self.loss_type = loss_type

    def get_distributions(self, mu_1, sigma_1, mu_2, sigma_2):
            dist1 = Normal(loc=mu_1, scale=sigma_1)
            dist2 = Normal(loc=mu_2, scale=sigma_2)

            return dist1, dist2

    def get_distance(self, dist1, dist2, n_samples):
        batch1 = dist1.rsample((n_samples,))
        batch2 = dist2.rsample((n_samples,))

        dist = batch1 - batch2 # (n_samples, dim)
        dist_l2_norm = torch.norm(dist, p=2, dim=1) # (n_samples)
        return dist_l2_norm
    
    def contrastive_loss(self, dist1, dist2, n_samples=2000, case="positive", M=1.0):
        dist_l2_norm = self.get_distance(dist1, dist2, n_samples)

        if case == 'positive':
            return torch.mean(torch.pow(dist_l2_norm, 2))
        elif case == 'negative':
            max_dist = torch.where((M - dist_l2_norm) > 0., (M - dist_l2_norm), 0)
            return torch.mean(torch.pow(max_dist, 2))
        
    def soft_contrastive_loss(self, dist1, dist2, n_samples=2000, case="positive"):
        dist_l2_norm = self.get_distance(dist1, dist2, n_samples)

        prob = F.sigmoid(-self.a * dist_l2_norm + self.b)

        if case == 'positive':
            return torch.mean(-torch.log(prob))
        elif case == 'negative':
            return torch.mean(-torch.log(1-prob))

    def forward(self, mu_1, sigma_1, mu_2, sigma_2):
        dist1, dist2 = self.get_distributions(mu_1, sigma_1, mu_2, sigma_2)

        if self.loss_type == 'cont':
            return self.contrastive_loss(dist1, dist2, self.n_samples, self.case)
        elif self.loss_type == 'soft_cont':
            return self.soft_contrastive_loss(dist1, dist2, self.n_samples, self.case)

def plot_diag_gaussian(mu, var, n_std=1, color='red', alpha=0.2):
    theta = np.linspace(0, 2*np.pi, 200)
    x = mu[0] + n_std * np.sqrt(var[0]) * np.cos(theta)
    y = mu[1] + n_std * np.sqrt(var[1]) * np.sin(theta)

    plt.fill(x, y, color=color, alpha=alpha)
    plt.plot(x, y, c=color)
    plt.scatter(mu[0], mu[1], c=color)

def plot_simulation(mu_1, sigma_1, mu_2, sigma_2):
    mu_1_np = mu_1.detach().numpy()
    sigma_1_np = sigma_1.detach().numpy()
    mu_2_np = mu_2.detach().numpy()
    sigma_2_np = sigma_2.detach().numpy()

    plot_diag_gaussian(mu_1_np, sigma_1_np ** 2, color='red')
    plot_diag_gaussian(mu_2_np, sigma_2_np ** 2, color='green')

def main(cfg):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
    
    model = MultiModel(
        model1=GaussianModel(init_mu=[1., 1.]),
        model2=GaussianModel(init_mu=[-1., 0.])
    ).to(device)


    num_iter = cfg['num_iter']

    loss_fn = LossFunc(n_samples=cfg['n_samples'],
                       case=cfg['case'],
                       loss_type=cfg['loss_type'])
    
    optimizer = torch.optim.SGD(list(model.parameters())+list(loss_fn.parameters()), lr=cfg['lr'])

    os.makedirs(f"experiments/expr3/assets/learn_dynamics/{cfg['expr_name']}")
    

    for iter in range(1, num_iter+1):
        optimizer.zero_grad()

        mu_1, sigma_1 = model.model1()
        mu_2, sigma_2 = model.model2()

        plt.figure(figsize=(6, 6))
        plt.xlim(-3.0, 3.0)
        plt.ylim(-3.0, 3.0)

        plot_simulation(mu_1, sigma_1, mu_2, sigma_2)

        loss = loss_fn(mu_1, sigma_1, mu_2, sigma_2)

        loss.backward()

        print(model.model1.mu.grad.norm().item())

        optimizer.step()


        print(f"\rIteration [{iter:05d}/{num_iter:05d}]", end="")
        plt.tight_layout()
        plt.savefig(f"experiments/expr3/assets/learn_dynamics/{cfg['expr_name']}/frame_{iter:03d}.png", dpi=200)
        plt.close() # Memory 때문에

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='cont_pos')
    
    args = parser.parse_args()

    with open(f'experiments/expr3/config/{args.config}.yaml') as f:
        cfg = yaml.full_load(f)

    main(cfg)