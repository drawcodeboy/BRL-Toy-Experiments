import torch
from torch.distributions import Normal
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

def get_dists(mu_1=torch.tensor([-0.1, 0]), sigma_1=torch.tensor([1, 1]), 
              mu_2=torch.tensor([0.1, 0]), sigma_2=torch.tensor([1, 1])):
    dist1 = Normal(loc=mu_1, scale=sigma_1)
    dist2 = Normal(loc=mu_2, scale=sigma_2)

    return dist1, dist2

class loss_fn:
    @staticmethod
    def get_dist(dist1, dist2, n_samples):
        # rsample -> gradient 0 방지
        batch1 = dist1.rsample((n_samples,))
        batch2 = dist2.rsample((n_samples,))

        dist = batch1 - batch2 # (n_samples, dim)
        dist_l2_norm = torch.norm(dist, p=2, dim=1) # (n_samples)
        return dist_l2_norm

    @staticmethod
    def contrastive_loss(dist1, dist2, n_samples=2000, case="positive", M=None):
        dist_l2_norm = loss_fn.get_dist(dist1, dist2, n_samples)

        if case == 'positive':
            return torch.mean(torch.pow(dist_l2_norm, 2))
        elif case == 'negative':
            max_dist = torch.where((M - dist_l2_norm) > 0., (M - dist_l2_norm), 0)
            return torch.mean(torch.pow(max_dist, 2))
        
    @staticmethod
    def soft_contrastive_loss(dist1, dist2, n_samples=2000, case="positive", a=1., b=0.):
        dist_l2_norm = loss_fn.get_dist(dist1, dist2, n_samples)

        prob = F.sigmoid(-a * dist_l2_norm + b)

        if case == 'positive':
            return torch.mean(-torch.log(prob))
        elif case == 'negative':
            return torch.mean(-torch.log(1-prob))

def plot_diag_gaussian(mu, var, n_std=1, color='red', alpha=0.2):
    theta = np.linspace(0, 2*np.pi, 200)
    x = mu[0] + n_std * np.sqrt(var[0]) * np.cos(theta)
    y = mu[1] + n_std * np.sqrt(var[1]) * np.sin(theta)

    plt.fill(x, y, color=color, alpha=alpha)
    plt.plot(x, y, c=color)
    plt.scatter(mu[0], mu[1], c=color)

def get_trajectory(start_point, end_point, steps=100):
    t = np.linspace(0, 1, steps)
    points = (1 - t)[:, None] * start_point + t[:, None] * end_point
    return points

def simulate(sim_name, mu_1_li, mu_2_li, sigma_1_li, sigma_2_li, loss_case):
    os.makedirs(f"experiments/expr1/{sim_name}")

    frame_li = []
    # Contrastive loss
    mu_1_grad_norm_li_1, mu_2_grad_norm_li_1, sigma_1_grad_norm_li_1, sigma_2_grad_norm_li_1 = [], [], [], []
    # Soft contrastive loss
    mu_1_grad_norm_li_2, mu_2_grad_norm_li_2, sigma_1_grad_norm_li_2, sigma_2_grad_norm_li_2 = [], [], [], []
    for frame, (mu_1, mu_2, sigma_1, sigma_2) in enumerate(zip(mu_1_li, mu_2_li, sigma_1_li, sigma_2_li), start=1):
        plt.figure(figsize=(18, 6))

        # figure 1: 2D Embedding space
        plt.subplot(1, 3, 1)
        plt.title('2D Embedding space')
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])

        # mu_1, mu_2 = np.array([-1.5, 0.]), np.array([1.5, 0])
        # sigma_1, sigma_2 = np.array([1, 1]), np.array([1, 1])

        plot_diag_gaussian(mu_1, sigma_1, color='red')
        plot_diag_gaussian(mu_2, sigma_2, color='green')

        mu_1 = torch.tensor(mu_1, requires_grad=True)
        sigma_1 = torch.tensor(sigma_1, requires_grad=True)
        mu_2 = torch.tensor(mu_2, requires_grad=True)
        sigma_2 = torch.tensor(sigma_2, requires_grad=True)

        dist1, dist2 = get_dists(mu_1, sigma_1, mu_2, sigma_2)
        c_loss = loss_fn.contrastive_loss(dist1, dist2, case=loss_case, M=10.0)
        c_loss.backward()

        sc_loss = loss_fn.soft_contrastive_loss(dist1, dist2, case=loss_case)

        dummies = [Line2D([], [], linestyle='none'),
                Line2D([], [], linestyle='none')]

        plt.legend(
            handles=dummies,
            labels=[rf"Contrastive loss = {c_loss:.4f}", 
                    rf"Soft contrastive loss = {sc_loss:.4f}"],
            handlelength=0,
            handletextpad=0
        )

        # figure 2: Contrastive loss gradient
        plt.subplot(1, 3, 2)
        plt.title("Contrastive loss gradient magnitude")
        plt.xlabel("Steps")
        plt.ylabel("Gradient")
        frame_li.append(frame)
        mu_1_grad_norm_li_1.append(torch.norm(mu_1.grad, p=2, dim=0).item())
        sigma_1_grad_norm_li_1.append(torch.norm(sigma_1.grad, p=2, dim=0).item())
        mu_2_grad_norm_li_1.append(torch.norm(mu_2.grad, p=2, dim=0).item())
        sigma_2_grad_norm_li_1.append(torch.norm(sigma_2.grad, p=2, dim=0).item())

        plt.plot(frame_li, mu_1_grad_norm_li_1, color='red')
        plt.scatter(frame_li, mu_1_grad_norm_li_1, color='red', label='mu_1')
        plt.plot(frame_li, sigma_1_grad_norm_li_1, color='red')
        plt.scatter(frame_li, sigma_1_grad_norm_li_1, color='red', marker='D', label='sigma_1')

        plt.plot(frame_li, mu_2_grad_norm_li_1, color='green')
        plt.scatter(frame_li, mu_2_grad_norm_li_1, color='green', label='mu_2')
        plt.plot(frame_li, sigma_2_grad_norm_li_1, color='green')
        plt.scatter(frame_li, sigma_2_grad_norm_li_1, color='green', marker='D', label='sigma_2')

        plt.legend()

        # figure 2: Soft contrastive loss gradient
        plt.subplot(1, 3, 3)
        mu_1.grad.zero_(), sigma_1.grad.zero_(), mu_2.grad.zero_(), sigma_2.grad.zero_()
        dist1, dist2 = get_dists(mu_1, sigma_1, mu_2, sigma_2)
        sc_loss = loss_fn.soft_contrastive_loss(dist1, dist2, case=loss_case)
        sc_loss.backward()
        plt.title("Soft contrastive loss gradient magnitude")
        plt.xlabel("Steps")
        plt.ylabel("Gradient")
        mu_1_grad_norm_li_2.append(torch.norm(mu_1.grad, p=2, dim=0).item())
        sigma_1_grad_norm_li_2.append(torch.norm(sigma_1.grad, p=2, dim=0).item())
        mu_2_grad_norm_li_2.append(torch.norm(mu_2.grad, p=2, dim=0).item())
        sigma_2_grad_norm_li_2.append(torch.norm(sigma_2.grad, p=2, dim=0).item())

        plt.plot(frame_li, mu_1_grad_norm_li_2, color='red')
        plt.scatter(frame_li, mu_1_grad_norm_li_2, color='red', label='mu_1')
        plt.plot(frame_li, sigma_1_grad_norm_li_2, color='red')
        plt.scatter(frame_li, sigma_1_grad_norm_li_2, color='red', marker='D', label='sigma_1')

        plt.plot(frame_li, mu_2_grad_norm_li_2, color='green')
        plt.scatter(frame_li, mu_2_grad_norm_li_2, color='green', label='mu_2')
        plt.plot(frame_li, sigma_2_grad_norm_li_2, color='green')
        plt.scatter(frame_li, sigma_2_grad_norm_li_2, color='green', marker='D', label='sigma_2')

        plt.legend()

        # plt.tight_layout() # bounding box들이 스케일 때문에 튀는 현상이 있음
        plt.subplots_adjust(left=0.05, right=0.98, bottom=0.10, top=0.90, wspace=0.30)
        plt.savefig(f"experiments/expr1/{sim_name}/frame_{frame:03d}.png", dpi=200)
        plt.close() # Memory 때문에 필요함

def main():
    '''
    mu_1_points = get_trajectory(np.array([-1.5, 0.0]), np.array([-3.0, 0.0]))
    mu_2_points = get_trajectory(np.array([1.5, 0.0]), np.array([3.0, 0.0]))
    sigma_1_points = get_trajectory(np.array([1., 1.]), np.array([1., 1.]))
    sigma_2_points = get_trajectory(np.array([1., 1.]), np.array([1., 1.]))
    simulate('mean_open_case', mu_1_points, mu_2_points, sigma_1_points, sigma_2_points, 'positive')

    mu_1_points = get_trajectory(np.array([-1.5, 0.0]), np.array([-1.5, 0.0]))
    mu_2_points = get_trajectory(np.array([1.5, 0.0]), np.array([1.5, 0.0]))
    sigma_1_points = get_trajectory(np.array([1., 1.]), np.array([3., 3.]))
    sigma_2_points = get_trajectory(np.array([1., 1.]), np.array([3., 3.]))
    simulate('var_open_case', mu_1_points, mu_2_points, sigma_1_points, sigma_2_points, 'positive')

    mu_1_points = get_trajectory(np.array([-1.5, 0.0]), np.array([-3.0, 0.0]))
    mu_2_points = get_trajectory(np.array([1.5, 0.0]), np.array([3.0, 0.0]))
    sigma_1_points = get_trajectory(np.array([1., 1.]), np.array([3., 3.]))
    sigma_2_points = get_trajectory(np.array([1., 1.]), np.array([3., 3.]))
    simulate('both_open_case', mu_1_points, mu_2_points, sigma_1_points, sigma_2_points, 'positive')
    '''
    
    mu_1_points = get_trajectory(np.array([-1.5, 0.0]), np.array([-3.0, 0.0]))
    mu_2_points = get_trajectory(np.array([1.5, 0.0]), np.array([3.0, 0.0]))
    sigma_1_points = get_trajectory(np.array([1., 1.]), np.array([1., 1.]))
    sigma_2_points = get_trajectory(np.array([1., 1.]), np.array([1., 1.]))
    simulate('mean_open_case_neg', mu_1_points, mu_2_points, sigma_1_points, sigma_2_points, 'negative')

if __name__ == '__main__':
    main()