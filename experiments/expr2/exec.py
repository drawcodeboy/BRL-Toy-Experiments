import torch
from torch import nn
from torch.distributions import Normal
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch.nn.functional as F

class loss_fn:
    @staticmethod
    def get_a(sigma_1, sigma_2):
        return (1 / (torch.pow(sigma_1, 2))) + (1 / (2 * torch.pow(sigma_2, 2)))
    
    @staticmethod
    def get_b(mu_1, sigma_1, mu_2, sigma_2):
        return ((2 * mu_1) / torch.pow(sigma_1, 2)) + (mu_2 / torch.pow(sigma_2, 2))

    @staticmethod
    def get_c(mu_1, sigma_1, mu_2, sigma_2):
        return (torch.pow(mu_1, 2) / torch.pow(sigma_1, 2)) + (torch.pow(mu_2, 2) / (2 * torch.pow(sigma_2, 2)))

    @staticmethod
    def inclusion(mu_1, sigma_1, mu_2, sigma_2):
        A = loss_fn.get_a(sigma_1, sigma_2)
        B = loss_fn.get_b(mu_1, sigma_1, mu_2, sigma_2)
        C = loss_fn.get_c(mu_1, sigma_1, mu_2, sigma_2)

        first_term = -2 * torch.log(torch.pow(sigma_1, 2))
        second_term = -torch.log(torch.pow(sigma_2, 2))
        third_term = -0.5 * torch.log(A)
        fourth_term = torch.pow(B, 2) / (4 * A)
        fifth_term = -C

        return first_term + second_term + third_term + fourth_term + fifth_term

    @staticmethod
    def hypothesis_test(mu_1, sigma_1, mu_2, sigma_2):
        first_inclusion = loss_fn.inclusion(mu_1, sigma_1, mu_2, sigma_2)
        second_inclusion = loss_fn.inclusion(mu_2, sigma_2, mu_1, sigma_1)
        return first_inclusion - second_inclusion
    
    @staticmethod
    def inclusion_loss(mu_1, sigma_1, mu_2, sigma_2, c=1000):
        return -torch.log(F.sigmoid(loss_fn.hypothesis_test(mu_1, sigma_1, mu_2, sigma_2)))
    
    @staticmethod
    def kl_divergence(mu_1, sigma_1, mu_2, sigma_2):
        return torch.log(sigma_2 / sigma_1) + (sigma_1**2 + (mu_1 - mu_2)**2) / (2 * sigma_2**2) - 0.5

def visualize(mu, sigma, label, color, lw = 1.5, linestyle='-', square=False):
    dist1 = Normal(loc=torch.tensor(mu), scale=torch.tensor(sigma))

    x = torch.linspace(mu -4*sigma, mu + 4*sigma, 1000)
    y = torch.exp(dist1.log_prob(x))
    y = y if square == False else y ** 2

    plt.plot(x.numpy(), y.numpy(), label=label, color=color, linestyle=linestyle, lw=lw, zorder=2)
    plt.xlabel("x")
    plt.ylabel("p(x)")

def main():
    plt.figure(figsize=(5, 4))
    plt.xlim(-2.2, 2.2)
    plt.ylim(0, 3.8)
    plt.grid(zorder=1)

    root = "experiments/expr2/assets"
    save_name = "paper_fig3_c"

    mu_1, sigma_1 = 1.1, 0.2
    mu_2, sigma_2 = -0.9, 0.2
    
    visualize(mu_1, sigma_1, label=rf"pdf of $\it{{Z}}_1=N$({mu_1}, {sigma_1})", color="red", linestyle='-')
    visualize(mu_1, sigma_1, label=rf"squared pdf of $\it{{Z}}_1$", color="red", linestyle='--', square=True)
    visualize(mu_2, sigma_2, label=rf"pdf of $\it{{Z}}_2=N$({mu_2}, {sigma_2})", color="blue", linestyle='-')

    mu_1, sigma_1, mu_2, sigma_2 = torch.tensor(mu_1), torch.tensor(sigma_1), torch.tensor(mu_2), torch.tensor(sigma_2)
    inclusion = loss_fn.inclusion(mu_1, sigma_1, mu_2, sigma_2)
    hypothesis_test = loss_fn.hypothesis_test(mu_1, sigma_1, mu_2, sigma_2)
    kld = loss_fn.kl_divergence(mu_1, sigma_1, mu_2, sigma_2)

    handles, labels = plt.gca().get_legend_handles_labels()

    dummies = [
        Line2D([], [], linestyle='none'),
        Line2D([], [], linestyle='none'),
        Line2D([], [], linestyle='none')
    ]

    handles += dummies
    labels += [
        rf"$H(Z_1\subset Z_2)={hypothesis_test:.3f}$",
        rf"inc($Z_1, Z_2)={inclusion:.3f}$",
        rf"$KL(Z_1, Z_2)$={kld:.3f}"
    ]

    plt.legend(
        handles=handles,
        labels=labels,
        # handlelength=0,
        handletextpad=1
    )

    plt.tight_layout()
    plt.savefig(f"{root}/{save_name}.png", dpi=200)

if __name__ == '__main__':
    main()