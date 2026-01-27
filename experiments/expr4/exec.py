import numpy as np
import matplotlib.pyplot as plt
import os

def plot_diag_gaussian(mu, var, n_std=1, color='red', alpha=0.2):
    theta = np.linspace(0, 2*np.pi, 200)
    x = mu[0] + n_std * np.sqrt(var[0]) * np.cos(theta)
    y = mu[1] + n_std * np.sqrt(var[1]) * np.sin(theta)

    # plt.fill(x, y, color=color, alpha=alpha)
    plt.fill(x, y, color=color, alpha=alpha)
    plt.plot(x, y, c=color)
    plt.scatter(mu[0], mu[1], c=color)

def l2_norm(x, axis=None, eps=1e-12):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)

def main():
    plt.figure(figsize=(6, 6))
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    sigma = 1
    filename = f"sigma_{sigma}"
    std = 0.1
    plt.title(rf'{sigma}$\sigma$', fontsize=14)

    plot_diag_gaussian(l2_norm([-1, 1]), [std, std], sigma, color='red')
    plot_diag_gaussian(l2_norm([-1, -1]), [std, std], sigma, color='orange')
    plot_diag_gaussian(l2_norm([1, -0.2]), [std, std], sigma, color='green')
    plot_diag_gaussian(l2_norm([1, 1]), [std, std], sigma, color='blue')

    plt.tight_layout()
    os.makedirs('experiments/expr4/assets', exist_ok=True)
    plt.savefig(f'experiments/expr4/assets/{filename}.png', dpi=200)

if __name__ == '__main__':
    main()