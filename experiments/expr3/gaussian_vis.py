from exec import GaussianModel
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(6, 6))

def plot_diag_gaussian(mu, var, n_std=1, color='red', alpha=0.2):
    theta = np.linspace(0, 2*np.pi, 200)
    x = mu[0] + n_std * np.sqrt(var[0]) * np.cos(theta)
    y = mu[1] + n_std * np.sqrt(var[1]) * np.sin(theta)

    # plt.fill(x, y, color=color, alpha=alpha)
    plt.plot(x, y, c=color)
    plt.scatter(mu[0], mu[1], c=color)

for _ in range(0, 100):
    model = GaussianModel()
    mu, logvar = model()
    mu, logvar = mu.detach().numpy(), logvar.detach().numpy()
    plot_diag_gaussian(mu, logvar, color=None)

plt.tight_layout()
plt.savefig('experiments/expr3/assets/vis_gaussian.png', dpi=200)