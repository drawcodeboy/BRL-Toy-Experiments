import numpy as np
import matplotlib.pyplot as plt
import os

def plot_diag_gaussian(mu, var, n_std=1, color='red', alpha=0.2, label='image', ambiguity=False):
    theta = np.linspace(0, 2*np.pi, 200)
    x = mu[0] + n_std * np.sqrt(var[0]) * np.cos(theta)
    y = mu[1] + n_std * np.sqrt(var[1]) * np.sin(theta)

    # plt.fill(x, y, color=color, alpha=alpha)
    if ambiguity == False:
        plt.fill(x, y, color=color, alpha=alpha)
        plt.plot(x, y, c=color)
    elif ambiguity == True:
        plt.plot(x, y, c=color, linestyle='--')
    plt.scatter(mu[0], mu[1], c=color, label=label)

def plot_box(center, size, color='red', alpha=0.2, lw=2, label='image'):
    cx, cy = center
    w, h = size
    x0, y0 = cx - w/2, cy - h/2

    rect = plt.Rectangle((x0, y0), w, h, fill=True, color=color, alpha=alpha, linewidth=0)
    plt.gca().add_patch(rect)

    rect_edge = plt.Rectangle((x0, y0), w, h, fill=False, edgecolor=color, linewidth=lw)
    plt.gca().add_patch(rect_edge)

    plt.scatter(cx, cy, c=color, label=label)

def main():
    plt.figure(figsize=(6, 6))
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

    embedding = 'Gaussian'

    if embedding == 'Gaussian':
        sigma = 1
        std = 0.5
        plt.title(rf'Gaussian embedding', fontsize=14)
        plot_diag_gaussian([-1, 0], [std, std], sigma, color='red', label='image')
        plot_diag_gaussian([-0.3, -0.5], [std, std], sigma, color='blue', label='tactile')
        plot_diag_gaussian([-0.65, -0.25], [0.1, 0.1], sigma, color='green', label='intersection candidate 1', ambiguity=True)
        plot_diag_gaussian([-0.65, -0.25], [0.2, 0.2], sigma, color='green', label='intersection candidate 2', ambiguity=True)
        plot_diag_gaussian([-0.65, -0.25], [std, std], sigma, color='green', label='intersection candidate 3', ambiguity=True)
        plt.legend(fontsize=14)

        plt.tight_layout()
        os.makedirs('experiments/expr5/assets', exist_ok=True)
        plt.savefig(f'experiments/expr5/assets/Gaussian.png', dpi=200)
    
    elif embedding == 'Box':
        plt.title('Box embedding', fontsize=14)
        plot_box(center=[-1, 0],   size=[1.0, 1.0], color='red', label='image')
        plot_box(center=[-0.3, -0.5], size=[1.0, 1.0], color='blue', label='tactile')
        plot_box(center=[-0.65, -0.25], size=[0.3, 0.5], color='green', label='intersection')
        plt.legend(fontsize=14)

        plt.tight_layout()
        os.makedirs('experiments/expr5/assets', exist_ok=True)
        plt.savefig('experiments/expr5/assets/Box.png', dpi=200)

if __name__ == '__main__':
    main()