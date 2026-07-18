import matplotlib.pyplot as plt
import seaborn as sns
import torch


V1 = 10   # Speed on first terrain
V2 = 1.2  # Speed on second terrain
D1 = 15
D2 = 20
H = 18


x_hist = []
t_hist = []


def reset_history():
    global x_hist, t_hist
    x_hist = []
    t_hist = []


def log_history(x, t):
    global x_hist, t_hist
    if isinstance(x, torch.Tensor):
        x = x.detach()

    if isinstance(t, torch.Tensor):
        t = t.detach()

    x_hist.append(float(x))
    t_hist.append(float(t))


def visualize_history():
    global x_hist, t_hist

    cmap = plt.get_cmap('Spectral_r')

    sns.set_style('darkgrid')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    rects = [
        plt.Rectangle((0, 0), D1, H, facecolor='#080041', alpha=.5),
        plt.Rectangle((D1, 0), D2, H, facecolor='#080041', alpha=.15),
    ]

    for i in range(len(x_hist)):
        axes[0].plot([0, D1, D1 + D2], [0, x_hist[i], H], c=cmap(1 - i / (len(t_hist) - 1)))

    for r in rects:
        axes[0].add_patch(r)

    axes[0].axis('off')
    axes[0].set_title('Path history')

    axes[1].plot(t_hist, c='r', marker='x')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Time [s]')
    axes[1].set_title('Traversal time history')

    plt.tight_layout()
    plt.show()
