import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch


V1 = .7   # Speed on first terrain
V2 = .3   # Speed on second terrain
V3 = .45  # Speed on third terrain
H = 20
D = 10


time_history = []
param_history = []


def reset_history():
    global time_history, param_history
    time_history = []
    param_history = []


def log_history(time, params):
    global time_history, param_history
    if isinstance(time, torch.Tensor):
        time = time.detach()

    if isinstance(params[0], torch.Tensor):
        params = [x.detach() for x in params]

    time_history.append(float(time))
    param_history.append([float(x) for x in params])


def _calc_time(x1, x2):
    x3 = H - x1 - x2
    return np.sqrt(x1 ** 2 + D ** 2) / V1 + np.sqrt(x2 ** 2 + D ** 2) / V2 + np.sqrt(x3 ** 2 + D ** 2) / V3


def visualize_history():
    global time_history, param_history
    all_x = all_y = np.linspace(0, H, 300)
    xx, yy = np.meshgrid(all_x, all_y)
    tt = _calc_time(xx, yy)

    cmap = plt.get_cmap('Spectral_r')

    sns.set_style('darkgrid')
    fig, axes = plt.subplots(3, 1, figsize=(7, 12))

    rects = [
        plt.Rectangle((0, 0), D, H, facecolor='#080041', alpha=V1 / 2),
        plt.Rectangle((D, 0), D, H, facecolor='#080041', alpha=V2 / 2),
        plt.Rectangle((2 * D, 0), D, H, facecolor='#080041', alpha=V3 / 2)
    ]

    for i in range(len(param_history)):
        x1, x2 = param_history[i]

        axes[0].plot([0, D, 2 * D, 3 * D], [0, x1, x1 + x2, H], c=cmap(1 - i / (len(param_history) - 1)))

    for r in rects:
        axes[0].add_patch(r)

    axes[0].axis('off')
    axes[0].set_title('Path history')

    axes[1].contourf(xx, yy, tt, levels=50, cmap=cmap)
    axes[1].plot(*zip(*param_history), c='r', linewidth=1, linestyle='--', marker='x')
    axes[1].set_xlabel('$ x_1 $ [m]')
    axes[1].set_ylabel('$ x_2 $ [m]')
    axes[1].set_yticks(range(0, 21, 5))
    axes[1].set_aspect(1)
    axes[1].set_title('Parameter history')

    axes[2].plot(time_history, c='r', marker='x')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Time [s]')
    axes[2].set_title('Traversal time history')

    fig.tight_layout()
    plt.show()
