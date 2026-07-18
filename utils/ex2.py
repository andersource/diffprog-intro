import imageio
import numpy as np
from skimage.transform import resize
from IPython.display import Video
import seaborn as sns
from skimage.io import imread
import matplotlib.pyplot as plt
import torch
import os
from ipywidgets import interact, FloatLogSlider


loss_hist = []
render_hist = []


def reset_history():
    global loss_hist, render_hist
    loss_hist = []
    render_hist = []


def log_history(render, loss):
    global loss_hist, render_hist
    if isinstance(render, torch.Tensor):
        render = render.detach().cpu().numpy()

    if isinstance(loss, torch.Tensor):
        loss = loss.detach()

    loss_hist.append(float(loss))
    render_hist.append(render)


def visualize_history():
    global loss_hist, render_hist

    sns.set_style('darkgrid')
    plt.plot(loss_hist, marker='x', c='r')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('Loss history')
    plt.show()

    if len(render_hist) == 0:
        print('No iterations logged')
        return

    os.makedirs('artifacts/ex2', exist_ok=True)

    gt = imread('data/ex2/gt.png') / 255
    scaled_gt = resize(gt, (480, 480))
    writer = imageio.get_writer('artifacts/ex2/video.mp4', fps=60)
    cmap = plt.get_cmap('PRGn')
    for r in render_hist + [render_hist[-1]] * 30:
        r = resize(r, (480, 480), order=0)
        img = (cmap((scaled_gt - r + 1) / 2) * 255).astype(np.uint8)
        writer.append_data(img)

    writer.close()

    return Video('artifacts/ex2/video.mp4')


def load_gt():
    gt = imread('data/ex2/gt.png') / 255
    plt.imshow(gt, cmap='PRGn', extent=(-10, 10, -10, 10), vmin=-1, vmax=1)
    plt.xticks((-10, 10), fontsize=14)
    plt.yticks((-10, 10), fontsize=14)
    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.title('Ground truth render', fontsize=18)
    plt.show()
    return gt


def visualize_render(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    plt.imshow(x, cmap='viridis', vmin=0, vmax=1)
    plt.axis('off')
    plt.show()


def interactive_sigmoid():
    def f(coef):
        x = np.linspace(-20, 20, 300)
        y = 1 / (1 + np.exp(-coef * x))
        a = b = np.linspace(-10, 10, 100)
        aa, bb = np.meshgrid(a, b)
        d = np.sqrt((aa - .5) ** 2 + (bb - .5) ** 2)
        z = 1 / (1 + np.exp(-coef * (3.5 - d)))

        sns.set_style('darkgrid')

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), width_ratios=(1, 1))
        axes[0].plot(x, y)
        axes[0].set_ylim((-.05, 1.05))
        axes[0].set_title(f'$ f(x) = \\frac{{1}}{{1+e^{{-{coef:.2f}x}}}} $', fontsize=16)
        axes[0].set_xlabel('$ x $')
        axes[0].set_ylabel('$ f(x) $')

        axes[1].imshow(z, extent=(-10, 10, -10, 10), cmap='viridis', vmin=0, vmax=1)
        axes[1].set_xticks((-10, 10))
        axes[1].set_yticks((-10, 10))
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].set_title('Rendering simulation')

        fig.tight_layout()
        plt.show()

    return interact(f, coef=FloatLogSlider(min=-3, max=3, step=.01, base=np.e))