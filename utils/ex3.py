import os
os.environ['OMP_NUM_THREADS'] = '1'
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm
import torch
import imageio
from IPython.display import Video

from utils.ex3_gen_mesh import POSE


param_hist = []
loss_hist = []
render_hist = []
vis = None


def load_gt():
    res = []

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3))
    cmap = np.concatenate([np.ones((1, 3)),
                           np.asarray(sns.color_palette('colorblind', 5))], axis=0)
    for gt_axes, ax in zip(['xy', 'xz', 'yz'], axes):
        gt = imread(f'data/ex3/{gt_axes}.png')
        res.append(np.stack([gt == i for i in range(1, 6)], axis=0).astype(float))

        img = cmap[gt]

        ax.imshow(img, extent=(-15, 15, -15, 15))
        ax.set_xticks((-15, 15), labels=(-15, 15), fontsize=14)
        ax.set_yticks((-15, 15), labels=(-15, 15), fontsize=14)
        ax.set_xlabel(gt_axes[0], fontsize=16)
        ax.set_ylabel(gt_axes[1], fontsize=16)

    plt.suptitle('Ground truth renders', fontsize=18)

    fig.tight_layout()
    plt.show()

    return res


def _load_gt_internal():
    res = []
    for gt_axes in ['xy', 'xz', 'yz']:
        gt = imread(f'data/ex3/{gt_axes}.png')
        res.append(np.stack([gt == i for i in range(1, 6)], axis=0).astype(float))

    return res


def reset_history():
    global param_hist, loss_hist, render_hist
    param_hist = []
    loss_hist = []
    render_hist = []


def log_history(renders, loss, params):
    global param_hist, loss_hist, render_hist

    if isinstance(params, torch.Tensor):
        params = params.detach().cpu().numpy()

    if isinstance(renders[0], torch.Tensor):
        renders = [r.detach().cpu().numpy() for r in renders]

    if isinstance(loss, torch.Tensor):
        loss = loss.detach()

    param_hist.append(params)
    loss_hist.append(float(loss))
    render_hist.append([r.max(axis=0) for r in renders])


def visualize_history():
    global param_hist, loss_hist, render_hist, vis

    sns.set_style('darkgrid')
    plt.plot(loss_hist, marker='x', c='r')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('Loss history')
    plt.show()

    if vis is None:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=512, height=368, visible=False)

    spheres = []
    for i in range(5):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1, resolution=100)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(np.array([139, 62, 184]) / 255)
        spheres.append(sphere)

    vis.clear_geometries()
    ctl = vis.get_view_control()

    for sphere in spheres:
        vis.add_geometry(sphere)

    if len(param_hist) == 0:
        print('No iterations logged')
        return

    xy, xz, yz = _load_gt_internal()
    os.makedirs('artifacts/ex3', exist_ok=True)
    writer = imageio.get_writer('artifacts/ex3/video.mp4', fps=120)
    cmap = plt.get_cmap('PRGn')
    for idx in tqdm(range(len(param_hist)), desc='Rendering optimization'):
        for i in range(5):
            curr = param_hist[idx][i]
            scale = curr[0]
            translation = curr[1:]
            center = np.zeros(3)
            if idx > 0:
                prev = param_hist[idx - 1][i]
                scale = curr[0] / prev[0]
                center = prev[1:]
                translation = curr[1:] - center

            spheres[i].scale(scale, center)
            spheres[i].translate(translation)
            vis.update_geometry(spheres[i])

            vis.reset_view_point(reset_bounding_box=True)
            ctl.set_lookat(POSE['lookat'])
            ctl.set_front(POSE['front'])
            ctl.set_up(POSE['up'])
            ctl.set_zoom(POSE['zoom'])

            img = np.clip(vis.capture_screen_float_buffer(do_render=True), 0, 1)
            img = (img * 255).astype(np.uint8)

            diffs = (np.concatenate([
                resize(
                    cmap((gt.max(axis=0) - render_hist[idx][j] + 1) / 2),
                    (122, 122), order=0)
                for j, gt in enumerate([xy, xz, yz])
            ] + [np.ones((2, 122, 4))], axis=0) * 255).astype(np.uint8)[..., :3]

            img = np.concatenate([img, diffs], axis=1)
            img = np.concatenate([img, np.ones((img.shape[0], 6, 3), dtype=np.uint8) * 255], axis=1)

            writer.append_data(img)

    writer.close()

    return Video('artifacts/ex3/video.mp4')
