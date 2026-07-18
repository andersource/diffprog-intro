import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave
import os
from ex3_gen_mesh import get_all_params


def render(axes):
    x = np.linspace(-15, 15, 150)
    y = np.linspace(-15, 15, 150)
    xx, yy = np.meshgrid(x, y)
    z = np.zeros_like(xx)
    for idx, p in enumerate(get_all_params()):
        r = p[0]
        px, py = np.array(p[1:])[axes]
        mask = (xx - px) ** 2 + (yy - py) ** 2 <= r ** 2
        z[mask] = idx + 1

    plt.axis('off')
    plt.imshow(z)
    plt.show()
    return z


def main():
    os.makedirs('renders', exist_ok=True)
    for axes, name in zip([(0, 1), (0, 2), (1, 2)], ['xy', 'xz', 'yz']):
        res = render(list(axes)).astype(np.uint8)
        imsave(f'../data/ex3/{name}.png', res)


if __name__ == '__main__':
    main()
