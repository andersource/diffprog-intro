import numpy as np
from skimage.io import imsave


def main():
    x = y = np.linspace(-10, 10, 150)
    xx, yy = np.meshgrid(x, y)
    z = (xx + 4.5) ** 2 + (yy - 6) ** 2 <= 3.5 ** 2

    imsave('../data/ex2/gt.png', z)


if __name__ == '__main__':
    main()
