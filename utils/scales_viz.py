import numpy as np
import matplotlib.pyplot as plt


def main():
    x = np.linspace(-3, 3, 300)
    y = np.linspace(-3, 3, 300)
    xx, yy = np.meshgrid(x, y)

    z = np.sin(3 * xx) + np.sin(3 * yy) + 2 * np.sqrt((xx - 2) ** 2 + (yy + 1) ** 2)

    plt.figure(figsize=(7, 5))
    plt.contourf(z, levels=20)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
