import numpy as np
import matplotlib.pyplot as plt


def plot_ndarray(array: np.ndarray) -> None:
    """
    Displays image from numpy's ndarray
    """

    plt.imshow(array, aspect="auto")
    plt.plot(array)
    plt.imsave('img_1.png', array)


if __name__ == "__main__":
    plot_ndarray(np.random.randint(1, size=(20, 20, 3), dtype=np.uint8))


