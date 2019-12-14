import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Used by gca
import numpy as np
import imageio
import os


def plot_creator(bounds, precision=0.1, function=None):
    x0, x1 = bounds[0]
    x = np.arange(x0, x1 + precision, precision)
    y0, y1 = bounds[1]
    y = np.arange(y0, y1 + precision, precision)
    if function is not None:
        x, y = np.meshgrid(x, y)
        z = function(x, y)

    def plot_population(population, title, file):
        x_p, y_p = np.transpose(population)
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        if function is not None:
            plt.pcolormesh(x, y, z)
        else:
            plt.xticks(x)
            plt.yticks(y)
        plt.plot(x_p, y_p, marker='o', linestyle='', color='r')
        plt.savefig(file)
        plt.clf()
    return plot_population


def plot_function(bounds, function, precision=0.1):
    x0, x1, y0, y1 = bounds
    x = np.arange(x0, x1 + precision, precision)
    y = np.arange(y0, y1 + precision, precision)
    x, y = np.meshgrid(x, y)
    z = function(x, y)
    # find maximum
    i = np.unravel_index(np.argmax(z, axis=None), z.shape)
    print('MAXIMUM VALUE')
    print('x:', round(x[i], 3),
          ' y:', round(y[i], 3),
          ' z:', round(z[i], 3))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.plot_surface(x, y, z, linewidth=0, antialiased=False)
    plt.show()


def save_gif(path, file):
    images = []
    filenames = os.listdir(path)
    for filename in sorted(filenames, key=lambda x: int(os.path.splitext(x)[0])):
        images.append(imageio.imread(path + filename))
    imageio.mimsave(file, images, duration=0.1)