import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Used by gca
import numpy as np
import imageio
import os


def plot_creator(bounds, precision=0.2, function=None):
    x = np.arange(*bounds[0], precision)
    y = np.arange(*bounds[1], precision)
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


def plot_function(bounds, function):
    x = np.arange(*bounds[:2], 0.1)
    y = np.arange(*bounds[2:], 0.1)
    x, y = np.meshgrid(x, y)
    z = function(x, y)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, z, linewidth=0, antialiased=False)
    plt.show()


def save_gif(path, file):
    images = []
    filenames = os.listdir(path)
    for filename in sorted(filenames, key=lambda x: int(os.path.splitext(x)[0])):
        images.append(imageio.imread(path + filename))
    imageio.mimsave(file, images, duration=0.1)