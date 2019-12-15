import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Used by gca
import numpy as np
import pandas as pd
import imageio
import os


def process_history(history, population):
    history = np.reshape(history, (-1, population, 3))
    maxes = np.max(history[:, :, 2], axis=1)
    means = np.mean(history[:, :, 2], axis=1)
    return pd.DataFrame(data=np.transpose([maxes, means]),
                        columns=['Max', 'Mean'])


def visualize_history(history, population, function, bounds, precision=0.1):
    if not os.path.exists('images/'):
        os.makedirs('images/')
    history = np.reshape(history, (-1, population, 3))
    x0, x1 = bounds[:2]
    x = np.arange(x0, x1 + precision, precision)
    y0, y1 = bounds[2:]
    y = np.arange(y0, y1 + precision, precision)
    x, y = np.meshgrid(x, y)
    z = function(x, y)
    visualize_function(x, y, z)
    for i, population in enumerate(history):
        x_p, y_p = np.transpose(population[:, :2])
        plt.title(str(i) + ' POPULATION')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.pcolormesh(x, y, z)
        plt.plot(x_p, y_p, marker='o', linestyle='', color='r')
        plt.savefig('images/' + str(i) + '.png')
        plt.clf()
    save_gif('images/', 'visualization.gif')


def visualize_function(x, y, z):
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
