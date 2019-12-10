from genetic_algorithm import GeneticAlgorithm
from plotting import plot_function, save_gif
import numpy as np


if __name__ == '__main__':
    function = lambda x, y: np.cos(x) * np.sin(x) * np.exp(y / 2)
    bounds = [-2., 2., -2., 2.]
    plot_function(bounds, function)
    gen_alg = GeneticAlgorithm(0.05, 0.5)
    result = gen_alg(function, bounds, iterations=100, log=True)
    save_gif('images/', 'visualisation.gif')