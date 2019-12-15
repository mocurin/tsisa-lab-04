from genetic_algorithm import GeneticAlgorithm
from results_processing import visualize_history, process_history
import numpy as np


if __name__ == '__main__':
    function = lambda x, y: np.cos(x) * np.cos(y) * np.exp(y / 2)
    bounds = [-2., 2., -2., 2.]
    gen_alg = GeneticAlgorithm(0.1, 0.25)
    gen_alg(function, bounds, iterations=100, log=True)
    history = gen_alg.history
    history.to_csv('history.csv')
    process_history(history.values, 4).to_csv('metrics.csv')
    visualize_history(history.values, 4, function, bounds)