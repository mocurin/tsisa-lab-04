import random as rd
from functools import reduce
import numpy as np
import pandas as pd


def roll(proba):
    return rd.uniform(0, 1) < proba


def switch_check(delta, element, bounds):
    return element + delta > bounds[0] if delta < 0 else element + delta < bounds[1]


class GeneticAlgorithm:
    def __init__(self, m_delta, m_proba):
        self._m_delta = m_delta
        self._m_proba = m_proba
        self._population = 4
        self._select = 3
        self._log = False
        self.history = None

    def __call__(self, function, bounds, iterations, log=False, seed=42):
        rd.seed(seed)
        self._bounds = [bounds[:2], bounds[2:]]
        self._function = function
        self._log = log
        population = self._init_population()
        for _ in range(iterations):
            population = reduce(lambda x, y: y(x), [population,
                                                    self._selector,
                                                    self._crossover,
                                                    self._mutator])
            if self._log:
                scores = self._score(population)[:, np.newaxis]
                df = pd.DataFrame(data=np.round(np.hstack([population, scores]), 3),
                                  columns=self.history.columns)
                self.history = self.history.append(df, ignore_index=True)
        return population

    def _selector(self, population):
        scores = self._score(population)
        if any(scores < 0):
            scores = np.subtract(scores, np.min(scores))
        # find probabilities
        probas = np.divide(scores, np.sum(scores))
        indices = np.argsort(probas).tolist()
        selected = []
        while len(selected) < self._select:
            rolled = rd.uniform(0, np.sum(probas[indices]))
            for i in indices:
                rolled -= probas[i]
                if rolled <= 0:
                    selected.append(i)
                    indices.remove(i)
                    break
        return population[selected]

    def _crossover(self, population):
        x, y = np.transpose(population)
        # Organize x column: [x[0], x[0], x[1], x[2]]
        x_top = np.hstack([np.repeat(x[0], len(x[1:])), x[1:]])
        # Organize y column: [y[1], y[2], y[0], y[0]]
        y_top = np.hstack([y[1:], np.repeat(y[0], len(y[1:]))])
        # Make pairs
        population = np.transpose([x_top, y_top])
        return population

    def _mutator(self, population):
        # Roll probability, roll whether delta will be negative or positive
        mutation = [[roll(self._m_proba) * self._m_delta * (-1 if roll(0.5) else 1)
                     for _ in pair]
                    for pair in population]
        # Check bounds. If delta is 0 - stay 0,
        # else check if delta applying breaks respective bound
        mutation = [[delta if switch_check(delta, population[i][j],
                                           self._bounds[j]) else 0
                     for j, delta in enumerate(pair)]
                    for i, pair in enumerate(mutation)]
        # Apply deltas element-wise
        population = np.add(population, mutation)
        return population

    def _init_population(self):
        # Roll random coordinates inside respective bounds
        population = [[rd.uniform(*bounds)
                       for bounds in self._bounds]
                      for _ in range(self._population)]
        if self._log:
            score = self._score(population)[:, np.newaxis]
            self.history = pd.DataFrame(data=np.round(np.hstack([population, score]), 3),
                                        columns=['X', 'Y', 'Fit'])
        return np.array(population)

    def _score(self, population):
        return np.array([self._function(x, y) for x, y in population])
