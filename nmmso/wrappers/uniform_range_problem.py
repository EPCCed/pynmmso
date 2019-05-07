import numpy as np


class UniformRangeProblem:

    def __init__(self, problem):
        self.problem = problem
        mn, mx = problem.get_bounds()
        self.mn = np.array(mn)
        mx = np.array(mx)
        self.range = mx - self.mn

    def fitness(self, x):
        return self.problem.fitness(self.remap_parameters(x))

    def get_bounds(self):
        return [0] * len(self.mn), [1] * len(self.mn)

    def remap_parameters(self,x):
        return (x * self.range) + self.mn

