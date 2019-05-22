import numpy as np


class UniformRangeProblem:
    """
    Class used to wrap a problem class so that all parameters have a uniform range.

    Arguments
    ---------

    problem
        The problem object to be wrapped.
    """

    def __init__(self, problem):
        self.problem = problem
        mn, mx = problem.get_bounds()
        self.min = np.array(mn)
        mx = np.array(mx)
        self.range = mx - self.min

    def fitness(self, location):
        """
        Calls the fitness function.

        Arguments
        ---------

        location : numpy array
            The location to evaluate the fitness of.

        Returns
        -------

        float
            The fitness value at the given location.
        """
        return self.problem.fitness(self.remap_parameters(location))

    def get_bounds(self):
        """
        Gets the bounds of the problem.

        Returns
        -------
        numpy array, numpy array
            One array with the lower bounds and an array with the upper bounds.
        """
        return [0] * len(self.min), [1] * len(self.min)

    def remap_parameters(self, location):
        """
        Maps a given location in the new parameter space to a location in the
        parameter space of the wrapped problem.

        Arguments
        ---------

        location : numpy array
            The location in the uniform parameter space.

        Returns
        -------

        numpy array
            The corresponding location in the parameter space of the
            wrapped problem.

        """
        return (location * self.range) + self.min
