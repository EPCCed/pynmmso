class SequentialFitnessCaller:
    """
    Fitness caller used for sequential implementation of NMMSO algorithm.
    """
    def __init__(self):
        self.problem = None
        self.data = []

    def set_problem(self, problem):
        """
        Sets the problem object to use to calculate the fitness.

        Arguments
        ---------

        problem
            Problem object implementing the fitness method.
        """
        self.problem = problem

    def add(self, location, userdata):
        """
        Add a location to be evaluated.

        Arguments
        ---------

        location : numpy array
            Location to be evaluated.

        userdata
            Userdata to be returned with the evaluation result.
        """
        self.data.append((location, userdata))

    def evaluate(self):
        """
        Evaluates all the locations.

        Returns
        -------

        list of (location, value, userdate) tuples
            Tuples containing the location, value and corresponding user data
        """
        result = []
        for location, userdata in self.data:
            value = self.problem.fitness(location)
            result.append((location, value, userdata))

        self.data = []
        return result

    def finish(self):
        """
        Terminates the fitness caller.
        """
        pass
