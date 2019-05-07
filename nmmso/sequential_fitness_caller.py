
class SequentialFitnessCaller:

    def __init__(self):
        self.problem = None
        self.data = []

        self.num_calls = 0
        self.max_nodes = 0
        self.min_num_calls = 0

    def set_problem(self, problem):
        self.problem = problem

    def add(self, location, userdata):
        self.data.append((location, userdata))

    def evaluate(self):
        if len(self.data) > self.max_nodes:
            self.max_nodes = len(self.data)
        self.min_num_calls += 1

        result = []
        for location, userdata in self.data:
            value = self.problem.fitness(location)
            result.append((location, value, userdata))
            self.num_calls += 1

        self.data = []
        return result

    def finish(self):
        pass

    def print_stats(self):
        print("Sequential Fitness Caller: num calls = {}, max_nodes = {}, min_num_calls = {}".format(self.num_calls, self.max_nodes, self.min_num_calls))
