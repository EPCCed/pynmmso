from multiprocess import Process, Queue


def multiprocessor_process(problem, task_queue, result_queue):
    """
    Worker task main function.

    Arguments
    ---------

    problem
        problem object with a fitness method to invoke.
    task_queue : Queue
        Queue on which to receive tasks.
    result_queue : Queue
        Queue on which to return results.

    """
    while True:
        task = task_queue.get()
        task_id = task[0]
        task_loc = task[1]
        if task_id == -1:
            break

        # do some work
        res = problem.fitness(task_loc)
        result_queue.put([task_id, res])


class MultiprocessorFitnessCaller:
    """
    Fitness caller used for multiprocessor parallelism.

    Arguments
    ---------

    num_workers : int
        Number of worker nodes to create.

    """
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.problem = None

        self.total_tasks = 0
        self.total_groups = 0
        self.max_group_size = 0

        self.tasks = []

        self.num_calls = 0
        self.max_nodes = 0
        self.min_num_calls = 0

        self.num_workers = self.num_workers
        self.processes = []

        self.task_queue = Queue()
        self.result_queue = Queue()

    def __enter__(self):
        return self

    def set_problem(self, problem):
        """
        Sets the problem object to use to calculate the fitness.

        Arguments
        ---------

        problem
            Problem object implementing the fitness method.
        """
        for _ in range(self.num_workers):
            p = Process(target=multiprocessor_process,
                        args=(problem, self.task_queue, self.result_queue))
            p.start()
            self.processes.append(p)

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
        self.tasks.append([location, userdata])

    def evaluate(self):
        """
        Evaluates all the locations.

        Returns
        -------

        list of (location, value, userdate) tuples
            Tuples containing the location, value and corresponding user data
        """
        num_tasks = len(self.tasks)

        self.total_tasks += num_tasks
        self.total_groups += 1
        if num_tasks > self.max_group_size:
            self.max_group_size = num_tasks

        for i in range(num_tasks):
            self.task_queue.put([i, self.tasks[i][0]]) # [index, loc]

        y = 0
        num_results = 0
        results = []
        while num_results < num_tasks:
            result = self.result_queue.get()  # [ index, y]
            index = result[0]
            y = result[1]
            results.append((self.tasks[index][0], y, self.tasks[index][1],))
            num_results += 1

        self.tasks = []
        return results

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """
        Terminates the fitness caller.
        """
        for p in self.processes:
            self.task_queue.put([-1, -1])
        for p in self.processes:
            p.join()
