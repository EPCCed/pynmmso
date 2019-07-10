from mpi4py import MPI
import numpy as np


class MpiFitnessCaller:

    """
    Fitness caller used for MPI parallelism.

    Arguments
    ---------

    comm : mpi4py comm object
        Comm object from mpi4py

    """
    def __init__(self, comm):
        self.comm = comm
        self.tasks = []
        self.num_workers = self.comm.Get_size()
        
    def set_problem(self, problem):
        """
        Sets the problem object to use to calculate the fitness.

        Arguments
        ---------

        problem
            Problem object implementing the fitness method.
        """
        self.problem = problem
        
        # Send the pickled problem to the workers
        self.comm.bcast(self.problem, root=0)
        
        
    def add(self, location, userdata):
        """
        Add a location to be evaluated.

        Arguments
        ---------
        location : numpy array
            Location to be evaluated.
        userdata
            User data to be returned with the evaluation result.
        """
        self.tasks.append([location, userdata])
        
        
    def evaluate(self):
        """
        Evaluates all the locations.

        Returns
        -------
        list of (location, value, userdata) tuples
            Tuples containing the location, value and corresponding user data
        """
        
        info = MPI.Status()
        
        num_tasks = len(self.tasks)
        task_index = 0
        
        # Give a task to each worker
        for worker in range(1, self.num_workers):
            if task_index < num_tasks:
                # Give task to worker
                self._send_task(worker, task_index)
                task_index += 1
        
        # Wait for results
        results = []
        fitness = np.zeros(1, dtype=np.float64)
        for result in range(num_tasks):
            self.comm.Recv(fitness, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=info)
            # Subtract 1 to get the correct index (doing so that 0 can be used as terminate tag)
            index = info.Get_tag()-1
            worker = info.Get_source()
            results.append((self.tasks[index][0], fitness[0], self.tasks[index][1],))
            
            if task_index < num_tasks:
                # Give task to worker
                self._send_task(worker, task_index)
                task_index += 1
            
        self.tasks = []
        return results            

    def _send_task(self, worker, task_index):
        """
        Sends the specified task to the specified worker.

        Arguments
        ---------

        worker: int
            Worker id

        task_index: int
            Index of task to be sent.
        """
        # Add 1 to the task index so 0 can be used as the terminate tag
        self.comm.Ssend(self.tasks[task_index][0], dest=worker, tag=task_index+1)
    
    def run_worker(self):
        """
        Runs the worker task to receive tasks and return the result.
        """
        rank = self.comm.Get_rank()
    
        # Receive the pickled problem
        self.problem = self.comm.bcast(None, root=0)
    
        info = MPI.Status()
        fitness = np.zeros(1, dtype=np.float64)
        params = np.zeros(len(self.problem.get_bounds()[0]), dtype=np.float64)
        
        # Receive data from controller
        tag = 1
        while tag > 0:
            self.comm.Recv(params, source=0, tag=MPI.ANY_TAG, status=info)
            tag = info.Get_tag()

            if tag > 0:
                fitness[0] = self.problem.fitness(params)
                self.comm.Ssend(fitness, dest=0, tag=tag)
            
    def finish(self):
        """
        Terminates the workers.
        """
        # Send quit tag to all workers
        dummy = np.zeros(1, dtype=np.float64)
        for worker in range(1, self.num_workers):
            self.comm.Ssend(dummy, dest=worker, tag=0) 
            
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
