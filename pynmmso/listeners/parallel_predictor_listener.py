import math
from pynmmso.listeners.base_listener import BaseListener


class ParallelPredictorListener(BaseListener):
    """
    Listener class used to predict how a parallel implementation may peform.
    """

    def __init__(self):
        self.num_workers = [2, 4, 6, 8, 10, 12, 16, 32, 64, 128]
        self.min_evaluations = [0] * len(self.num_workers)
        self.max_active_workers = 0
        self.total_evaluations = 0
        super().__init__()

    def iteration_ended(
            self, n_new_locations, n_mid_evals, n_evol_modes, n_rand_modes, n_hive_samples):
        self.total_evaluations += \
            n_new_locations + n_mid_evals + n_evol_modes + n_rand_modes + n_hive_samples

        for i in range(len(self.num_workers)):
            self.min_evaluations[i] += n_evol_modes + n_rand_modes + n_hive_samples
            self.min_evaluations[i] += math.ceil(n_new_locations/self.num_workers[i])
            self.min_evaluations[i] += math.ceil(n_mid_evals/self.num_workers[i])
            if n_new_locations > self.max_active_workers:
                self.max_active_workers = n_new_locations
            if n_mid_evals > self.max_active_workers:
                self.max_active_workers = n_mid_evals

    def print_summary(self):
        """
        Prints a summary of the results.
        """
        print("Total number of fitness evaluations: {}".format(self.total_evaluations))
        print("Maximum number of workers: {}".format(self.max_active_workers))
        for i in range(len(self.num_workers)):
            print("{} workers, effective evaluations {}, "
                  "minimum runtime as percentage of sequential runtime: {:0.2f}%".format(
                      self.num_workers[i], self.min_evaluations[i],
                      self.min_evaluations[i]/self.total_evaluations * 100.0))
