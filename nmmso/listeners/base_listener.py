

class BaseListener:

    """Empty base class for listeners to NMMSO """

    def __init__(self):
        pass

    def set_nmmso(self, nmmso):
        pass

    def iteration_started(self):
        pass

    def location_evaluated(self, location, value):
        pass

    def swarm_peak_changed(self, swarm, old_location, old_value):
        pass

    def swarm_created_at_random(self, new_swarm):
        pass

    def swarm_created_from_crossover(self, new_swarm, parent_swarm1, parent_swarm2):
        pass

    def merging_started(self):
        pass

    def merged_close_swarms(self, swarm1, swarm2):
        pass

    def merged_saddle_swarms(self, swarm1, swarm2):
        pass

    def merging_ended(self):
        pass

    def incrementing_swarms_started(self):
        pass

    def swarm_added_particle(self, swarm):
        pass

    def swarm_moved_particle(self, swarm):
        pass

    def incrementing_swarms_ended(self):
        pass

    def hiving_swams_started(self):
        pass

    def hiving_new_swarm(self, new_swarm, parent_swarm):
        pass

    def hiving_swarms_ended(self):
        pass

    def iteration_ended(self, n_new_locations, n_mid_evals, n_evol_modes, n_rand_modes, n_hive_samples):
        pass

    def max_evaluations_reached(self):
        pass