from nmmso.listeners.base_listener import BaseListener


class MultiListener(BaseListener):
    """
    Listener that contains multiple listeners and calls them for each event.
    """

    def __init__(self):
        super().__init__()
        self.listeners = []

    def add_listener(self, listener):
        """
        Adds a new listener.

        Arguments
        ---------

        listener
            The listener to add.
        """
        self.listeners.append(listener)

    def set_nmmso(self, nmmso):
        for listener in self.listeners:
            listener.set_nmmso(nmmso)

    def iteration_started(self):
        for listener in self.listeners:
            listener.iteration_started()

    def location_evaluated(self, location, value):
        for listener in self.listeners:
            listener.location_evaluated(location, value)

    def swarm_peak_changed(self, swarm, old_location, old_value):
        for listener in self.listeners:
            listener.swarm_peak_changed(swarm, old_location, old_value)

    def swarm_created_at_random(self, new_swarm):
        for listener in self.listeners:
            listener.swarm_created_at_random(new_swarm)

    def swarm_created_from_crossover(self, new_swarm, parent_swarm1, parent_swarm2):
        for listener in self.listeners:
            listener.swarm_created_from_crossover(new_swarm, parent_swarm1, parent_swarm2)

    def merging_started(self):
        for listener in self.listeners:
            listener.merging_started()

    def merged_close_swarms(self, swarm1, swarm2):
        for listener in self.listeners:
            listener.merged_close_swarms(swarm1, swarm2)

    def merged_saddle_swarms(self, swarm1, swarm2):
        for listener in self.listeners:
            listener.merged_saddle_swarms(swarm1, swarm2)

    def merging_ended(self):
        for listener in self.listeners:
            listener.merging_ended()

    def incrementing_swarms_started(self):
        for listener in self.listeners:
            listener.incrementing_swarms_started()

    def swarm_added_particle(self, swarm):
        for listener in self.listeners:
            listener.swarm_added_particle(swarm)

    def swarm_moved_particle(self, swarm):
        for listener in self.listeners:
            listener.swarm_moved_particle(swarm)

    def incrementing_swarms_ended(self):
        for listener in self.listeners:
            listener.incrementing_swarms_ended()

    def hiving_swams_started(self):
        for listener in self.listeners:
            listener.hiving_swams_started()

    def hiving_new_swarm(self, new_swarm, parent_swarm):
        for listener in self.listeners:
            listener.hiving_new_swarm(new_swarm, parent_swarm)

    def hiving_swarms_ended(self):
        for listener in self.listeners:
            listener.hiving_swarms_ended()

    def iteration_ended(
            self, n_new_locations, n_mid_evals, n_evol_modes, n_rand_modes, n_hive_samples):
        for listener in self.listeners:
            listener.iteration_ended(
                n_new_locations, n_mid_evals, n_evol_modes, n_rand_modes, n_hive_samples)

    def max_evaluations_reached(self):
        for listener in self.listeners:
            listener.max_evaluations_reached()
