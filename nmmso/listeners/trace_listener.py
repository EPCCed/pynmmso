from nmmso.listeners.base_listener import BaseListener


class TraceListener(BaseListener):

    def __init__(self, level = 2):
        self.nmmso = None
        self.iteration_number = 1
        self.evaluations = 0
        self.level = level

    def set_nmmso(self, nmmso):
        self.nmmso = nmmso

    def iteration_started(self):
        if self.level >= 3:
            print(80*"=")
            print("Starting iteration {}".format(self.iteration_number))

    def location_evaluated(self, location, value):
        self.evaluations += 1
        if self.level >= 5:
            print("Evaluation {}: location {}, value is {}".format(self.evaluations, location, value))

    def swarm_peak_changed(self, swarm, old_location, old_value):
        if self.level >= 3:
            print("Swarm {} has found a new peak at location {} with value {}, old location was {} old value was {}".format(
                swarm.id, swarm.mode_location,swarm.mode_value, old_location, old_value))

    def swarm_created_at_random(self, swarm):
        if self.level >= 3:
            print("Created swarm {} at random location {}, value is {}".format(swarm.id, swarm.mode_location, swarm.mode_value))

    def swarm_created_from_crossover(self, swarm, parent_swarm1, parent_swarm2):
        if self.level >= 3:
            print("Created swarm {} by crossover of swarms {} and {} at location {}, value is {}".format(
                swarm.id, parent_swarm1.id, parent_swarm2.id, swarm.mode_location, swarm.mode_value, ))

    def merging_started(self):
        if self.level >= 4:
            print("Merging swarms...")

    def merged_close_swarms(self, swarm1, swarm2):
        if self.level >= 3:
            print("Merged swarm {} into swarm {} as they were close".format(swarm2.id, swarm1.id))

    def merged_saddle_swarms(self, swarm1, swarm2):
        if self.level >= 3:
            print("Merged swarm {} into swarm {} as midpoint was fitter".format(swarm2.id, swarm1.id))

    def merging_ended(self):
        if self.level >= 4:
            print("Finished merging swarms")

    def incrementing_swarms_started(self):
        if self.level >= 4:
            print("Incrementing swarms...")

    def swarm_added_particle(self, swarm):
        if self.level >= 4:
            print("Added particle to swarm {}, it now has {} particles".format(swarm.id, swarm.number_of_particles))

    def swarm_moved_particle(self, swarm):
        if self.level >= 4:
            print("Moved particle of swarm {}".format(swarm.id))

    def incrementing_swarms_ended(self):
        if self.level >= 4:
            print("Finished incrementing swarms")

    def hiving_swams_started(self):
        if self.level >= 4:
            print("Hiving swarms...")

    def hiving_new_swarm(self, new_swarm, parent_swarm):
        if self.level >= 3:
            print("Hiving new swarm {} from swarm {}".format(new_swarm.id, parent_swarm.id))

    def hiving_swarms_ended(self):
        if self.level >= 4:
            print("Finishing hiving swarms")

    def iteration_ended(self, n_new_locations, n_mid_evals, n_evol_modes, n_rand_modes, n_hive_samples):
        total_this_iteration = n_new_locations + n_mid_evals + n_evol_modes + n_rand_modes + n_hive_samples

        if self.level >= 1:
            print("Finished iteration {}, evaluations this iteration: {}, total evaluations: {}, number of swarms: {}".format(
                self.iteration_number, total_this_iteration, self.nmmso.evaluations, len(self.nmmso.swarms)))

        if self.level >= 3:
            print("  This iteration: new location evals = {} mid evals = {} evol modes = {} rand modes = {} hive samples = {}".format(
                n_new_locations, n_mid_evals, n_evol_modes, n_rand_modes, n_hive_samples))

        if self.level >= 2:
            for swarm in self.nmmso.swarms:
                print("Swarm {} : location: {}  value {}".format(swarm.id, swarm.mode_location, swarm.mode_value))

        self.iteration_number += 1

    def max_evaluations_reached(self):
        if self.level >= 1:
            print("Maximum number of evaluations reached.  Total evaluations: {}".format(self.nmmso.evaluations))
