class BaseListener:
    """
    Empty base class for listeners to NMMSO
    """

    def __init__(self):
        pass

    def set_nmmso(self, nmmso):
        """
        Sets the Nmmso object.

        Arguments
        ---------

        nmmso : Nmmso
            The Nmmso object.
        """
        pass

    def iteration_started(self):
        """
        Called when a new iteration is started.
        """
        pass

    def location_evaluated(self, location, value):
        """
        Called whenever a new location in parameter space has been evaluated.

        Arguments
        ---------

        location : numpy array
            Array of parameter values
        value : float
            Fitness value for this location.
        """
        pass

    def swarm_peak_changed(self, swarm, old_location, old_value):
        """
        Called when a swarm's peak location has changed.

        Arguments
        ---------

        swarm : Swarm
            The swarm that has changed.  This will contain the new location and fitness values.
        old_location : numpy array
            Previous location of the swarm.
        old_value :
            Previous fitness value of the swarm.
        """
        pass

    def swarm_created_at_random(self, new_swarm):
        """
         Called when a new swarm has been created at a random location.

        Arguments
        ---------

        new_swarm : Swarm
            The swarm that has been created.
        """
        pass

    def swarm_created_from_crossover(self, new_swarm, parent_swarm1, parent_swarm2):
        """
        Called when a new swarm has been created at a location produced by crossover from two
        other swarms.

        Arguments
        ---------

        new_swarm : Swarm
            Newly created swarm
        parent_swarm1 : Swarm
            Parent swarm
        parent_swarm2 : Swarm
            Parent swarm
        """
        pass

    def merging_started(self):
        """
        Called when the merging stage has started.
        """
        pass

    def merged_close_swarms(self, swarm1, swarm2):
        """
        Called when two swarms have been merged because them were close.

        Arguments
        ---------

        swarm1 : Swarm
            The swarm that will be kept.
        swarm2 : Swarm
            The swarm that will be merged into the other swarm.
        """
        pass

    def merged_saddle_swarms(self, swarm1, swarm2):
        """
        Called when two swarms have been merged because the midpoint between them is fitter
        than both swarms.

        Arguments
        ---------

        swarm1 : Swarm
            The swarm that will be kept.
        swarm2 : Swarm
            The swarm that will be merged into the other swarm.
        """
        pass

    def merging_ended(self):
        """
        Called when the merging stage has completed.
        """
        pass

    def incrementing_swarms_started(self):
        """
        Called when the swarm incrementing stage has started.
        """
        pass

    def swarm_added_particle(self, swarm):
        """
        Called when a particle has been added to a swarm.

        Arguments
        ---------

        swarm : Swarm
            The swarm to which the particle has been added.
        """
        pass

    def swarm_moved_particle(self, swarm):
        """
        Called when a particle has been moved.

        Arguments
        ---------

        swarm : Swarm
            The swarm in which the particle has been moved.
        """
        pass

    def incrementing_swarms_ended(self):
        """
        Called when the swarm incrementing stage has finished.
        """
        pass

    def hiving_swams_started(self):
        """
        Called when the swarm hiving stage has completed.
        """
        pass

    def hiving_new_swarm(self, new_swarm, parent_swarm):
        """
        Called when a new swarm is hived from a parent swarm.

        Arguments
        ---------

        new_swarm : Swarm
            The swarm in which been hived off.
        parent_swarm : Swarm
            The parent swarm.
        """
        pass

    def hiving_swarms_ended(self):
        """
        Called when the swarm hiving stage has completed.
        """
        pass

    def iteration_ended(
            self, n_new_locations, n_mid_evals, n_evol_modes, n_rand_modes, n_hive_samples):
        """
        Called when an iteration has completed.

        Arguments
        ---------

        n_new_locations : int
            Number of fitness evaluations when incrementing the swarms in this iteration
        n_mid_evals : int
            Number of midpoint fitness evaluations when merging in this iteration
        n_evol_modes : int
            Number of fitness evaluations when evolving new swarms in this iteration
        n_rand_modes : int
            Number of fitness evaluations when generating a new swarm at a random location in this iteration
        n_hive_samples : int
            Number of fitness evaluation during the hive process in this iteration

        """
        pass

    def max_evaluations_reached(self):
        """
        Called when the specified maximum number of evaluation has been reached.
        """
        pass
