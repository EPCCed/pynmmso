import random
import math
import numpy as np
import pynmmso.swarm as s
from pynmmso.listeners import MultiListener
from pynmmso.sequential_fitness_caller import SequentialFitnessCaller


class ModeResult:
    """
    Contains a mode result.

    Parameters
    ----------

    location: Numpy array
        Location of this mode in parameter space.

    value: float
        Fitness score of this mode.

    Attributes
    ----------

    location : Numpy array
        Location of this mode in parameter space.

    value : float
        Fitness score of this mode.

    """
    def __init__(self, location, value):
        self.location = location
        self.value = value


class Nmmso:
    """
    Niching Migratory Multi-Swarm Optimser.

    Parameters
    ----------

    problem :
        Instance of the problem class. Must implement get_bounds and fitness functions.

    swarm_size : int
        The maximum number of particles in a swarm. The default is -1 which specifies
        the number of particles will be 4 + floor(3*log(D)), where D is the number of
        dimensions of the problem.

    max_evol : int
        Maximum number of swarms that are updated in each iteration. The default is
        100.

    tol_val: float
        Tolerance of Euclidean distance between swarms. Swarms whose distance is below
        this threshold will be merged. Default value is  1e-06.

    fitness_caller :
        Used to specify various approaches to parallelism. Default is to use no parallelism.

    Attributes
    ----------

    evaluations: int
        Number of evaluations of the fitness function that have been performed so far.

    swarms:  Set of Swarm objects
        The currently active swarms.

    """
    def __init__(self, problem, swarm_size=-1, max_evol=100, tol_val=1e-06,
                 fitness_caller=SequentialFitnessCaller()):

        self.problem = problem
        self.max_evol = max_evol if max_evol > 0 else 100

        self.tol_val = tol_val
        self.min, self.max = problem.get_bounds()
        self.min = np.array(self.min)  # convert to numpy array
        self.max = np.array(self.max)  # convert to numpy array

        # Validate the input bounds
        if not np.all(len(self.min) == len(self.max)):
            raise ValueError("Lower bounds list must be the same length as the upper bounds list.")
        if not np.all(self.min < self.max):
            raise ValueError("Problem lower bound must be less than upper bound for all dimensions.")

        self.num_dimensions = len(self.min)

        if swarm_size < 1:
            self.swarm_size = 4 + math.floor(3 * math.log(self.num_dimensions))
        else:
            self.swarm_size = swarm_size

        self.converged_modes = 0
        self.next_swarm_id = 1
        self.evaluations = 0

        self.swarms = set()
        self.total_mid_evals = 0
        self.total_new_locations = 0
        self.total_evol_modes = 0
        self.total_rand_modes = 0
        self.total_hive_samples = 0

        self.fitness_caller = fitness_caller
        self.fitness_caller.set_problem(self.problem)

        self.listener = None

    def add_listener(self, listener):
        """
        Add a listener object that will receive notifications of events that occurs
        during the optimisation process.

        Parameters
        ----------

        listener : subclass of nmmso.listeners.BaseListener
            Listener object to receive notification of events.

        """
        if self.listener is None:
            self.listener = MultiListener()

        self.listener.add_listener(listener)
        listener.set_nmmso(self)

    def run(self, max_evaluations):
        """
        Runs is the optimisation algorithm until the specified number of fitness
        evaluations has been exceeded.

        If the algorithm has already been run it will continue from where it
        left off.

        Parameters
        ----------

        max_evaluations : int
            The maximum number of fitness evaluations.  The aglorithm may execute
            more evaluations that this number but will stop at the end of the
            iteration when this limit is exceeded.

        Returns
        -------

        list of ModeResult
            All the modes found by the algorithm.

        """
        while self.evaluations < max_evaluations:
            self.iterate()

        if self.listener is not None:
            self.listener.max_evaluations_reached()

        return self.get_result()

    def get_result(self):
        """
        Obtains the modes found by the algorithm.

        Returns
        -------

        list of ModeResult
            All the modes found by the algorithm.
        """
        has_remap_parameters_function = False
        remap_func = getattr(self.problem, "remap_parameters", None)
        if callable(remap_func):
            has_remap_parameters_function = True

        result = []
        for swarm in self.swarms:
            loc = swarm.mode_location
            if has_remap_parameters_function:
                loc = self.problem.remap_parameters(loc)
            result.append(ModeResult(loc, swarm.mode_value))
        return result

    def iterate(self):
        """
        Executes a single iteration of the algorithm.

        """
        if self.listener is not None:
            self.listener.iteration_started()

        if self.evaluations == 0:
            swarm = self._new_swarm()
            swarm.set_initial_location()
            swarm.evaluate_first()
            self._add_swarm(swarm)
            if self.listener is not None:
                self.listener.swarm_created_at_random(swarm)
            self.evaluations = 0
            num_of_evol_modes = 0
            num_rand_modes = 1
        else:
            # create speculative new swarm, either at random in design space, or via crossover
            if random.random() < 0.5 or \
                    len(self.swarms) == 1 or len(self.max) == 1 or self.max_evol == 1:
                num_of_evol_modes = 0
                num_rand_modes = self._random_new()  # this currently *always* returns 1
            else:
                num_rand_modes = 0
                num_of_evol_modes = self._evolve()

        if self.listener is not None:
            self.listener.merging_started()

        # See if modes should be merged together
        num_of_mid_evals = 0
        while sum([mode.changed for mode in self.swarms]) > 0:
            merge_evals = self._merge_swarms()
            num_of_mid_evals = num_of_mid_evals + merge_evals

        if self.listener is not None:
            self.listener.merging_ended()

        # Now increment the swarms
        if self.listener is not None:
            self.listener.incrementing_swarms_started()

        # if we have more than max_evol, then only increment a subset
        limit = min(self.max_evol, len(self.swarms))
        if len(self.swarms) > self.max_evol:
            if np.random.random() < 0.5:
                # select the fittest
                swarms_to_increment = \
                    sorted(self.swarms, key=lambda x: x.mode_value, reverse=True)[0:limit]
            else:
                # select at random
                swarms_to_increment = random.sample(self.swarms, limit)

        else:
            # use all swarms when incrementing
            swarms_to_increment = self.swarms

        for swarm in swarms_to_increment:
            swarm.increment()

        # evaluate new member/new location of swarm member
        num_of_new_locations = self._evaluate_new_locations(swarms_to_increment)

        if self.listener is not None:
            self.listener.incrementing_swarms_ended()

        # attempt to split off a member from one of the swarms to seed a new
        # swarm (if detected to be on another peak)
        if self.listener is not None:
            self.listener.hiving_swams_started()
        num_of_hive_samples = self._hive()
        if self.listener is not None:
            self.listener.hiving_swarms_ended()

        # update the total number of function evaluations used, with those
        # required at each of the algorithm stages
        self.evaluations = self.evaluations + \
                           num_of_mid_evals + \
                           num_of_new_locations + \
                           num_of_evol_modes + \
                           num_rand_modes + \
                           num_of_hive_samples

        self.total_mid_evals += num_of_mid_evals
        self.total_new_locations += num_of_new_locations
        self.total_evol_modes += num_of_evol_modes
        self.total_rand_modes += num_rand_modes
        self.total_hive_samples += num_of_hive_samples

        if self.listener is not None:
            self.listener.iteration_ended(
                num_of_new_locations, num_of_mid_evals, num_of_evol_modes,
                num_rand_modes, num_of_hive_samples)

    def _merge_swarms(self):

        # Only concern ourselves with modes that have actually shifted, or are new
        # since the last generation, as no need to check others
        swarms_changed = list(filter(lambda x: x.changed, self.swarms))
        self._reset_changed_flags()

        n = len(swarms_changed)

        closest_swarms = set()

        number_of_mid_evals = 0

        # only compare if there is a changed mode, and more than one mode in system
        if n >= 1 and len(self.swarms) > 1:
            for swarm in swarms_changed:
                closest_swarm, _ = swarm.find_nearest(self.swarms)
                tuple_of_swarms = Nmmso._create_swarm_tuple(swarm, closest_swarm)
                closest_swarms.add(tuple_of_swarms)
                if swarm.number_of_particles == 1:
                    swarm.initialise_new_swarm_velocities()

            to_merge = set()
            number_of_mid_evals = 0

            for swarm1, swarm2 in closest_swarms:
                if swarm1.distance_to(swarm2) < self.tol_val:
                    # merge if sufficiently close
                    to_merge.add((swarm1, swarm2, True))
                else:
                    # otherwise merge if midpoint is fitter
                    mid_loc = 0.5 * (swarm1.mode_location - swarm2.mode_location) + \
                              swarm2.mode_location
                    self.fitness_caller.add(mid_loc, (swarm1, swarm2))
                    number_of_mid_evals += 1

            # Get the results from the fitness caller
            for loc, y, userdata in self.fitness_caller.evaluate():
                swarm1, swarm2 = userdata
                if self.listener is not None:
                    self.listener.location_evaluated(loc, y)

                if y > swarm2.mode_value:
                    swarm2.update_location_and_value(loc, y)
                    to_merge.add((swarm1, swarm2, False))
                elif swarm1.mode_value <= y:
                    to_merge.add((swarm1, swarm2, False))

            # Merge the swarms - it is possible that we get one swarm involved in two merges.
            # In such situation depending on order we can get three combos:
            #   2 -> 1 and 3 -> 1  : do two merges into swarm 1, swarms 2 and 3 are deleted
            #   2 -> 1 and 3 -> 2  : merge 2 into 1, swarms 2 and 3 are deleted (3's
            #                        particles go to 2 but it is pointless)
            #   3 -> 2 and 2 -> 1  : all good, 2 and 3 are deleted but 3's particles have a
            #                        chance of reaching swarm 1
            for swarm1, swarm2, are_close in to_merge:
                # print('Merging {} and {}'.format(swarm1.id, swarm2.id))
                if swarm1.mode_value > swarm2.mode_value:
                    swarm1.merge(swarm2)
                    swarm1.changed = True
                    if swarm2 in self.swarms:
                        self.swarms.remove(swarm2)
                    if self.listener is not None:
                        if are_close:
                            self.listener.merged_close_swarms(swarm1, swarm2)
                        else:
                            self.listener.merged_saddle_swarms(swarm1, swarm2)
                else:
                    swarm2.merge(swarm1)
                    swarm2.changed = True
                    if swarm1 in self.swarms:
                        self.swarms.remove(swarm1)
                    if self.listener is not None:
                        if are_close:
                            self.listener.merged_close_swarms(swarm2, swarm1)
                        else:
                            self.listener.merged_saddle_swarms(swarm2, swarm1)

        # If only one mode, choose arbitrary dist for it
        if len(self.swarms) == 1:
            # for swarm in self.swarms:
            [swarm] = self.swarms
            swarm.set_arbitrary_distance()

        return number_of_mid_evals

    @staticmethod
    def _distance_to(location1, location2):
        """Euclidean distance between two locations"""
        return np.linalg.norm(location1-location2)

    @staticmethod
    def _create_swarm_tuple(swarm1, swarm2):
        """Create swarm tuples in a consistent way so we don't have (a,b) and
        (b,a) as separate keys"""
        if swarm1.id < swarm2.id:
            tuple_of_swarms = (swarm1, swarm2)
        else:
            tuple_of_swarms = (swarm2, swarm1)
        return tuple_of_swarms

    def _reset_changed_flags(self):
        for mode in self.swarms:
            mode.changed = False

    def _evaluate_new_locations(self, swarm_subset):
        """Evaluates the new locations of all swarms in the given subset of swarms."""

        # clear changed flag of all swarms (not just this subset)
        for swarm in self.swarms:
            swarm.changed = False

        # evaluate the subset of swarms
        for swarm in swarm_subset:
            self.fitness_caller.add(swarm.new_location, swarm)

        for loc, y, swarm in self.fitness_caller.evaluate():
            if self.listener is not None:
                self.listener.location_evaluated(loc, y)
            swarm.evaluate(y)

        return len(swarm_subset)

    def _evolve(self):
        """Evolves a new swarm by crossing over two existing swarms."""
        candidate_swarms = self.swarms

        if len(candidate_swarms) > self.max_evol and random.random() < 0.5:
            # if have a lot of swarms occasionally reduce the candidate set to be the fittest ones
            candidate_swarms = \
                sorted(candidate_swarms, key=lambda x: x.mode_value, reverse=True)[:self.max_evol]

        # select two at random from the candidate swarms
        swarms_to_cross = random.sample(candidate_swarms, 2)

        swarm = self._new_swarm()
        swarm.initialise_with_uniform_crossover(swarms_to_cross[0], swarms_to_cross[1])
        self._add_swarm(swarm)

        if self.listener is not None:
            self.listener.swarm_created_from_crossover(
                swarm, swarms_to_cross[0], swarms_to_cross[1])

        number_of_new_modes = 1
        return number_of_new_modes

    def _hive(self):

        # Comments by CW:
        #  - some of this probably could be put into the swarm class, but we would then need to pass
        #    quite a lot of stuff about the current swarms to the swarm class, *and* the [r] swarm
        #    needs updating too - lots of the logic is intertwined

        number_of_new_samples = 0

        limit = min(self.max_evol, len(self.swarms))

        # first identify those swarms who are at capacity, and therefore may be
        # considered for splitting off a member
        candidates = random.sample(self.swarms, limit)

        candidates = {x for x in candidates if x.number_of_particles >= self.swarm_size}

        if candidates:
            # select swarm at random
            [swarm] = random.sample(candidates, 1)

            # select an active swarm member at random (i.e. particle)
            k = random.randrange(swarm.swarm_size)

            particle_location = np.copy(swarm.history_locations[k, :])
            particle_value = swarm.history_values[k]

            # only look at splitting off member who is greater than tol_value
            # distance away -- otherwise will be merged right in again at the
            # next iteration
            if Nmmso._distance_to(particle_location, swarm.mode_location) > self.tol_val:

                mid_loc = 0.5 * (swarm.mode_location - particle_location) + particle_location

                mid_loc_val = self.problem.fitness(mid_loc)
                if self.listener is not None:
                    self.listener.location_evaluated(mid_loc, mid_loc_val)

                # if valley between, then hive off the old swarm member to create new swarm
                if mid_loc_val < particle_value:

                    new_swarm = self._new_swarm()

                    # allocate new
                    new_swarm.mode_location = particle_location
                    new_swarm.mode_value = particle_value

                    new_swarm.history_locations[0, :] = particle_location
                    new_swarm.history_values[0] = particle_value

                    new_swarm.pbest_locations[0, :] = particle_location
                    new_swarm.pbest_values[0] = particle_value

                    new_swarm.changed = True
                    new_swarm.converged = False

                    self._add_swarm(new_swarm)

                    # remove from existing swarm and replace with mid_eval
                    swarm.history_locations[k, :] = mid_loc
                    swarm.history_values[k] = mid_loc_val
                    swarm.pbest_locations[k, :] = mid_loc
                    swarm.pbest_values[k] = mid_loc_val

                    temp_vel = self.min - 1
                    d = Nmmso._distance_to(swarm.mode_location, particle_location)
                    reject = 0
                    while np.any(temp_vel < self.min) or np.any(temp_vel > self.max):
                        temp_vel = swarm.mode_location + \
                                   Nmmso.uniform_sphere_points(1, self.num_dimensions)[0] * (d / 2)

                        reject += 1
                        if reject > 20:
                            temp_vel = \
                                np.random.random(self.num_dimensions) * (self.max - self.min) + \
                                self.min

                    swarm.velocities[k, :] = temp_vel

                    if self.listener is not None:
                        self.listener.hiving_new_swarm(new_swarm, swarm)

                elif mid_loc_val > swarm.mode_value:
                    swarm.update_location_and_value(mid_loc, mid_loc_val)

                number_of_new_samples = 1

        return number_of_new_samples

    def _random_new(self):
        number_rand_modes = 1

        x = np.random.random_sample(np.shape(self.max)) * (self.max - self.min) + self.min

        new_swarm = self._new_swarm()
        new_swarm.changed = True
        new_swarm.converged = False
        new_swarm.new_location = x
        new_swarm.evaluate_first()
        self._add_swarm(new_swarm)

        if self.listener is not None:
            self.listener.swarm_created_at_random(new_swarm)

        return number_rand_modes

    @staticmethod
    def uniform_sphere_points(n, d):
        """
        Constructs uniform sphere points.

        Internal algorithm unity function.  Not to be called by users.

        Parameters:
        -----------

        n : int
            The number of points to construct

        d : int
            The number of dimensions of each point.

        Returns:
        numpy array
            2D array of uniform sphere points.

        """

        import operator

        z = np.random.normal(size=(n, d))

        # keepdims arg seems important; matlab explicitly seems to do this
        r1 = np.sqrt(np.sum(pow(z, 2), 1, keepdims=True))
        reps = np.tile(r1, (1, d))
        # using the function to make clear that it's element wise division
        # (/ symbol sometimes used for mrdivide)
        x = np.divide(z, reps)
        r = pow(np.random.rand(n, 1), (operator.truediv(1, d)))
        res = np.multiply(x, np.tile(r, (1, d)))  # np.multiply is element-wise
        return res

    def _add_swarm(self, swarm):
        self.swarms.add(swarm)
        self.next_swarm_id += 1

    def _new_swarm(self):
        return s.Swarm(self.next_swarm_id, self.swarm_size, self.problem, self.listener)
