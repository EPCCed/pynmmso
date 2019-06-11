import random
import numpy as np
import pynmmso as nmmso


class Swarm:
    """
    Represents a swarm in the NMMSO algorithm.

    Arguments
    ---------

    id : int
        Id used to refer to the swarm
    swarm_size : int
        Maximum number of particles in the swarm
    problem :
        Instance of the problem class. Must implement get_bounds and fitness functions.
    listener : subclass of nmmso.listeners.BaseListener
            Listener object to receive notification of events. Optional.


    Attributes
    ----------

    id : int
        A unique identification number of this swarm.
    mode_location : numpy array
        The location of this mode.
    mode_value : float
        The fitness of the mode location.
    number_of_particles : int
        Number of particles in the swarm.
    history_locations : 2D Numpy array
        The current locations of each particle in the swarm.
    history_values : 1D Numpy array
        The fitness values for current locations of each particle in the swarm.
    velocities : 2D Numpy array
        Current velocity of each particle in the swarm.
    pbest_location : 2D Numpy array
        The best location discovered for each particle.
    pbest_value : 1D Numpy array
        The fitness value associated with the best location for each particle in the swarm.

    """
    def __init__(self, id, swarm_size, problem, listener=None):
        self.id = id
        self.swarm_size = swarm_size
        self.problem = problem
        self.listener = listener

        self.mn = np.array(problem.get_bounds()[0])
        self.mx = np.array(problem.get_bounds()[1])

        self.changed = True
        self.converged = False
        self.num_dimensions = len(self.mn)

        self.mode_location = None  # Will be populated later on
        self.new_location = None   # Will be populated later on
        self.mode_value = None     # Will be populated later on

        # Initialize locations for swarm elements

        # current locations of swarm
        self.history_locations = np.zeros((self.swarm_size, self.num_dimensions))
        # current values of swarm
        self.history_values = np.full(self.swarm_size, -np.inf)

        # current best locations of swarm
        self.pbest_locations = np.zeros((self.swarm_size, self.num_dimensions))
        # current best values of swarm
        self.pbest_values = np.full(self.swarm_size, -np.inf)

        self.velocities = np.zeros((swarm_size, self.num_dimensions))
        self.number_of_particles = 1

        self.shifted_loc = None  # Will be populated later on
        self.dist = None  # Will be populated later on

    def set_initial_location(self):
        """Sets the initial location of a swarm."""
        self.changed = True
        self.new_location = (np.random.rand(self.num_dimensions) * (self.mx-self.mn)) + self.mn
        # random initial velocities of swarm
        self.velocities[0, :] = (np.random.rand(self.num_dimensions) * (self.mx-self.mn)) + self.mn

    def set_arbitrary_distance(self):
        """Set an arbitrary distance - this is done when we only have one swarm"""
        self.dist = np.min(self.mx-self.mn)

    def increment(self):
        """ Increments the swarm. """
        new_location = self.mn - 1

        d = self.dist
        shifted = False
        omega = 0.1
        reject = 0

        r = random.randrange(self.swarm_size)   # select particle at random to move

        while np.sum(new_location < self.mn) > 0 or np.sum(new_location > self.mx) > 0:

            # if swarm is not yet at capacity, simply add a new particle
            if self.number_of_particles < self.swarm_size:
                usp = nmmso.Nmmso.uniform_sphere_points(1, self.num_dimensions)[0]
                new_location = self.mode_location + usp * (d/2)
            else:
                # move an existing particle
                shifted = True
                self.shifted_loc = r
                r1 = np.random.rand(self.num_dimensions)
                r2 = np.random.rand(self.num_dimensions)

                temp_vel = omega * self.velocities[self.shifted_loc, :] + \
                           2.0 * r1 * \
                           (self.mode_location - self.history_locations[self.shifted_loc, :]) + \
                           2.0 * r2 * \
                           (self.pbest_locations[self.shifted_loc, :] -
                            self.history_locations[self.shifted_loc, :])

                if reject > 20:
                    # if we keep rejecting then put at extreme any violating design parameters
                    i_max = np.flatnonzero(
                        np.asarray(
                            self.history_locations[self.shifted_loc, :] + temp_vel > self.mx))
                    i_min = np.flatnonzero(
                        np.asarray(
                            self.history_locations[self.shifted_loc, :] + temp_vel < self.mn))
                    if i_max.size > 0:
                        temp_vel[i_max] = \
                            np.random.rand(i_max.size) * \
                            (self.mx[i_max] - self.history_locations[self.shifted_loc, i_max])
                    if i_min.size > 0:
                        temp_vel[i_min] = \
                            np.random.rand(i_min.size) * \
                            (self.history_locations[self.shifted_loc, i_min] - self.mn[i_min])

                new_location = self.history_locations[self.shifted_loc, :] + temp_vel
                reject = reject + 1

        if shifted:
            self.velocities[self.shifted_loc, :] = temp_vel
        else:
            # otherwise initialise velocity in sphere based on distance from gbest to next
            # closest mode
            self.number_of_particles = self.number_of_particles + 1
            self.shifted_loc = self.number_of_particles - 1
            temp_vel = self.mn - 1

            reject = 0
            while np.sum(temp_vel < self.mn) > 0 or np.sum(temp_vel > self.mx) > 0:
                temp_vel = \
                    self.mode_location + \
                    nmmso.Nmmso.uniform_sphere_points(1, self.num_dimensions)[0] * (d / 2)
                reject = reject + 1
                if reject > 20:  # resolve if keep rejecting
                    temp_vel = np.random.rand(self.num_dimensions)*(self.mx-self.mn) + self.mn

            self.velocities[self.shifted_loc, :] = temp_vel

        self.new_location = new_location

        if self.listener is not None:
            if shifted:
                self.listener.swarm_moved_particle(self)
            else:
                self.listener.swarm_added_particle(self)

    def initialise_with_uniform_crossover(self, swarm1, swarm2):
        """
        Initialise a new swarm with the uniform crossover of the given swarms.

        Arguments
        ---------

        swarm1 : Swarm
        swarm2 : Swarm
        """
        self.new_location, _ = Swarm.uni(swarm1.mode_location, swarm2.mode_location)
        self.evaluate_first()
        self.changed = True
        self.converged = False

    def distance_to(self, swarm):
        """
        Euclidean distance between this swarm and the given swarm, based on their mode locations.

        Returns
        -------
        float
            The distance between the two swarms.
        """
        return np.linalg.norm(self.mode_location-swarm.mode_location)

    def merge(self, swarm):
        """
        Merges the give swarm into this swarm.

        Arguments
        ----------

        swarm : Swarm
            Swarm to merge into this swarm.

        """
        n1 = self.number_of_particles
        n2 = swarm.number_of_particles

        if n1 + n2 < self.swarm_size:
            # simplest solution, where the combined active members of both populations
            # are below the total size they can grow to
            self.number_of_particles = n1 + n2
            self.history_locations[n1:n1 + n2, :] = swarm.history_locations[0:n2, :]
            self.history_values[n1:n1 + n2] = swarm.history_values[0:n2]
            self.pbest_locations[n1:n1 + n2, :] = swarm.pbest_locations[0:n2, :]
            self.pbest_values[n1:n1 + n2] = swarm.pbest_values[0:n2]
            self.velocities[n1:n1 + n2, :] = swarm.velocities[0:n2, :]
        else:
            # select best out of combines population, based on current location (rather than pbest)
            self.number_of_particles = self.swarm_size
            temp_h_loc = \
                np.concatenate((self.history_locations[0:n1, :], swarm.history_locations[0:n2, :]))
            temp_h_v = \
                np.concatenate((self.history_values[0:n1], swarm.history_values[0:n2]))
            temp_p_loc = \
                np.concatenate((self.pbest_locations[0:n1, :], swarm.pbest_locations[0:n2, :]))
            temp_p_v = np.concatenate((self.pbest_values[0:n1], swarm.pbest_values[0:n2]))
            temp_vel = np.concatenate((self.velocities[0:n1, :], swarm.velocities[0:n2, :]))


            # get the indices of highest values
            I = np.argsort(temp_h_v)[len(temp_h_v) - self.swarm_size:]

            self.history_locations = temp_h_loc[I, :]
            self.history_values = temp_h_v[I]
            self.pbest_locations = temp_p_loc[I, :]
            self.pbest_values = temp_p_v[I]
            self.velocities = temp_vel[I, :]

    def initialise_new_swarm_velocities(self):
        """Initialises velocities of a new swarm."""

        reject = 0
        temp_vel = self.mn - 1
        while np.sum(temp_vel < self.mn) > 0 or np.sum(temp_vel > self.mx) > 0:
            temp_vel = self.mode_location + \
                       nmmso.Nmmso.uniform_sphere_points(1, self.num_dimensions)[0] * \
                       (self.dist / 2)
            reject += 1

            if reject > 20:
                temp_vel = (np.random.rand(self.num_dimensions) * (self.mx-self.mn)) + self.mn

        self.velocities[0, :] = temp_vel

    def update_location_and_value(self, location, value):
        """
        Updates the location and value of this swarm.

        Arguments
        ---------

        location : numpy arrya
            New location of swarm
        value : float
            New fitness value at swarm location.
        """
        previous_location = self.mode_location
        previous_value = self.mode_value
        self.mode_location = location
        self.mode_value = value
        if self.listener is not None:
            self.listener.swarm_peak_changed(self, previous_location, previous_value)

    def evaluate_first(self):
        """
        Evaluates the new location.  This is the first evaluation so no need to examine
        if a shift has occurred
        """

        # new location is the only solution thus far in mode, so by definition
        # is also the mode estimate, and the only history thus far

        y = self.problem.fitness(self.new_location)

        if not np.isscalar(y):
            raise ValueError("Problem class's fitness method must return a scalar value.")

        if self.listener is not None:
            self.listener.location_evaluated(self.new_location, y)

        self.mode_location = self.new_location  # gbest location
        self.mode_value = y  # gbest value

        self.history_locations[0, :] = self.mode_location
        self.history_values[0] = y

        self.pbest_locations[0, :] = self.mode_location
        self.pbest_values[0] = y

    def evaluate(self, y):
        """
        Takes the value at the new location and updates the swarm statistics and history.

        Arguments
        ---------

        y : float
            fitness value at the new location.
        """
        if y > self.mode_value:
            self.update_location_and_value(self.new_location, y)
            self.changed = True

        self.history_locations[self.shifted_loc, :] = self.new_location
        self.history_values[self.shifted_loc] = y

        if y > self.pbest_values[self.shifted_loc]:
            self.pbest_values[self.shifted_loc] = y
            self.pbest_locations[self.shifted_loc, :] = self.new_location

    def find_nearest(self, swarms):
        """
        Finds the nearest swarm from the given set of swarms.

        Returns
        -------
        swarm
            The nearest swarm this this swarm.
        """
        best_swarm = None
        distance = np.inf

        for s in swarms:
            if self != s:
                d = np.sum((self.mode_location - s.mode_location) ** 2)
                if d < distance:
                    distance = d
                    best_swarm = s

        self.dist = np.sqrt(distance)  # track Euc distance to nearest neighbour

        return best_swarm, self.dist

    @staticmethod
    def uni(x1, x2):
        """
        Uniform binary crossover.

        Arguments
        ---------

        x1 : numpy array of parameters
        x2 : numpy array of parameters

        Returns:
        numpy array
            New array of parameters formed from uniform crossover.
        """

        # simulated binary crossover
        x_c = x1.copy()
        x_d = x2.copy()
        l = len(x1)
        r = np.flatnonzero(np.random.rand(l, 1) > 0.5)
        # ensure at least one is swapped
        if r.size == 0 or r.size == l:
            r = np.random.randint(l)

        x_c[r] = x2[r]
        x_d[r] = x1[r]

        return x_c, x_d
