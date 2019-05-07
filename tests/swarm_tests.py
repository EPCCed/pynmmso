"""
    Testing module for the Swarm class
"""

import unittest
from unittest import mock
import numpy as np
import nmmso.swarm as swarm
import math


class BasicProblem:

    def __init__(self, inverse=False):
        self.inverse = inverse

    def fitness(self, x):
        res = sum(x * x)
        if self.inverse:
            res = -res
        return res

    @staticmethod
    def get_bounds():
        return [-1, -1], [1, 1]

class BasicProblemWithRangeExcludingZero:

    def __init__(self, inverse=False):
        self.inverse = inverse

    def fitness(self, x):
        res = sum(x * x)
        if self.inverse:
            res = -res
        return res

    @staticmethod
    def get_bounds():
        return [2, 2], [4, 4]


class SwarmTests(unittest.TestCase):

    def setUp(self):
        pass

    def test_distance_to(self):
        """Tests distance_to method"""
        s1 = swarm.Swarm(1, 10, BasicProblem())
        s1.mode_location = np.array([0.2, 0.5])
        s2 = swarm.Swarm(2, 10, BasicProblem())
        s2.mode_location = np.array([0.1, 0.9])

        self.assertAlmostEqual(s1.distance_to(s2), np.sqrt(0.1**2 + 0.4**2))

    @staticmethod
    def add_particle_to_swarm(swarm, index, history_location, history_value, pbest_location, pbest_value, velocity):
        swarm.number_of_particles = index+1
        swarm.history_locations[index, :] = np.array(history_location)
        swarm.history_values[index] = history_value
        swarm.pbest_locations[index, :] = np.array(pbest_location)
        swarm.pbest_values[index] = pbest_value
        swarm.velocities[index, :] = velocity

    def test_merge_simply_way(self):
        """Test the merge function when both swarms' particles can be kept."""

        s1 = swarm.Swarm(1, 10, BasicProblem())
        s2 = swarm.Swarm(2, 10, BasicProblem())

        # give swarm 1 2 particles
        s1.mode_location = np.array([0, 1])
        SwarmTests.add_particle_to_swarm(s1, 0, [1.1, 1.2], 10.0, [1.3, 1.4], 11.0, [1.5, 1.6])
        SwarmTests.add_particle_to_swarm(s1, 1, [2.1, 2.2], 20.0, [2.3, 2.4], 21.0, [2.5, 2.6])

        # give swarm 2 3 particles
        SwarmTests.add_particle_to_swarm(s2, 0, [3.1, 3.2], 30.0, [3.3, 3.4], 31.0, [3.5, 3.6])
        SwarmTests.add_particle_to_swarm(s2, 1, [4.1, 4.2], 40.0, [4.3, 4.4], 41.0, [4.5, 4.6])
        SwarmTests.add_particle_to_swarm(s2, 2, [5.1, 5.2], 50.0, [5.3, 5.4], 51.0, [5.5, 5.6])

        # merge
        s1.merge(s2)

        # result should have 5 particles
        self.assertEqual(s1.number_of_particles, 5)

        for i in range(5):
            np.testing.assert_array_almost_equal(s1.history_locations[i, :], np.array([i+1.1, i+1.2]))
            self.assertAlmostEqual(s1.history_values[i], (i+1)*10.0)
            np.testing.assert_array_almost_equal(s1.pbest_locations[i, :], np.array([i+1.3, i+1.4]))
            self.assertAlmostEqual(s1.pbest_values[i], (i+1)*10.0+1.0)
            np.testing.assert_array_almost_equal(s1.velocities[i, :], np.array([i+1.5, i+1.6]))

    def test_merge_that_requires_combining(self):
        """Tests the merge function when not all particles can be kept."""
        """Test the merge function when both swarms' particles can be kept."""

        s1 = swarm.Swarm(1, 3, BasicProblem())
        s2 = swarm.Swarm(2, 3, BasicProblem())

        # give swarm 1 2 particles
        SwarmTests.add_particle_to_swarm(s1, 0, [1.1, 1.2], 10.0, [1.3, 1.4], 11.0, [1.5, 1.6])
        SwarmTests.add_particle_to_swarm(s1, 1, [3.1, 3.2], 30.0, [3.3, 3.4], 31.0, [3.5, 3.6])

        # give swarm 2 3 particles
        SwarmTests.add_particle_to_swarm(s2, 0, [5.1, 5.2], 50.0, [5.3, 5.4], 51.0, [5.5, 5.6])
        SwarmTests.add_particle_to_swarm(s2, 1, [4.1, 4.2], 40.0, [4.3, 4.4], 41.0, [4.5, 4.6])
        SwarmTests.add_particle_to_swarm(s2, 2, [2.1, 2.2], 20.0, [2.3, 2.4], 21.0, [2.5, 2.6])

        # merge
        s1.merge(s2)

        # result should have 5 particles
        self.assertEqual(s1.number_of_particles, 3)

        for i in range(3):
            ii = i+2
            np.testing.assert_array_almost_equal(s1.history_locations[i, :], np.array([ii+1.1, ii+1.2]))
            self.assertAlmostEqual(s1.history_values[i], (ii+1)*10.0)
            np.testing.assert_array_almost_equal(s1.pbest_locations[i, :], np.array([ii+1.3, ii+1.4]))
            self.assertAlmostEqual(s1.pbest_values[i], (ii+1)*10.0+1.0)
            np.testing.assert_array_almost_equal(s1.velocities[i, :], np.array([ii+1.5, ii+1.6]))

    @mock.patch('nmmso.Nmmso.uniform_sphere_points')
    def test_initialise_new_swarm_velocities_without_rejection(self, mock_uniform_sphere_points):
        """Tests initialise_new_swarm function where the uniform sphere approach works."""
        mock_uniform_sphere_points.return_value = [np.array([0.5,0.6])]

        s = swarm.Swarm(1, 10, BasicProblem())
        s.mode_location = np.array([0.01, -0.02])
        s.dist = 0.5

        s.initialise_new_swarm_velocities()

        np.testing.assert_array_almost_equal(s.velocities[0, :], np.array([0.135, 0.13]))

    @mock.patch('numpy.random.rand')
    @mock.patch('nmmso.Nmmso.uniform_sphere_points')
    def test_initialise_new_swarm_velocities_with_rejection(self, mock_uniform_sphere_points, mock_random_rand):
        """Tests initialise_new_swarm code to ensure the rejections occur and the second approach is used."""
        mock_random_rand.return_value = np.array([0.9, 0.2])
        mock_uniform_sphere_points.return_value = np.array([0.5, 0.6])

        s = swarm.Swarm(1, 10, BasicProblem())
        s.mode_location = np.array([0.01, -0.02])
        s.dist = 10.0

        s.initialise_new_swarm_velocities()

        np.testing.assert_array_almost_equal(s.velocities[0, :], np.array([0.8, -0.6]))

    def test_evaluate_first(self):
        """Tests evaluate_first"""
        s = swarm.Swarm(1, 10, BasicProblem())

        new_location = np.array([0.4, 0.4])
        y = 0.4*0.4*2
        s.new_location = new_location

        s.evaluate_first()

        self.assertAlmostEqual(s.mode_value, y)
        np.testing.assert_array_almost_equal(s.mode_location, new_location)

        np.testing.assert_array_almost_equal(s.history_locations[0, :], new_location)
        self.assertAlmostEqual(s.history_values[0], y)

        np.testing.assert_array_almost_equal(s.pbest_locations[0, :], new_location)
        self.assertAlmostEqual(s.pbest_values[0], y)

    def test_evaluate_when_new_value_is_best(self):
        """Tests the basic evaluate method when the new value is greater than the current best."""
        s = swarm.Swarm(1, 10, BasicProblem())
        s.shifted_loc = 1
        s.mode_value = 0

        new_location = np.array([0.4, 0.4])
        y = 0.4*0.4*2
        s.new_location = new_location

        result_mode_shift, result_y = s.evaluate(y)

        self.assertEqual(result_mode_shift, 1)
        self.assertEqual(result_y, y)
        self.assertAlmostEqual(s.mode_value, y)
        np.testing.assert_array_almost_equal(s.mode_location, new_location)

        np.testing.assert_array_almost_equal(s.history_locations[1, :], new_location)
        self.assertAlmostEqual(s.history_values[1], y)

        np.testing.assert_array_almost_equal(s.pbest_locations[1, :], new_location)
        self.assertAlmostEqual(s.pbest_values[1], y)

    def test_evaluate_when_new_value_is_not_best(self):
        """Tests the basic evaluate method when the new value is not greater than the current best."""
        s = swarm.Swarm(1, 10, BasicProblem())

        original_mode_location = np.array([0.9, 0.8])
        original_y = 0.9*0.9 + 0.8*0.8

        s.shifted_loc = 1
        s.mode_location = original_mode_location
        s.mode_value = original_y
        s.pbest_locations[1, :] = original_mode_location
        s.pbest_values[1] = original_y

        new_location = np.array([0.4, 0.4])
        y = 0.4*0.4*2
        s.new_location = new_location

        result_mode_shift, result_y = s.evaluate(y)

        self.assertEqual(result_mode_shift, 0)
        self.assertEqual(result_y, y)
        self.assertAlmostEqual(s.mode_value, original_y)
        np.testing.assert_array_almost_equal(s.mode_location, original_mode_location)

        np.testing.assert_array_almost_equal(s.history_locations[1, :], new_location)
        self.assertAlmostEqual(s.history_values[1], y)

        np.testing.assert_array_almost_equal(s.pbest_locations[1, :], original_mode_location)
        self.assertAlmostEqual(s.pbest_values[1], original_y)

    def test_find_nearest(self):

        swarms = set()

        swarm1 = swarm.Swarm(1, 10, BasicProblem())
        swarm1.mode_location = np.array([0, 0])
        swarms.add(swarm1)
        swarm2 = swarm.Swarm(1, 10, BasicProblem())
        swarm2.mode_location = np.array([0.1, 0.2])
        swarms.add(swarm2)
        swarm3 = swarm.Swarm(1, 10, BasicProblem())
        swarm3.mode_location = np.array([-0.2, -0.5])
        swarms.add(swarm3)
        swarm4 = swarm.Swarm(1, 10, BasicProblem())
        swarm4.mode_location = np.array([-0.09, 0.05])
        swarms.add(swarm4)

        best_swarm, distance = swarm1.find_nearest(swarms)

        expected_distance = math.sqrt(-0.09*-0.09 + 0.05*0.05)

        # Swarm 4 is the nearest
        self.assertEqual(best_swarm, swarm4)
        self.assertAlmostEqual(distance, expected_distance, msg="Unexpected distance returned")

        # Should update the internal store of the distance
        self.assertAlmostEqual(swarm1.dist, expected_distance, msg="Unexpected distance stored in class")

    @mock.patch('numpy.random.rand')
    def test_uni_basic(self, mock_rand):
        """Test the basic operation of the uni method."""

        mock_rand.return_value = np.array([0.3, 0.6, 0.2, 0.55, 0.9])
        params1 = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
        params2 = np.array([2.1, 2.2, 2.3, 2.4, 2.5])

        out1, out2 = swarm.Swarm.uni(params1, params2)

        np.testing.assert_array_almost_equal(out1, np.array([1.1, 2.2, 1.3, 2.4, 2.5]))
        np.testing.assert_array_almost_equal(out2, np.array([2.1, 1.2, 2.3, 1.4, 1.5]))

    @mock.patch('numpy.random.rand')
    @mock.patch('numpy.random.randint')
    def test_uni_none_selected_for_crossover(self, mock_randint, mock_rand):
        """Tests the uni method when the initial random choice selects no crossover"""

        # Produce random numbers such that nothing is selected
        mock_rand.return_value = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        # When selecting a random index then return 2
        mock_randint.return_value = 2

        params1 = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
        params2 = np.array([2.1, 2.2, 2.3, 2.4, 2.5])

        out1, out2 = swarm.Swarm.uni(params1, params2)

        np.testing.assert_array_almost_equal(out1, np.array([1.1, 1.2, 2.3, 1.4, 1.5]))
        np.testing.assert_array_almost_equal(out2, np.array([2.1, 2.2, 1.3, 2.4, 2.5]))

    @mock.patch('numpy.random.rand')
    @mock.patch('numpy.random.randint')
    def test_uni_all_selected_for_crossover(self, mock_randint, mock_rand):
        """Tests the uni method when the initial random choice selects everything for crossover"""

        # Produce random numbers such that everything is selected
        mock_rand.return_value = np.array([0.7, 0.7, 0.7, 0.7, 0.7])
        # When selecting a random index then return 2
        mock_randint.return_value = 2

        params1 = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
        params2 = np.array([2.1, 2.2, 2.3, 2.4, 2.5])

        out1, out2 = swarm.Swarm.uni(params1, params2)

        np.testing.assert_array_almost_equal(out1, np.array([1.1, 1.2, 2.3, 1.4, 1.5]))
        np.testing.assert_array_almost_equal(out2, np.array([2.1, 2.2, 1.3, 2.4, 2.5]))

    @mock.patch('numpy.random.rand')
    def test_initialise_with_uniform_crossover(self, mock_rand):
        """Tests initialise_with_uniform_crossover"""
        swarm1 = swarm.Swarm(1, 10, BasicProblem())
        swarm1.mode_location = np.array([0.0, 0.0])
        swarm2 = swarm.Swarm(2, 10, BasicProblem())
        swarm2.mode_location = np.array([0.1, 0.2])

        mock_rand.return_value = np.array([0.6, 0.1])
        s = swarm.Swarm(3, 10, BasicProblem())

        s.initialise_with_uniform_crossover(swarm1, swarm2)

        np.testing.assert_array_almost_equal(s.new_location, [0.1, 0])
        self.assertAlmostEqual(s.mode_value, 0.1*0.1)
        self.assertEqual(s.converged, False)
        self.assertEqual(s.changed, True)

    @mock.patch('nmmso.Nmmso.uniform_sphere_points')
    def test_increment_when_add_new_particle(self, mock_uniform_sphere_points):
        """Tests increment method when a new particle is added."""
        my_swarm = swarm.Swarm(1, 10, BasicProblem())

        my_swarm.dist = 0.5
        my_swarm.number_of_particles = 1
        my_swarm.mode_location = np.array([0.1, 0.2])

        # We mock two calls to uniform sphere points, first to compute the new location, then to compute
        # the velocity
        mock_uniform_sphere_points.side_effect = [[np.array([0.3, 0.4])], [np.array([0.2, 0.1])]]

        my_swarm.increment()

        self.assertEqual(my_swarm.number_of_particles, 2)
        self.assertEqual(my_swarm.shifted_loc, 1)

        np.testing.assert_array_almost_equal(my_swarm.new_location, np.array([0.1+0.3*0.5/2.0, 0.2+0.4*0.5/2.0]))
        np.testing.assert_array_almost_equal(my_swarm.velocities[1, :], np.array([0.1+0.2*0.5/2.0, 0.2+0.1*0.5/2.0]))

    @mock.patch('numpy.random.rand')
    @mock.patch('nmmso.Nmmso.uniform_sphere_points')
    def test_increment_when_add_new_particle_and_initial_velocities_are_rejected(self, mock_uniform_sphere_points, mock_random_rand):
        """Tests increment method when a new particle is added and initial velocities are rejected."""
        my_swarm = swarm.Swarm(1, 10, BasicProblem())

        my_swarm.dist = 0.5
        my_swarm.number_of_particles = 1
        my_swarm.mode_location = np.array([0.1, 0.2])

        # We mock two calls to uniform sphere points, first to compute the new location, then to compute
        # the velocity
        side_effect = list()
        side_effect.append([np.array([0.3, 0.4])])  # for new location
        for i in range(22):
            side_effect.append([np.array([300, 400])])  # for velocity
        mock_uniform_sphere_points.side_effect = side_effect

        mock_random_rand.return_value = np.array([0.22, 0.33])

        my_swarm.increment()

        self.assertEqual(my_swarm.number_of_particles, 2)
        self.assertEqual(my_swarm.shifted_loc, 1)

        np.testing.assert_array_almost_equal(my_swarm.new_location, np.array([0.1+0.3*0.5/2.0, 0.2+0.4*0.5/2.0]))
        np.testing.assert_array_almost_equal(my_swarm.velocities[1, :], np.array([0.22*2.0-1, 0.33*2.0-1]))

    @mock.patch('numpy.random.rand')
    @mock.patch('random.randrange')
    def test_increment_when_move_an_existing_particle(self, mock_random_range, mock_random_rand):
        """Tests increment method when a existing particle is moved"""
        my_swarm = swarm.Swarm(1, 4, BasicProblem())

        mode_loc = np.array([0.1, 0.2])
        hist_loc = np.array([0.3, 0.4])
        pbest_loc = np.array([0.22, 0.23])
        vel = np.array([0.01, 0.02])

        my_swarm.dist = 0.5
        my_swarm.number_of_particles = 4
        my_swarm.mode_location = mode_loc
        my_swarm.history_locations[2, :] = hist_loc
        my_swarm.pbest_locations[2, :] = pbest_loc
        my_swarm.velocities[2, :] = vel

        r1 = np.array([0.03, 0.04])
        r2 = np.array([0.01, 0.02])

        # We mock the choice of which particle to move
        mock_random_range.return_value = 2

        # Mock the two calls to np.random.rand
        mock_random_rand.side_effect = [r1, r2]

        my_swarm.increment()

        self.assertEqual(my_swarm.number_of_particles, 4)
        self.assertEqual(my_swarm.shifted_loc, 2)

        temp_vel = 0.1 * vel + 2.0 * r1 * (mode_loc - hist_loc) + 2.0 * r2 * (pbest_loc - hist_loc)
        new_loc = hist_loc + temp_vel

        np.testing.assert_array_almost_equal(my_swarm.new_location, new_loc)
        np.testing.assert_array_almost_equal(my_swarm.velocities[2, :], temp_vel)

    @mock.patch('numpy.random.rand')
    @mock.patch('random.randrange')
    def test_increment_when_move_an_existing_particle_and_locations_rejected(self, mock_random_range, mock_random_rand):
        """Tests increment method when a existing particle is moved and locations are rejected"""
        my_swarm = swarm.Swarm(1, 4, BasicProblem())

        mode_loc = np.array([0.1, 0.2])
        hist_loc = np.array([0.3, 0.4])
        pbest_loc = np.array([0.22, 0.23])
        vel = np.array([0.01, 0.02])

        my_swarm.dist = 0.5
        my_swarm.number_of_particles = 4
        my_swarm.mode_location = mode_loc
        my_swarm.history_locations[2, :] = hist_loc
        my_swarm.pbest_locations[2, :] = pbest_loc
        my_swarm.velocities[2, :] = vel

        r1 = np.array([300, 0.04])
        r2 = np.array([0.01, -200])

        side_effect = []
        for x in range(22):
            side_effect.append(r1)
            side_effect.append(r2)
        side_effect.append(np.array([0.25]))
        side_effect.append(np.array([0.50]))

        # We mock the choice of which particle to move
        mock_random_range.return_value = 2

        # Mock the two calls to np.random.rand
        mock_random_rand.side_effect = side_effect

        my_swarm.increment()

        self.assertEqual(my_swarm.number_of_particles, 4)
        self.assertEqual(my_swarm.shifted_loc, 2)

        temp_vel = np.array([0.5 * (0.3 - (-1.0)), 0.25 * (1.0 - 0.4)])
        new_loc = hist_loc + temp_vel

        np.testing.assert_array_almost_equal(my_swarm.new_location, new_loc)
        np.testing.assert_array_almost_equal(my_swarm.velocities[2, :], temp_vel)


    def test_set_initial_location(self):
        """Tests that the initial location is inside the range when the range does not include zero."""
        my_swarm = swarm.Swarm(1, 4, BasicProblemWithRangeExcludingZero())

        # ensure the new location is always in bounds
        for i in range(1000):
            my_swarm.set_initial_location()
            if sum(my_swarm.new_location < [2,2]) + sum(my_swarm.new_location > [4, 4]) > 0:
                self.fail("Initial location was out of bounds")
