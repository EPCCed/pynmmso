"""
    Testing module for the Nmmso class
"""

import math

import unittest
from unittest import mock
import numpy as np
import pynmmso.swarm as swarm
import pynmmso as nmmso
import swarm_tests

class ProblemThatReturnsNonScalar:
    def fitness(self, x):
        return np.array([1])

    @staticmethod
    def get_bounds():
        return [-1, -1], [1, 1]

class ProblemWithDifferentMinAndMaxDimensions:
    def fitness(self, x):
        return 1

    @staticmethod
    def get_bounds():
        return [-1, -1], [1, 1, 3]

class ProblemWithInvalidBounds:
    def fitness(self, x):
        return 1

    @staticmethod
    def get_bounds():
        return [0,10], [1,3]

class NmmsoTests(unittest.TestCase):

    def test_merge_swarms_two_swarms_merge(self):

        problem = swarm_tests.BasicProblem()

        my_nmmso = nmmso.Nmmso(problem, 10, 100, tol_val=0.2)
        my_nmmso.swarms = set()

        s1 = swarm.Swarm(1, 10, problem)
        s2 = swarm.Swarm(2, 10, problem)

        # give swarm 1 2 particles
        s1.mode_location = np.array([0.50, 0.60])
        s1.mode_value = sum(pow(s1.mode_location, 2))
        swarm_tests.SwarmTests.add_particle_to_swarm(s1, 0, [1.1, 1.2], 10.0, [1.3, 1.4], 11.0, [1.5, 1.6])
        swarm_tests.SwarmTests.add_particle_to_swarm(s1, 1, [2.1, 2.2], 20.0, [2.3, 2.4], 21.0, [2.5, 2.6])

        # give swarm 2 3 particles
        s2.mode_location = np.array([0.51, 0.61])
        s2.mode_value = sum(pow(s2.mode_location, 2))

        swarm_tests.SwarmTests.add_particle_to_swarm(s2, 0, [3.1, 3.2], 30.0, [3.3, 3.4], 31.0, [3.5, 3.6])
        swarm_tests.SwarmTests.add_particle_to_swarm(s2, 1, [4.1, 4.2], 40.0, [4.3, 4.4], 41.0, [4.5, 4.6])
        swarm_tests.SwarmTests.add_particle_to_swarm(s2, 2, [5.1, 5.2], 50.0, [5.3, 5.4], 51.0, [5.5, 5.6])

        my_nmmso.swarms.add(s1)
        my_nmmso.swarms.add(s2)

        my_nmmso._merge_swarms()

        # Merge to give only one active mode
        self.assertEqual(len(my_nmmso.swarms), 1)

        # That active mode should have 5 particles
        s = next(iter(my_nmmso.swarms))  # Just getting hold of the item in the set
        self.assertEqual(s.number_of_particles, 5)

        # No need to test the rest of the inside of the swarm there is a unit test to test that code in
        # more detail as part of the Swarm class unit tests.

    def test_merge_swarms_two_swarms_do_not_merge(self):

        problem = swarm_tests.BasicProblem()

        my_nmmso = nmmso.Nmmso(problem, 10, 100, tol_val=0.2)
        my_nmmso.swarms = set()

        s1 = swarm.Swarm(1, 10, problem)

        # give swarm 1 2 particles
        s1.mode_location = np.array([0.50, 0.60])
        swarm_tests.SwarmTests.add_particle_to_swarm(s1, 0, [1.1, 1.2], 10.0, [1.3, 1.4], 11.0, [1.5, 1.6])
        swarm_tests.SwarmTests.add_particle_to_swarm(s1, 1, [2.1, 2.2], 20.0, [2.3, 2.4], 21.0, [2.5, 2.6])

        my_nmmso.swarms.add(s1)

        # there's only 1 swarm, so merging shouldn't do anything
        my_nmmso._merge_swarms()

        self.assertEqual(len(my_nmmso.swarms), 1)  # there should still be only 1 swarm
        self.assertEqual(s1.number_of_particles, 2)  # nothing should have changed....

        # now generate a new swarm
        s2 = swarm.Swarm(2, 10, problem)

        # give swarm 2 3 particles
        s2.mode_location = np.array([0.51, 0.61])
        swarm_tests.SwarmTests.add_particle_to_swarm(s2, 0, [3.1, 3.2], 30.0, [3.3, 3.4], 31.0, [3.5, 3.6])
        swarm_tests.SwarmTests.add_particle_to_swarm(s2, 1, [4.1, 4.2], 40.0, [4.3, 4.4], 41.0, [4.5, 4.6])
        swarm_tests.SwarmTests.add_particle_to_swarm(s2, 2, [5.1, 5.2], 50.0, [5.3, 5.4], 51.0, [5.5, 5.6])

        my_nmmso.swarms.add(s2)

        # If neither swarm has changed, they shouldn't merge
        s1.changed = False
        s2.changed = False

        my_nmmso._merge_swarms()

        self.assertEqual(len(my_nmmso.swarms), 2)  # there should still be 2 swarms
        self.assertEqual(s1.number_of_particles, 2)  # nothing should have changed....
        self.assertEqual(s2.number_of_particles, 3)  # nothing should have changed....

    def test_merge_swarm_fitter_midpoint_midpoint_greater_swarm1_greater_swarm2(self):
        """Test merges when midpoint is greater than swarm 1 and which greater than swarm 2"""

        # Using our inverse fitness function
        # Swarm  1 [0.3, 0.3]     fitness = -0.18
        # Swarm  2 [-0.4, -0.4]   fitness = -0.32
        # midpoint will be: [-0.05,-0.05]  fitness = -0.005

        # To make the midpoint greater than the values of the swarms easier to use the inverse fitness
        problem = swarm_tests.BasicProblem(inverse=True)

        my_nmmso = nmmso.Nmmso(problem, 10, 100, tol_val=0.1)
        my_nmmso.swarms = set()

        s1 = swarm.Swarm(1, 10, problem)
        s2 = swarm.Swarm(2, 10, problem)

        s1.mode_location = np.array([-0.40, -0.40])
        s1.mode_value = -1.0 * (0.4*0.4 + 0.4*0.4)

        swarm_tests.SwarmTests.add_particle_to_swarm(s1, 0, [3.1, 3.2], 30.0, [3.3, 3.4], 31.0, [3.5, 3.6])
        swarm_tests.SwarmTests.add_particle_to_swarm(s1, 1, [4.1, 4.2], 40.0, [4.3, 4.4], 41.0, [4.5, 4.6])
        swarm_tests.SwarmTests.add_particle_to_swarm(s1, 2, [5.1, 5.2], 50.0, [5.3, 5.4], 51.0, [5.5, 5.6])

        s2.mode_location = np.array([0.30, 0.30])

        s2.mode_value = -1 * (0.3 * 0.3 + 0.30 * 0.30)

        # Don't really care about these values - just need to get them back in merged swarm
        swarm_tests.SwarmTests.add_particle_to_swarm(s2, 0, [-1.1, 1.2], 10.0, [-1.3, -1.4], 11.0, [-1.5, 1.6])
        swarm_tests.SwarmTests.add_particle_to_swarm(s2, 1, [-2.1, 2.2], 20.0, [-2.3, -2.4], 21.0, [-2.5, 2.6])

        my_nmmso.swarms.add(s1)
        my_nmmso.swarms.add(s2)

        my_nmmso._merge_swarms()
        self.assertEqual(len(my_nmmso.swarms), 1)

        # That active mode should have 5 particles
        s = next(iter(my_nmmso.swarms))  # Just getting hold of the item in the set
        self.assertEqual(s.number_of_particles, 5)

        # midpoint now swarm location
        midpoint = np.array([(0.3 - -0.4) / 2 + -0.4, (0.3 - -0.4) / 2 + -0.4])
        np.testing.assert_array_almost_equal(s.mode_location, midpoint)
        self.assertAlmostEqual(s.mode_value, -sum(pow(midpoint, 2)))

    def test_merge_swarm_fitter_midpoint_midpoint_greater_swarm2_greater_swarm1(self):
        """Test merges when midpoint is greater than swarm 1 and which greater than swarm 2"""

        # Using our inverse fitness function
        # Swarm  1 [-0.4, -0.4]   fitness = -0.32
        # Swarm  2 [0.3, 0.3]     fitness = -0.18
        # midpoint will be: [-0.05,-0.05]  fitness = -0.005

        # To make the midpoint greater than the values of the swarms easier to use the inverse fitness
        problem = swarm_tests.BasicProblem(inverse=True)

        my_nmmso = nmmso.Nmmso(problem, 10, 100, tol_val=0.1)
        my_nmmso.swarms = set()

        s1 = swarm.Swarm(1, 10, problem)
        s2 = swarm.Swarm(2, 10, problem)

        s1.mode_location = np.array([0.30, 0.30])

        s1.mode_value = -1 * (0.3 * 0.3 + 0.30 * 0.30)

        # Don't really care about these values - just need to get them back in merged swarm
        swarm_tests.SwarmTests.add_particle_to_swarm(s1, 0, [-1.1, 1.2], 10.0, [-1.3, -1.4], 11.0, [-1.5, 1.6])
        swarm_tests.SwarmTests.add_particle_to_swarm(s1, 1, [-2.1, 2.2], 20.0, [-2.3, -2.4], 21.0, [-2.5, 2.6])

        s2.mode_location = np.array([-0.40, -0.40])

        s2.mode_value = -1.0 * (0.4*0.4 + 0.4*0.4)

        swarm_tests.SwarmTests.add_particle_to_swarm(s2, 0, [3.1, 3.2], 30.0, [3.3, 3.4], 31.0, [3.5, 3.6])
        swarm_tests.SwarmTests.add_particle_to_swarm(s2, 1, [4.1, 4.2], 40.0, [4.3, 4.4], 41.0, [4.5, 4.6])
        swarm_tests.SwarmTests.add_particle_to_swarm(s2, 2, [5.1, 5.2], 50.0, [5.3, 5.4], 51.0, [5.5, 5.6])

        my_nmmso.swarms.add(s1)
        my_nmmso.swarms.add(s2)

        my_nmmso._merge_swarms()
        self.assertEqual(len(my_nmmso.swarms), 1)

        # That active mode should have 5 particles
        s = next(iter(my_nmmso.swarms))  # Just getting hold of the item in the set
        self.assertEqual(s.number_of_particles, 5)

        # midpoint now swarm location
        midpoint = np.array([(0.3 - -0.4) / 2 + -0.4, (0.3 - -0.4) / 2 + -0.4])
        np.testing.assert_array_almost_equal(s.mode_location, midpoint)
        self.assertAlmostEqual(s.mode_value, -sum(pow(midpoint, 2)))

    def test_merge_swarm_fitter_midpoint_swarm2_greater_midpoint_greater_swarm1(self):
        """Test merges when midpoint is greater than swarm 1 and which greater than swarm 2"""

        # Swarm  1 [0.1, 0.1]   fitness = 0.02
        # Swarm  2 [0.4, 0.4]   fitness = 0.32
        # midpoint will be: [0.25,0.25]  fitness = 0.125

        problem = swarm_tests.BasicProblem()

        my_nmmso = nmmso.Nmmso(problem, 10, 100, tol_val=0.1)
        my_nmmso.swarms = set()

        s1 = swarm.Swarm(1, 10, problem)
        s2 = swarm.Swarm(2, 10, problem)

        s1.mode_location = np.array([0.1, 0.1])

        s1.mode_value = 0.1 * 0.1 + 0.1 * 0.1

        # Don't really care about these values - just need to get them back in merged swarm
        swarm_tests.SwarmTests.add_particle_to_swarm(s1, 0, [-1.1, 1.2], 10.0, [-1.3, -1.4], 11.0, [-1.5, 1.6])
        swarm_tests.SwarmTests.add_particle_to_swarm(s1, 1, [-2.1, 2.2], 20.0, [-2.3, -2.4], 21.0, [-2.5, 2.6])

        s2.mode_location = np.array([0.40, 0.40])

        s2.mode_value = 0.4*0.4 + 0.4*0.4

        swarm_tests.SwarmTests.add_particle_to_swarm(s2, 0, [3.1, 3.2], 30.0, [3.3, 3.4], 31.0, [3.5, 3.6])
        swarm_tests.SwarmTests.add_particle_to_swarm(s2, 1, [4.1, 4.2], 40.0, [4.3, 4.4], 41.0, [4.5, 4.6])
        swarm_tests.SwarmTests.add_particle_to_swarm(s2, 2, [5.1, 5.2], 50.0, [5.3, 5.4], 51.0, [5.5, 5.6])

        my_nmmso.swarms.add(s1)
        my_nmmso.swarms.add(s2)

        my_nmmso._merge_swarms()
        self.assertEqual(len(my_nmmso.swarms), 1)

        # That active mode should have 5 particles
        s = next(iter(my_nmmso.swarms))  # Just getting hold of the item in the set
        self.assertEqual(s.number_of_particles, 5)

        #  swarm 2 does not move
        loc = np.array([0.4, 0.4])
        np.testing.assert_array_almost_equal(s.mode_location, loc)
        self.assertAlmostEqual(s.mode_value, sum(pow(loc, 2)))

    def test_merge_swarm_fitter_midpoint_swarm1_greater_midpoint_greater_swarm2(self):
        """Test merges when midpoint is greater than swarm 1 and which greater than swarm 2"""

        # Swarm  2 [0.1, 0.1]     fitness = 0.02
        # Swarm  1 [0.4, 0.4]     fitness = 0.32
        # midpoint will be: [0.25,0.25]  fitness = 0.125

        problem = swarm_tests.BasicProblem()

        my_nmmso = nmmso.Nmmso(problem, 10, 100, tol_val=0.1)
        my_nmmso.swarms = set()

        s1 = swarm.Swarm(1, 10, problem)
        s2 = swarm.Swarm(2, 10, problem)

        s1.mode_location = np.array([0.40, 0.40])

        s1.mode_value = 0.4 * 0.4 + 0.4 * 0.4

        swarm_tests.SwarmTests.add_particle_to_swarm(s1, 0, [3.1, 3.2], 30.0, [3.3, 3.4], 31.0, [3.5, 3.6])
        swarm_tests.SwarmTests.add_particle_to_swarm(s1, 1, [4.1, 4.2], 40.0, [4.3, 4.4], 41.0, [4.5, 4.6])
        swarm_tests.SwarmTests.add_particle_to_swarm(s1, 2, [5.1, 5.2], 50.0, [5.3, 5.4], 51.0, [5.5, 5.6])

        s2.mode_location = np.array([0.1, 0.1])

        s2.mode_value = 0.1 * 0.1 + 0.1 * 0.1

        # Don't really care about these values - just need to get them back in merged swarm
        swarm_tests.SwarmTests.add_particle_to_swarm(s2, 0, [-1.1, 1.2], 10.0, [-1.3, -1.4], 11.0, [-1.5, 1.6])
        swarm_tests.SwarmTests.add_particle_to_swarm(s2, 1, [-2.1, 2.2], 20.0, [-2.3, -2.4], 21.0, [-2.5, 2.6])

        my_nmmso.swarms.add(s1)
        my_nmmso.swarms.add(s2)

        my_nmmso._merge_swarms()
        self.assertEqual(len(my_nmmso.swarms), 1)

        # That active mode should have 5 particles
        s = next(iter(my_nmmso.swarms))  # Just getting hold of the item in the set
        self.assertEqual(s.number_of_particles, 5)

        #  swarm 1 does not move
        loc = np.array([0.4, 0.4])
        np.testing.assert_array_almost_equal(s.mode_location, loc)
        self.assertAlmostEqual(s.mode_value, sum(pow(loc, 2)))

    def test_uniform_sphere_points_in_two_dimensions(self):
        """Tests uniform_sphere_points to ensure all within the sphere."""
        num_samples = 10000
        res = nmmso.Nmmso.uniform_sphere_points(num_samples, 2)

        x = res[:, 0]
        y = res[:, 1]

        z = np.sqrt(x*x + y*y)

        # ensure no values outside the sphere
        values = np.where(z > 1)
        self.assertEqual(len(values[0]), 0)

        # Radius = 1, area of a circle is 3.14.  Radius = 0.9, area is 2.54.  Thus 1 - 2.54/3.14 = 0.191.  So
        # 19.1% of the data points should lie beyond the radius of 0.9.  We check between 17 and 21.

        values = np.where(z > 0.9)
        self.assertGreater(len(values[0]), 0.17*num_samples)
        self.assertLess(len(values[0]), 0.21*num_samples)

    def test_uniform_sphere_points_in_one_dimension(self):
        """Tests uniform_sphere_points to ensure all within the sphere."""

        res = nmmso.Nmmso.uniform_sphere_points(1000, 1)

        x = res[:, 0]

        z = np.sqrt(x*x)
        bad_values = np.where(z > 1)
        self.assertEqual(len(bad_values[0]), 0)

        # For 1 dimension not sure how things fit in the sphere so not trying to get the percentage

    def test_uniform_sphere_points_in_three_dimensions(self):
        """Tests uniform_sphere_points to ensure all within the sphere."""
        num_samples = 10000
        res = nmmso.Nmmso.uniform_sphere_points(num_samples, 3)

        x = res[:, 0]
        y = res[:, 1]
        w = res[:, 2]

        z = np.sqrt(x*x + y*y + w*w)

        # ensure no values outside the sphere
        values = np.where(z > 1)
        self.assertEqual(len(values[0]), 0)

        # Radius = 1, volume of sphere is 4.19.  Radius = 0.9, volume is 3.05.  Thus 1 - 3.05/4.19 = 0.272.  So
        # 27.2% of the data points should lie beyond the radius of 0.9.  We check between 25.2 and 29.2%

        values = np.where(z > 0.9)
        self.assertGreater(len(values[0]), 0.252*num_samples)
        self.assertLess(len(values[0]), 0.292*num_samples)

    @mock.patch('numpy.random.random_sample')
    def test_random_new(self, mock_random_sample):

        problem = swarm_tests.BasicProblem()

        my_nmmso = nmmso.Nmmso(problem, 10, 100, tol_val=0.2)
        my_nmmso.swarms = set()

        mock_random_sample.return_value = np.array([0.3, 0.9])

        my_nmmso._random_new()

        self.assertEqual(len(my_nmmso.swarms), 1)
        swarm = next(iter(my_nmmso.swarms))

        np.testing.assert_array_almost_equal(swarm.new_location, np.array([-0.4, 0.8]))
        self.assertEqual(swarm.converged, False)
        self.assertEqual(swarm.changed, True)
        self.assertAlmostEqual(swarm.mode_value, 0.4*0.4 + 0.8*0.8)

    @mock.patch('numpy.random.rand')
    @mock.patch('random.sample')
    def test_evolve_with_low_number_of_swarms(self, random_sample_mock, random_rand_mock):

        problem = swarm_tests.BasicProblem()

        my_nmmso = nmmso.Nmmso(problem, 10, tol_val=0.2, max_evol=10)
        my_nmmso.swarms = set()

        s1 = swarm.Swarm(1, 10, problem)
        s2 = swarm.Swarm(2, 10, problem)
        s3 = swarm.Swarm(3, 10, problem)

        # Mock call in test_evolve select swarms 1 and 3
        random_sample_mock.return_value = [s1, s3]

        # Mock call in Swarm.UNI to select to crossover 2nd element but not first
        random_rand_mock.return_value = np.array([0.1, 0.6])

        s1.mode_location = np.array([0.01, 0.11])
        s2.mode_location = np.array([0.02, 0.12])
        s3.mode_location = np.array([0.03, 0.13])

        my_nmmso._add_swarm(s1)
        my_nmmso._add_swarm(s2)
        my_nmmso._add_swarm(s3)

        my_nmmso._evolve()

        self.assertEqual(len(my_nmmso.swarms), 4)

        new_swarm = NmmsoTests.get_swarm_by_id(my_nmmso, 4)
        np.testing.assert_array_almost_equal(new_swarm.new_location, np.array([0.01, 0.13]))

    @mock.patch('random.random')
    @mock.patch('numpy.random.rand')
    def test_evolve_with_high_number_of_swarms_and_choose_to_select_fittest(self, random_rand_mock, random_random_mock):

        problem = swarm_tests.BasicProblem()

        my_nmmso = nmmso.Nmmso(problem, 10, tol_val=0.2, max_evol=2)
        my_nmmso.swarms = set()

        s1 = swarm.Swarm(1, 10, problem)
        s2 = swarm.Swarm(2, 10, problem)
        s3 = swarm.Swarm(3, 10, problem)

        s1.mode_location = np.array([0.01, 0.11])
        s1.mode_value = 10.0
        s2.mode_location = np.array([0.02, 0.12])
        s2.mode_value = 0.0
        s3.mode_location = np.array([0.03, 0.13])
        s3.mode_value = 20.0

        my_nmmso._add_swarm(s1)
        my_nmmso._add_swarm(s2)
        my_nmmso._add_swarm(s3)

        # Mock call to random.random to specify that the option of choosing the fittest is to be used
        random_random_mock.return_value = 0.2

        # Because max_evol is set to 3 and swarms 1 and 3 are the fittest they should always be the two selected
        # so no need to mock the random selection - the code has to choose 2 from 2.

        # Mock call in Swarm.UNI to select to crossover 2nd element but not first
        random_rand_mock.return_value = np.array([0.1, 0.6])

        my_nmmso._evolve()

        self.assertEqual(len(my_nmmso.swarms), 4)

        new_swarm = NmmsoTests.get_swarm_by_id(my_nmmso, 4)

        # The crossover could happen either way round so there are two options for the new location
        self.assertTrue(
            np.allclose(new_swarm.new_location, np.array([0.01, 0.13])) or
            np.allclose(new_swarm.new_location, np.array([0.03, 0.11])))

    def test_evaluate_new_locations(self):
        """Test for evaluate_new_locations"""
        problem = swarm_tests.BasicProblem()

        my_nmmso = nmmso.Nmmso(problem, 10, tol_val=0.2, max_evol=10)
        my_nmmso.swarms = set()

        # Create 4 swarms - 1 and 2 will have a new location whose value is better than the current best,
        # for swarms 3 and 4 this is not the case.  We will give the subset of only swarms 1 and 2 to
        # the evaluate_new_locations function.  This swarm 1 should be shifted by one of the other three
        # should be.
        s1 = swarm.Swarm(1, 10, problem)
        s2 = swarm.Swarm(2, 10, problem)
        s3 = swarm.Swarm(3, 10, problem)
        s4 = swarm.Swarm(4, 10, problem)

        s1.mode_location = np.array([0.01, 0.11])
        s1.new_location = np.array([0.50, 0.60])  # this is a location with a better value
        s1.mode_value = sum(s1.mode_location*s1.mode_location)
        s1.shifted_loc = 0

        s2.mode_location = np.array([0.02, 0.12])
        s2.new_location = np.array([0.01, 0.11])  # this is a location with a worse value
        s2.mode_value = sum(s2.mode_location*s2.mode_location)
        s2.shifted_loc = 0

        s3.mode_location = np.array([0.03, 0.13])
        s3.new_location = np.array([0.4, 0.6])  # this is a location with a better value
        s3.mode_value = sum(s3.mode_location*s3.mode_location)
        s3.shifted_loc = 0

        s4.mode_location = np.array([0.04, 0.14])
        s4.new_location = np.array([0.01, 0.01])  # this is a location with a worse value
        s4.mode_value = sum(s4.mode_location*s4.mode_location)
        s4.shifted_loc = 0

        my_nmmso._add_swarm(s1)
        my_nmmso._add_swarm(s2)
        my_nmmso._add_swarm(s3)
        my_nmmso._add_swarm(s4)

        number_of_new_locations = my_nmmso._evaluate_new_locations(set([s1, s2]))

        # Test the changed flag
        self.assertEqual(s1.changed, True)
        self.assertEqual(s2.changed, False)
        self.assertEqual(s3.changed, False)
        self.assertEqual(s4.changed, False)

        # Test that locations have been updated where appropriate
        np.testing.assert_allclose(s1.mode_location, np.array([0.50, 0.60]))  # only that that should have been updated
        np.testing.assert_allclose(s2.mode_location, np.array([0.02, 0.12]))
        np.testing.assert_allclose(s3.mode_location, np.array([0.03, 0.13]))
        np.testing.assert_allclose(s4.mode_location, np.array([0.04, 0.14]))

        # Check the mode values are correct
        self.assertEqual(s1.mode_value, sum(s1.mode_location * s1.mode_location))
        self.assertEqual(s2.mode_value, sum(s2.mode_location * s2.mode_location))
        self.assertEqual(s3.mode_value, sum(s3.mode_location * s3.mode_location))
        self.assertEqual(s4.mode_value, sum(s4.mode_location * s4.mode_location))

        self.assertEqual(number_of_new_locations, 2)  # 2 swarms were in the evaluate call

    @mock.patch('pynmmso.Nmmso.uniform_sphere_points')
    @mock.patch('random.randrange')
    def test_hive_when_midpoint_in_valley(self, mock_randrange, mock_usp):

        seed_particle_location = np.array([0.31, 0.32])
        seed_particle_value = sum(np.power(seed_particle_location, 2))

        problem = swarm_tests.BasicProblem()

        s1 = swarm.Swarm(1, 4, problem)
        s2 = swarm.Swarm(2, 4, problem)

        s2.number_of_particles = 3  # fewer than max swarm size

        s1.number_of_particles = 4
        s1.history_locations[0, :] = np.array([0.01, 0.02])
        s1.history_locations[1, :] = np.array([0.11, 0.12])
        s1.history_locations[2, :] = np.array([0.21, 0.22])
        s1.history_locations[3, :] = seed_particle_location

        for i in range(4):
            s1.history_values[i] = sum(np.power(s1.history_locations[i, :], 2))

        s1.mode_location = np.array([-0.4, -0.4])
        s1.mode_value = sum(np.power(s1.mode_location, 2))

        my_nmmso = nmmso.Nmmso(problem, 4, tol_val=0.2, max_evol=10)
        my_nmmso.swarms = set()

        my_nmmso._add_swarm(s1)
        my_nmmso._add_swarm(s2)

        mock_randrange.return_value = 3
        mock_usp.return_value = [np.array([0.02, 0.03])]

        number_of_new_samples = my_nmmso._hive()

        self.assertEqual(len(my_nmmso.swarms), 3)

        new_swarm = NmmsoTests.get_swarm_by_id(my_nmmso, 3)

        np.testing.assert_allclose(new_swarm.mode_location, seed_particle_location)
        self.assertAlmostEqual(new_swarm.mode_value, seed_particle_value)

        self.assertEqual(new_swarm.number_of_particles, 1)

        np.testing.assert_allclose(new_swarm.history_locations[0, :], seed_particle_location)
        np.testing.assert_allclose(new_swarm.pbest_locations[0, :], seed_particle_location)

        self.assertAlmostEqual(new_swarm.pbest_values[0], seed_particle_value)
        self.assertAlmostEqual(new_swarm.history_values[0], seed_particle_value)

        self.assertEqual(new_swarm.changed, True)
        self.assertEqual(new_swarm.converged, False)
        self.assertEqual(number_of_new_samples, 1)

        original_swarm = NmmsoTests.get_swarm_by_id(my_nmmso, 1)

        mid_loc_x = 0.5*(-0.4-0.31)+0.31
        mid_loc_y = 0.5*(-0.4-0.32)+0.32
        mid_loc = np.array([mid_loc_x, mid_loc_y])
        mid_loc_val = sum(np.power(mid_loc, 2))

        np.testing.assert_allclose(original_swarm.history_locations[3, :], mid_loc)
        self.assertAlmostEqual(original_swarm.history_values[3], mid_loc_val)
        np.testing.assert_allclose(original_swarm.pbest_locations[3, :], mid_loc)
        self.assertAlmostEqual(original_swarm.pbest_values[3], mid_loc_val)

        d = math.sqrt(sum(np.power(s1.mode_location-seed_particle_location, 2)))
        vel_x = -0.4 + (0.02 * d/2.0)
        vel_y = -0.4 + (0.03 * d/2.0)

        vel = np.array([vel_x, vel_y])

        np.testing.assert_allclose(original_swarm.velocities[3, :], vel)

    @mock.patch('numpy.random.random')
    @mock.patch('pynmmso.Nmmso.uniform_sphere_points')
    @mock.patch('random.randrange')
    def test_hive_when_midpoint_in_valley_velocity_rejected(self, mock_randrange, mock_usp, mock_random_random):

        seed_particle_location = np.array([0.31, 0.32])
        seed_particle_value = sum(np.power(seed_particle_location, 2))

        problem = swarm_tests.BasicProblem()

        s1 = swarm.Swarm(1, 4, problem)
        s2 = swarm.Swarm(2, 4, problem)

        s2.number_of_particles = 3  # fewer than max swarm size

        s1.number_of_particles = 4
        s1.history_locations[0, :] = np.array([0.01, 0.02])
        s1.history_locations[1, :] = np.array([0.11, 0.12])
        s1.history_locations[2, :] = np.array([0.21, 0.22])
        s1.history_locations[3, :] = seed_particle_location

        for i in range(4):
            s1.history_values[i] = sum(np.power(s1.history_locations[i, :], 2))

        s1.mode_location = np.array([-0.4, -0.4])
        s1.mode_value = sum(np.power(s1.mode_location, 2))

        my_nmmso = nmmso.Nmmso(problem, 4, tol_val=0.2, max_evol=10)
        my_nmmso.swarms = set()

        my_nmmso._add_swarm(s1)
        my_nmmso._add_swarm(s2)

        mock_randrange.return_value = 3
        mock_usp.return_value = [np.array([20, 20])]  # huge numbers so will be out of bounds and rejected
        mock_random_random.return_value = np.array([0.33, 0.44])
        number_of_new_samples = my_nmmso._hive()

        self.assertEqual(len(my_nmmso.swarms), 3)

        new_swarm = NmmsoTests.get_swarm_by_id(my_nmmso, 3)

        np.testing.assert_allclose(new_swarm.mode_location, seed_particle_location)
        self.assertAlmostEqual(new_swarm.mode_value, seed_particle_value)

        self.assertEqual(new_swarm.number_of_particles, 1)

        np.testing.assert_allclose(new_swarm.history_locations[0, :], seed_particle_location)
        np.testing.assert_allclose(new_swarm.pbest_locations[0, :], seed_particle_location)

        self.assertAlmostEqual(new_swarm.pbest_values[0], seed_particle_value)
        self.assertAlmostEqual(new_swarm.history_values[0], seed_particle_value)

        self.assertEqual(new_swarm.changed, True)
        self.assertEqual(new_swarm.converged, False)
        self.assertEqual(number_of_new_samples, 1)

        original_swarm = NmmsoTests.get_swarm_by_id(my_nmmso, 1)

        mid_loc_x = 0.5*(-0.4-0.31)+0.31
        mid_loc_y = 0.5*(-0.4-0.32)+0.32
        mid_loc = np.array([mid_loc_x, mid_loc_y])
        mid_loc_val = sum(np.power(mid_loc, 2))

        np.testing.assert_allclose(original_swarm.history_locations[3, :], mid_loc)
        self.assertAlmostEqual(original_swarm.history_values[3], mid_loc_val)
        np.testing.assert_allclose(original_swarm.pbest_locations[3, :], mid_loc)
        self.assertAlmostEqual(original_swarm.pbest_values[3], mid_loc_val)

        # d = math.sqrt(sum(np.power(s1.mode_location-seed_particle_location, 2)))
        vel_x = 0.33 * 2 - 1
        vel_y = 0.44 * 2 - 1

        vel = np.array([vel_x, vel_y])

        np.testing.assert_allclose(original_swarm.velocities[3, :], vel)

    @staticmethod
    def get_swarm_by_id(my_nmmso, id):
        """Gets the swam object with the given id."""
        return next((filter(lambda x: x.id == id, my_nmmso.swarms)))

    @mock.patch('random.randrange')
    def test_hive_when_midpoint_not_valley(self, mock_randrange):

        problem = swarm_tests.BasicProblem()

        s1 = swarm.Swarm(1, 4, problem)
        s2 = swarm.Swarm(2, 4, problem)

        s2.number_of_particles = 3  # fewer than max swarm size

        s1.number_of_particles = 4
        s1.history_locations[0, :] = np.array([0.01, 0.02])
        s1.history_locations[1, :] = np.array([0.11, 0.12])
        s1.history_locations[2, :] = np.array([0.21, 0.22])
        s1.history_locations[3, :] = np.array([0.31, 0.32])

        for i in range(4):
            s1.history_values[i] = sum(np.power(s1.history_locations[i, :], 2))

        s1.mode_location = np.array([-0.4, -0.4])
        s1.mode_value = sum(np.power(s1.mode_location, 2))

        my_nmmso = nmmso.Nmmso(problem, 4, tol_val=0.2, max_evol=10)
        my_nmmso.swarms = set()

        my_nmmso._add_swarm(s1)
        my_nmmso._add_swarm(s2)

        mock_randrange.return_value = 0

        number_of_new_samples = my_nmmso._hive()

        self.assertEqual(number_of_new_samples, 1)

        original_swarm = NmmsoTests.get_swarm_by_id(my_nmmso, 1)

        # Nothing should have changed
        self.assertEqual(len(my_nmmso.swarms), 2)
        np.testing.assert_allclose(original_swarm.mode_location, np.array([-0.4, -0.4]))
        np.testing.assert_allclose(original_swarm.history_locations[0, :], np.array([0.01, 0.02]))

    @mock.patch('random.randrange')
    def test_hive_when_midpoint_is_maximum(self, mock_randrange):

        problem = swarm_tests.BasicProblem(inverse=True)

        s1 = swarm.Swarm(1, 4, problem)
        s2 = swarm.Swarm(2, 4, problem)

        s2.number_of_particles = 3  # fewer than max swarm size

        s1.number_of_particles = 4
        s1.history_locations[0, :] = np.array([0.01, 0.02])
        s1.history_locations[1, :] = np.array([0.11, 0.12])
        s1.history_locations[2, :] = np.array([0.21, 0.22])
        s1.history_locations[3, :] = np.array([0.31, 0.32])

        for i in range(4):
            s1.history_values[i] = -sum(np.power(s1.history_locations[i, :], 2))

        s1.mode_location = np.array([-0.4, -0.4])
        s1.mode_value = -sum(np.power(s1.mode_location, 2))

        my_nmmso = nmmso.Nmmso(problem, 4, tol_val=0.2, max_evol=10)
        my_nmmso.swarms = set()

        my_nmmso._add_swarm(s1)
        my_nmmso._add_swarm(s2)

        mock_randrange.return_value = 3

        number_of_new_samples = my_nmmso._hive()

        self.assertEqual(number_of_new_samples, 1)

        original_swarm = NmmsoTests.get_swarm_by_id(my_nmmso, 1)

        # Should have stored the mid point as the now mode location and value, no other changes
        self.assertEqual(len(my_nmmso.swarms), 2)

        mid_loc_x = 0.5*(-0.4-0.31)+0.31
        mid_loc_y = 0.5*(-0.4-0.32)+0.32
        mid_loc = np.array([mid_loc_x, mid_loc_y])
        mid_loc_val = -sum(np.power(mid_loc, 2))
        np.testing.assert_allclose(original_swarm.mode_location, mid_loc)
        self.assertAlmostEqual(original_swarm.mode_value, mid_loc_val)
        np.testing.assert_allclose(original_swarm.history_locations[0, :], np.array([0.01, 0.02]))

    def test_swarm_tuples_in_a_set(self):
        """Tests that tuples created by create_swarm_tuple work as expected in a set"""
        problem = swarm_tests.BasicProblem()

        s1 = swarm.Swarm(1, 4, problem)
        s2 = swarm.Swarm(2, 4, problem)

        s = set()

        t1 = nmmso.Nmmso._create_swarm_tuple(s1, s2)
        t2 = nmmso.Nmmso._create_swarm_tuple(s1, s2)

        s.add(t1)
        s.add(t2)

        self.assertEqual(len(s),1)

    def test_error_if_fitness_function_returns_non_scalar(self):
        my_nmmso = nmmso.Nmmso(ProblemThatReturnsNonScalar())
        with self.assertRaises(ValueError):
            my_nmmso.run(5)

    def test_error_if_min_and_max_different_dimensions(self):
        with self.assertRaises(ValueError):
            my_nmmso = nmmso.Nmmso(ProblemWithDifferentMinAndMaxDimensions())

    def test_error_if_min_bound_is_larger_than_max(self):
        with self.assertRaises(ValueError):
            my_nmmso = nmmso.Nmmso(ProblemWithInvalidBounds())





