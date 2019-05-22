# Niching Migratory Multi-Swarm Optimser for Python (pynmmso)

Python implementation of the Niching Migratory Multi-Swarm Optimser, described
in: "*Running Up Those Hills: Multi-Modal Search with the Niching Migratory Multi-Swarm Optimiser*"
by Jonathan E. Fieldsend published in Proceedings of the IEEE Congress on Evolutionary Computation, 
pages 2593-2600, 2014 (http://hdl.handle.net/10871/15247)

Please reference this paper if you undertake work utilising this code.

Documentation for pynmmso can be found at: https://github.com/EPCCed/pynmmso/wiki/NMMSO

## Install pynmmso

The Python implementation of NMMSO requires Python 3 and Numpy (https://www.numpy.org/).  

You can install `pynmmso` using `pip`:

```
pip install pynmmso
```

## Using NMMSO

We will demonstrate using NMMSO to solve a one-dimensional optimisation problem. The function will we optimise is:

-x^4 + x^3 + 3x^2

Plotting this function with x in the range [-2, 3] gives:

![](https://github.com/EPCCed/pynmmso/wiki/images/1D-function.png)

This function has two optima (one global and one local).  We can use NMMSO to find these optima.

First we need to write Python code that captures the problem we wish to solve. Problems must be written as a Python class that implements two functions: `fitness` and `get_bounds`.

The `fitness` function takes one argument. This argument is a 1D Numpy array containing a value for each parameter of the problem.  Since our problem is one dimensional this array will contain a single value. The function must return a single scalar value which is the *fitness* for the given parameter values.  This is where we implement the function to be optimised.

The ```get_bounds``` function takes no arguments and returns two Python lists that define the bounds of the parameter search.  The first list specifies the minimum value for each parameter, the second list specifies the maximum value for each parameter.  As our problem is one dimensional there will only be one value in each list.

The implementation of our problem in Python is therefore:

```
class MyProblem:
    @staticmethod
    def fitness(params):
        x = params[0]
        return -x**4 + x**3 + 3 * x**2

    @staticmethod
    def get_bounds():
        return [-2], [3]
```

The following code uses NMMSO to solve this problem. The `Nmmso` object is constructed such that each swarm the algorithm creates will have a maximum of 10 particles and algorithm will terminate at the end of the iteration where the number of fitness evaluations exceeds 1000.  When run the algorithm returns a list of objects that contain the location and value for each of the discovered modes.

```
from pynmmso import Nmmso


class MyProblem:
    @staticmethod
    def fitness(params):
        x = params[0]
        return -x**4 + x**3 + 3 * x**2

    @staticmethod
    def get_bounds():
        return [-2], [3]


def main():
    number_of_fitness_evaluations = 1000

    nmmso = Nmmso(MyProblem())
    my_result = nmmso.run(number_of_fitness_evaluations)
    for mode_result in my_result:
        print("Mode at {} has value {}".format(mode_result.location, mode_result.value))


if __name__ == "__main__":
    main()
```



Running this code produces output similar to the following:

```
Mode at [1.65586203] has value 5.247909824656198
Mode at [-0.90586887] has value 1.0450589249496887
```

Further documentation is available at: https://github.com/EPCCed/pynmmso/wiki/NMMSO
