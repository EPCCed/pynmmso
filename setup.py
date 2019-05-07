from setuptools import setup

setup(name='nmmso',
      version='0.1',
      description='Multi-Modal Search with the Niching Migratory Multi-Swarm Optimiser',
      url='http://TODO',
      author='Ally Hume',
      author_email='a.hume@epcc.ed.ac.uk',
      license='MIT',
      packages=['nmmso','nmmso.wrappers','nmmso.listeners'],
      install_requires=[
          'numpy',
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
