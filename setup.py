from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()
        
setup(name='nmmso',
      version='0.1',
      description='Multi-Modal Search with the Niching Migratory Multi-Swarm Optimizer',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Operating System :: OS Independent',
      ],
      keywords='optimization, optimisation, optimizer, optimiser, multimodal, multi-modal, GA, genetic',
      url='https://github.com/EPCCed/NMMSO/wiki/NMMSO',
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
