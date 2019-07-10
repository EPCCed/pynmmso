from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()
        
setup(name='pynmmso',
      version='1.0.0',
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
      url='https://github.com/EPCCed/pynmmso/wiki',
      author='Ally Hume',
      author_email='a.hume@epcc.ed.ac.uk',
      license='MIT',
      packages=['pynmmso','pynmmso.wrappers','pynmmso.listeners'],
      install_requires=[
          'numpy',
          'multiprocess'
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
