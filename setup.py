#!/usr/bin/env python
# encoding: utf-8
"""
This file contains the setup for setuptools to distribute everything as a
(PyPI) package.

"""

from setuptools import setup, find_packages

import glob
import numpy as np

# define version
version = '0.0.1'

# define scripts to be installed by the PyPI package
scripts = glob.glob('bin/*')

# define the models to be included in the PyPI package
package_data = ['models/*' ]

# some PyPI metadata
classifiers = ['Development Status :: 3 - Alpha',
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7',
               'Environment :: Console',
               'License :: OSI Approved :: BSD License',
               'License :: Free for non-commercial use',
               'Topic :: Multimedia :: Sound/Audio :: Analysis',
               'Topic :: Scientific/Engineering :: Artificial Intelligence']

# requirements
requirements = ['stempeg>=0.2.3',
                'spleeter>=2.2.2',
                'mutagen>=1.45.1',
                ]

# docs to be included
try:
    long_description = open('README.md', encoding='utf-8').read()
    long_description += '\n' + open('CHANGES.md', encoding='utf-8').read()
except TypeError:
    long_description = open('README.md').read()
    long_description += '\n' + open('CHANGES.md').read()

# the actual setup routine
setup(name='stemutil',
      version=version,
      description='Stem conversion utilities',
      long_description=long_description,
      author='Max Vohra',
      author_email='max-opensource@seattlenetworks.com',
      url='https://github.com/cryptopunk/stemutil',
      license='BSD, CC BY-NC-SA',
      packages=find_packages(exclude=['tests', 'docs']),
      package_data={'': package_data},
      exclude_package_data={'': ['tests', 'docs']},
      scripts=scripts,
      install_requires=requirements,
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      classifiers=classifiers)
