from setuptools import setup

import os
BASEPATH = os.path.dirname(os.path.abspath(__file__))

setup(name='pfbayes',
      py_modules=['pfbayes'],
      install_requires=[
          'torch',
          'torchdiffeq'
      ],
)
