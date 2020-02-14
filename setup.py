import os
import setuptools
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()
    
mapel_dir_1 = os.path.join('controllers','models')
mapel_dir_2 = os.path.join('results','points')

setup(name='map-of-elections',
      version='1.0.7',
      description='Map of Elections',
      url='https://github.com/szufix/mapel',      
      download_url='https://github.com/szufix/mapel/archive/v1.0.7.tar.gz',
      author='Stanislaw Szufa',
      author_email='s.szufa@gmail.com',
      package_data={'mapel': [mapel_dir_1 + '/*.txt', mapel_dir_2 + '/*.txt']},
      packages=setuptools.find_packages(),
      install_requires=['matplotlib', 'numpy
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        ],
        python_requires='>=3.6',
    )
