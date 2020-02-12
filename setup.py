
import setuptools
from setuptools import setup

setup(name='map-of-elections',
      version='1.0.0',
      description='Map of Elections',
      url='https://github.com/szufix/mapel',      
      download_url='https://github.com/Koozco/pmp/archive/v1.0.0.tar.gz',
      author='Stanislaw Szufa',
      author_email='s.szufa@gmail.com',
      packages=setuptools.find_packages(),
      install_requires=['numpy'],
      zip_safe=False)
