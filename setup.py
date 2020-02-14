
import setuptools
from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()
      
setup(name='map-of-elections',
      version='1.0.3',
      description='Map of Elections',
      url='https://github.com/szufix/mapel',      
      download_url='https://github.com/szufix/mapel/archive/v1.0.3.tar.gz',
      author='Stanislaw Szufa',
      author_email='s.szufa@gmail.com',
      packages=setuptools.find_packages(),
      install_requires=['numpy'],
      zip_safe=False)
