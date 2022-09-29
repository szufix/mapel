import os
import sys

py_dot_version = '.'.join([str(sys.version_info[0]), str(sys.version_info[1])])
py_version = ''.join([str(sys.version_info[0]), str(sys.version_info[1])])

from setuptools import setup, find_packages
from distutils.core import Extension

cpp_dist_module = Extension('mapel.elections.metrics.cppdistances', sources =
['mapel/elections/metrics/cppdistances.cpp'], libraries = ['boost_python' + \
py_version], include_dirs = ['/usr/include/python' + py_dot_version],
extra_compile_args = ['-std=gnu++17'])

rootDir = os.path.abspath(os.path.dirname(__file__))
reqPath = os.path.join(rootDir, 'requirements.txt')
readmePath = os.path.join(rootDir, 'README.md')
dir_1 = os.path.join(rootDir, 'mapel')

 
with open(reqPath, encoding = 'utf-8') as f:
    required = f.read().splitlines()
 
with open(readmePath, "r", encoding = 'utf-8') as f:
    long_description = f.read()
 
setup(
    name='mapel',
    version='1.3.0',
    license='MIT',
    author='Stanislaw Szufa',
    author_email='s.szufa@gmail.com',
    description='Map of Elections',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/szufix/mapel',
    download_url='https://github.com/szufix/mapel',
    packages=find_packages(exclude=['*private.py']),
    include_package_data=True,
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    ext_modules = [cpp_dist_module]
)
