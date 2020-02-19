from setuptools import setup, find_packages
import os

rootDir = os.path.abspath(os.path.dirname(__file__))
reqPath = os.path.join(rootDir, 'requirements.txt')
readmePath = os.path.join(rootDir, 'README.md')
mapel_dir_1 = os.path.join(rootDir, 'mapel', 'controllers', 'models')
mapel_dir_2 = os.path.join(rootDir, 'mapel', 'results', 'points')

with open(reqPath) as f:
    required = f.read().splitlines()

with open(readmePath, "r") as f:
    long_description = f.read()

setup(
    name='mapel',
    version='1.0.1',
    license='MIT',
    author='Stanislaw Szufa',
    author_email='s.szufa@gmail.com',
    description='Map of Elections',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/szufix/mapel',
    download_url='https://github.com/szufix/mapel',
    package_data={'mapel': [mapel_dir_1 + '/*.txt', mapel_dir_2 + '/*.txt']},
    packages=find_packages(),
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
