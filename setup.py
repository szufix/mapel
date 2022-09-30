from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

cpp_dist_module = Pybind11Extension('mapel.elections.metrics.cppdistances',
['mapel/elections/metrics/cppdistances.cpp'])

print("asdfasdfasdfasdfasdf")

print(find_packages())

print(find_packages(exclude = ['mapel.elections.not_in_the_package*',
    'mapel.elections.not_in_the_package']))

setup(
    packages=find_packages(exclude = ['mapel.elections.not_in_the_package*',
    'mapel.elections.not_in_the_package']),
    ext_modules = [cpp_dist_module],
    cmdclass={"build_ext": build_ext},
)
