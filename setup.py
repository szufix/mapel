from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

cpp_dist_module = Pybind11Extension('elections.metrics.cppdistances',
['mapel/elections/metrics/cppdistances.cpp'], extra_compile_args=["-Ofast"])

setup(
    ext_modules = [cpp_dist_module],
    cmdclass={"build_ext": build_ext},
)
