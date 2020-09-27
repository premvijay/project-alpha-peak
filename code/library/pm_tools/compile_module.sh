#!/bin/bash
cd ~/project-alpha-peak/code/library/pm_tools/
pwd
export pybind_include=./src/pybind11/include/
g++ -O3 -Wall -shared -std=c++11 -fPIC `python3.7m-config --includes` \
-I $pybind_include ./src/grid_assign_pybind11.cpp -o ./particles_to_grid`python3.7m-config --extension-suffix`