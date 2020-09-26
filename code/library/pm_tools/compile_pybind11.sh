#!/bin/bash
g++ -O3 -Wall -shared -std=c++11 -fPIC `python -m pybind11 --includes` ./grid_assign_pybind11.cpp -o ./particles_to_grid`python3-config --extension-suffix`