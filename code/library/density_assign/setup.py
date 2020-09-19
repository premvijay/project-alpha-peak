from setuptools import setup
from Cython.Build import cythonize
import os


setup(
    ext_modules = cythonize("cic_cy.pyx")
)