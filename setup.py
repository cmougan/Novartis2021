from pip._internal.req import parse_requirements
from setuptools import setup

import os
from setuptools import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="novartis",
    version="0.1",
    description="Novartis Datathon Package",
    packages=["novartis"],
    entry_points={},
    install_requires=required,
)
