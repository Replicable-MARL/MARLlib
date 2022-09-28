import os
from setuptools import find_packages, setup

with open(os.path.join("marl", "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

setup(
    name="marllib",
    version=__version__,
    packages=find_packages(),
    license="MIT",
    python_requires=">=3.8",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
)