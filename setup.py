import os
from setuptools import find_packages, setup

with open("VERSION.txt", "r") as file_handler:
    __version__ = file_handler.read().strip()

setup(
    name="marllib",
    version=__version__,
    packages=find_packages(),
    license="MIT",
    python_requires=">=3.8",
    package_data={'': ['*.yaml']},
    include_package_data=True,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: POSIX :: Linux",
    ],
)