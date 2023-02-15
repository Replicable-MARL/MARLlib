import os
from setuptools import find_packages, setup

import os
lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = lib_folder + '/requirements.txt'
install_requires = [] # Here we'll get: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

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
    install_requires=install_requires,
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