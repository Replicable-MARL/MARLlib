import os
from setuptools import find_packages, setup
import pathlib

lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = lib_folder + '/requirements.txt'
install_requires = []  # Here we'll get: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="marllib",
    version="1.0.3",
    long_description=README,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["examples", "docs", "tests"]),
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
