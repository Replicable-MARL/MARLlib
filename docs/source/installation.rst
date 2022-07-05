.. _basic-installation:

Installation Guides
===================

The installation of MARLlib is very easy. We've tested MARLlib on Python 3.6 and 3.7. This guide is based on ubuntu 18.04 or above.


Conda Environment
-----------------

We strongly recommend using `conda <https://docs.conda.io/en/latest/miniconda.html>`_ to manage your dependencies, and avoid version conflicts. Here we show the example of building python 3.7 based conda environment.

.. code-block:: shell

    conda create -n marllib python==3.7 -y
    conda activate marllib

    # install dependencies
    cmake --version # must be >=3.12
    clang++ --version   # must be >=7.0.0
    sudo apt-get install graphviz cmake clang

    # install marllib
    pip install -e .


External Environments
---------------------

External environments are integrated in MARLlib, such as `StarCraftII <https://github.com/oxwhirl/smac>`_ and `Mujoco <https://mujoco.org/>`_. You can intall them by following the official guides on their project homepage.


Development requirements
------------------------

For users who wanna contribute to our repository, run ``pip install -e .[dev]`` to complete the development dependencies, also refer the contributing guide.

