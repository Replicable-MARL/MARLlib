.. _basic-installation:

Installation
===================

The installation of MARLlib has two parts: common installation and external environment installation.
We've tested the installation on Python 3.8 with Ubuntu 18.04 and Ubuntu 20.04.


MARLlib Installation
--------------------

We strongly recommend using `conda <https://docs.conda.io/en/latest/miniconda.html>`_ to manage your dependencies and avoid version conflicts.
Here we show the example of building python 3.8 based conda environment.

.. code-block:: shell

    conda create -n marllib python=3.8
    conda activate marllib
    git clone https://github.com/Replicable-MARL/MARLlib.git
    cd MARLlib
    pip install --upgrade pip
    pip install -r requirements.txt

    # recommend always keeping the gym version at 0.21.0.
    pip install gym==0.21.0

    # add patch files to MARLlib
    python patch/add_patch.py -y


External Environments Requirements
------------------------------------------

External environments are not auto-integrated (except MPE). However, you can install them by following.

* `our simplified guides <https://marllib.readthedocs.io/en/latest/handbook/env.html>`_.
* the official guide of each environment.
