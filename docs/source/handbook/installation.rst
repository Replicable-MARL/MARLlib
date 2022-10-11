.. _basic-installation:

Installation
===================

The installation of MARLlib has two parts: common installation and external environment installation.
We've tested the installation on Python >= 3.7.10 with Ubuntu 18.04 and Ubuntu 20.04.


MARLlib Installation
--------------------

We strongly recommend using `conda <https://docs.conda.io/en/latest/miniconda.html>`_ to manage your dependencies and avoid version conflicts.
Here we show the example of building python 3.8 based conda environment.

.. code-block:: shell

    conda create -n marllib python==3.8
    conda activate marllib
    # please install pytorch <= 1.9.1 compatible with your hardware.

    pip install ray==1.8.0
    pip install ray[tune]
    pip install ray[rllib]

    git clone MARLlib_git_url
    export PYTHONPATH="$PWD" # set /Your/Path/To/MARLlib as python path


External Environments Requirements
------------------------------------------

External environments are not auto-integrated. However, you can install them by following.

* `our simplified guides <https://marllib.readthedocs.io/en/latest/handbook/env.html>`_.
* the official guide of each environment.


Contribute
----------------------------

Please refer to the `Contribute <https://github.com/Replicable-MARL/MARLlib>`_ in our repository cover.

