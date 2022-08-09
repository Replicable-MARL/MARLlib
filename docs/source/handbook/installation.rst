.. _basic-installation:

Installation
===================

The installation of MARLlib has two parts: common installation and external environment installation.
We've tested the installation on Python 3.6 with ubuntu 18.04 or above.


Basic Installation
--------------------

We strongly recommend using `conda <https://docs.conda.io/en/latest/miniconda.html>`_ to manage your dependencies and avoid version conflicts.
Here we show the example of building python 3.6 based conda environment.

.. code-block:: shell

    conda create -n marllib python==3.6
    conda activate marllib

    # install Ray/RLlib
    pip install ray==1.8.0 # version sensitive
    # or just the package needed
    # pip install ray[tune]==1.8.0
    # pip install ray[rllib]==1.8.0

    # add patches to fix ray bugs
    cd PathToMARLlib/patch
    python add_patch.py

    # recommended gym version for all envs
    pip install gym==0.21.0


External Environments Requirements
------------------------------------------

External environments are not auto-integrated. However, you can install them by following.

* `our simplified guides <https://marllib.readthedocs.io/en/latest/handbook/env.html>`_.
* the official guide of each environment.


Contribute
----------------------------

Please refer to the `Contribute <https://github.com/Replicable-MARL/MARLlib>`_ in our repository cover.

