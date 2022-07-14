.. _basic-installation:

Installation Guides
===================

The installation of MARLlib has two part: common installation and external environments installation.
We've tested the installation on Python 3.6 with ubuntu 18.04 or above.


Basic Installation
--------------------

We strongly recommend using `conda <https://docs.conda.io/en/latest/miniconda.html>`_ to manage your dependencies, and avoid version conflicts.
Here we show the example of building python 3.6 based conda environment.

.. code-block:: shell

    conda create -n marllib python==3.6
    conda activate marllib

    # install Ray/RLlib
    pip install ray==1.8.0 # version sensitive
    (optional) pip install ray[tune]
    (optional) pip install ray[rllib]

    # add patches to fix ray bugs
    cd PathToMARLlib/patch
    python add_patch.py
    or
    python add_patch.py -y

    # recommended gym version for all envs
    pip install gym==0.21.0


External Environments Requirements
------------------------------------------

External environments are integrated in MARLlib, such as `StarCraftII <https://github.com/oxwhirl/smac>`_ and `MaMujoco <https://github.com/schroederdewitt/multiagent_mujoco>`_. You can install them by following

* our simplified guides
* the official guides on their project homepage

The related content can be found in in :ref:`env`.


Development requirements
----------------------------

For users who wanna contribute to our repository, please refer the contributing guide.

