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

    conda create -n marllib python==3.8
    conda activate marllib
    # (recommended) please install pytorch <= 1.9.1 if compatible with your hardware.

    pip install ray==1.8.0
    pip install ray[tune]
    pip install ray[rllib]

    git clone https://github.com/Replicable-MARL/MARLlib.git
    cd MARLlib
    pip install -e .
    pip install icecream && pip install supersuit && pip install gym==0.21.0 && pip install importlib-metadata==4.13.0

    # add patch files to MARLlib
    python patch/add_patch.py -y


(Optional) We also provide docker-based MARLlib usage. Make sure `docker <https://docs.docker.com/desktop/install/linux-install/>`_  is installed and run

.. code-block:: shell

    git clone https://github.com/Replicable-MARL/MARLlib.git
    cd MARLlib
    bash docker/build.sh
    docker run -d -it marllib:1.0
    docker exec -it [your_container_name] # you can get container_name by this command: docker ps
    python patch/add_patch.py -y
    # launch the training in docker under project directory
    python marl/main.py --algo_config=mappo --env_config=lbf with env_args.map_name=lbf-8x8-2p-2f-3s-c

Note we only pre-install :ref:`LBF` in the target container marllib:1.0 as a fast example. All running/algorithm/task configurations are kept unchanged.
You may also need root access to use docker or add `sudo`.

External Environments Requirements
------------------------------------------

External environments are not auto-integrated. However, you can install them by following.

* `our simplified guides <https://marllib.readthedocs.io/en/latest/handbook/env.html>`_.
* the official guide of each environment.


Contribute
----------------------------

Please refer to the `Contribute <https://github.com/Replicable-MARL/MARLlib>`_ in our repository cover.
