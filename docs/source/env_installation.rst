.. _env-installation:

Env Installation Guides
=========================


Note: make sure you have read and complete the :ref:`basic-installation` part.

SMAC
-----------------

.. code-block:: shell

    bash install_sc2.sh # https://github.com/oxwhirl/pymarl/blob/master/install_sc2.sh
    pip3 install numpy scipy pyyaml matplotlib
    pip3 install imageio
    pip3 install tensorboard-logger
    pip3 install pygame
    pip3 install jsonpickle==0.9.6
    pip3 install setuptools
    pip3 install sacred

    git clone https://github.com/oxwhirl/smac.git
    cd smac
    pip install .

Note: the location of the StarcraftII game directory should be pre-defined,
or you can just follow the error log (when the process can not found the game's location)
and put it in the right place.

MaMujoco
-----------------

.. code-block:: shell

    mkdir /home/YourUserName/.mujoco
    cd /home/YourUserName/.mujoco
    wget https://roboti.us/download/mujoco200_linux.zip
    unzip mujoco200_linux.zip
    export LD_LIBRARY_PATH=/home/YourUserName/.mujoco/mujoco200/bin;
    pip install mujoco-py==2.0.2.8

    git clone https://github.com/schroederdewitt/multiagent_mujoco
    cd multiagent_mujoco
    mv multiagent_mujoco /home/YourPathTo/MARLlib/multiagent_mujoco

    # optional
    sudo apt-get install libosmesa6-dev # If you meet GCC error with exit status 1
    pip install patchelf-wrapper

Note: To access the MuJoCo API, you may have to get a mjkey (which is free now) and put it under /home/YourUserName/.mujoco.

Google Research Football
-----------------

Google Research Football is somehow hard to be easily installed. We wish you good luck.

.. code-block:: shell

    sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev libdirectfb-dev libst-dev mesa-utils xvfb x11vnc python3-pip
    python3 -m pip install --upgrade pip setuptools psutil wheel

We provide solutions (may work) for potential bugs

* `Compiler error on /usr/lib/x86_64-linux-gnu/libGL.so <https://github.com/RobotLocomotion/drake/issues/2087>`_
* `apt-get, unmet dependencies, ... "but it is not going to be installed" <https://askubuntu.com/questions/564282/apt-get-unmet-dependencies-but-it-is-not-going-to-be-installed>`_

MPE
-----------------

We use pettingzoo version of MPE

.. code-block:: shell

    pip install pettingzoo[mpe]

LBF
---------------------

.. code-block:: shell

    pip install lbforaging==1.0.15

RWARE
------------------------

.. code-block:: shell

    pip install rware==1.0.1


MAgent
------------------------

We use pettingzoo version of MAgent

.. code-block:: shell

    pip install pettingzoo[magent]

Pommerman
------------------------

.. code-block:: shell

    git clone https://github.com/MultiAgentLearning/playground
    cd playground
    pip install .
    cd /home/YourPathTo/MARLlib/patch
    python add_patch.py --pommerman
    pip install gym==0.21.0


MetaDrive
------------------------

.. code-block:: shell

    pip install metadrive-simulator==0.2.3

Hanabi
------------------------

From `Compiler error on /usr/lib/x86_64-linux-gnu/libGL.so <https://github.com/marlbenchmark/on-policy>`_

Environment code for Hanabi is developed from the open-source environment code, but has been slightly modified to fit the algorithms used here.
To install, execute the following:

.. code-block:: shell

    pip install cffi
    cd /home/YourPathTo/MARLlib/patch/hanabi
    mkdir build
    cd build
    cmake ..
    make -j


