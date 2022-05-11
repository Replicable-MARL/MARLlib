# MaMujoco for Ray

### Getting Started

Install Ray

> pip install ray==1.8.0 # version is important


Download Mujoco-200

https://roboti.us/download/mujoco200_linux.zip

extract it to 

> /home/YourUserName/.mujoco/

Note: you have to get you licence key of mujoco

Set env variable

> LD_LIBRARY_PATH=/home/YourUserName/.mujoco/mujoco200/bin;

Install Mujoco python api

> pip install mujoco-py==2.0.2.8 # version is important, has to be >2.0 and <2.1

Set up multi-agent Mujoco according to https://github.com/schroederdewitt/multiagent_mujoco

If you meet GCC error with exit status 1, try:

> sudo apt-get install libosmesa6-dev

