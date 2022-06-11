# Pommerman for Ray

### Getting Started

Install Pommerman
> git clone https://github.com/MultiAgentLearning/playground

> cd playground

> pip install .


Install Ray
> pip install ray==1.8.0 # version is important


Pommerman require gym=0.10.11, which is a version too old.

Here we solve this conflict by modifying some source code of Pommerman as follows:

you can find the replace file in *patch/pommerman* directory

Pattern: the original file -> replace file

- **pommerman/graphics.py**  ->  **graphics.py**
- **pommerman/\_\_init\_\_.py**  ->  **\_\_init\_\_.py** 
- **pommerman/forward_model.py**  ->  **forward_model.py** 
- **pommerman/env/v0.py**  ->  **v0.py** 






