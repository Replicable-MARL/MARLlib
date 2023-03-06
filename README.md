<div align="center">
<img src=docs/source/images/logo1.png width=65% />
</div>

<h1 align="center"> MARLlib: The Multi-agent Reinforcement Learning Library </h1>

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)]()
![test](https://github.com/Replicable-MARL/MARLlib/workflows/test/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/marllib/badge/?version=latest)](https://marllib.readthedocs.io/en/latest/)
[![GitHub issues](https://img.shields.io/github/issues/Replicable-MARL/MARLlib)](https://github.com/Replicable-MARL/MARLlib/issues) 
[![GitHub stars](https://img.shields.io/github/stars/Replicable-MARL/MARLlib)](https://github.com/Replicable-MARL/MARLlib/stargazers) 
[![GitHub forks](https://img.shields.io/github/forks/Replicable-MARL/MARLlib)](https://github.com/Replicable-MARL/MARLlib/network)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Replicable-MARL/MARLlib/blob/sy_dev/marllib.ipynb)

**Multi-agent Reinforcement Learning Library (MARLlib)** is ***a comprehensive MARL algorithm library*** based
on [**Ray**](https://github.com/ray-project/ray) and one of its toolkits [**RLlib**](https://github.com/ray-project/ray/tree/master/rllib). It provides MARL research community with a unified
platform for building, training, and evaluating MARL algorithms on almosty all kinds of diverse tasks and environments.

A simple case of MARLlib usage:
```py
from marllib import marl

# prepare env
env = marl.make_env(environment_name="mpe", map_name="simple_spread")

# initialize algorithm with appointed hyper-parameters
mappo = marl.algos.mappo(hyperparam_source='mpe')

# build agent model based on env + algorithms + user preference
model = marl.build_model(env, mappo, {"core_arch": "gru", "encode_layer": "128-256"})

# start training
mappo.fit(env, model, stop={'timesteps_total': 1000000}, share_policy='group')

# ready to control
mappo.render(env, model, share_policy='group', restore_path='path_to_checkpoint')
```


## Why MARLlib?

Here we provide a table for the comparison of MARLlib and existing work.

|   Library   | Github Stars | Supported Env | Algorithm | Parameter Sharing  | Model | Framework
|:-------------:|:-------------:|:-------------:|:-------------:|:--------------:|:----------------:|:-----------------:|
|     [PyMARL](https://github.com/oxwhirl/pymarl) | [![GitHub stars](https://img.shields.io/github/stars/oxwhirl/pymarl)](https://github.com/oxwhirl/pymarl/stargazers)    |       1 cooperative       |       5       |         share        |      GRU           | *
|   [MARL-Algorithms](https://github.com/starry-sky6688/MARL-Algorithms)| [![GitHub stars](https://img.shields.io/github/stars/starry-sky6688/MARL-Algorithms)](https://github.com/starry-sky6688/MARL-Algorithms/stargazers)       |       1 cooperative       |     9   |         share        |  RNN  | *
| [MAPPO Benchmark](https://github.com/marlbenchmark/on-policy)| [![GitHub stars](https://img.shields.io/github/stars/marlbenchmark/on-policy)](https://github.com/marlbenchmark/on-policy/stargazers)    |       4 cooperative       |      1     |          share + separate        |          MLP / GRU        |         pytorch-a2c-ppo-acktr-gail              |
| [MAlib](https://github.com/sjtu-marl/malib) | [![GitHub stars](https://img.shields.io/github/stars/sjtu-marl/malib)](https://github.com/hijkzzz/sjtu-marl/malib/stargazers) | 4 self-play  | 10 | share + group + separate | MLP / LSTM | *
|    [EPyMARL](https://github.com/uoe-agents/epymarl)| [![GitHub stars](https://img.shields.io/github/stars/uoe-agents/epymarl)](https://github.com/hijkzzz/uoe-agents/epymarl/stargazers)         |       4 cooperative      |    9    |        share + separate       |      GRU             |           PyMARL            |
|    [MARLlib](https://github.com/Replicable-MARL/MARLlib)|  [![GitHub stars](https://img.shields.io/github/stars/Replicable-MARL/MARLlib)](https://github.com/Replicable-MARL/MARLlib/stargazers)  |       any task with **no task mode restriction**     |    18     |   share + group + separate + customizable         |         MLP / CNN / GRU / LSTM          |           Ray/Rllib           |


[comment]: <> (<div align="center">)

[comment]: <> (<img src=docs/source/images/overview.png width=100% />)

[comment]: <> (</div>)

What **MARLlib** brings to MARL community:

- MARLlib unifies diverse algorithm pipeline with agent-level distributed dataflow.
- MARLlib supports all task modes 
- MARLlib unifies multi-agent environment interfaces with a new interface following Gym standard.
- MARLlib provides flexible and customizable parameter sharing strategies.

With MARLlib, you can exploit the advantages not limited to:

- out of the box **18 algorithms** including common baselines and recent state of the arts!
- **all task modes** available! cooperative, collaborative, competitive, and mixed (team-based
  competition)
- easy to incorporate new multi-agent environment!
- **customizable model arch**! or pick your favorite one from MARLlib
- **customizable policy sharing** among agents! or grouped by MARLlib automatically
- more than a thousand experiments are conducted and released!

## Installation

- install dependencies
- install environments
- install patches

### Install dependencies (basic)

First install MARLlib dependencies to guarantee basic usage.
following [this guide](https://marllib.readthedocs.io/en/latest/handbook/env.html), finally install patches for RLlib.
After installation, training can be launched by following the usage section below.

```bash
conda create -n marllib python=3.8
conda activate marllib
git clone https://github.com/Replicable-MARL/MARLlib.git
cd MARLlib
pip install --upgrade pip
pip install -r requirements.txt
```
Note: **MPE** is included in basic installation.

### Install environments (optional)

Please follow [this guide](https://marllib.readthedocs.io/en/latest/handbook/env.html).

### Install patches (basic)

Fix bugs of RLlib using patches by run the following command:

```bash
cd /Path/To/MARLlib/marl/patch
python add_patch.py -y
```

[comment]: <> (If pommerman is installed and used as your testing bed, run)

[comment]: <> (```bash)

[comment]: <> (cd /Path/To/MARLlib/marl/patch)

[comment]: <> (python add_patch.py -y -p)

[comment]: <> (```)

[comment]: <> (follow the guide [here]&#40;https://marllib.readthedocs.io/en/latest/handbook/env.html#pommerman&#41; before you starting)

[comment]: <> (training.)

## Learning with MARLlib

There are four parts of configurations that take charge of the whole training process.

- scenario: specify the environment/task settings
- algorithm: choose the hyperparameters of the algorithm 
- model: customize the model architecture
- ray/rllib: change the basic training settings

<div align="center">
<img src=docs/source/images/configurations.png width=100% />
</div>

*Note: You can modify all the pre-set parameters via MARLLib api.*


### Pre-training

Making sure all the dependency are installed for the environment you are running with.
Otherwise, please refer to the [doc](https://marllib.readthedocs.io/en/latest/handbook/env.html). 

>  **Note: Keep your Gym version to 0.21.0.**

All environments MARLlib supported should work fine with this version.

### MARLlib API

```py
from marllib import marl
# prepare env
env = marl.make_env(environment_name="mpe", map_name="simple_spread")
# initialize algorithm with appointed hyper-parameters
mappo = marl.algos.mappo(hyperparam_source="mpe")
# build agent model based on env + algorithms + user preference
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})
# start training
mappo.fit(env, model, stop={"timesteps_total": 1000000}, checkpoint_freq=100, share_policy="group")
```

### prepare the ```environment```

|   task mode   | api example |
| :-----------: | ----------- |
| cooperative | ```marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)``` |
| collaborative | ```marl.make_env(environment_name="mpe", map_name="simple_spread")``` |
| competitive | ```marl.make_env(environment_name="mpe", map_name="simple_adversary")``` |
| mixed | ```marl.make_env(environment_name="mpe", map_name="simple_crypto")``` |

Most of the popular environments in MARL research are supported by MARLlib:

| Env Name | Learning Mode | Observability | Action Space | Observations |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| **[LBF](https://github.com/semitable/lb-foraging)**  | cooperative + collaborative | Both | Discrete | 1D  |
| **[RWARE](https://github.com/semitable/robotic-warehouse)**  | cooperative | Partial | Discrete | 1D  |
| **[MPE](https://github.com/openai/multiagent-particle-envs)**  | cooperative + collaborative + mixed | Both | Both | 1D  |
| **[SMAC](https://github.com/oxwhirl/smac)**  | cooperative | Partial | Discrete | 1D |
| **[MetaDrive](https://github.com/decisionforce/metadrive)**  | collaborative | Partial | Continuous | 1D |
| **[MAgent](https://www.pettingzoo.ml/magent)** | collaborative + mixed | Partial | Discrete | 2D |
| **[Pommerman](https://github.com/MultiAgentLearning/playground)**  | collaborative + competitive + mixed | Both | Discrete | 2D |
| **[MAMuJoCo](https://github.com/schroederdewitt/multiagent_mujoco)**  | cooperative | Partial | Continuous | 1D |
| **[GRF](https://github.com/google-research/football)**  | collaborative + mixed | Full | Discrete | 2D |
| **[Hanabi](https://github.com/deepmind/hanabi-learning-environment)** | cooperative | Partial | Discrete | 1D |

Each environment has a readme file, standing as the instruction for this task, including env settings, installation,
and important notes.

### initialize the  ```algorithm```

|  running target   | api example |
| :-----------: | ----------- |
| train & finetune  | ```marl.algos.mappo(hyperparam_source=$ENV)``` |
| develop & debug | ```marl.algos.mappo(hyperparam_source="test")``` |
| 3rd party env | ```marl.algos.mappo(hyperparam_source="common")``` |

Here is a chart describing the characteristics of each algorithm:

| algorithm                                                    | support task mode | discrete action   | continuous action |  policy type        |
| :------------------------------------------------------------: | :-----------------: | :----------: | :--------------------: | :----------: | 
| *IQL**                                                         | all four               | :heavy_check_mark:   |    |  off-policy |
| *[PG](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)* | all four                  | :heavy_check_mark:       | :heavy_check_mark:   |  on-policy  |
| *[A2C](https://arxiv.org/abs/1602.01783)*                      | all four              | :heavy_check_mark:       | :heavy_check_mark:   |  on-policy  |
| *[DDPG](https://arxiv.org/abs/1509.02971)*                     | all four             |  | :heavy_check_mark:   |  off-policy |
| *[TRPO](http://proceedings.mlr.press/v37/schulman15.pdf)*      | all four            | :heavy_check_mark:       | :heavy_check_mark:   |  on-policy  |
| *[PPO](https://arxiv.org/abs/1707.06347)*                      | all four            | :heavy_check_mark:       | :heavy_check_mark:   |  on-policy  |
| *[COMA](https://ojs.aaai.org/index.php/AAAI/article/download/11794/11653)* | all four                           | :heavy_check_mark:       |   |  on-policy  |
| *[MADDPG](https://arxiv.org/abs/1706.02275)*                   | all four                     |  | :heavy_check_mark:   |  off-policy |
| *MAA2C**                                                       | all four                        | :heavy_check_mark:       | :heavy_check_mark:   |  on-policy  |
| *MATRPO**                                                      | all four                         | :heavy_check_mark:       | :heavy_check_mark:   |  on-policy  |
| *[MAPPO](https://arxiv.org/abs/2103.01955)*                    | all four                         | :heavy_check_mark:       | :heavy_check_mark:   |  on-policy  |
| *[HATRPO](https://arxiv.org/abs/2109.11251)*                   | cooperative                     | :heavy_check_mark:       | :heavy_check_mark:   |  on-policy  |
| *[HAPPO](https://arxiv.org/abs/2109.11251)*                    | cooperative                     | :heavy_check_mark:       | :heavy_check_mark:   |  on-policy  |
| *[VDN](https://arxiv.org/abs/1706.05296)*                      | cooperative         | :heavy_check_mark:   |    |  off-policy |
| *[QMIX](https://arxiv.org/abs/1803.11485)*                     | cooperative                    | :heavy_check_mark:   |   |  off-policy |
| *[FACMAC](https://arxiv.org/abs/2003.06709)*                   | cooperative                    |  | :heavy_check_mark:   |  off-policy |
| *[VDAC](https://arxiv.org/abs/2007.12306)*                    | cooperative                    | :heavy_check_mark:       | :heavy_check_mark:   |  on-policy  |
| *VDPPO**                                                      | cooperative                | :heavy_check_mark:       | :heavy_check_mark:   |  on-policy  |

***all four**: cooperative collaborative competitive mixed

[comment]: <> (18 MARL algorithms are implemented and categorized as follows:)

[comment]: <> (**Independent Learning:**)

[comment]: <> (*IQL DDPG PG A2C TRPO PPO*)

[comment]: <> (**Centralized Critic:**)

[comment]: <> (*COMA MADDPG MAAC MAPPO MATRPO HATRPO HAPPO*)

[comment]: <> (**Value Decomposition:**)

[comment]: <> (*VDN QMIX FACMAC VDAC VDPPO*)

*IQL* is the multi-agent version of Q learning.
*MAA2C* and *MATRPO* are the centralized version of A2C and TRPO.
*VDPPO* is the value decomposition version of PPO.

### construct the agent  ```model```

|  model arch   | api example |
| :-----------: | ----------- |
| MLP  | ```marl.build_model(env, algo, {"core_arch": "mlp")``` |
| GRU | ```marl.build_model(env, algo, {"core_arch": "gru"})```  |
| LSTM | ```marl.build_model(env, algo, {"core_arch": "lstm"})```  |
| encoder arch | ```marl.build_model(env, algo, {"core_arch": "gru", "encode_layer": "128-256"})```  |

### kick off the training ```algo.fit```

|  setting   | api example |
| :-----------: | ----------- |
| train  | ```algo.fit(env, model)``` |
| debug  | ```algo.fit(env, model, local_mode=True)``` |
| stop condition | ```algo.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 10000000})```  |
| policy sharing | ```algo.fit(env, model, share_policy='all') # or 'group' / 'individual'```  |
| save model | ```algo.fit(env, model, checkpoint_freq=100, checkpoint_end=True)```  |
| GPU accelerate  | ```algo.fit(env, model, local_mode=False, num_gpus=1)``` |
| CPU accelerate | ```algo.fit(env, model, local_mode=False, num_workers=5)```  |

policy inference ```algo.render```

|  setting   | api example |
| :-----------: | ----------- |
| render  | `algo.render(env, model, local_mode=True, restore_path='path_to_model')` |

By default, all the models will be saved at ```/home/username/ray_results/experiment_name/checkpoint_xxxx```


## Tutorials

Try MPE + MAPPO examples on Google Colaboratory!
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Replicable-MARL/MARLlib/blob/sy_dev/marllib.ipynb)




[comment]: <> (## Docker)

[comment]: <> (We also provide docker-based usage for MARLlib. Before use, make)

[comment]: <> (sure [docker]&#40;https://docs.docker.com/desktop/install/linux-install/&#41; is installed on your machine.)

[comment]: <> (Note: You need root access to use docker.)

[comment]: <> (### Ready to Go Image)

[comment]: <> (We prepare a docker image ready for MARLlib to)

[comment]: <> (run. [link]&#40;https://hub.docker.com/repository/docker/iclr2023paper4242/marllib&#41;)

[comment]: <> (```bash)

[comment]: <> (docker pull iclr2023paper4242/marl:1.0)

[comment]: <> (docker run -d -it --rm --gpus all iclr2023paper4242/marl:1.0)

[comment]: <> (docker exec -it [container_name] # you can get container_name by this command: docker ps)

[comment]: <> (# launch the training)

[comment]: <> (python main.py)

[comment]: <> (```)

[comment]: <> (### Alternatively, you can build your image on your local machine with two options: GPU or CPU only)

[comment]: <> (#### Use GPU in docker)

[comment]: <> (To use CUDA in MARLlib docker container, please first)

[comment]: <> (install [NVIDIA Container Toolkit]&#40;https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html&#41;)

[comment]: <> (.)

[comment]: <> (To build MARLlib docker image, use the following command:)

[comment]: <> (```bash)

[comment]: <> (git clone https://github.com/Replicable-MARL/MARLlib.git)

[comment]: <> (cd MARLlib)

[comment]: <> (bash docker/build.sh)

[comment]: <> (```)

[comment]: <> (Run `docker run --itd --rm --gpus all marllib:1.0` to create a new container and make GPU visible inside the container.)

[comment]: <> (Then attach into the container and run experiments:)

[comment]: <> (```bash)

[comment]: <> (docker attach [your_container_name] # you can get container_name by this command: docker ps)

[comment]: <> (# now we are in docker /workspace/MARLlib)

[comment]: <> (# modify config file ray.yaml to enable GPU use)

[comment]: <> (# launch the training)

[comment]: <> (python main.py)

[comment]: <> (```)

[comment]: <> (#### Only use CPU in docker)

[comment]: <> (To build MARLlib docker image, use the following command:)

[comment]: <> (```bash)

[comment]: <> (git clone https://github.com/Replicable-MARL/MARLlib.git)

[comment]: <> (cd MARLlib)

[comment]: <> (bash docker/build.sh)

[comment]: <> (```)

[comment]: <> (Run `docker run -d -it marllib:1.0` to create a new container. Then attach into the container and run experiments:)

[comment]: <> (```bash)

[comment]: <> (docker attach [your_container_name] # you can get container_name by this command: docker ps)

[comment]: <> (# now we are in docker /workspace/MARLlib)

[comment]: <> (# launch the training)

[comment]: <> (python main.py)

[comment]: <> (```)

[comment]: <> (Note we only pre-install [LBF]&#40;https://iclr2023marllib.readthedocs.io/en/latest/handbook/env.html#lbf&#41; in the target)

[comment]: <> (container marllib:1.0 as a fast example. All running/algorithm/task configurations are kept unchanged.)

## Experiment Results

All results are listed [here](https://github.com/Replicable-MARL/MARLlib/tree/main/results).


[comment]: <> (## Bug Shooting)

[comment]: <> (- Environment side bug: e.g., SMAC is not installed properly.)

[comment]: <> (    - Cause of bug: environment not installed properly &#40;dependency, version, ...&#41;)

[comment]: <> (    - Solution: find the bug description in the log printed, especailly the table status at the initial part.)

[comment]: <> (- Gym related bug:)

[comment]: <> (    - Cause of bug: gym version required by RLlib and Environment has conflict)

[comment]: <> (    - Solution: always change gym version back to 0.21.0 after new package installation.)

[comment]: <> (- Package missing:)

[comment]: <> (    - Cause of bug: miss installing package or incorrect Python Path)

[comment]: <> (    - Solution: install the package and check you current PYTHONPATH)
    
    
