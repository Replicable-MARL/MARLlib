<div align="center">
<img src=image/logo1.png width=70% />
</div>

<h1 align="center"> MARLlib: The MARL Extension for RLlib </h1>

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)]()

**Multi-Agent RLlib (MARLlib)** is ***a comprehensive Multi-Agent Reinforcement Learning algorithm library*** based
on [**Ray**](https://github.com/ray-project/ray) and one of its toolkits [**
RLlib**](https://github.com/ray-project/ray/tree/master/rllib). It provides MARL research community with a unified
platform for building, training, and evaluating MARL algorithms.

There are four core features of **MARLlib**.

- It unifies diverse algorithm pipeline with a newly proposed agent-level distributed dataflow. Currently, MARLlib
  delivers 18 algorithms and is able to handle cooperative (team-reward-only cooperation), collaborative (
  individual-reward-accessible cooperation), competitive (individual competition), and mixed (teamwork-based
  competition) tasks.
- It unifies multi-agent environment interfaces with a new interface following Gym standard and supports both
  synchronous and asynchronous agent-environment interaction. Currently, MARLlib provides support to ten environments.
- It provides three parameter sharing strategies, namely full-sharing, non-sharing, and group-sharing, by implementing
  the policy mapping API of RLlib. This is implemented to be fully decoupled from algorithms and environments, and is
  completely controlled by configuration files.
- It provides standard 2 or 20 millions timesteps learning curve in the form of CSV of each task-algorithm for
  reference. These results are reproducible as configuration files for each experiment are provided along. In total,
  more than a thousand experiments are conducted and released.

<div align="center">
<img src=image/overview.png width=100% />
</div>

## Overview

### Environments

Most of the popular environments in MARL research are supported by MARLlib:

| Env Name | Learning Mode | Observability | Action Space | Observations |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| [LBF](https://github.com/semitable/lb-foraging)  | cooperative + collaborative | Both | Discrete | Discrete  |
| [RWARE](https://github.com/semitable/robotic-warehouse)  | cooperative | Partial | Discrete | Discrete  |
| [MPE](https://github.com/openai/multiagent-particle-envs)  | cooperative + collaborative + mixed | Both | Both | Continuous  |
| [SMAC](https://github.com/oxwhirl/smac)  | cooperative | Partial | Discrete | Continuous |
| [MetaDrive](https://github.com/decisionforce/metadrive)  | collaborative | Partial | Continuous | Continuous |
|[MAgent](https://www.pettingzoo.ml/magent) | collaborative + mixed | Partial | Discrete | Discrete |
| [Pommerman](https://github.com/MultiAgentLearning/playground)  | collaborative + competitive + mixed | Both | Discrete | Discrete |
| [MAMuJoCo](https://github.com/schroederdewitt/multiagent_mujoco)  | cooperative | Partial | Continuous | Continuous |
| [GRF](https://github.com/google-research/football)  | collaborative + mixed | Full | Discrete | Continuous |
| [Hanabi](https://github.com/deepmind/hanabi-learning-environment) | cooperative | Partial | Discrete | Discrete |

Each environment has a readme file, standing as the instruction for this task, talking about env settings, installation,
and some important notes.

### Algorithms

We provide three types of MARL algorithms as our baselines including:

**Independent Learning:**
IQL DDPG PG A2C TRPO PPO

**Centralized Critic:**
COMA MADDPG MAAC MAPPO MATRPO HATRPO HAPPO

**Value Decomposition:**
VDN QMIX FACMAC VDAC VDPPO

Here is a chart describing the characteristics of each algorithm:

| Algorithm                                                    | Support Task Mode | Need Central Information | Discrete Action   | Continuous Action | Learning Categorize        | Type       |
| ------------------------------------------------------------ | ----------------- | ----------------- | ---------- | -------------------- | ---------- | ---------- |
| IQL*                                                         | cooperative collaborative competitive mixed             |                 | :heavy_check_mark:   |    | Independent Learning | Off Policy |
| [PG](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf) | cooperative collaborative competitive mixed             |                 | :heavy_check_mark:       | :heavy_check_mark:   | Independent Learning | On Policy  |
| [A2C](https://arxiv.org/abs/1602.01783)                      | cooperative collaborative competitive mixed             |                 | :heavy_check_mark:       | :heavy_check_mark:   | Independent Learning | On Policy  |
| [DDPG](https://arxiv.org/abs/1509.02971)                     | cooperative collaborative competitive mixed             |                 |  | :heavy_check_mark:   | Independent Learning | Off Policy |
| [TRPO](http://proceedings.mlr.press/v37/schulman15.pdf)      | cooperative collaborative competitive mixed             |                 | :heavy_check_mark:       | :heavy_check_mark:   | Independent Learning | On Policy  |
| [PPO](https://arxiv.org/abs/1707.06347)                      | cooperative collaborative competitive mixed             |                 | :heavy_check_mark:       | :heavy_check_mark:   | Independent Learning | On Policy  |
| [COMA](https://ojs.aaai.org/index.php/AAAI/article/download/11794/11653) | cooperative collaborative competitive mixed              | :heavy_check_mark:               | :heavy_check_mark:       |   | Centralized Critic   | On Policy  |
| [MADDPG](https://arxiv.org/abs/1706.02275)                   | cooperative collaborative competitive mixed             | :heavy_check_mark:               |  | :heavy_check_mark:   | Centralized Critic   | Off Policy |
| MAA2C*                                                       | cooperative collaborative competitive mixed             | :heavy_check_mark:               | :heavy_check_mark:       | :heavy_check_mark:   | Centralized Critic   | On Policy  |
| MATRPO*                                                      | cooperative collaborative competitive mixed             | :heavy_check_mark:               | :heavy_check_mark:       | :heavy_check_mark:   | Centralized Critic   | On Policy  |
| [MAPPO](https://arxiv.org/abs/2103.01955)                    | cooperative collaborative competitive mixed             | :heavy_check_mark:               | :heavy_check_mark:       | :heavy_check_mark:   | Centralized Critic   | On Policy  |
| [HATRPO](https://arxiv.org/abs/2109.11251)                   | Cooperative       | :heavy_check_mark:               | :heavy_check_mark:       | :heavy_check_mark:   | Centralized Critic   | On Policy  |
| [HAPPO](https://arxiv.org/abs/2109.11251)                    | Cooperative       | :heavy_check_mark:               | :heavy_check_mark:       | :heavy_check_mark:   | Centralized Critic   | On Policy  |
| [VDN](https://arxiv.org/abs/1706.05296)                      | Cooperative       |                 | :heavy_check_mark:   |    | Value Decomposition  | Off Policy |
| [QMIX](https://arxiv.org/abs/1803.11485)                     | Cooperative       | :heavy_check_mark:               | :heavy_check_mark:   |   | Value Decomposition  | Off Policy |
| [FACMAC](https://arxiv.org/abs/2003.06709)                   | Cooperative       | :heavy_check_mark:               |  | :heavy_check_mark:   | Value Decomposition  | Off Policy |
| [VDAC](https://arxiv.org/abs/2007.12306)                     | Cooperative       | :heavy_check_mark:               | :heavy_check_mark:       | :heavy_check_mark:   | Value Decomposition  | On Policy  |
| VDPPO*                                                       | Cooperative       | :heavy_check_mark:               | :heavy_check_mark:       | :heavy_check_mark:   | Value Decomposition  | On Policy  |

*IQL* is the multi-agent version of Q learning.
*MAA2C* and *MATRPO* are the centralized version of A2C and TRPO.
*VDPPO* is the value decomposition version of PPO.

### Environment-Algorithm Combination

Y for available, N for not suitable, P for partially available on some scenarios.
(Note: in our code, independent algorithms may not have **I** as prefix. For instance, PPO = IPPO)

| Env w Algorithm | IQL  | PG   | A2C  | DDPG | TRPO | PPO  | COMA | MADDPG | MAAC | MATRPO | MAPPO | HATRPO | HAPPO | VDN  | QMIX | FACMAC | VDAC | VDPPO |
| --------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ------ | ---- | ------ | ----- | ------ | ----- | ---- | ---- | ------ | ---- | ----- |
| LBF             | Y    | Y    | Y    | N    | Y    | Y    | Y    | N      | Y    | Y      | Y     | Y      | Y     | P    | P    | P      | P    | P     |
| RWARE           | Y    | Y    | Y    | N    | Y    | Y    | Y    | N      | Y    | Y      | Y     | Y      | Y     | Y    | Y    | Y      | Y    | Y     |
| MPE             | P    | Y    | Y    | P    | Y    | Y    | P    | P      | Y    | Y      | Y     | Y      | Y     | Y    | Y    | Y      | Y    | Y     |
| SMAC            | Y    | Y    | Y    | N    | Y    | Y    | Y    | N      | Y    | Y      | Y     | Y      | Y     | Y    | Y    | Y      | Y    | Y     |
| MetaDrive       | N    | Y    | Y    | Y    | Y    | Y    | N    | N      | N    | N      | N     | N      | N     | N    | N    | N      | N    | N     |
| MAgent          | Y    | Y    | Y    | N    | Y    | Y    | Y    | N      | Y    | Y      | Y     | Y      | Y     | N    | N    | N      | N    | N     |
| Pommerman       | Y    | Y    | Y    | N    | Y    | Y    | P    | N      | Y    | Y      | Y     | Y      | Y     | P    | P    | P      | P    | P     |
| MAMuJoCo        | N    | Y    | Y    | Y    | Y    | Y    | N    | Y      | Y    | Y      | Y     | Y      | Y     | N    | N    | Y      | Y    | Y     |
| GRF             | Y    | Y    | Y    | N    | Y    | Y    | Y    | N      | Y    | Y      | Y     | Y      | Y     | Y    | Y    | Y      | Y    | Y     |
| Hanabi          | Y    | Y    | Y    | N    | Y    | Y    | Y    | N      | Y    | Y      | Y     | Y      | Y     | N    | N    | N      | N    | N     |

You can find a comprehensive list of existing MARL algorithms in different
environments  [here](https://github.com/Replicable-MARL/MARLlib/tree/main/envs).

### Why MARLlib?

Here we provide a table for the comparison of MARLlib and existing work.

|   Library   | Github Stars | Task Mode | Supported Env | Algorithm | Parameter Sharing  | Asynchronous Interact | Framework
|:-------------:|:-------------:|:-------------:|:-------------:|:--------------:|:----------------:|:-----------------:|:---------------------:
|     [PyMARL](https://github.com/oxwhirl/pymarl) | [![GitHub stars](https://img.shields.io/github/stars/oxwhirl/pymarl)](https://github.com/oxwhirl/pymarl/stargazers)    |       cooperative      |       1       |       Independent Learning(1) + Centralized Critic(1) + Value Decomposition(3)       |         full-sharing        |                   | *
|    [PyMARL2](https://github.com/hijkzzz/pymarl2) | [![GitHub stars](https://img.shields.io/github/stars/hijkzzz/pymarl2)](https://github.com/hijkzzz/pymarl2/stargazers)    |       cooperative      |       1       |       Independent Learning(1) + Centralized Critic(1) + Value Decomposition(9)     |         full-sharing        |                   |   PyMARL
|   [MARL-Algorithms](https://github.com/starry-sky6688/MARL-Algorithms)| [![GitHub stars](https://img.shields.io/github/stars/starry-sky6688/MARL-Algorithms)](https://github.com/starry-sky6688/MARL-Algorithms/stargazers)  |       cooperative      |       1       |     CTDE(6) + Communication(1) + Graph(1) + Multi-task(1)   |         full-sharing        |  | *
|    [EPyMARL](https://github.com/uoe-agents/epymarl)| [![GitHub stars](https://img.shields.io/github/stars/uoe-agents/epymarl)](https://github.com/hijkzzz/uoe-agents/epymarl/stargazers)    |       cooperative      |       4       |    Independent Learning(3) + Value Decomposition(4) + Centralized Critic(2)    |        full-sharing + non-sharing       |                   |           PyMARL            | 
| [MAlib](https://github.com/sjtu-marl/malib) | [![GitHub stars](https://img.shields.io/github/stars/sjtu-marl/malib)](https://github.com/hijkzzz/sjtu-marl/malib/stargazers) | self-play | 2 +  [PettingZoo](https://www.pettingzoo.ml/) + [OpenSpiel](https://github.com/deepmind/open_spiel) | Population-based | full-sharing + group-sharing + non-sharing | :heavy_check_mark: | *
| [MAPPO Benchmark](https://github.com/marlbenchmark/on-policy)| [![GitHub stars](https://img.shields.io/github/stars/marlbenchmark/on-policy)](https://github.com/marlbenchmark/on-policy/stargazers) |     cooperative     |       4       |      MAPPO(1)     |         full-sharing + non-sharing        |         :heavy_check_mark:         |         pytorch-a2c-ppo-acktr-gail              |
|    [MARLlib](https://github.com/Replicable-MARL/MARLlib)| |  cooperative collaborative competitive mixed  |       10 + [PettingZoo](https://www.pettingzoo.ml/)      |    Independent Learning(6) + Centralized Critic(7) + Value Decomposition(5)     |        full-sharing + group-sharing + non-sharing        |         :heavy_check_mark:         |           Ray/Rllib           |

## Installation

To use MARLlib, first install MARLlib, then install desired environments
following [this guide](https://marllib.readthedocs.io/en/latest/handbook/env.html), finally install patches for RLlib.
After installation, training can be launched by following the usage section below.

### Install MARLlib

```bash
conda create -n marl python==3.8
conda activate marl
# please install pytorch <= 1.9.1 compatible with your hardware.

pip install ray==1.8.0
pip install ray[tune]
pip install ray[rllib]

git clone https://github.com/Replicable-MARL/MARLlib.git
cd MARLlib
pip install -e .
pip install icecream && pip install supersuit && pip install gym==0.21.0 && pip install importlib-metadata==4.13.0 
```

### Install environments

Please follow [this guide](https://marllib.readthedocs.io/en/latest/handbook/env.html).

### Install patches for RLlib

We fix bugs of RLlib by providing patches. After installing Ray, run the following command:

```bash
cd /Path/To/MARLlib/patch
python add_patch.py -y
```

If pommerman is installed and used as your testing bed, run

```bash
cd /Path/To/MARLlib/patch
python add_patch.py -y -p
```

follow the guide [here](https://marllib.readthedocs.io/en/latest/handbook/env.html#pommerman) before you starting
training.

## Quick Start

### Step 1. Prepare the configuration files

<div align="center">
<img src=image/configurations.png width=100% />
</div>

There are four configuration files you need to ensure correctness for your training demand.

- scenario: specify your environment/task settings
- algorithm: finetune your algorithm hyperparameters
- model: customize the model architecture
- ray/rllib: changing the basic training settings

### Step 2. Making sure all the dependency are installed for your environment.

You can refer to [here](https://marllib.readthedocs.io/en/latest/handbook/env.html) to install the environment. After
everything settled, make sure to change back you Gym version to 0.21.0. All environment MARLlib supported should work
fine with this version.

```bash
pip install gym==0.21.0
```

### Step 3. Start training with MARLlib API

```python
import marl

env = marl.make_env(environment_name="smac", map_name="5m_vs_6m")
mappo = marl.algos.mappo(hyperparam_source='smac')
mappo.fit(env, stop={'timesteps_total': 10000000})
```

or refer to **main.py** for detailed instruction.

### Step 4. Policy inference & Env rendering

Here we provide a mamujoco example.

```python
import marl

env = marl.make_env(environment_name="mamujoco", map_name="2AgentAnt")
mappo = marl.algos.mappo(hyperparam_source='common')
mappo.render(env, stop={'training_iteration': 1}, local_mode=True, num_gpus=1, num_workers=2, share_policy='all',
          restore_path='mappo_2AgentAnt_checkpoint_000015/checkpoint-15', checkpoint_end=False)

```


## Docker

We also provide docker-based usage for MARLlib. Before use, make
sure [docker](https://docs.docker.com/desktop/install/linux-install/) is installed on your machine.

Note: You need root access to use docker.

### Ready to Go Image

We prepare a docker image ready for MARLlib to
run. [link](https://hub.docker.com/repository/docker/iclr2023paper4242/marllib)

```bash
docker pull iclr2023paper4242/marl:1.0
docker run -d -it --rm --gpus all iclr2023paper4242/marl:1.0
docker exec -it [container_name] # you can get container_name by this command: docker ps
# launch the training
python main.py
```

### Alternatively, you can build your image on your local machine with two options: GPU or CPU only

#### Use GPU in docker

To use CUDA in MARLlib docker container, please first
install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
.

To build MARLlib docker image, use the following command:

```bash
git clone https://github.com/Replicable-MARL/MARLlib.git
cd MARLlib
bash docker/build.sh
```

Run `docker run --itd --rm --gpus all marllib:1.0` to create a new container and make GPU visible inside the container.
Then attach into the container and run experiments:

```bash
docker attach [your_container_name] # you can get container_name by this command: docker ps
# now we are in docker /workspace/MARLlib
# modify config file ray.yaml to enable GPU use
# launch the training
python main.py
```

#### Only use CPU in docker

To build MARLlib docker image, use the following command:

```bash
git clone https://github.com/Replicable-MARL/MARLlib.git
cd MARLlib
bash docker/build.sh
```

Run `docker run -d -it marllib:1.0` to create a new container. Then attach into the container and run experiments:

```bash
docker attach [your_container_name] # you can get container_name by this command: docker ps
# now we are in docker /workspace/MARLlib
# launch the training
python main.py
```

Note we only pre-install [LBF](https://iclr2023marllib.readthedocs.io/en/latest/handbook/env.html#lbf) in the target
container marllib:1.0 as a fast example. All running/algorithm/task configurations are kept unchanged.

## Experiment Results

All results are listed [here](https://github.com/Replicable-MARL/MARLlib/tree/main/results).

## Bug Shooting

- Environment side bug: e.g., SMAC is not installed properly.
    - Cause of bug: environment not installed properly (dependency, version, ...)
    - Solution: find the bug description in the log printed, especailly the table status at the initial part.
- Gym related bug:
    - Cause of bug: gym version required by RLlib and Environment has conflict
    - Solution: always change gym version back to 0.21.0 after new package installation.
- Package missing:
    - Cause of bug: miss installing package or incorrect Python Path
    - Solution: install the package and check you current PYTHONPATH
    
    
