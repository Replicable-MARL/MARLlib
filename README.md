<div align="center">

<img src=docs/source/images/logo1.png width=75% />
</div>



<h1 align="center"> MARLlib: A Multi-agent Reinforcement Learning Library </h1>

<div align="center">

<img src=docs/source/images/allenv.gif width=99% />

</div>

&emsp;

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)]()
![coverage](https://github.com/Replicable-MARL/MARLlib/blob/rllib_1.8.0_dev/coverage.svg)
[![Documentation Status](https://readthedocs.org/projects/marllib/badge/?version=latest)](https://marllib.readthedocs.io/en/latest/)
[![GitHub issues](https://img.shields.io/github/issues/Replicable-MARL/MARLlib)](https://github.com/Replicable-MARL/MARLlib/issues)
[![PyPI version](https://badge.fury.io/py/marllib.svg)](https://badge.fury.io/py/marllib)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Replicable-MARL/MARLlib/blob/master/marllib.ipynb)
[![Organization](https://img.shields.io/badge/Organization-ReLER_RL-blue.svg)](https://github.com/Replicable-MARL/MARLlib)
[![Organization](https://img.shields.io/badge/Organization-PKU_MARL-blue.svg)](https://github.com/Replicable-MARL/MARLlib)
[![Awesome](https://awesome.re/badge.svg)](https://marllib.readthedocs.io/en/latest/resources/awesome.html)


| :exclamation:  News |
|:-----------------------------------------|
| **March 2023** :anchor:We are excited to announce that a major update has just been released. For detailed version information, please refer to the [version info](https://github.com/Replicable-MARL/MARLlib/releases/tag/1.0.2).|
| **May 2023** Exciting news! MARLlib now supports five more tasks: [MATE](https://marllib.readthedocs.io/en/latest/handbook/env.html#mate), [GoBigger](https://marllib.readthedocs.io/en/latest/handbook/env.html#gobigger), [Overcooked-AI](https://marllib.readthedocs.io/en/latest/handbook/env.html#overcooked-ai), [MAPDN](https://marllib.readthedocs.io/en/latest/handbook/env.html#power-distribution-networks), and [AirCombat](https://marllib.readthedocs.io/en/latest/handbook/env.html#air-combat). Give them a try!|
| **June 2023** [OpenAI: Hide and Seek](https://marllib.readthedocs.io/en/latest/handbook/env.html#hide-and-seek) and [SISL](https://marllib.readthedocs.io/en/latest/handbook/env.html#sisl) environments are incorporated into MARLlib.|
| **Aug 2023** :tada:MARLlib has been accepted for publication in [JMLR](https://www.jmlr.org/mloss/).|
| **Sept 2023** Latest [PettingZoo](https://marllib.readthedocs.io/en/latest/handbook/env.html#pettingzoo) with [Gymnasium](https://gymnasium.farama.org) are compatiable within MARLlib.|
| **Nov 2023** We are currently in the process of creating a hands-on MARL book and aim to release the draft by the end of 2023.|


**Multi-agent Reinforcement Learning Library ([MARLlib](https://arxiv.org/abs/2210.13708))** is ***a MARL library*** that utilizes [**Ray**](https://github.com/ray-project/ray) and one of its toolkits [**RLlib**](https://github.com/ray-project/ray/tree/master/rllib). It offers a comprehensive platform for developing, training, and testing MARL algorithms across various tasks and environments. 

Here's an example of how MARLlib can be used:

```py
from marllib import marl

# prepare env
env = marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)

# initialize algorithm with appointed hyper-parameters
mappo = marl.algos.mappo(hyperparam_source='mpe')

# build agent model based on env + algorithms + user preference
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})

# start training
mappo.fit(env, model, stop={'timesteps_total': 1000000}, share_policy='group')
```



## Why MARLlib?

Here we provide a table for the comparison of MARLlib and existing work.

|   Library   |  Supported Env | Algorithm | Parameter Sharing  | Model 
|:-------------:|:-------------:|:-------------:|:--------------:|:----------------:|
|     [PyMARL](https://github.com/oxwhirl/pymarl) |        1 cooperative       |       5       |         share        |      GRU           | :x:
|   [PyMARL2](https://github.com/hijkzzz/pymarl2)|        2 cooperative       |     11   |         share        |  MLP + GRU  | :x:
| [MAPPO Benchmark](https://github.com/marlbenchmark/on-policy) |       4 cooperative       |      1     |          share + separate        |          MLP + GRU        |         :x:              |
| [MAlib](https://github.com/sjtu-marl/malib) |  4 self-play  | 10 | share + group + separate | MLP + LSTM | [![Documentation Status](https://readthedocs.org/projects/malib/badge/?version=latest)](https://malib.readthedocs.io/en/latest/?badge=latest)
|    [EPyMARL](https://github.com/uoe-agents/epymarl)|        4 cooperative      |    9    |        share + separate       |      GRU             |           :x:            |
|    [HARL](https://github.com/PKU-MARL/HARL)|        8 cooperative      |    9    |        share + separate       |      MLP + CNN + GRU           |           :x:            |
|    **[MARLlib](https://github.com/Replicable-MARL/MARLlib)** |       17 **no task mode restriction**     |    18     |   share + group + separate + **customizable**         |         MLP + CNN + GRU + LSTM          |           [![Documentation Status](https://readthedocs.org/projects/marllib/badge/?version=latest)](https://marllib.readthedocs.io/en/latest/) |

|   Library   | Github Stars  | Documentation | Issues Open | Activity | Last Update
|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|     [PyMARL](https://github.com/oxwhirl/pymarl) | [![GitHub stars](https://img.shields.io/github/stars/oxwhirl/pymarl)](https://github.com/oxwhirl/pymarl)    |       :x: | ![GitHub opened issue](https://img.shields.io/github/issues/oxwhirl/pymarl.svg) | ![GitHub commit-activity](https://img.shields.io/github/commit-activity/y/oxwhirl/pymarl?label=commit) | ![GitHub last commit](https://img.shields.io/github/last-commit/oxwhirl/pymarl?label=last%20update)  
|   [PyMARL2](https://github.com/hijkzzz/pymarl2)| [![GitHub stars](https://img.shields.io/github/stars/hijkzzz/pymarl2)](https://github.com/hijkzzz/pymarl2)       |       :x:  | ![GitHub opened issue](https://img.shields.io/github/issues/hijkzzz/pymarl2.svg) | ![GitHub commit-activity](https://img.shields.io/github/commit-activity/y/hijkzzz/pymarl2?label=commit) | ![GitHub last commit](https://img.shields.io/github/last-commit/hijkzzz/pymarl2?label=last%20update)  
| [MAPPO Benchmark](https://github.com/marlbenchmark/on-policy)| [![GitHub stars](https://img.shields.io/github/stars/marlbenchmark/on-policy)](https://github.com/marlbenchmark/on-policy)   |        :x:              | ![GitHub opened issue](https://img.shields.io/github/issues/marlbenchmark/on-policy.svg) | ![GitHub commit-activity](https://img.shields.io/github/commit-activity/y/marlbenchmark/on-policy?label=commit)| ![GitHub last commit](https://img.shields.io/github/last-commit/marlbenchmark/on-policy?label=last%20update)  
| [MAlib](https://github.com/sjtu-marl/malib) | [![GitHub stars](https://img.shields.io/github/stars/sjtu-marl/malib)](https://github.com/sjtu-marl/malib) | [![Documentation Status](https://readthedocs.org/projects/malib/badge/?version=latest)](https://malib.readthedocs.io/en/latest/?badge=latest) | ![GitHub opened issue](https://img.shields.io/github/issues/sjtu-marl/malib.svg) | ![GitHub commit-activity](https://img.shields.io/github/commit-activity/y/sjtu-marl/malib?label=commit) | ![GitHub last commit](https://img.shields.io/github/last-commit/sjtu-marl/malib?label=last%20update)  
|    [EPyMARL](https://github.com/uoe-agents/epymarl)| [![GitHub stars](https://img.shields.io/github/stars/uoe-agents/epymarl)](https://github.com/uoe-agents/epymarl)        |           :x:            | ![GitHub opened issue](https://img.shields.io/github/issues/uoe-agents/epymarl.svg) | ![GitHub commit-activity](https://img.shields.io/github/commit-activity/y/uoe-agents/epymarl?label=commit) | ![GitHub last commit](https://img.shields.io/github/last-commit/uoe-agents/epymarl?label=last%20update)  
|    [HARL](https://github.com/PKU-MARL/HARL)**\***| [![GitHub stars](https://img.shields.io/github/stars/PKU-MARL/HARL)](https://github.com/PKU-MARL/HARL)        |           :x:            | ![GitHub opened issue](https://img.shields.io/github/issues/PKU-MARL/HARL.svg) | ![GitHub commit-activity](https://img.shields.io/github/commit-activity/y/PKU-MARL/HARL?label=commit) | ![GitHub last commit](https://img.shields.io/github/last-commit/PKU-MARL/HARL?label=last%20update)  
|    **[MARLlib](https://github.com/Replicable-MARL/MARLlib)** |  [![GitHub stars](https://img.shields.io/github/stars/Replicable-MARL/MARLlib)](https://github.com/Replicable-MARL/MARLlib)  |           [![Documentation Status](https://readthedocs.org/projects/marllib/badge/?version=latest)](https://marllib.readthedocs.io/en/latest/) | ![GitHub opened issue](https://img.shields.io/github/issues/Replicable-MARL/MARLlib.svg) | ![GitHub commit-activity](https://img.shields.io/github/commit-activity/y/Replicable-MARL/MARLlib?label=commit) | ![GitHub last commit](https://img.shields.io/github/last-commit/Replicable-MARL/MARLlib?label=last%20update)  

> **_\*_**  **HARL** is the latest MARL library that has been recently released:fire:. If cutting-edge MARL algorithms with state-of-the-art performance are your target, HARL is definitely worth [a look](https://github.com/PKU-MARL/HARL)!

[comment]: <> (<div align="center">)

[comment]: <> (<img src=docs/source/images/overview.png width=100% />)

[comment]: <> (</div>)

## key features

:beginner: MARLlib offers several key features that make it stand out:

- MARLlib unifies diverse algorithm pipelines with agent-level distributed dataflow, allowing researchers to develop, test, and evaluate MARL algorithms across different tasks and environments.
- MARLlib supports all task modes, including cooperative, collaborative, competitive, and mixed. This makes it easier for researchers to train and evaluate MARL algorithms across a wide range of tasks.
- MARLlib provides a new interface that follows the structure of Gym, making it easier for researchers to work with multi-agent environments.
- MARLlib provides flexible and customizable parameter-sharing strategies, allowing researchers to optimize their algorithms for different tasks and environments.

:rocket: Using MARLlib, you can take advantage of various benefits, such as:

- **Zero knowledge of MARL**: MARLlib provides 18 pre-built algorithms with an intuitive API, allowing researchers to start experimenting with MARL without prior knowledge of the field.
- **Support for all task modes**: MARLlib supports almost all multi-agent environments, making it easier for researchers to experiment with different task modes.
- **Customizable model architecture**: Researchers can choose their preferred model architecture from the model zoo, or build their own.
- **Customizable policy sharing**: MARLlib provides grouping options for policy sharing, or researchers can create their own.
- **Access to over a thousand released experiments**: Researchers can access over a thousand released experiments to see how other researchers have used MARLlib.

## Installation

> __Note__:
> Please note that at this time, MARLlib is only compatible with Linux operating systems.

### Step-by-step  (recommended)

- install dependencies
- install environments
- install patches

#### 1. install dependencies (basic)

First, install MARLlib dependencies to guarantee basic usage.
following [this guide](https://marllib.readthedocs.io/en/latest/handbook/env.html), finally install patches for RLlib.

```bash
$ conda create -n marllib python=3.8 # or 3.9
$ conda activate marllib
$ git clone https://github.com/Replicable-MARL/MARLlib.git && cd MARLlib
$ pip install -r requirements.txt
```

#### 2. install environments (optional)

Please follow [this guide](https://marllib.readthedocs.io/en/latest/handbook/env.html).

> __Note__:
> We recommend the gym version around 0.20.0.
```bash
pip install "gym==0.20.0"
```

#### 3. install patches (basic)

Fix bugs of RLlib using patches by running the following command:

```bash
$ cd /Path/To/MARLlib/marllib/patch
$ python add_patch.py -y
```

### PyPI

```bash
$ pip install --upgrade pip
$ pip install marllib
```

### Docker-based usage

We provide a Dockerfile for building the MARLlib docker image in `MARLlib/docker/Dockerfile` and a devcontainer setup in `MARLlib/.devcontainer` folder. If you use the devcontainer, one thing to note is that you may need to customise certain arguments in `runArgs`  of `devcontainer.json` according to your hardware, for example the `--shm-size` argument.

## Getting started

<details>
<summary><b><big>Prepare the configuration</big></b></summary>

There are four parts of configurations that take charge of the whole training process.

- scenario: specify the environment/task settings
- algorithm: choose the hyperparameters of the algorithm
- model: customize the model architecture
- ray/rllib: change the basic training settings

<div align="center">
<img src=docs/source/images/configurations.png width=100% />
</div>

Before training, ensure all the parameters are set correctly, especially those you don't want to change.
> __Note__:
> You can also modify all the pre-set parameters via MARLLib API.*

</details>

<details>
<summary><b><big>Register the environment</big></b></summary>

Ensure all the dependencies are installed for the environment you are running with. Otherwise, please refer to
[MARLlib documentation](https://marllib.readthedocs.io/en/latest/handbook/env.html).


|   task mode   | api example |
| :-----------: | ----------- |
| cooperative | ```marl.make_env(environment_name="mpe", map_name="simple_spread", force_coop=True)``` |
| collaborative | ```marl.make_env(environment_name="mpe", map_name="simple_spread")``` |
| competitive | ```marl.make_env(environment_name="mpe", map_name="simple_adversary")``` |
| mixed | ```marl.make_env(environment_name="mpe", map_name="simple_crypto")``` |

Most of the popular environments in MARL research are supported by MARLlib:

| Env Name | Learning Mode | Observability | Action Space | Observations |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| **[LBF](https://marllib.readthedocs.io/en/latest/handbook/env.html#lbf)**  | cooperative + collaborative | Both | Discrete | 1D  |
| **[RWARE](https://marllib.readthedocs.io/en/latest/handbook/env.html#rware)**  | cooperative | Partial | Discrete | 1D  |
| **[MPE](https://marllib.readthedocs.io/en/latest/handbook/env.html#mpe)**  | cooperative + collaborative + mixed | Both | Both | 1D  |
| **[SISL](https://marllib.readthedocs.io/en/latest/handbook/env.html#sisl)** | cooperative + collaborative | Full | Both | 1D |
| **[SMAC](https://marllib.readthedocs.io/en/latest/handbook/env.html#smac)**  | cooperative | Partial | Discrete | 1D |
| **[MetaDrive](https://marllib.readthedocs.io/en/latest/handbook/env.html#metadrive)**  | collaborative | Partial | Continuous | 1D |
| **[MAgent](https://marllib.readthedocs.io/en/latest/handbook/env.html#magent)** | collaborative + mixed | Partial | Discrete | 2D |
| **[Pommerman](https://marllib.readthedocs.io/en/latest/handbook/env.html#pommerman)**  | collaborative + competitive + mixed | Both | Discrete | 2D |
| **[MAMuJoCo](https://marllib.readthedocs.io/en/latest/handbook/env.html#mamujoco)**  | cooperative | Full | Continuous | 1D |
| **[GRF](https://marllib.readthedocs.io/en/latest/handbook/env.html#google-research-football)**  | collaborative + mixed | Full | Discrete | 2D |
| **[Hanabi](https://marllib.readthedocs.io/en/latest/handbook/env.html#hanabi)** | cooperative | Partial | Discrete | 1D |
| **[MATE](https://marllib.readthedocs.io/en/latest/handbook/env.html#mate)** | cooperative + mixed | Partial | Both | 1D |
| **[GoBigger](https://marllib.readthedocs.io/en/latest/handbook/env.html#gobigger)** | cooperative + mixed | Both | Continuous | 1D |
| **[Overcooked-AI](https://marllib.readthedocs.io/en/latest/handbook/env.html#overcooked-ai)** | cooperative | Full | Discrete | 1D |
| **[PDN](https://marllib.readthedocs.io/en/latest/handbook/env.html#power-distribution-networks)** | cooperative | Partial | Continuous | 1D |
| **[AirCombat](https://marllib.readthedocs.io/en/latest/handbook/env.html#air-combat)** | cooperative + mixed | Partial | MultiDiscrete | 1D |
| **[HideAndSeek](https://marllib.readthedocs.io/en/latest/handbook/env.html#hide-and-seek)** | competitive + mixed | Partial | MultiDiscrete | 1D |

Each environment has a readme file, standing as the instruction for this task, including env settings, installation, and
important notes.
</details>

<details>
<summary><b><big>Initialize the algorithm</big></b></summary>


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

*IQL* is the multi-agent version of Q learning.
*MAA2C* and *MATRPO* are the centralized version of A2C and TRPO.
*VDPPO* is the value decomposition version of PPO.

</details>

<details>
<summary><b><big>Build the agent model</big></b></summary>

An agent model consists of two parts, `encoder` and `core arch`. 
`encoder` will be constructed by MARLlib according to the observation space.
Choose `mlp`, `gru`, or `lstm` as you like to build the complete model.

|  model arch   | api example |
| :-----------: | ----------- |
| MLP  | ```marl.build_model(env, algo, {"core_arch": "mlp")``` |
| GRU | ```marl.build_model(env, algo, {"core_arch": "gru"})```  |
| LSTM | ```marl.build_model(env, algo, {"core_arch": "lstm"})```  |
| Encoder Arch | ```marl.build_model(env, algo, {"core_arch": "gru", "encode_layer": "128-256"})```  |


</details>

<details>
<summary><b><big>Kick off the training</big></b></summary>

|  setting   | api example |
| :-----------: | ----------- |
| train  | ```algo.fit(env, model)``` |
| debug  | ```algo.fit(env, model, local_mode=True)``` |
| stop condition | ```algo.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 10000000})```  |
| policy sharing | ```algo.fit(env, model, share_policy='all') # or 'group' / 'individual'```  |
| save model | ```algo.fit(env, model, checkpoint_freq=100, checkpoint_end=True)```  |
| GPU accelerate  | ```algo.fit(env, model, local_mode=False, num_gpus=1)``` |
| CPU accelerate | ```algo.fit(env, model, local_mode=False, num_workers=5)```  |

</details>

<details>
<summary><b><big>Training & rendering API</big></b></summary>

```py
from marllib import marl

# prepare env
env = marl.make_env(environment_name="smac", map_name="5m_vs_6m")
# initialize algorithm with appointed hyper-parameters
mappo = marl.algos.mappo(hyperparam_source="smac")
# build agent model based on env + algorithms + user preference
model = marl.build_model(env, mappo, {"core_arch": "gru", "encode_layer": "128-256"})
# start training
mappo.fit(
  env, model, 
  stop={"timesteps_total": 1000000}, 
  checkpoint_freq=100, 
  share_policy="group"
)
# rendering
mappo.render(
  env, model, 
  local_mode=True, 
  restore_path={'params_path': "checkpoint/params.json",
                'model_path': "checkpoint/checkpoint-10"}
)
```
</details>

## Results

Under the current working directory, you can find all the training data (logging and TensorFlow files) as well as the saved models. To visualize the learning curve, you can use Tensorboard. Follow the steps below:

1. Install Tensorboard by running the following command:
```bash
pip install tensorboard
```

2. Use the following command to launch Tensorboard and visualize the results:
```bash
tensorboard --logdir .
```

Alternatively, you can refer to [this tutorial](https://github.com/tensorflow/tensorboard/blob/master/docs/get_started.ipynb) for more detailed instructions.

For a list of all the existing results, you can visit [this link](https://github.com/Replicable-MARL/MARLlib/tree/main/results). Please note that these results were obtained from an older version of MARLlib, which may lead to inconsistencies when compared to the current results.

## Quick examples

MARLlib provides some practical examples for you to refer to.

- [Detailed API usage](https://github.com/Replicable-MARL/MARLlib/blob/rllib_1.8.0_dev/examples/api_basic_usage.py): show how to use MARLlib api in
  detail, e.g. cmd + api combined running.
- [Policy sharing cutomization](https://github.com/Replicable-MARL/MARLlib/blob/rllib_1.8.0_dev/examples/customize_policy_sharing.py):
  define your group policy-sharing strategy as you like based on current tasks.
- [Loading model](https://github.com/Replicable-MARL/MARLlib/blob/rllib_1.8.0_dev/examples/load_model.py):
  load the pre-trained model and keep training.
- [Loading model and rendering](https://github.com/Replicable-MARL/MARLlib/blob/rllib_1.8.0_dev/examples/load_and_render_model.py):
  render the environment based on the pre-trained model.
- [Incorporating new environment](https://github.com/Replicable-MARL/MARLlib/blob/rllib_1.8.0_dev/examples/add_new_env.py):
  add your new environment following MARLlib's env-agent interaction interface.
- [Incorporating new algorithm](https://github.com/Replicable-MARL/MARLlib/blob/rllib_1.8.0_dev/examples/add_new_algorithm.py):
  add your new algorithm following MARLlib learning pipeline.
- [Parallelized finetuning](https://github.com/Replicable-MARL/MARLlib/blob/rllib_1.8.0_dev/examples/grid_search_usage.py):
  fintune your policy/model performance with `ray.tune`.
  
## Tutorials

Try MPE + MAPPO examples on Google Colaboratory!
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Replicable-MARL/MARLlib/blob/master/marllib.ipynb)
More tutorial documentations are available [here](https://marllib.readthedocs.io/).

## Awesome List

A collection of research and review papers of multi-agent reinforcement learning (MARL) is available. The papers have been organized based on their publication date and their evaluation of the corresponding environments.

Algorithms: [![Awesome](https://awesome.re/badge.svg)](https://marllib.readthedocs.io/en/latest/resources/awesome.html)
Environments: [![Awesome](https://awesome.re/badge.svg)](https://marllib.readthedocs.io/en/latest/handbook/env.html)


## Community

|  Channel   | Link |
| :----------- | :----------- |
| Issues | [GitHub Issues](https://github.com/Replicable-MARL/MARLlib/issues) |

## Roadmap

The roadmap to the future release is available in [ROADMAP.md](https://github.com/Replicable-MARL/MARLlib/blob/main/ROADMAP.md).

## Contributing

We are a small team on multi-agent reinforcement learning, and we will take all the help we can get! 
If you would like to get involved, here is information on [contribution guidelines and how to test the code locally](https://github.com/Replicable-MARL/MARLlib/blob/rllib_1.8.0_dev/CONTRIBUTING.md).

You can contribute in multiple ways, e.g., reporting bugs, writing or translating documentation, reviewing or refactoring code, requesting or implementing new features, etc.

## Citation

If you use MARLlib in your research, please cite the [MARLlib paper](https://arxiv.org/abs/2210.13708).

```tex
@article{hu2022marllib,
  author  = {Siyi Hu and Yifan Zhong and Minquan Gao and Weixun Wang and Hao Dong and Xiaodan Liang and Zhihui Li and Xiaojun Chang and Yaodong Yang},
  title   = {MARLlib: A Scalable and Efficient Multi-agent Reinforcement Learning Library},
  journal = {Journal of Machine Learning Research},
  year    = {2023},
}
```

Works that are based on or closely collaborate with MARLlib <[link](https://github.com/PKU-MARL/HARL)>

```tex
@InProceedings{hu2022policy,
      title={Policy Diagnosis via Measuring Role Diversity in Cooperative Multi-agent {RL}},
      author={Hu, Siyi and Xie, Chuanlong and Liang, Xiaodan and Chang, Xiaojun},
      booktitle={Proceedings of the 39th International Conference on Machine Learning},
      year={2022},
}
@misc{zhong2023heterogeneousagent,
      title={Heterogeneous-Agent Reinforcement Learning}, 
      author={Yifan Zhong and Jakub Grudzien Kuba and Siyi Hu and Jiaming Ji and Yaodong Yang},
      archivePrefix={arXiv},
      year={2023},
}
```


