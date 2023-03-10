<div align="center">
<img src=docs/source/images/logo1.png width=65% />
</div>

<h1 align="center"> MARLlib: An Extensive Multi-agent Reinforcement Learning Library </h1>

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)]()
![test](https://github.com/Replicable-MARL/MARLlib/workflows/test/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/marllib/badge/?version=latest)](https://marllib.readthedocs.io/en/latest/)
[![GitHub issues](https://img.shields.io/github/issues/Replicable-MARL/MARLlib)](https://github.com/Replicable-MARL/MARLlib/issues)
[![GitHub stars](https://img.shields.io/github/stars/Replicable-MARL/MARLlib)](https://github.com/Replicable-MARL/MARLlib/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Replicable-MARL/MARLlib)](https://github.com/Replicable-MARL/MARLlib/network)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Replicable-MARL/MARLlib/blob/sy_dev/marllib.ipynb)
[![PyPI version](https://badge.fury.io/py/marllib.svg)](https://badge.fury.io/py/marllib)

**Multi-agent Reinforcement Learning Library ([MARLlib](https://arxiv.org/abs/2210.13708))** is ***a MARL library*** based
on [**Ray**](https://github.com/ray-project/ray) and one of its toolkits [**RLlib**](https://github.com/ray-project/ray/tree/master/rllib). It provides the MARL research community a unified
platform for building, training, and evaluating MARL algorithms on almost all diverse tasks and environments.

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

|   Library   | Github Stars | Supported Env | Algorithm | Parameter Sharing  | Model | Documentation
|:-------------:|:-------------:|:-------------:|:-------------:|:--------------:|:----------------:|:-----------------:|
|     [PyMARL](https://github.com/oxwhirl/pymarl) | [![GitHub stars](https://img.shields.io/github/stars/oxwhirl/pymarl)](https://github.com/oxwhirl/pymarl/stargazers)    |       1 cooperative       |       5       |         share        |      GRU           | :x:
|   [PyMARL2](https://github.com/hijkzzz/pymarl2)| [![GitHub stars](https://img.shields.io/github/stars/hijkzzz/pymarl2)](https://github.com/hijkzzz/pymarl2)       |       2 cooperative       |     11   |         share        |  MLP + GRU  | :x:
| [MAPPO Benchmark](https://github.com/marlbenchmark/on-policy)| [![GitHub stars](https://img.shields.io/github/stars/marlbenchmark/on-policy)](https://github.com/marlbenchmark/on-policy/stargazers)    |       4 cooperative       |      1     |          share + separate        |          MLP + GRU        |         :x:              |
| [MAlib](https://github.com/sjtu-marl/malib) | [![GitHub stars](https://img.shields.io/github/stars/sjtu-marl/malib)](https://github.com/hijkzzz/sjtu-marl/malib/stargazers) | 4 self-play  | 10 | share + group + separate | MLP + LSTM | [![Documentation Status](https://readthedocs.org/projects/malib/badge/?version=latest)](https://malib.readthedocs.io/en/latest/?badge=latest)
|    [EPyMARL](https://github.com/uoe-agents/epymarl)| [![GitHub stars](https://img.shields.io/github/stars/uoe-agents/epymarl)](https://github.com/hijkzzz/uoe-agents/epymarl/stargazers)         |       4 cooperative      |    9    |        share + separate       |      GRU             |           :x:            |
|    [MARLlib](https://github.com/Replicable-MARL/MARLlib)|  [![GitHub stars](https://img.shields.io/github/stars/Replicable-MARL/MARLlib)](https://github.com/Replicable-MARL/MARLlib/stargazers)  |       10 **no task mode restriction**     |    18     |   share + group + separate + **customizable**         |         MLP + CNN + GRU + LSTM          |           [![Documentation Status](https://readthedocs.org/projects/marllib/badge/?version=latest)](https://marllib.readthedocs.io/en/latest/) |


[comment]: <> (<div align="center">)

[comment]: <> (<img src=docs/source/images/overview.png width=100% />)

[comment]: <> (</div>)

## key features

:beginner: What **MARLlib** brings to MARL community:

- it unifies diverse algorithm pipelines with agent-level distributed dataflow.
- it supports all task modes: cooperative, collaborative, competitive, and mixed.
- it unifies multi-agent environment interfaces with a new interface following Gym.
- it provides flexible and customizable parameter-sharing strategies.

:rocket: With MARLlib, you can exploit the advantages not limited to:

- **zero knowledge of MARL**: out of the box 18 algorithms with intuitive API!
- **all task modes available**: support almost all multi-agent environment!
- **customizable model arch**: pick your favorite one from the model zoo!
- **customizable policy sharing**: grouped by MARLlib or build your own!
- more than a thousand experiments are conducted and released!

## Installation

> __Note__:
> MARLlib supports Linux only.

### Step-by-step  (recommended)

- install dependencies
- install environments
- install patches

#### 1. install dependencies (basic)

First, install MARLlib dependencies to guarantee basic usage.
following [this guide](https://marllib.readthedocs.io/en/latest/handbook/env.html), finally install patches for RLlib.

```bash
$ conda create -n marllib python=3.8
$ conda activate marllib
$ git clone https://github.com/Replicable-MARL/MARLlib.git && cd MARLlib
$ pip install -r requirements.txt
```

#### 2. install environments (optional)

Please follow [this guide](https://marllib.readthedocs.io/en/latest/handbook/env.html).

#### 3. install patches (basic)

Fix bugs of RLlib using patches by running the following command:

```bash
$ cd /Path/To/MARLlib/marl/patch
$ python add_patch.py -y
```

### PyPI

```bash
$ pip install --upgrade pip
$ pip install marllib
```

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
| encoder arch | ```marl.build_model(env, algo, {"core_arch": "gru", "encode_layer": "128-256"})```  |


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
env = marl.make_env(environment_name="mpe", map_name="simple_spread")
# initialize algorithm with appointed hyper-parameters
mappo = marl.algos.mappo(hyperparam_source="mpe")
# build agent model based on env + algorithms + user preference
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})
# start training
mappo.fit(env, model, stop={"timesteps_total": 1000000}, checkpoint_freq=100, share_policy="group")
# rendering(optional)
mappo.render(env, model, local_mode=True, restore_path='path_to_model')
```
</details>

## Benchmark results

All results are listed [here](https://github.com/Replicable-MARL/MARLlib/tree/main/results).

## Quick examples

MARLlib provides some practical examples for you to refer to.

- [Detailed API usage](https://github.com/Replicable-MARL/MARLlib/blob/sy_dev/examples/api_basic_usage.py): show how to use MARLlib api in
  detail, e.g. cmd + api combined running.
- [Policy sharing cutomization](https://github.com/Replicable-MARL/MARLlib/blob/sy_dev/examples/customize_policy_sharing.py):
  define your group policy-sharing strategy as you like based on current tasks.
- [Loading model and rendering](https://github.com/Replicable-MARL/MARLlib/blob/sy_dev/examples/load_and_render_model.py):
  render the environment based on the pre-trained model.
- [Incorporating new environment to MARLlib](https://github.com/Replicable-MARL/MARLlib/blob/sy_dev/examples/add_new_env.py):
  add your new environment following MARLlib's env-agent interaction interface.

## Tutorials

Try MPE + MAPPO examples on Google Colaboratory!
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Replicable-MARL/MARLlib/blob/sy_dev/marllib.ipynb)

More tutorial documentations are available [here](https://marllib.readthedocs.io/).

## Community

|  Channel   | Link |
| :----------- | :----------- |
| Issues | [GitHub Issues](https://github.com/Replicable-MARL/MARLlib/issues) |


## Contributing

We are a small team on multi-agent reinforcement learning, and we will take all the help we can get! 
If you would like to get involved, here is information on [contribution guidelines and how to test the code locally](https://github.com/Replicable-MARL/MARLlib/blob/sy_dev/CONTRIBUTING.md).

You can contribute in multiple ways, e.g., reporting bugs, writing or translating documentation, reviewing or refactoring code, requesting or implementing new features, etc.

[comment]: <> (## Paper)

[comment]: <> (If you use MARLlib in your research, please cite the [MARLlib paper]&#40;https://arxiv.org/abs/2210.13708&#41;.)

[comment]: <> (```tex)

[comment]: <> (@article{hu2022marllib,)

[comment]: <> (  title={MARLlib: Extending RLlib for Multi-agent Reinforcement Learning},)

[comment]: <> (  author={Hu, Siyi and Zhong, Yifan and Gao, Minquan and Wang, Weixun and Dong, Hao and Li, Zhihui and Liang, Xiaodan and Chang, Xiaojun and Yang, Yaodong},)

[comment]: <> (  journal={arXiv preprint arXiv:2210.13708},)

[comment]: <> (  year={2022})

[comment]: <> (})

[comment]: <> (```)

