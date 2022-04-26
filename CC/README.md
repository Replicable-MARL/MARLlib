### Part IV. Getting started

Install Ray 
```
pip install -U ray==1.8.0 # version important
```


clone our fork of Ray
```
git clone https://github.com/Theohhhu/ray_marl.git
git checkout marl_in_one
```

follow https://docs.ray.io/en/latest/ray-contribute/development.html#python-develop
```
cd ray_marl
python python/ray/setup-dev.py
```
**Y** to replace source-packages code with local one (replacing **rllib** is enough)

**Attention**: Above is the common installation. Follow the README file under each task's directory to meet the requirement before you run the code.

After everything you need is settled down, you can run the code. A typical run script can be like
```
CUDA_VISIBLE_DEVICES=1 python IL/run.py --algo_config=ppo --env-config=football with env_args.map_name=academy_3_vs_1_with_keeper
```

The basic structure of the repository. Here we take **[SMAC](HTTPS://GITHUB.COM/OXWHIRL/SMAC)** task as an example (name may be slightly different)

```
/
└───SMAC
        └───env     [**env compatible with RLLIB**]
                └───starcraft2_rllib.py
                
        └───model   [**agent architecture**]
                └───gru.py
                └───gru_cc.py
                
        └───util    [**algorithm module**]
                └───mappo_tools.py
                └───vda2c_tools.py
        
        └───policy  [**algorithm config**]
                └───mappo.py
                └───vda2c.py
                
        └───metrics [**logging**]
                └───callback.py
                └───reporter.py
                
        └───README.md
        └───run.py
        └───config.py 

```
