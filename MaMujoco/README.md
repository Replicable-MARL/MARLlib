# Multi-agent Mujoco in Ray
This is Multi-agent Mujoco baseline built for **ray[rllib]**

## Getting Started

Install Ray

> pip install ray==1.8.0 # version is important


Download Mujoco-200

https://roboti.us/download/mujoco200_linux.zip

extract it to 

> /home/YourUserName/.mujoco/

Note: you have to get you licence key of mujoco

Set env variable

> LD_LIBRARY_PATH=/home/YourUserName/.mujoco/mujoco200/bin;

> LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

Install Mujoco python api

> pip install mujoco-py==2.0.2.8 # version is important, has to >2.0 and <2.1

Set up multi-agent Mujoco according to https://github.com/schroederdewitt/multiagent_mujoco

Please annotate one line source code to avoid parallel env seed bug (some env need this)
at ray.rllib.evaluation.rollout_worker.py line 508

> _update_env_seed_if_necessary(self.env, seed, worker_index, 0)

Fix Multi-agent + RNN bug according to

https://github.com/ray-project/ray/pull/20743/commits/70861dae9398823814b5b28e3593cc98159c9c44

In **ray/rllib/policy/rnn_sequencing.py** about line 130-150

        for i, k in enumerate(feature_keys_):
            batch[k] = tree.unflatten_as(batch[k], feature_sequences[i])
        for i, k in enumerate(state_keys):
            batch[k] = initial_states[i]
        batch[SampleBatch.SEQ_LENS] = np.array(seq_lens)

        # add two lines here
        if dynamic_max:
            batch.max_seq_len = max(seq_lens)

        if log_once("rnn_ma_feed_dict"):
            logger.info("Padded input for RNN/Attn.Nets/MA:\n\n{}\n".format(
                summarize({
                    "features": feature_sequences,
                    "initial_states": initial_states,
                    "seq_lens": seq_lens,
                    "max_seq_len": max_seq_len,
                })))

## current support algo
- PG
- A2C
- A3C
- MAA2C
- DDPG 
- MADDPG   
- PPO
- MAPPO
- VDA2C-SUM/MIX 
- VDPPO-SUM/MIX 
  
### with neural arch
- GRU
- LSTM
- MLP (only for DDPG/MADDPG)

