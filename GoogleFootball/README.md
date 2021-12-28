# Google Football in Ray

This is Google Football baseline built for **ray[rllib]**

## Getting Started

Install Ray
> pip install ray==1.8.0 # version is important

Install C++ compiler tools for Google Football
> sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
libdirectfb-dev libst-dev mesa-utils xvfb x11vnc python3-pip
>
> python3 -m pip install --upgrade pip setuptools psutil wheel
>

Install Google Football Simulator
> python3 -m pip install gfootball==2.10.1

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

- R2D2(IQL)
- VDN
- QMIX
- PG
- A2C
- A3C
- PPO

Note: Google Football is strictly MDP. Information is fully observed. 
Observation of different agents is almost same(except the current agent's location). 
Therefore, centralized critic algorithms (e.g. MAPPO) is not supported here.
As for joint Q learning algorithms like QMIX and VDN, the global reward is calculated
as the sum of the individual reward.

### with neural arch

- CNN
- CNN + GRU
- CNN + LSTM
- CNN + UPDeT

Note: although there is no partial observation setting in Google Football, 
We still provide the standard [**CNN + popular recurrent architecture**] here.



