from collections import defaultdict
import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks


###############################################################################
### Logging
class NMMO_Callbacks(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        assert len(base_env.envs) == 1, 'One env per worker'
        env = base_env.envs[0]

        logs = env.terminal()
        for key, vals in logs['Stats'].items():
            episode.custom_metrics[key] = np.mean(vals)

        if not env.config.EVALUATE:
            return

        agents = defaultdict(list)

        stats = logs['Stats']
        policy_ids = stats['PolicyID']
        scores = stats['Achievement']

        invMap = {agent.policyID: agent for agent in env.config.AGENTS}

        for policyID, score in zip(policy_ids, scores):
            policy = invMap[policyID]
            agents[policy].append(score)

        for agent in agents:
            agents[agent] = np.mean(agents[agent])

        policies = list(agents.keys())
        scores = list(agents.values())

        idxs = np.argsort(-np.array(scores))

        for rank, idx in enumerate(idxs):
            key = 'Rank_{}'.format(policies[idx].__name__)
            episode.custom_metrics[key] = rank
