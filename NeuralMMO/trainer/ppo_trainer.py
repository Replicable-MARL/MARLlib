''' copied and modified from Neural-MMO RLlib_Wrapper '''

import trueskill
import ray.rllib.agents.ppo.ppo as ppo
from NeuralMMO.model.utils.output import *


class Customized_PPOTrainer(ppo.PPOTrainer):
    def __init__(self, config, env=None, logger_creator=None):
        super().__init__(config, env, logger_creator)
        self.env_config = config['env_config']['config']

        # 1/sqrt(2)=76% win chance within beta, 95% win chance vs 3*beta=100 SR
        trueskill.setup(mu=1000, sigma=2 * 100 / 3, beta=100 / 3, tau=2 / 3, draw_probability=0)

        self.ratings = [{agent.__name__: trueskill.Rating(mu=1000, sigma=2 * 100 / 3)}
                        for agent in self.env_config.EVAL_AGENTS]

        self.reset_scripted()

    def reset_scripted(self):
        for rating_dict in self.ratings:
            for agent, rating in rating_dict.items():
                if agent == 'Combat':
                    rating_dict[agent] = trueskill.Rating(mu=1500, sigma=1)

    def post_mean(self, stats):
        for key, vals in stats.items():
            if type(vals) == list:
                stats[key] = np.mean(vals)

    def train(self):
        stats = super().train()
        self.post_mean(stats['custom_metrics'])
        return stats

    def evaluate(self):
        stat_dict = super().evaluate()
        stats = stat_dict['evaluation']['custom_metrics']

        ranks = {agent.__name__: -1 for agent in self.env_config.EVAL_AGENTS}
        for key in list(stats.keys()):
            if key.startswith('Rank_'):
                stat = stats[key]
                del stats[key]
                agent = key[5:]
                ranks[agent] = stat

        ranks = list(ranks.values())
        nEnvs = len(ranks[0])

        # Once RLlib adds better custom metric support,
        # there should be a cleaner way to divide episodes into blocks
        for i in range(nEnvs):
            env_ranks = [e[i] for e in ranks]
            self.ratings = trueskill.rate(self.ratings, env_ranks)
            self.reset_scripted()

        for rating in self.ratings:
            key = 'SR_{}'.format(list(rating.keys())[0])
            val = list(rating.values())[0]
            stats[key] = val.mu

        return stat_dict
