''' copied and modified from Neural-MMO RLlib_Wrapper '''

from collections import defaultdict

from tqdm import tqdm
import numpy as np
from ray.rllib import MultiAgentEnv

from neural_mmo.forge.trinity import Env as NeuralMMO
from neural_mmo.forge.trinity.dataframe import DataType
from neural_mmo.forge.trinity.overlay import Overlay, OverlayRegistry


###############################################################################
### RLlib Wrappers: Env, Overlays
class NeuralMMO_RLlib(NeuralMMO, MultiAgentEnv):
    def __init__(self, config):
        self.config = config['config']
        super().__init__(self.config)

    def reward(self, ent):
        config = self.config

        ACHIEVEMENT = config.REWARD_ACHIEVEMENT
        SCALE = config.ACHIEVEMENT_SCALE
        COOPERATIVE = config.COOPERATIVE

        individual = 0 if ent.entID in self.realm.players else -1
        team = 0

        if ACHIEVEMENT:
            individual += SCALE * ent.achievements.update(self.realm, ent, dry=True)
        if COOPERATIVE:
            nDead = len([p for p in self.dead.values() if p.population == ent.pop])
            team = -nDead / config.TEAM_SIZE
        if COOPERATIVE and ACHIEVEMENT:
            pre, post = [], []
            for p in self.realm.players.corporeal.values():
                if p.population == ent.pop:
                    pre.append(p.achievements.score(aggregate=False))
                    post.append(p.achievements.update(
                        self.realm, ent, aggregate=False, dry=True))

            pre = np.array(pre).max(0)
            post = np.array(post).max(0)
            team += SCALE * (post - pre).sum()

        ent.achievements.update(self.realm, ent)

        alpha = config.TEAM_SPIRIT
        return alpha * team + (1.0 - alpha) * individual

    def step(self, decisions, preprocess=None, omitDead=False):
        preprocess = {entID for entID in decisions}
        obs, rewards, dones, infos = super().step(decisions, preprocess, omitDead)

        config = self.config
        dones['__all__'] = False
        test = config.EVALUATE or config.RENDER

        if config.EVALUATE:
            horizon = config.EVALUATION_HORIZON
        else:
            horizon = config.TRAIN_HORIZON

        population = len(self.realm.players) == 0
        hit_horizon = self.realm.tick >= horizon

        if not config.RENDER and (hit_horizon or population):
            dones['__all__'] = True

        return obs, rewards, dones, infos
