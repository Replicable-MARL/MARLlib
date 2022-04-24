from IndependentLearning.envs.mpe_rllib import RllibMPE
from IndependentLearning.envs.mamujoco_rllib import RllibMAMujoco
from IndependentLearning.envs.smac_rllib import RLlibSMAC
from IndependentLearning.envs.football_rllib import RllibGFootball
from IndependentLearning.envs.rware_rllib import RllibRWARE

ENV_REGISTRY = {
    "mpe": RllibMPE,
    "rware": RllibRWARE,
    "mamujoco": RllibMAMujoco,
    "smac": RLlibSMAC,
    "football": RllibGFootball
}
