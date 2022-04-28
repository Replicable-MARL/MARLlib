from IL.envs.mpe_rllib import RllibMPE
from IL.envs.mamujoco_rllib import RllibMAMujoco
from IL.envs.smac_rllib import RLlibSMAC
from IL.envs.football_rllib import RllibGFootball
from IL.envs.rware_rllib import RllibRWARE

ENV_REGISTRY = {
    "mpe": RllibMPE,
    "rware": RllibRWARE,
    "mamujoco": RllibMAMujoco,
    "smac": RLlibSMAC,
    "football": RllibGFootball
}
