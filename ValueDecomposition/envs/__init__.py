from ValueDecomposition.envs.mpe_rllib import RllibMPE
from ValueDecomposition.envs.mamujoco_rllib import RllibMAMujoco
from ValueDecomposition.envs.smac_rllib import RLlibSMAC
from ValueDecomposition.envs.football_rllib import RllibGFootball
from ValueDecomposition.envs.rware_rllib import RllibRWARE

# cooperative only scenarios
ENV_REGISTRY = {
    "mpe": RllibMPE,
    "rware": RllibRWARE,
    "mamujoco": RllibMAMujoco,
    "smac": RLlibSMAC,
    "football": RllibGFootball
}
