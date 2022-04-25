from CC.envs.mpe_rllib import RllibMPE
from CC.envs.mamujoco_rllib import RllibMAMujoco
from CC.envs.smac_rllib import RLlibSMAC
from CC.envs.football_rllib import RllibGFootball
from CC.envs.rware_rllib import RllibRWARE

# cooperative only scenarios
ENV_REGISTRY = {
    "mpe": RllibMPE,
    "rware": RllibRWARE,
    "mamujoco": RllibMAMujoco,
    "smac": RLlibSMAC,
    "football": RllibGFootball
}
