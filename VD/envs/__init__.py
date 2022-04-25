from VD.envs.mpe_rllib import RllibMPE
from VD.envs.mamujoco_rllib import RllibMAMujoco
from VD.envs.smac_rllib import RLlibSMAC
from VD.envs.football_rllib import RllibGFootball
from VD.envs.rware_rllib import RllibRWARE

# cooperative only scenarios
ENV_REGISTRY = {
    "mpe": RllibMPE,
    "rware": RllibRWARE,
    "mamujoco": RllibMAMujoco,
    "smac": RLlibSMAC,
    "football": RllibGFootball
}
