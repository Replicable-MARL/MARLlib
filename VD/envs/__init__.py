from VD.envs.mpe_rllib import RllibMPE
from VD.envs.mamujoco_rllib import RllibMAMujoco
from VD.envs.smac_rllib import RLlibSMAC
from VD.envs.football_rllib import RllibGFootball
from VD.envs.rware_rllib import RllibRWARE
from VD.envs.lbf_rllib import RllibLBF
from VD.envs.pommerman_rllib import RllibPommerman

# cooperative only scenarios
ENV_REGISTRY = {
    "mpe": RllibMPE,
    "rware": RllibRWARE,
    "mamujoco": RllibMAMujoco,
    "smac": RLlibSMAC,
    "football": RllibGFootball,
    "lbf": RllibLBF,
    "pommerman": RllibPommerman
}
