from IL.envs.mpe_rllib import RllibMPE
from IL.envs.mamujoco_rllib import RllibMAMujoco
from IL.envs.smac_rllib import RLlibSMAC
from IL.envs.football_rllib import RllibGFootball
from IL.envs.rware_rllib import RllibRWARE
from IL.envs.lbf_rllib import RllibLBF
from IL.envs.magent_rllib import RllibMAgent
from IL.envs.metadrive_rllib import RllibMetaDrive
from IL.envs.pommerman_rllib import RllibPommerman

ENV_REGISTRY = {
    "mpe": RllibMPE,
    "rware": RllibRWARE,
    "lbf": RllibLBF,
    "mamujoco": RllibMAMujoco,
    "smac": RLlibSMAC,
    "football": RllibGFootball,
    "magent": RllibMAgent,
    "metadrive": RllibMetaDrive,
    "pommerman": RllibPommerman
}
