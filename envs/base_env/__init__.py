from envs.base_env.mpe import RllibMPE
from envs.base_env.mamujoco import RllibMAMujoco
from envs.base_env.smac import RLlibSMAC
from envs.base_env.football import RllibGFootball
from envs.base_env.rware import RllibRWARE
from envs.base_env.lbf import RllibLBF
from envs.base_env.magent import RllibMAgent
# from EnvZoo.base_env.metadrive import RllibMetaDrive  # disable when not using
from envs.base_env.pommerman import RllibPommerman
from envs.base_env.hanabi import RLlibHanabi

ENV_REGISTRY = {
    "mpe": RllibMPE,
    "rware": RllibRWARE,
    "lbf": RllibLBF,
    "mamujoco": RllibMAMujoco,
    "smac": RLlibSMAC,
    "football": RllibGFootball,
    "magent": RllibMAgent,
    # "metadrive": RllibMetaDrive,  # disable when not using
    "pommerman": RllibPommerman,
    "hanabi": RLlibHanabi
}
