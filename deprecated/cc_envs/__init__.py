from CC.envs.mpe_rllib import RllibMPE
from CC.envs.mamujoco_rllib import RllibMAMujoco
from CC.envs.smac_rllib import RLlibSMAC
from CC.envs.football_rllib import RllibGFootball
from CC.envs.rware_rllib import RllibRWARE
from CC.envs.hanabi_rllib import RLlibHanabi
from CC.envs.magent_rllib import RllibMAgent
from CC.envs.pommerman_rllib import RllibPommerman
from CC.envs.lbf_rllib import RllibLBF
# cooperative only scenarios
ENV_REGISTRY = {
    "mpe": RllibMPE,
    "rware": RllibRWARE,
    "mamujoco": RllibMAMujoco,
    "smac": RLlibSMAC,
    "football": RllibGFootball,
    "pommerman": RllibPommerman,
    "magent": RllibMAgent,
    "hanabi": RLlibHanabi,
    "lbf": RllibLBF,
}
