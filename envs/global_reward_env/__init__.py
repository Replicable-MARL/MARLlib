from envs.global_reward_env.mpe_fcoop import RllibMPE_FCOOP
from envs.global_reward_env.mamujoco_fcoop import RllibMAMujoco_FCOOP
from envs.global_reward_env.smac_fcoop import RLlibSMAC_FCOOP
from envs.global_reward_env.football_fcoop import RllibGFootball_FCOOP
from envs.global_reward_env.rware_fcoop import RllibRWARE_FCOOP
from envs.global_reward_env.lbf_fcoop import RllibLBF_FCOOP
from envs.global_reward_env.pommerman_fcoop import RllibPommerman_FCOOP

# cooperative only scenarios
COOP_ENV_REGISTRY = {
    "mpe": RllibMPE_FCOOP,
    "rware": RllibRWARE_FCOOP,
    "mamujoco": RllibMAMujoco_FCOOP,
    "smac": RLlibSMAC_FCOOP,
    "football": RllibGFootball_FCOOP,
    "lbf": RllibLBF_FCOOP,
    "pommerman": RllibPommerman_FCOOP
}
