COOP_ENV_REGISTRY = {}

try:
    from envs.global_reward_env.mpe_fcoop import RllibMPE_FCOOP
    COOP_ENV_REGISTRY["mpe"] = RllibMPE_FCOOP
except:
    COOP_ENV_REGISTRY["mpe"] = False

try:
    from envs.global_reward_env.mamujoco_fcoop import RllibMAMujoco_FCOOP
    COOP_ENV_REGISTRY["mamujoco"] = RllibMAMujoco_FCOOP
except:
    COOP_ENV_REGISTRY["mamujoco"] = False

try:
    from envs.global_reward_env.smac_fcoop import RLlibSMAC_FCOOP
    COOP_ENV_REGISTRY["smac"] = RLlibSMAC_FCOOP
except:
    COOP_ENV_REGISTRY["smac"] = False

try:
    from envs.global_reward_env.football_fcoop import RllibGFootball_FCOOP
    COOP_ENV_REGISTRY["football"] = RllibGFootball_FCOOP
except:
    COOP_ENV_REGISTRY["football"] = False

try:
    from envs.global_reward_env.rware_fcoop import RllibRWARE_FCOOP
    COOP_ENV_REGISTRY["rware"] = RllibRWARE_FCOOP
except:
    COOP_ENV_REGISTRY["rware"] = False

try:
    from envs.global_reward_env.lbf_fcoop import RllibLBF_FCOOP
    COOP_ENV_REGISTRY["lbf"] = RllibLBF_FCOOP
except:
    COOP_ENV_REGISTRY["lbf"] = False

try:
    from envs.global_reward_env.pommerman_fcoop import RllibPommerman
    COOP_ENV_REGISTRY["pommerman"] = RllibPommerman
except:
    COOP_ENV_REGISTRY["pommerman"] = False

