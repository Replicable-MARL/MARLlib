COOP_ENV_REGISTRY = {}

try:
    from envs.global_reward_env.mpe_fcoop import RllibMPE_FCOOP
    COOP_ENV_REGISTRY["mpe"] = RllibMPE_FCOOP
except Exception as e:
    COOP_ENV_REGISTRY["mpe"] = str(e)

try:
    from envs.global_reward_env.mamujoco_fcoop import RllibMAMujoco_FCOOP
    COOP_ENV_REGISTRY["mamujoco"] = RllibMAMujoco_FCOOP
except Exception as e:
    COOP_ENV_REGISTRY["mamujoco"] = str(e)

try:
    from envs.global_reward_env.smac_fcoop import RLlibSMAC_FCOOP
    COOP_ENV_REGISTRY["smac"] = RLlibSMAC_FCOOP
except Exception as e:
    COOP_ENV_REGISTRY["smac"] = str(e)

try:
    from envs.global_reward_env.football_fcoop import RllibGFootball_FCOOP
    COOP_ENV_REGISTRY["football"] = RllibGFootball_FCOOP
except Exception as e:
    COOP_ENV_REGISTRY["football"] = str(e)

try:
    from envs.global_reward_env.rware_fcoop import RllibRWARE_FCOOP
    COOP_ENV_REGISTRY["rware"] = RllibRWARE_FCOOP
except Exception as e:
    COOP_ENV_REGISTRY["rware"] = str(e)

try:
    from envs.global_reward_env.lbf_fcoop import RllibLBF_FCOOP
    COOP_ENV_REGISTRY["lbf"] = RllibLBF_FCOOP
except Exception as e:
    COOP_ENV_REGISTRY["lbf"] = str(e)

try:
    from envs.global_reward_env.pommerman_fcoop import RllibPommerman
    COOP_ENV_REGISTRY["pommerman"] = RllibPommerman
except Exception as e:
    COOP_ENV_REGISTRY["pommerman"] = str(e)

