ENV_REGISTRY = {}

try:
    from envs.base_env.mpe import RllibMPE
    ENV_REGISTRY["mpe"] = RllibMPE
except:
    ENV_REGISTRY["mpe"] = False

try:
    from envs.base_env.mamujoco import RllibMAMujoco
    ENV_REGISTRY["mamujoco"] = RllibMAMujoco
except:
    ENV_REGISTRY["mamujoco"] = False

try:
    from envs.base_env.smac import RLlibSMAC
    ENV_REGISTRY["smac"] = RLlibSMAC
except:
    ENV_REGISTRY["smac"] = False

try:
    from envs.base_env.football import RllibGFootball
    ENV_REGISTRY["football"] = RllibGFootball
except:
    ENV_REGISTRY["football"] = False

try:
    from envs.base_env.magent import RllibMAgent
    ENV_REGISTRY["magent"] = RllibMAgent
except:
    ENV_REGISTRY["magent"] = False

try:
    from envs.base_env.rware import RllibRWARE
    ENV_REGISTRY["rware"] = RllibRWARE
except:
    ENV_REGISTRY["rware"] = False

try:
    from envs.base_env.lbf import RllibLBF
    ENV_REGISTRY["lbf"] = RllibLBF
except:
    ENV_REGISTRY["lbf"] = False

try:
    from envs.base_env.pommerman import RllibPommerman
    ENV_REGISTRY["pommerman"] = RllibPommerman
except:
    ENV_REGISTRY["pommerman"] = False

try:
    from envs.base_env.hanabi import RLlibHanabi
    ENV_REGISTRY["hanabi"] = RLlibHanabi
except:
    ENV_REGISTRY["hanabi"] = False

try:
    from envs.base_env.metadrive import RllibMetaDrive
    ENV_REGISTRY["metadrive"] = RllibMetaDrive
except:
    ENV_REGISTRY["metadrive"] = False

