# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

COOP_ENV_REGISTRY = {}

try:
    from marllib.envs.global_reward_env.gymnasium_mamujoco_fcoop import RLlibGymnasiumRoboticsMAMujoco_FCOOP
    COOP_ENV_REGISTRY["gymnasium_mamujoco"] = RLlibGymnasiumRoboticsMAMujoco_FCOOP

except Exception as e:
    COOP_ENV_REGISTRY["gymnasium_mamujoco"] = str(e)

try:
    from marllib.envs.global_reward_env.mpe_fcoop import RLlibMPE_FCOOP

    COOP_ENV_REGISTRY["mpe"] = RLlibMPE_FCOOP
except Exception as e:
    COOP_ENV_REGISTRY["mpe"] = str(e)

try:
    from marllib.envs.global_reward_env.gymnasium_mpe_fcoop import RLlibMPE_Gymnasium_FCOOP

    COOP_ENV_REGISTRY["gymnasium_mpe"] = RLlibMPE_Gymnasium_FCOOP
except Exception as e:
    COOP_ENV_REGISTRY["gymnasium_mpe"] = str(e)

try:
    from marllib.envs.global_reward_env.magent_fcoop import RLlibMAgent_FCOOP

    COOP_ENV_REGISTRY["magent"] = RLlibMAgent_FCOOP
except Exception as e:
    COOP_ENV_REGISTRY["magent"] = str(e)

try:
    from marllib.envs.global_reward_env.mamujoco_fcoop import RLlibMAMujoco_FCOOP

    COOP_ENV_REGISTRY["mamujoco"] = RLlibMAMujoco_FCOOP
except Exception as e:
    COOP_ENV_REGISTRY["mamujoco"] = str(e)

try:
    from marllib.envs.global_reward_env.smac_fcoop import RLlibSMAC_FCOOP

    COOP_ENV_REGISTRY["smac"] = RLlibSMAC_FCOOP
except Exception as e:
    COOP_ENV_REGISTRY["smac"] = str(e)

try:
    from marllib.envs.global_reward_env.football_fcoop import RLlibGFootball_FCOOP

    COOP_ENV_REGISTRY["football"] = RLlibGFootball_FCOOP
except Exception as e:
    COOP_ENV_REGISTRY["football"] = str(e)

try:
    from marllib.envs.global_reward_env.rware_fcoop import RLlibRWARE_FCOOP

    COOP_ENV_REGISTRY["rware"] = RLlibRWARE_FCOOP
except Exception as e:
    COOP_ENV_REGISTRY["rware"] = str(e)

try:
    from marllib.envs.global_reward_env.lbf_fcoop import RLlibLBF_FCOOP

    COOP_ENV_REGISTRY["lbf"] = RLlibLBF_FCOOP
except Exception as e:
    COOP_ENV_REGISTRY["lbf"] = str(e)

try:
    from marllib.envs.global_reward_env.pommerman_fcoop import RLlibPommerman_FCOOP

    COOP_ENV_REGISTRY["pommerman"] = RLlibPommerman_FCOOP
except Exception as e:
    COOP_ENV_REGISTRY["pommerman"] = str(e)

try:
    from marllib.envs.global_reward_env.mate_fcoop import RLlibMATE_FCOOP

    COOP_ENV_REGISTRY["mate"] = RLlibMATE_FCOOP
except Exception as e:
    COOP_ENV_REGISTRY["mate"] = str(e)

try:
    from marllib.envs.global_reward_env.gobigger_fcoop import RLlibGoBigger_FCOOP

    COOP_ENV_REGISTRY["gobigger"] = RLlibGoBigger_FCOOP
except Exception as e:
    COOP_ENV_REGISTRY["gobigger"] = str(e)

try:
    from marllib.envs.global_reward_env.overcooked_fcoop import RLlibOverCooked_FCOOP

    COOP_ENV_REGISTRY["overcooked"] = RLlibOverCooked_FCOOP
except Exception as e:
    COOP_ENV_REGISTRY["overcooked"] = str(e)

try:
    from marllib.envs.global_reward_env.voltage_fcoop import RLlibVoltageControl_FCOOP

    COOP_ENV_REGISTRY["voltage"] = RLlibVoltageControl_FCOOP
except Exception as e:
    COOP_ENV_REGISTRY["voltage"] = str(e)

try:
    from marllib.envs.global_reward_env.aircombat_fcoop import RLlibCloseAirCombatEnv_FCOOP

    COOP_ENV_REGISTRY["aircombat"] = RLlibCloseAirCombatEnv_FCOOP
except Exception as e:
    COOP_ENV_REGISTRY["aircombat"] = str(e)

try:
    from marllib.envs.global_reward_env.sisl_fcoop import RLlibSISL_FCOOP

    COOP_ENV_REGISTRY["sisl"] = RLlibSISL_FCOOP
except Exception as e:
    COOP_ENV_REGISTRY["sisl"] = str(e)
