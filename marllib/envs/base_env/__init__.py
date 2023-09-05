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

ENV_REGISTRY = {}

try:
    from marllib.envs.base_env.gymnasium_mamujoco import RLlibGymnasiumRoboticsMAMujoco
    ENV_REGISTRY["gymnasium_mamujoco"] = RLlibGymnasiumRoboticsMAMujoco

except Exception as e:
    ENV_REGISTRY["gymnasium_mamujoco"] = str(e)


try:
    from marllib.envs.base_env.mpe import RLlibMPE

    ENV_REGISTRY["mpe"] = RLlibMPE
except Exception as e:
    ENV_REGISTRY["mpe"] = str(e)


try:
    from marllib.envs.base_env.gymnasium_mpe import RLlibMPE_Gymnasium

    ENV_REGISTRY["gymnasium_mpe"] = RLlibMPE_Gymnasium
except Exception as e:
    ENV_REGISTRY["gymnasium_mpe"] = str(e)

try:
    from marllib.envs.base_env.mamujoco import RLlibMAMujoco

    ENV_REGISTRY["mamujoco"] = RLlibMAMujoco
except Exception as e:
    ENV_REGISTRY["mamujoco"] = str(e)




try:
    from marllib.envs.base_env.smac import RLlibSMAC

    ENV_REGISTRY["smac"] = RLlibSMAC
except Exception as e:
    ENV_REGISTRY["smac"] = str(e)

try:
    from marllib.envs.base_env.football import RLlibGFootball

    ENV_REGISTRY["football"] = RLlibGFootball
except Exception as e:
    ENV_REGISTRY["football"] = str(e)

try:
    from marllib.envs.base_env.magent import RLlibMAgent

    ENV_REGISTRY["magent"] = RLlibMAgent
except Exception as e:
    ENV_REGISTRY["magent"] = str(e)

try:
    from marllib.envs.base_env.rware import RLlibRWARE

    ENV_REGISTRY["rware"] = RLlibRWARE
except Exception as e:
    ENV_REGISTRY["rware"] = str(e)

try:
    from marllib.envs.base_env.lbf import RLlibLBF

    ENV_REGISTRY["lbf"] = RLlibLBF
except Exception as e:
    ENV_REGISTRY["lbf"] = str(e)

try:
    from marllib.envs.base_env.pommerman import RLlibPommerman

    ENV_REGISTRY["pommerman"] = RLlibPommerman
except Exception as e:
    ENV_REGISTRY["pommerman"] = str(e)

try:
    from marllib.envs.base_env.hanabi import RLlibHanabi

    ENV_REGISTRY["hanabi"] = RLlibHanabi
except Exception as e:
    ENV_REGISTRY["hanabi"] = str(e)

try:
    from marllib.envs.base_env.metadrive import RLlibMetaDrive

    ENV_REGISTRY["metadrive"] = RLlibMetaDrive
except Exception as e:
    ENV_REGISTRY["metadrive"] = str(e)

try:
    from marllib.envs.base_env.mate import RLlibMATE

    ENV_REGISTRY["mate"] = RLlibMATE
except Exception as e:
    ENV_REGISTRY["mate"] = str(e)

try:
    from marllib.envs.base_env.gobigger import RLlibGoBigger

    ENV_REGISTRY["gobigger"] = RLlibGoBigger
except Exception as e:
    ENV_REGISTRY["gobigger"] = str(e)

try:
    from marllib.envs.base_env.overcooked import RLlibOverCooked

    ENV_REGISTRY["overcooked"] = RLlibOverCooked
except Exception as e:
    ENV_REGISTRY["overcooked"] = str(e)

try:
    from marllib.envs.base_env.voltage import RLlibVoltageControl

    ENV_REGISTRY["voltage"] = RLlibVoltageControl
except Exception as e:
    ENV_REGISTRY["voltage"] = str(e)

try:
    from marllib.envs.base_env.aircombat import RLlibCloseAirCombatEnv

    ENV_REGISTRY["aircombat"] = RLlibCloseAirCombatEnv
except Exception as e:
    ENV_REGISTRY["aircombat"] = str(e)


try:
    from marllib.envs.base_env.hns import RLlibHideAndSeek

    ENV_REGISTRY["hns"] = RLlibHideAndSeek
except Exception as e:
    ENV_REGISTRY["hns"] = str(e)


try:
    from marllib.envs.base_env.sisl import RLlibSISL

    ENV_REGISTRY["sisl"] = RLlibSISL
except Exception as e:
    ENV_REGISTRY["sisl"] = str(e)