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
    from marllib.envs.base_env.mpe import RllibMPE
    ENV_REGISTRY["mpe"] = RllibMPE
except Exception as e:
    ENV_REGISTRY["mpe"] = str(e)

try:
    from marllib.envs.base_env.mamujoco import RllibMAMujoco
    ENV_REGISTRY["mamujoco"] = RllibMAMujoco
except Exception as e:
    ENV_REGISTRY["mamujoco"] = str(e)

try:
    from marllib.envs.base_env.smac import RLlibSMAC
    ENV_REGISTRY["smac"] = RLlibSMAC
except Exception as e:
    ENV_REGISTRY["smac"] = str(e)

try:
    from marllib.envs.base_env.football import RllibGFootball
    ENV_REGISTRY["football"] = RllibGFootball
except Exception as e:
    ENV_REGISTRY["football"] = str(e)

try:
    from marllib.envs.base_env.magent import RllibMAgent
    ENV_REGISTRY["magent"] = RllibMAgent
except Exception as e:
    ENV_REGISTRY["magent"] = str(e)

try:
    from marllib.envs.base_env.rware import RllibRWARE
    ENV_REGISTRY["rware"] = RllibRWARE
except Exception as e:
    ENV_REGISTRY["rware"] = str(e)

try:
    from marllib.envs.base_env.lbf import RllibLBF
    ENV_REGISTRY["lbf"] = RllibLBF
except Exception as e:
    ENV_REGISTRY["lbf"] = str(e)

try:
    from marllib.envs.base_env.pommerman import RllibPommerman
    ENV_REGISTRY["pommerman"] = RllibPommerman
except Exception as e:
    ENV_REGISTRY["pommerman"] = str(e)

try:
    from marllib.envs.base_env.hanabi import RLlibHanabi
    ENV_REGISTRY["hanabi"] = RLlibHanabi
except Exception as e:
    ENV_REGISTRY["hanabi"] = str(e)

try:
    from marllib.envs.base_env.metadrive import RllibMetaDrive
    ENV_REGISTRY["metadrive"] = RllibMetaDrive
except Exception as e:
    ENV_REGISTRY["metadrive"] = str(e)

