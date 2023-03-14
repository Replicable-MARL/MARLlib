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

from .vda2c import run_vda2c
from .vdppo import run_vdppo
from .vdn_qmix_iql import run_joint_q
from .maa2c import run_maa2c
from .mappo import run_mappo
from .coma import run_coma
from .ia2c import run_ia2c
from .ippo import run_ippo
from .iddpg import run_iddpg
from .maddpg import run_maddpg
from .facmac import run_facmac
from .happo import run_happo
from .itrpo import run_itrpo
from .hatrpo import run_hatrpo
from .matrpo import run_matrpo


POlICY_REGISTRY = {
    "ia2c": run_ia2c,
    "ippo": run_ippo,
    "iql": run_joint_q,
    "qmix": run_joint_q,
    "vdn": run_joint_q,
    "vda2c": run_vda2c,
    "vdppo": run_vdppo,
    "maa2c": run_maa2c,
    "mappo": run_mappo,
    "coma": run_coma,
    "iddpg": run_iddpg,
    "maddpg": run_maddpg,
    "facmac": run_facmac,
    'happo': run_happo,
    'itrpo': run_itrpo,
    'hatrpo': run_hatrpo,
    'matrpo': run_matrpo
}



