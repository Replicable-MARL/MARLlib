from .vda2c import run_vda2c
from .vdppo import run_vdppo
from .vdn_qmix_iql import run_joint_q
from .maa2c import run_maa2c
from .mappo import run_mappo
from .coma import run_coma
from .a2c import run_a2c
from .ppo import run_ppo
from .ddpg import run_ddpg
from .maddpg import run_maddpg
from .facmac import run_facmac
from .happo import run_happo
from .trpo import run_trpo
from .hatrpo import run_hatrpo
from .matrpo import run_matrpo


POlICY_REGISTRY = {
    "a2c": run_a2c,
    "ppo": run_ppo,
    "iql": run_joint_q,
    "qmix": run_joint_q,
    "vdn": run_joint_q,
    "vda2c": run_vda2c,
    "vdppo": run_vdppo,
    "maa2c": run_maa2c,
    "mappo": run_mappo,
    "coma": run_coma,
    "ddpg": run_ddpg,
    "maddpg": run_maddpg,
    "facmac": run_facmac,
    'happo': run_happo,
    'trpo': run_trpo,
    'hatrpo': run_hatrpo,
    'matrpo': run_matrpo
}



