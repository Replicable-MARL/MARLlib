from marl.algos.scripts.vda2c import run_vda2c
from marl.algos.scripts.vdppo import run_vdppo
from marl.algos.scripts.vdn_qmix_iql import run_joint_q
from marl.algos.scripts.maa2c import run_maa2c
from marl.algos.scripts.mappo import run_mappo
from marl.algos.scripts.coma import run_coma
from marl.algos.scripts.a2c import run_a2c
from marl.algos.scripts.pg import run_pg
from marl.algos.scripts.ppo import run_ppo
from marl.algos.scripts.ddpg import run_ddpg
from marl.algos.scripts.maddpg import run_maddpg
from marl.algos.scripts.facmac import run_facmac
from marl.algos.scripts.happo import run_happo
from marl.algos.scripts.trpo import run_trpo
from marl.algos.scripts.hatrpo import run_hatrpo
from marl.algos.scripts.matrpo import run_matrpo


POlICY_REGISTRY = {
    "pg": run_pg,
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



