from marl.algos.scripts.vda2c import run_vda2c
from marl.algos.scripts.vdppo import run_vdppo
from marl.algos.scripts.vdn_qmix_iql import run_joint_q
from marl.algos.scripts.maa2c import run_maa2c
from marl.algos.scripts.mappo import run_mappo
from marl.algos.scripts.coma import run_coma
from marl.algos.scripts.pg_a2c_a3c import run_pg_a2c_a3c
from marl.algos.scripts.ppo import run_ppo
from marl.algos.scripts.ddpg import run_ddpg
from marl.algos.scripts.maddpg import run_maddpg


POlICY_REGISTRY = {
    "pg": run_pg_a2c_a3c,
    "a2c": run_pg_a2c_a3c,
    "a3c": run_pg_a2c_a3c,
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
    "maddpg": run_maddpg
}



