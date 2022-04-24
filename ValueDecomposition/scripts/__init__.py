from ValueDecomposition.scripts.vda2c import run_vda2c
from ValueDecomposition.scripts.vdppo import run_vdppo
from ValueDecomposition.scripts.vdn_qmix_iql import run_joint_q

POlICY_REGISTRY = {
    "iql": run_joint_q,
    "qmix": run_joint_q,
    "vdn": run_joint_q,
    "vda2c": run_vda2c,
    "vdppo": run_vdppo
}
