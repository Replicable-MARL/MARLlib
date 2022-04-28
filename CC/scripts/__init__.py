from CC.scripts.mappo import run_mappo
from CC.scripts.maa2c import run_maa2c
from CC.scripts.coma import run_coma

POlICY_REGISTRY = {
    "maa2c": run_maa2c,
    "mappo": run_mappo,
    "coma": run_coma
}
