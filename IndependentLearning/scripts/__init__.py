from IndependentLearning.scripts.pg_a2c_a3c import run_pg_a2c_a3c
from IndependentLearning.scripts.ppo import run_ppo

POlICY_REGISTRY = {
    "pg": run_pg_a2c_a3c,
    "a2c": run_pg_a2c_a3c,
    "a3c": run_pg_a2c_a3c,
    "ppo": run_ppo,
}
