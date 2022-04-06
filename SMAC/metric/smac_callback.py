from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from typing import Dict, Optional, TYPE_CHECKING

from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.evaluation import MultiAgentEpisode

from ray.rllib.utils.typing import AgentID, PolicyID

from queue import Queue

# Import psutil after ray so the packaged version is used.
import psutil

if TYPE_CHECKING:
    from ray.rllib.evaluation import RolloutWorker


class SmacCallbacks(DefaultCallbacks):

    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
        super().__init__(legacy_callbacks_dict)
        self.battle_win_queue = Queue(maxsize=100)
        self.ally_survive_queue = Queue(maxsize=100)
        self.enemy_killing_queue = Queue(maxsize=100)

    def on_episode_end(self,
                       *,
                       worker: "RolloutWorker",
                       base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode,
                       env_index: Optional[int] = None,
                       **kwargs) -> None:
        """Runs when an episode is done.

        Args:
            worker (RolloutWorker): Reference to the current rollout worker.
            base_env (BaseEnv): BaseEnv running the episode. The underlying
                env object can be gotten by calling base_env.get_unwrapped().
            policies (Dict[PolicyID, Policy]): Mapping of policy id to policy
                objects. In single agent mode there will only be a single
                "default_policy".
            episode (MultiAgentEpisode): Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            env_index (EnvID): Obsoleted: The ID of the environment, which the
                episode belongs to.
            kwargs: Forward compatibility placeholder.
        """

        if self.legacy_callbacks.get("on_episode_end"):
            self.legacy_callbacks["on_episode_end"]({
                "env": base_env,
                "policy": policies,
                "episode": episode,
            })

        if "GroupAgentsWrapper" in worker.env.__class__.__name__: # QMIX VDN
            ally_state = worker.env.env.env.death_tracker_ally
            enemy_state = worker.env.env.env.death_tracker_enemy
        else:
            ally_state = worker.env.env.death_tracker_ally
            enemy_state = worker.env.env.death_tracker_enemy

        # count battle win rate in recent 100 games
        if self.battle_win_queue.full():
            self.battle_win_queue.get()  # pop FIFO

        battle_win_this_episode = int(all(enemy_state == 1))  # all enemy died / win
        self.battle_win_queue.put(battle_win_this_episode)

        episode.custom_metrics["battle_win_rate"] = sum(self.battle_win_queue.queue) / self.battle_win_queue.qsize()

        # count ally survive in recent 100 games
        if self.ally_survive_queue.full():
            self.ally_survive_queue.get()  # pop FIFO

        ally_survive_this_episode = sum(ally_state == 0) / ally_state.shape[0]  # all enemy died / win
        self.ally_survive_queue.put(ally_survive_this_episode)

        episode.custom_metrics["ally_survive_rate"] = sum(
            self.ally_survive_queue.queue) / self.ally_survive_queue.qsize()

        # count enemy killing rate in recent 100 games
        if self.enemy_killing_queue.full():
            self.enemy_killing_queue.get()  # pop FIFO

        enemy_killing_this_episode = sum(enemy_state == 1) / enemy_state.shape[0]  # all enemy died / win
        self.enemy_killing_queue.put(enemy_killing_this_episode)

        episode.custom_metrics["enemy_kill_rate"] = sum(
            self.enemy_killing_queue.queue) / self.enemy_killing_queue.qsize()
