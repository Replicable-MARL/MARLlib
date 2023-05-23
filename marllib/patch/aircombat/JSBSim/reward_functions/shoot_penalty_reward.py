from .reward_function_base import BaseRewardFunction


class ShootPenaltyReward(BaseRewardFunction):
    """
    ShootPenaltyReward
    when launching a missile, give -10 reward for penalty, 
    to avoid launching all missiles at once 
    """
    def __init__(self, config):
        super().__init__(config)

    def reset(self, task, env):
        self.pre_remaining_missiles = {agent_id: agent.num_missiles for agent_id, agent in env.agents.items()}
        return super().reset(task, env)

    def get_reward(self, task, env, agent_id):
        """
        Reward is the sum of all the events.

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """
        reward = 0
        if task.remaining_missiles[agent_id] == self.pre_remaining_missiles[agent_id] - 1:
            reward -= 10
        self.pre_remaining_missiles[agent_id] = task.remaining_missiles[agent_id]
        return self._process(reward, agent_id)
